#!/usr/bin/env python
"""Training step (fwd+bwd+opt) throughput: NCHW vs channels_last. train.py uses
net.to(channels_last) + boards.to(channels_last). channels_last hurt INFERENCE
~15% here (GroupNorm transposes); training adds a backward pass through the same
norms. If it hurts training too, dropping it is a free training speedup.
Mirrors train_epoch: bf16 autocast, WDL CE + policy CE + mlh + future, grad clip.
"""
import os, sys, copy, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python_src"))
from model import build_net_from_state_dict, CapablancaNet, POLICY_SIZE  # noqa: E402

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
C, H, W = CapablancaNet.INPUT_PLANES, CapablancaNet.BOARD_H, CapablancaNet.BOARD_W


def load_src(ckpt):
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    net, sd = build_net_from_state_dict(sd)
    net.load_state_dict(sd, strict=False)
    return net


def fake_batch(bs, device, channels_last):
    boards = torch.randn(bs, C, H, W, device=device)
    if channels_last:
        boards = boards.to(memory_format=torch.channels_last)
    # soft policy target (sparse-ish), WDL one-hot-ish, mlh, future
    pol = torch.rand(bs, POLICY_SIZE, device=device); pol /= pol.sum(1, keepdim=True)
    wdl = torch.rand(bs, 3, device=device); wdl /= wdl.sum(1, keepdim=True)
    mlh = torch.rand(bs, device=device)
    fut = torch.randint(0, POLICY_SIZE, (bs,), device=device)
    return boards, pol, wdl, mlh, fut


def bench(net, opt, bs, device, channels_last, iters=40, warmup=12):
    boards, pol, wdl, mlh, fut = fake_batch(bs, device, channels_last)
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, wdl_logits, mlh_raw, fut_logits = net(boards)
            logits = logits.float(); wdl_logits = wdl_logits.float()
            ploss = -(pol * F.log_softmax(logits, 1)).sum(1).mean()
            vloss = -(wdl * F.log_softmax(wdl_logits, 1)).sum(1).mean()
            mloss = F.mse_loss(torch.sigmoid(mlh_raw.float().squeeze(-1)), mlh)
            floss = F.cross_entropy(fut_logits.float(), fut)
            loss = ploss + vloss + 0.1 * mloss + 0.15 * floss
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return bs / dt, dt * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../../python_src/checkpoints_big/latest.pth")
    ap.add_argument("--batches", default="512,1024")
    args = ap.parse_args()
    device = torch.device("cuda")
    print(f"GPU {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    src = load_src(args.ckpt)
    print(f"net: {src.num_channels}ch×{src.num_res_blocks}bl {src.num_transformer_blocks}tb\n")

    print(f"{'batch':>6} | {'NCHW samp/s':>12} | {'chan_last samp/s':>16} | {'CL/NCHW':>8}")
    for bs in [int(b) for b in args.batches.split(",")]:
        net_n = copy.deepcopy(src).to(device).train()
        opt_n = torch.optim.SGD(net_n.parameters(), lr=2e-4)  # no optimizer state → less VRAM
        tn, _ = bench(net_n, opt_n, bs, device, False)
        del net_n, opt_n; torch.cuda.empty_cache()

        net_c = copy.deepcopy(src).to(device).to(memory_format=torch.channels_last).train()
        opt_c = torch.optim.SGD(net_c.parameters(), lr=2e-4)
        tc, _ = bench(net_c, opt_c, bs, device, True)
        del net_c, opt_c; torch.cuda.empty_cache()

        print(f"{bs:>6} | {tn:>12,.0f} | {tc:>16,.0f} | {tc/tn:>7.2f}x", flush=True)


if __name__ == "__main__":
    main()
