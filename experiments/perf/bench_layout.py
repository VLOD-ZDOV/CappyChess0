#!/usr/bin/env python
"""Does converting the self-play inference net to channels_last (NHWC) speed it
up? The hot loop in mcts.py feeds channels_last INPUT but never converts the
MODEL — unlike eval.py / play_fsf.py / train.py which all do
`net.to(memory_format=channels_last)`. This measures the gap on the real net.

Forward-only throughput (the part layout affects; PCIe/D2H is ~8% and constant
across layouts per the team's 2026-06-05 attribution). bf16 weights + bf16
channels_last input, exactly mirroring UltraFastMCTS._infer_raw_nn.

Run with the project venv from experiments/perf/.
"""
import os, sys, copy, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python_src"))
from model import build_net_from_state_dict, CapablancaNet  # noqa: E402

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

C, H, W = CapablancaNet.INPUT_PLANES, CapablancaNet.BOARD_H, CapablancaNet.BOARD_W


def load_src(ckpt):
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    net, sd = build_net_from_state_dict(sd)
    net.load_state_dict(sd, strict=False)
    return net.eval()


def make_net(src, channels_last, compile_mode):
    net = copy.deepcopy(src).to(torch.bfloat16).cuda().eval()
    if channels_last:
        net = net.to(memory_format=torch.channels_last)
    if compile_mode:
        net = torch.compile(net, mode=compile_mode, dynamic=False)
    return net


def make_input(bs):
    x = torch.randn(bs, C, H, W, dtype=torch.bfloat16, device="cuda")
    return x.to(memory_format=torch.channels_last)


@torch.no_grad()
def bench(net, x, iters=50, warmup=15):
    for _ in range(warmup):
        net(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        net(x)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))  # ms
    t = float(np.median(times))
    return t, x.shape[0] / (t / 1000.0)


@torch.no_grad()
def correctness(src, bs=512):
    """argmax-policy agreement and Q drift: NHWC vs NCHW weights, same input."""
    x = make_input(bs)
    n_nchw = make_net(src, False, None)
    n_nhwc = make_net(src, True, None)
    p0, w0 = n_nchw(x)[0].float(), n_nchw(x)[1].float()
    p1, w1 = n_nhwc(x)[0].float(), n_nhwc(x)[1].float()
    top1 = (p0.argmax(1) == p1.argmax(1)).float().mean().item()
    q0 = torch.softmax(w0, 1); q0 = (q0[:, 0] - q0[:, 2])
    q1 = torch.softmax(w1, 1); q1 = (q1[:, 0] - q1[:, 2])
    qmae = (q0 - q1).abs().mean().item()
    pmae = (torch.softmax(p0, 1) - torch.softmax(p1, 1)).abs().max().item()
    print(f"\n[correctness NHWC vs NCHW @bs{bs}]  top1-agree={top1*100:.2f}%  "
          f"Q-MAE={qmae:.2e}  policy-max-abs={pmae:.2e}")
    del n_nchw, n_nhwc
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../../python_src/checkpoints_big/latest.pth")
    ap.add_argument("--batches", default="256,512,1024,2048,4096")
    ap.add_argument("--compile", default="none",
                    help="none | default | max-autotune | all")
    args = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    src = load_src(args.ckpt)
    nparams = sum(p.numel() for p in src.parameters())
    print(f"net: {src.num_channels}ch × {src.num_res_blocks}bl  "
          f"{src.num_transformer_blocks}tb  {nparams/1e6:.1f}M params")

    batches = [int(b) for b in args.batches.split(",")]
    correctness(src)

    modes = ["none"] if args.compile == "none" else \
            (["none", "default", "max-autotune"] if args.compile == "all"
             else ["none", args.compile])

    for cm in modes:
        compile_mode = None if cm == "none" else cm
        label = "eager" if cm == "none" else f"compile:{cm}"
        print(f"\n========== {label} ==========")
        net_nchw = make_net(src, False, compile_mode)
        net_nhwc = make_net(src, True, compile_mode)
        print(f"{'batch':>6} | {'NCHW pos/s':>12} | {'NHWC pos/s':>12} | "
              f"{'NHWC/NCHW':>9}")
        for bs in batches:
            x = make_input(bs)
            try:
                _, t0 = bench(net_nchw, x)
                _, t1 = bench(net_nhwc, x)
                print(f"{bs:>6} | {t0:>12,.0f} | {t1:>12,.0f} | "
                      f"{t1/t0:>8.2f}x", flush=True)
            except Exception as ex:
                print(f"{bs:>6} | FAILED: {ex}", flush=True)
            del x
        del net_nchw, net_nhwc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
