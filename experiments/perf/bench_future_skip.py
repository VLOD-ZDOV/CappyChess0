#!/usr/bin/env python
"""The self-play path (mcts._infer_raw_nn) calls net.forward(), which always
computes the future_head (conv1x1 -> GN -> Mish -> Linear(2560,7000)) and then
DISCARDS it (`logits, values, mlh_raw, _ = out`). future is training-only.
Setting net.enable_future=False makes forward() skip it. Measure the free gain.

Also checks correctness: policy/value/mlh are bit-identical with future skipped.
"""
import os, sys, copy, argparse
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


@torch.no_grad()
def bench(net, x, iters=60, warmup=20):
    for _ in range(warmup):
        net(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); net(x); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return x.shape[0] / (float(np.median(ts)) / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../../python_src/checkpoints_big/latest.pth")
    ap.add_argument("--batches", default="512,1024,2048,4096")
    ap.add_argument("--compile", default="none")
    args = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    src = load_src(args.ckpt)
    has_future = any("future_head" in n for n, _ in src.named_parameters())
    print(f"net: {src.num_channels}ch×{src.num_res_blocks}bl {src.num_transformer_blocks}tb  "
          f"future_head present: {has_future}")

    net_full = copy.deepcopy(src).to(torch.bfloat16).cuda().eval()
    net_skip = copy.deepcopy(src).to(torch.bfloat16).cuda().eval()
    net_skip.enable_future = False   # forward() returns future=None, skips the head

    # correctness: policy/value/mlh identical
    x = torch.randn(256, C, H, W, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        of = net_full(x); osk = net_skip(x)
    pol_id = torch.equal(of[0], osk[0])
    val_id = torch.equal(of[1], osk[1])
    mlh_id = torch.equal(of[2], osk[2]) if of[2] is not None else True
    print(f"[correctness] policy identical={pol_id}  wdl identical={val_id}  "
          f"mlh identical={mlh_id}  future(full)={'tensor' if of[3] is not None else None}"
          f"/skip={'tensor' if osk[3] is not None else None}")

    cm = None if args.compile == "none" else args.compile
    if cm:
        net_full = torch.compile(net_full, mode=cm, dynamic=False)
        net_skip = torch.compile(net_skip, mode=cm, dynamic=False)
        print(f"(compiled mode={cm})")

    print(f"\n{'batch':>6} | {'full pos/s':>11} | {'skip-future pos/s':>17} | {'gain':>6}")
    for bs in [int(b) for b in args.batches.split(",")]:
        x = torch.randn(bs, C, H, W, dtype=torch.bfloat16, device="cuda")
        ff = bench(net_full, x)
        sf = bench(net_skip, x)
        print(f"{bs:>6} | {ff:>11,.0f} | {sf:>17,.0f} | {sf/ff:>5.2f}x", flush=True)


if __name__ == "__main__":
    main()
