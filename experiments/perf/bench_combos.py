#!/usr/bin/env python
"""Follow-up: channels_last on the model HURT (measured ~0.9x). Two questions:
  1. Why? (profiler: is GroupNorm forcing NHWC<->NCHW transposes?)
  2. What's actually fastest? 4 layout combos (input × weights) in eager, plus
     torch.compile default vs the unexplored max-autotune.
Representative batch sizes only, to keep compile warmup bounded.
"""
import os, sys, copy, argparse
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

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


def make_net(src, weights_cl, compile_mode=None):
    net = copy.deepcopy(src).to(torch.bfloat16).cuda().eval()
    if weights_cl:
        net = net.to(memory_format=torch.channels_last)
    if compile_mode:
        net = torch.compile(net, mode=compile_mode, dynamic=False)
    return net


def make_input(bs, channels_last):
    x = torch.randn(bs, C, H, W, dtype=torch.bfloat16, device="cuda")
    return x.to(memory_format=torch.channels_last) if channels_last else x.contiguous()


@torch.no_grad()
def bench(net, x, iters=50, warmup=15):
    for _ in range(warmup):
        net(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); net(x); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return x.shape[0] / (float(np.median(ts)) / 1000.0)


@torch.no_grad()
def profile_layout(src, weights_cl, bs=2048):
    net = make_net(src, weights_cl)
    x = make_input(bs, True)
    for _ in range(10):
        net(x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(20):
            net(x)
        torch.cuda.synchronize()
    tag = "NHWC-weights" if weights_cl else "NCHW-weights"
    print(f"\n=== profiler ({tag}, bs{bs}, 20 iters) — top CUDA ops ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
    # count transpose/copy time as a fraction
    ka = prof.key_averages()
    total = sum(k.cuda_time_total for k in ka)
    copyt = sum(k.cuda_time_total for k in ka
                if any(s in k.key.lower() for s in
                       ("copy", "transpose", "contiguous", "permute", "to_copy")))
    print(f"  layout-shuffle (copy/transpose) share: {copyt/max(total,1)*100:.1f}%")
    del net, x; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../../python_src/checkpoints_big/latest.pth")
    ap.add_argument("--batches", default="512,2048")
    ap.add_argument("--do", default="all", help="combos | profile | compile | all")
    args = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    src = load_src(args.ckpt)
    print(f"net: {src.num_channels}ch × {src.num_res_blocks}bl {src.num_transformer_blocks}tb")
    batches = [int(b) for b in args.batches.split(",")]

    if args.do in ("combos", "all"):
        print("\n########## eager: 4 layout combos (pos/s) ##########")
        print(f"{'batch':>6} | {'NCHW-w/NCHW-in':>14} | {'NCHW-w/NHWC-in':>14} | "
              f"{'NHWC-w/NCHW-in':>14} | {'NHWC-w/NHWC-in':>14}")
        net_nchw = make_net(src, False)
        net_nhwc = make_net(src, True)
        for bs in batches:
            xc = make_input(bs, False)
            xl = make_input(bs, True)
            r = []
            for net, x in [(net_nchw, xc), (net_nchw, xl),
                           (net_nhwc, xc), (net_nhwc, xl)]:
                try: r.append(f"{bench(net, x):>14,.0f}")
                except Exception as ex: r.append(f"ERR:{str(ex)[:8]}")
            print(f"{bs:>6} | " + " | ".join(r), flush=True)
        del net_nchw, net_nhwc; torch.cuda.empty_cache()

    if args.do in ("profile", "all"):
        profile_layout(src, False)   # NCHW weights (current mcts.py)
        profile_layout(src, True)    # NHWC weights (the slower one) — show why

    if args.do in ("compile", "all"):
        print("\n########## torch.compile: default vs max-autotune ##########")
        print("(NCHW weights — the eager winner; NHWC input as mcts.py feeds)")
        base = make_net(src, False)
        baselines = {}
        for bs in batches:
            x = make_input(bs, True)
            baselines[bs] = bench(base, x)
        del base; torch.cuda.empty_cache()
        for cm in ("default", "max-autotune"):
            print(f"\n--- compile mode={cm} ---")
            try:
                net = make_net(src, False, cm)
            except Exception as ex:
                print(f"  build failed: {ex}"); continue
            print(f"{'batch':>6} | {'eager pos/s':>12} | {'compiled pos/s':>14} | {'speedup':>8}")
            for bs in batches:
                x = make_input(bs, True)
                try:
                    c = bench(net, x)
                    print(f"{bs:>6} | {baselines[bs]:>12,.0f} | {c:>14,.0f} | "
                          f"{c/baselines[bs]:>7.2f}x", flush=True)
                except Exception as ex:
                    print(f"{bs:>6} | compiled FAILED: {str(ex)[:60]}", flush=True)
            del net; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
