#!/usr/bin/env python
"""Where does the self-play loop's non-forward wall-clock go? The real loop runs
at ~68% of pure-forward throughput; this attributes the rest. Mirrors
generate_games' inner loop with per-phase timers (CUDA-synced around inference).
"""
import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python_src"))
from mcts import UltraFastMCTS                  # noqa: E402
from model import build_net_from_state_dict     # noqa: E402
from capablanca_engine import CapablancaEngine, RustMCTS  # noqa: E402

T = {}  # phase -> seconds
def tic(): return time.perf_counter()
def add(k, t0): T[k] = T.get(k, 0.0) + (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../../python_src/checkpoints_big/latest.pth")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--psims", type=int, default=32)
    ap.add_argument("--moves", type=int, default=10)
    ap.add_argument("--compile", default="default")
    args = ap.parse_args()

    device = torch.device("cuda")
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    net, sd = build_net_from_state_dict(sd); net.load_state_dict(sd, strict=False)
    net = net.eval().cuda()
    cm = None if args.compile == "none" else args.compile
    mcts = UltraFastMCTS(net, device, c_puct=1.25, batch_size=args.games,
                         parallel_sims=args.psims, compile_mode=cm, bf16_weights=True)
    engines = [CapablancaEngine() for _ in range(args.games)]
    rust = RustMCTS(engines, args.psims)

    _parallel = args.psims
    steps = max(1, (args.sims + _parallel - 1) // _parallel)
    total_pos = 0
    n_infer = 0

    # warmup move (compile)
    for _step in range(steps):
        lm = rust.collect_leaves(args.sims)
        if lm.shape[0] == 0: break
        rp, rv, rd, rm = mcts._infer(lm)
        rust.apply_inference_buffered(
            np.ascontiguousarray(rp, np.float32), np.ascontiguousarray(rv, np.float32),
            np.ascontiguousarray(rd, np.float32), np.ascontiguousarray(rm, np.float32),
            rust.get_current_batch_counts())
    torch.cuda.synchronize()

    t_all = tic()
    for mv in range(args.moves):
        for _step in range(steps):
            t0 = tic(); lm = rust.collect_leaves(args.sims); add("1_collect_leaves", t0)
            if lm.shape[0] == 0: break
            total_pos += lm.shape[0]; n_infer += 1
            t0 = tic()
            rp, rv, rd, rm = mcts._infer(lm)
            torch.cuda.synchronize(); add("2_infer(fwd+softmax+D2H)", t0)
            t0 = tic()
            cnts = rust.get_current_batch_counts()
            rust.apply_inference_buffered(
                np.ascontiguousarray(rp, np.float32), np.ascontiguousarray(rv, np.float32),
                np.ascontiguousarray(rd, np.float32), np.ascontiguousarray(rm, np.float32),
                cnts)
            add("3_apply_inference", t0)
        t0 = tic(); sparse = rust.get_policies_sparse(); add("4_get_policies", t0)
        t0 = tic(); vals = rust.get_values(); _ = rust.get_draws(); add("5_get_values", t0)
        t0 = tic()
        for i, eng in enumerate(engines):
            if eng.is_game_over(): continue
            idx, val = sparse[i]
            if len(idx) == 0: continue
            mv_int = int(idx[int(np.argmax(val))])
            legal = eng.get_legal_moves_int()
            if legal:
                eng.make_move_int(mv_int if mv_int in legal else legal[0])
                rust.make_move(i, mv_int if mv_int in legal else legal[0])
        add("6_move_select+make_move", t0)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_all

    print(f"\ncompile={args.compile} games={args.games} sims={args.sims} "
          f"psims={args.psims} moves={args.moves}")
    print(f"total positions inferred: {total_pos:,}  ({n_infer} infer calls)")
    print(f"wall: {wall:.2f}s   throughput: {total_pos/wall:,.0f} pos/s\n")
    print(f"{'phase':32s} {'sec':>8} {'%wall':>7}")
    for k in sorted(T):
        print(f"{k:32s} {T[k]:8.2f} {T[k]/wall*100:6.1f}%")
    accounted = sum(T.values())
    print(f"{'(unaccounted: py loop/sync)':32s} {wall-accounted:8.2f} {(wall-accounted)/wall*100:6.1f}%")
    print(f"\nNON-forward overhead = {(wall - T.get('2_infer(fwd+softmax+D2H)',0))/wall*100:.0f}% of wall")


if __name__ == "__main__":
    main()
