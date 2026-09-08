#!/usr/bin/env python
"""End-to-end self-play throughput: compile 'default' vs 'max-autotune', inside
the real UltraFastMCTS loop (Rust engine, dynamic batches, D2H). Main venv 3.14.
Honest number — guards against the forward-only-vs-real-loop gap that bit the
TRT experiment (1.85x clean -> 1.13x real).
"""
import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python_src"))
from mcts import UltraFastMCTS                     # noqa: E402
from model import build_net_from_state_dict        # noqa: E402
from capablanca_engine import CapablancaEngine      # noqa: E402


def make_counter(mcts):
    orig = mcts._infer
    st = {"pos": 0}
    def wrapped(t, hashes=None):
        st["pos"] += t.shape[0]
        return orig(t, hashes=hashes)
    mcts._infer = wrapped
    return st


def run(net, device, mode, games, sims, psims, moves):
    mcts = UltraFastMCTS(net, device, c_puct=1.25, batch_size=games,
                         add_dirichlet=True, parallel_sims=psims,
                         compile_mode=mode, bf16_weights=True)
    st = make_counter(mcts)
    engines = [CapablancaEngine() for _ in range(games)]
    mcts.search_games_with_values(engines, sims)   # warmup (compile)
    st["pos"] = 0
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(moves):
        pols, _ = mcts.search_games_with_values(engines, sims)
        for i, eng in enumerate(engines):
            if eng.is_game_over():
                continue
            mv = int(np.asarray(pols[i]).argmax())
            legal = eng.get_legal_moves_int()
            if legal:
                eng.make_move_int(mv if mv in legal else legal[0])
    torch.cuda.synchronize(); dt = time.time() - t0
    del mcts; torch.cuda.empty_cache()
    return dt, st["pos"], st["pos"] / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../../python_src/checkpoints_big/latest.pth")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--psims", type=int, default=32)
    ap.add_argument("--moves", type=int, default=12)
    args = ap.parse_args()

    device = torch.device("cuda")
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    net, sd = build_net_from_state_dict(sd)
    net.load_state_dict(sd, strict=False)
    net = net.eval().cuda()
    print(f"games={args.games} sims={args.sims} psims={args.psims} moves={args.moves}")

    res = {}
    for mode in ("default", "max-autotune"):
        print(f"\n[{mode}] warming + timing...", flush=True)
        dt, pos, tput = run(net, device, mode, args.games, args.sims,
                            args.psims, args.moves)
        res[mode] = tput
        print(f"  {mode:13s}: {dt:6.1f}s  {pos:,} pos  ->  {tput:,.0f} pos/s", flush=True)

    print(f"\nREAL self-play  max-autotune / default = "
          f"{res['max-autotune']/res['default']:.3f}x")


if __name__ == "__main__":
    main()
