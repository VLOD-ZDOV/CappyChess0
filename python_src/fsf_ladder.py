#!/usr/bin/env python3
"""Where a checkpoint stands against Fairy-Stockfish, on a scale with rungs.

The obvious knob, `go nodes N`, barely weakens the engine: even at one node it
answers with a static evaluation plus quiescence search. Every network in this
project scored 0% against "FSF-1" regardless of its actual strength, which made
the measurement useless — there was no rung between a random mover and a full
engine. `Skill Level` (-20..20) and `UCI_LimitStrength` + `UCI_Elo` (500..2850)
are the real knobs.

    python fsf_ladder.py <weights.pth> --skill -5 -4 -3 -2
    python fsf_ladder.py <weights.pth> --elo 500 700 900 1100

Moves are picked by argmax. `train.generate_fsf_games` samples with tau=0.8,
which is right for collecting training data and wrong for measuring strength —
`--tau` restores sampling if you want it.

Skill Level below about -5 is not "plays weakly", it is "hangs pieces", so the
useful span for this project is -5..0. If the network sits inside a cliff on
that scale (the -4 to -3 step drops from 80% to 33%), a large Elo change barely
moves the 50% crossing — switch to --elo, which has finer steps.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T                                              # noqa: E402
from model import build_net_from_state_dict, describe_arch     # noqa: E402

FSF_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fairy-stockfish-largeboard_x86-64-bmi2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--skill", type=int, nargs="+", default=None,
                    help="Skill Level (-20..20); 0 = полная сила")
    ap.add_argument("--elo", type=int, nargs="+", default=None,
                    help="UCI_Elo (500..2850) — мельче ступени, чем у Skill")
    ap.add_argument("--nodes", type=int, nargs="+", default=[0, 1],
                    help="0 = случайный игрок. Ступеней силы почти не даёт.")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="0 = argmax (замер силы); >0 = сэмплирование")
    ap.add_argument("--fsf-path", default=FSF_DEFAULT)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.tau <= 0.0:
        import numpy as np

        def _argmax_move(eng, pol_lookup, legal, tau=None):
            raw = np.array([pol_lookup.get(eng.move_int_to_policy_idx(m), 0.0)
                            for m in legal], dtype=np.float64)
            if not np.isfinite(raw).all() or raw.max(initial=0.0) <= 0.0:
                return int(np.random.choice(legal))
            return int(legal[int(np.argmax(raw))])
        T._sample_move_from_policy = _argmax_move
    else:
        _orig = T._sample_move_from_policy
        T._sample_move_from_policy = lambda e, p, l, tau=a.tau: _orig(e, p, l, tau)

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    net, sd = build_net_from_state_dict(ck["model"] if "model" in ck else ck)
    net.load_state_dict(sd, strict=False)
    net = net.to(dev).eval()
    print(f"{a.checkpoint}\n{describe_arch(net)}\n"
          f"{a.sims} симуляций, parallel {a.parallel}, "
          f"{'argmax' if a.tau <= 0 else f'tau={a.tau}'}\n")

    cfg = T.Config()
    cfg.mcts_batch = a.games
    cfg.mcts_parallel_sims = a.parallel
    cfg.contempt = 0.0

    levels = ([("skill", v) for v in a.skill] if a.skill else
              [("elo", v) for v in a.elo] if a.elo else
              [("nodes", v) for v in a.nodes])

    print(f"{'соперник':>12} {'+W':>4} {'=D':>4} {'-L':>4} {'счёт':>8}")
    for kind, v in levels:
        cfg.fsf_skill = v if kind == "skill" else None
        cfg.fsf_elo = v if kind == "elo" else None
        nodes = v if kind == "nodes" else 1000
        with torch.inference_mode():
            _, w, d, l = T.generate_fsf_games(net, dev, cfg, a.games,
                                              a.fsf_path, nodes, a.sims)
        tot = w + d + l
        score = (w + 0.5 * d) / tot if tot else 0.0
        label = ("Random" if (kind == "nodes" and v == 0) else
                 f"FSF-{v}n" if kind == "nodes" else
                 f"skill {v}" if kind == "skill" else f"Elo {v}")
        print(f"{label:>12} {w:>4} {d:>4} {l:>4} {score:>7.1%}", flush=True)


if __name__ == "__main__":
    main()
