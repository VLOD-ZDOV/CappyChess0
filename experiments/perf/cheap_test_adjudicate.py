#!/usr/bin/env python
"""Cheap test #2: does --adjudicate (with Q-gate) fire on the trained net, and
how much does it shorten games? Runs a small batch of real self-play with the
trained checkpoint, off vs on, comparing avg positions/game + adjudication rate.
Quantifies the compute upside directly (minutes, no full training).
"""
import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python_src"))
import train as T
from model import build_net_from_state_dict


def cfg_for(adj, qgate):
    c = T.Config()
    c.games_per_iter = 64
    c.simulations = 128
    c.fast_simulations = 64
    c.mcts_parallel_sims = 32
    c.mcts_batch = 64
    c.max_game_length = 200
    c.compile_inference = None
    c.adjudicate = adj
    c.adjudicate_q_gate = qgate
    return c


def run(net, device, adj, qgate):
    c = cfg_for(adj, qgate)
    t0 = time.time()
    with torch.inference_mode():
        samples = T.generate_games(net, c, device, iteration=999)  # high iter → real resign thresh
    dt = time.time() - t0
    return len(samples), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../../python_src/checkpoints_big/latest.pth")
    args = ap.parse_args()
    device = torch.device("cuda")
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    net, sd = build_net_from_state_dict(sd); net.load_state_dict(sd, strict=False)
    net = net.eval().cuda()

    print("Running 64-game self-play batches with the trained net...\n")
    configs = [("adjudicate OFF", False, 0.0),
               ("adjudicate ON, material-only (q-gate=0)", True, 0.0),
               ("adjudicate ON, Q-gate=0.5", True, 0.5)]
    base_pos = None
    for name, adj, qg in configs:
        npos, dt = run(net, device, adj, qg)
        tag = ""
        if base_pos is None:
            base_pos = npos
        else:
            tag = f"  ({npos/base_pos*100:.0f}% of baseline positions)"
        print(f"{name:42s}: {npos:6,} positions  in {dt:5.1f}s{tag}", flush=True)
    print("\nNote: fewer positions with adjudicate ON = games ended earlier = "
          "compute saved. Q-gate=0.5 should adjudicate fewer than material-only "
          "(rejects positions the net doesn't confirm as won).")


if __name__ == "__main__":
    main()
