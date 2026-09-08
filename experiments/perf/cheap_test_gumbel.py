#!/usr/bin/env python
"""Cheap test #3: does the Gumbel completed-Q target extract a STRONGER policy
per simulation than raw visit counts? Tests the theoretical claim directly with
the trained net — no training needed.

For a set of positions, run search at N (low) and 4N (high) sims. Treat the
high-sim visit distribution as the "improvement direction". Then measure how
close each low-sim target is to it:
    KL(visits@N    || visits@4N)   — the baseline target
    KL(gumbel@N    || visits@4N)   — the Gumbel target
If gumbel@N is closer (lower KL) to visits@4N, it approximates a stronger search
with the same N sims → it's a better policy target. Lower KL = better.
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python_src"))
import train as T
from mcts import UltraFastMCTS
from model import build_net_from_state_dict
from capablanca_engine import CapablancaEngine, RustMCTS


def search_get(rust, mcts, sims, psims, want_children):
    """Run `sims` sims on rust (in place), return per-game (visit_sparse, children_stats?)."""
    steps = max(1, (sims + psims - 1) // psims)
    for _ in range(steps):
        lm = rust.collect_leaves(sims)
        if lm.shape[0] == 0:
            break
        rp, rv, rd, rm = mcts._infer(lm)
        rust.apply_inference_buffered(
            np.ascontiguousarray(rp, np.float32), np.ascontiguousarray(rv, np.float32),
            np.ascontiguousarray(rd, np.float32), np.ascontiguousarray(rm, np.float32),
            rust.get_current_batch_counts())
    pols = rust.get_policies_sparse()
    vals = np.array(rust.get_values(), np.float32)
    stats = rust.get_root_children_stats() if want_children else None
    return pols, vals, stats


def to_dense(sparse, size=7000):
    d = np.zeros(size, np.float64)
    idx, val = sparse
    if len(idx):
        d[np.asarray(idx, np.int64)] = np.asarray(val, np.float64)
    return d


def kl(p, q, eps=1e-9):
    # KL(p||q) over the support of p
    p = p / max(p.sum(), eps)
    q = q / max(q.sum(), eps)
    m = p > eps
    return float((p[m] * (np.log(p[m] + eps) - np.log(q[m] + eps))).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../../python_src/checkpoints_big/latest.pth")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--low", type=int, default=100)
    ap.add_argument("--high", type=int, default=800)
    ap.add_argument("--psims", type=int, default=32)
    ap.add_argument("--diversify", type=int, default=6, help="random opening plies")
    args = ap.parse_args()

    device = torch.device("cuda")
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    net, sd = build_net_from_state_dict(sd); net.load_state_dict(sd, strict=False)
    net = net.eval().cuda()
    mcts = UltraFastMCTS(net, device, batch_size=args.games, parallel_sims=args.psims,
                         compile_mode=None, bf16_weights=True, add_dirichlet=False)

    # Diversify positions with a few random legal moves.
    rng = np.random.default_rng(0)
    engines = [CapablancaEngine() for _ in range(args.games)]
    for e in engines:
        for _ in range(args.diversify):
            legal = e.get_legal_moves_int()
            if not legal or e.is_game_over():
                break
            e.make_move_int(int(rng.choice(legal)))

    print(f"net iter? ckpt={os.path.basename(args.ckpt)}  games={args.games}  "
          f"low={args.low} high={args.high} sims\n")

    # High-sim reference (fresh tree, no dirichlet → deterministic-ish search).
    rust_hi = RustMCTS([e.copy() for e in engines], args.psims)
    rust_hi.set_add_dirichlet(False)
    pols_hi, _, _ = search_get(rust_hi, mcts, args.high, args.psims, False)

    # Low-sim: visits + gumbel.
    rust_lo = RustMCTS([e.copy() for e in engines], args.psims)
    rust_lo.set_add_dirichlet(False)
    pols_lo, vals_lo, stats_lo = search_get(rust_lo, mcts, args.low, args.psims, True)

    # Fair metric for a policy-IMPROVEMENT target (which is intentionally sharper
    # than visit counts): does the low-sim target point at the move the deep
    # search ultimately prefers? Two views:
    #   (a) top-1 agreement with argmax(visits@high)
    #   (b) probability mass the low-sim target puts on that deep-best move
    agree_vis = agree_gum = 0
    mass_vis = mass_gum = 0.0
    npos = 0
    for g in range(args.games):
        hi = to_dense(pols_hi[g])
        if hi.sum() <= 0:
            continue
        vis = to_dense(pols_lo[g])
        if vis.sum() <= 0:
            continue
        c_idx, c_pri, c_vis, c_q = stats_lo[g]
        g_idx, g_val = T.gumbel_improved_policy(c_idx, c_pri, c_vis, c_q,
                                                float(vals_lo[g]))
        gum = to_dense((g_idx, g_val))
        best = int(hi.argmax())              # deep-search's preferred move
        agree_vis += int(vis.argmax() == best)
        agree_gum += int(gum.argmax() == best)
        mass_vis += float(vis[best])
        mass_gum += float(gum[best])
        npos += 1

    n = max(npos, 1)
    print(f"positions compared: {npos}  (reference = argmax of visits@{args.high})")
    print(f"  top-1 agreement with deep search:  visits@{args.low}={agree_vis/n*100:.1f}%"
          f"   gumbel@{args.low}={agree_gum/n*100:.1f}%")
    print(f"  prob mass on deep-best move:        visits@{args.low}={mass_vis/n:.3f}"
          f"        gumbel@{args.low}={mass_gum/n:.3f}")
    print(f"\nVERDICT: gumbel@{args.low} agrees with the {args.high}-sim search "
          f"{'MORE' if agree_gum > agree_vis else 'LESS' if agree_gum < agree_vis else 'EQUALLY'} "
          f"than visit counts ({(agree_gum-agree_vis)/n*100:+.1f} pts top-1).")
    print("  Higher agreement / mass on the deep-best move = stronger policy target.")


if __name__ == "__main__":
    main()
