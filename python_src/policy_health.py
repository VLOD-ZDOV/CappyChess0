#!/usr/bin/env python3
"""Is the policy head still ranking moves, or has it flattened out again?

The failure this catches: when a search gets too few sequential PUCT rounds the
stored target goes uniform, and a net trained on it stops ranking moves — see
`docs/experiments/policy_target_collapse.md`. It is invisible in `policy_loss` alone,
because a flat policy sits at ln(legal moves) and that looks like a plateau.

Reports, over a fixed set of positions:

  max(p)*n   how much sharper than uniform the best move is. 1.0 = uniform,
             1.04 = a randomly initialised head, 3.5+ = healthy.
  illegal    probability mass the head leaks onto illegal moves.
  ln(n)-H(p) information the head carries above uniform, in nats.

Pass several checkpoints to see the trend:

    python policy_health.py checkpoints_v3/model_iter*.pth
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from capablanca_engine import CapablancaEngine
from model import build_net_from_state_dict

PLANES, H, W = 139, 8, 10


def positions(n, seed):
    """Random-opening positions — a fixed, net-independent yardstick."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        e = CapablancaEngine()
        for _ in range(int(rng.integers(6, 40))):
            lm = e.get_legal_moves_int()
            if e.is_game_over() or len(lm) == 0:
                break
            e.make_move_int(int(lm[rng.integers(len(lm))]))
        if not e.is_game_over():
            out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--positions", type=int, default=128)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    engines = positions(a.positions, a.seed)
    x = torch.from_numpy(np.stack([np.asarray(e.get_board_tensor(), dtype=np.float32)
                                   for e in engines])).to(device)
    x = x.reshape(len(engines), PLANES, H, W)
    legal = [[j for j in (e.move_int_to_policy_idx(m) for m in e.get_legal_moves_int())
              if j is not None] for e in engines]

    print(f"{len(engines)} позиций, в среднем {np.mean([len(l) for l in legal]):.1f} "
          f"легальных ходов\n")
    print(f"{'чекпоинт':<34} {'max(p)*n':>9} {'нелегальные':>12} {'ln(n)-H(p)':>11}")
    for path in a.checkpoints:
        try:
            ck = torch.load(path, map_location=device, weights_only=False)
            net, sd = build_net_from_state_dict(ck["model"] if "model" in ck else ck)
            net.load_state_dict(sd, strict=False)
            net = net.to(device).eval()
        except Exception as exc:                       # noqa: BLE001
            print(f"{path.split('/')[-1]:<34} не загрузился: {exc}")
            continue
        # Raw logits: net.inference() already returns a softmax, and softmaxing
        # it again flattens everything toward uniform.
        with torch.inference_mode():
            lg = net(x)[0].float()
        sharp, ill, deficit = [], [], []
        for i, L in enumerate(legal):
            row = lg[i]
            m = torch.zeros(row.shape[0], dtype=torch.bool, device=device)
            m[L] = True
            ill.append(float(F.softmax(row, 0)[~m].sum()))
            p = F.softmax(row[m], 0)
            sharp.append(float(p.max()) * len(L))
            deficit.append(np.log(len(L)) +
                           float((p * torch.log(p.clamp_min(1e-12))).sum()))
        flag = "" if np.mean(sharp) >= 2.0 else "  ⚠️ плоская"
        print(f"{path.split('/')[-1]:<34} {np.mean(sharp):>9.2f} "
              f"{np.mean(ill):>11.1%} {np.mean(deficit):>11.4f}{flag}")
        del net
        torch.cuda.empty_cache()

    print("\n1.00 = равномерное, 1.04 = случайная инициализация, 3.5+ = здоровая голова")


if __name__ == "__main__":
    main()
