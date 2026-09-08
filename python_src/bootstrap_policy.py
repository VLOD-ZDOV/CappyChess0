#!/usr/bin/env python3
"""Re-teach the policy head from a deep search, without touching the value head.

Why this exists: with too few sequential PUCT rounds the stored policy target is
nearly uniform, and a net trained on it un-learns move ranking — see
`docs/experiments/policy_target_collapse.md`. Fixing the self-play settings stops the
damage but does not undo it. This script does.

Two phases:

  collect  play games where EVERY move comes from a deep search (default 800
           simulations at parallel 2, which agrees with a 6400-simulation
           reference on ~53% of positions vs ~11% for the training-time
           search).  Recording the search that also picks the move costs
           nothing extra and keeps the position distribution sane.

  fit      minimise policy cross-entropy against those targets.  The trunk
           moves at a reduced learning rate and the value head is anchored to
           the original net by a KL term, so a policy repair cannot silently
           cost value strength.

    python bootstrap_policy.py checkpoints_v2/latest.pth out.pth \\
        --positions 30000 --sims 800 --parallel 2
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from capablanca_engine import CapablancaEngine
from model import build_net_from_state_dict, describe_arch
from mcts import UltraFastMCTS

PLANES, H, W = 139, 8, 10
POLICY_SIZE = 7000


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net, sd = build_net_from_state_dict(raw)
    net.load_state_dict(sd, strict=False)
    return net.to(device), ckpt


def collect(net, device, n_positions, games, sims, parallel, max_plies, seed):
    """Games played by the deep search; every visited position is a sample."""
    rng = np.random.default_rng(seed)
    mcts = UltraFastMCTS(net, device, batch_size=games, add_dirichlet=True,
                         parallel_sims=parallel)
    boards, idxs, vals = [], [], []
    t0 = time.time()
    while len(boards) < n_positions:
        engines = [CapablancaEngine() for _ in range(games)]
        for ply in range(max_plies):
            live = [i for i, e in enumerate(engines) if not e.is_game_over()]
            if not live or len(boards) >= n_positions:
                break
            tree = mcts.new_tree(engines)
            mcts.run_search(tree, sims)
            sparse = tree.get_policies_sparse()
            for i in live:
                pi, pv = sparse[i]
                if len(pi) == 0:
                    continue
                boards.append(np.asarray(engines[i].get_board_tensor(),
                                         dtype=np.float32).astype(np.float16))
                idxs.append(np.asarray(pi, dtype=np.int16))
                v = np.asarray(pv, dtype=np.float64)
                vals.append((v / v.sum()).astype(np.float16))
                # Sample early, argmax later: the target is recorded either way,
                # so exploration costs nothing but position diversity.
                legal = engines[i].get_legal_moves_int()
                look = {int(a): float(b) for a, b in zip(pi, pv)}
                w = np.array([look.get(engines[i].move_int_to_policy_idx(m), 0.0)
                              for m in legal])
                if w.sum() <= 0:
                    w = np.ones(len(legal))
                if ply < 20:
                    mv = int(legal[rng.choice(len(legal), p=w / w.sum())])
                else:
                    mv = int(legal[int(np.argmax(w))])
                engines[i].make_move_int(mv)
            done = len(boards)
            if done % 2000 < games:
                el = time.time() - t0
                print(f"  собрано {done:,}/{n_positions:,}  "
                      f"({done / max(el, 1e-9):.1f} поз/с, {el / 60:.1f} мин)",
                      flush=True)
    return boards[:n_positions], idxs[:n_positions], vals[:n_positions]


def pack(idxs, vals):
    """Ragged sparse targets → one dense (N, K) index/value pair."""
    k = max(len(x) for x in idxs)
    ii = np.zeros((len(idxs), k), dtype=np.int64)
    vv = np.zeros((len(idxs), k), dtype=np.float32)
    for n, (a, b) in enumerate(zip(idxs, vals)):
        ii[n, :len(a)] = a
        vv[n, :len(b)] = b
    return torch.from_numpy(ii), torch.from_numpy(vv)


def fit(net, ref, device, boards, ii, vv, epochs, batch, lr, trunk_lr_mult,
        anchor):
    head = [p for n, p in net.named_parameters()
            if n.startswith(("policy_head", "future_head"))]
    trunk = [p for n, p in net.named_parameters()
             if not n.startswith(("policy_head", "future_head"))]
    opt = torch.optim.AdamW([
        {"params": head, "lr": lr},
        {"params": trunk, "lr": lr * trunk_lr_mult},
    ], weight_decay=1e-4)
    n = len(boards)
    x_all = torch.from_numpy(np.stack(boards))
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot_p = tot_a = steps = 0.0
        for s in range(0, n - batch + 1, batch):
            sl = perm[s:s + batch]
            x = x_all[sl].to(device, non_blocking=True).float().view(-1, PLANES, H, W)
            ti, tv = ii[sl].to(device), vv[sl].to(device)
            logits, wdl, _, _ = net(x)
            logp = F.log_softmax(logits, dim=1)
            p_loss = -(tv * logp.gather(1, ti)).sum(dim=1).mean()
            if anchor > 0:
                with torch.no_grad():
                    ref_wdl = F.softmax(ref(x)[1], dim=1)
                a_loss = F.kl_div(F.log_softmax(wdl, dim=1), ref_wdl,
                                  reduction="batchmean")
            else:
                a_loss = torch.zeros((), device=device)
            opt.zero_grad(set_to_none=True)
            (p_loss + anchor * a_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot_p += p_loss.detach().item(); tot_a += float(a_loss); steps += 1
        print(f"  эпоха {ep + 1}/{epochs}: policy_loss={tot_p / steps:.4f} "
              f"value-якорь={tot_a / steps:.5f}", flush=True)


def report(net, device, boards, ii, vv, tag):
    """max(p)·n over the target's own support — 1.0 means uniform."""
    net.eval()
    x = torch.from_numpy(np.stack(boards[:512])).to(device).float().view(-1, PLANES, H, W)
    with torch.inference_mode():
        logits = net(x)[0].float()
    r, agree = [], []
    for i in range(x.shape[0]):
        k = int((vv[i] > 0).sum())
        idx = ii[i, :k].to(device)
        p = F.softmax(logits[i, idx], 0)
        r.append(float(p.max()) * k)
        agree.append(float(int(p.argmax()) == int(vv[i, :k].argmax())))
    print(f"  {tag}: max(p)·n = {np.mean(r):.2f} (равномерное = 1.0), "
          f"argmax совпал с целью {np.mean(agree):.1%}")
    net.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--positions", type=int, default=30000)
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--trunk-lr-mult", type=float, default=0.1,
                    help="LR ствола как доля от LR головы (0 = заморозить)")
    ap.add_argument("--anchor", type=float, default=1.0,
                    help="Вес KL-якоря value-головы к исходной сети (0 = выкл)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default=None,
                    help="Файл .npz для переиспользования собранных целей")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    net, ckpt = load(args.checkpoint, device)
    ref, _ = load(args.checkpoint, device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    print(f"сеть: {describe_arch(net)}")

    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache)
        boards = list(z["boards"])
        ii, vv = torch.from_numpy(z["ii"]).long(), torch.from_numpy(z["vv"]).float()
        print(f"цели загружены из {args.cache}: {len(boards):,} позиций")
    else:
        print(f"сбор целей: {args.sims} симуляций, parallel {args.parallel} "
              f"({args.sims // max(args.parallel, 1)} раундов PUCT)")
        net.eval()
        with torch.inference_mode():
            boards, idxs, vals = collect(net, device, args.positions, args.games,
                                         args.sims, args.parallel, args.max_plies,
                                         args.seed)
        ii, vv = pack(idxs, vals)
        if args.cache:
            np.savez(args.cache.replace(".npz", ""), boards=np.stack(boards),
                     ii=ii.numpy().astype(np.int32), vv=vv.numpy().astype(np.float16))
            print(f"цели сохранены в {args.cache}")

    print("\nдо дообучения:")
    report(net, device, boards, ii, vv, "исходная сеть")
    net.train()
    fit(net, ref, device, boards, ii, vv, args.epochs, args.batch, args.lr,
        args.trunk_lr_mult, args.anchor)
    print("\nпосле дообучения:")
    report(net, device, boards, ii, vv, "дообученная")

    out = dict(ckpt) if isinstance(ckpt, dict) else {}
    out["model"] = net.state_dict()
    out.pop("optimizer", None)      # LR-состояние больше не соответствует весам
    out.pop("scheduler", None)
    # EMA переписываем на новые веса, а НЕ оставляем и НЕ выбрасываем.
    #
    # Оставить: train.py делает `ema.apply_to(net)` перед self-play с
    # iteration >= ema_start_iter, то есть партии игрались бы старой сетью, и
    # вся починка не доехала бы до данных. Именно так и вышло в прогоне v4:
    # голова стала острее (max(p)*n 4.31 -> 6.93), а сила упала на 96 Elo,
    # потому что цели генерировала догоняющая EMA, отстающая от весов на 18%.
    #
    # Выбросить: ModelEMA создаётся из ЕЩЁ НЕ загруженной сети (train.py:1816
    # против 1951), и без ключа "ema" тень осталась бы случайной.
    #
    # ModelEMA.state_dict() — это просто shadow, поэтому формат совпадает.
    out["ema"] = {k: v.clone() for k, v in net.state_dict().items()}
    torch.save(out, args.out)
    print(f"\nсохранено: {args.out}")


if __name__ == "__main__":
    main()
