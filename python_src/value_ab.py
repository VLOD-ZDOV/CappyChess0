"""Какой способ обучения меньше заставляет value-голову зубрить.

Оффлайн A/B без self-play. Берём чекпоинт, дообучаем его на старой части
буфера разными способами и меряем value CE на свежих позициях, которых этот
чекпоинт никогда не видел — они появились в буфере уже после него. Минуты на
вариант вместо часов на прогон. Разрыв «виденные / отложенные» — мера
заучивания, CE на отложенных — мера того, что сеть на самом деле выучила.

Позиции одной партии лежат в буфере подряд (генерация идёт циклом по партиям,
потом по ходам). Границу партии даёт скачок mlh вверх: «осталось ходов» внутри
партии убывает. Проверено на буфере v8: 3840 партий в 82 353 свежих строках
при ожидаемых 256 партий × 15 итераций.

Варианты задаются токенами:
    base      как в прогоне
    vw0.5     вес value-лосса 0.5
    k4        не больше 4 случайных позиций с партии
    wd1e-3    затухание весов 1e-3

    python value_ab.py checkpoints_v8/buffer.npz checkpoints_v8/model_iter00025.pth \\
        --holdout 82353 --variants base vw0.5 k4 wd1e-3
"""
import argparse
import time

import numpy as np
import torch

from model import build_net_from_state_dict, pick_state_dict, split_weights
from overfit_check import age_order, losses, slice_rows
from train import Config, ReplayBuffer, build_optimizer, train_epoch


def game_starts(mlh):
    """Индексы начала партий в последовательности строк, идущих по возрасту."""
    return np.concatenate([[0], np.flatnonzero(np.diff(mlh) > 1e-4) + 1])


def per_game_subsample(rows, mlh_of_rows, k, rng):
    """Не больше k случайных позиций с каждой партии."""
    starts = game_starts(mlh_of_rows)
    ends = np.concatenate([starts[1:], [len(rows)]])
    keep = []
    for a, b in zip(starts, ends):
        idx = np.arange(a, b)
        keep.append(idx if len(idx) <= k else rng.choice(idx, k, replace=False))
    return rows[np.sort(np.concatenate(keep))]


def rows_to_buffer(D, rows):
    """ReplayBuffer из выбранных строк — те же кортежи, что строит load_npz."""
    buf = ReplayBuffer(len(rows))
    samples = []
    for i in rows:
        mask = D["pol_idx"][i] >= 0
        samples.append((
            D["boards"][i],
            (D["pol_idx"][i][mask].astype(np.int16, copy=False),
             D["pol_val"][i][mask].astype(np.float16, copy=False)),
            float(D["values"][i]), float(D["mlhs"][i]), int(D["futures"][i]),
            float(D["draws"][i]),
        ))
    buf.push(samples)
    return buf


def load_net(path, dev):
    path, which = split_weights(path)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    net, sd = build_net_from_state_dict(pick_state_dict(raw, which))
    res = net.load_state_dict(sd, strict=False)
    assert not res.missing_keys and not res.unexpected_keys, res
    return net.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("buffer")
    ap.add_argument("checkpoint")
    ap.add_argument("--holdout", type=int, required=True,
                    help="сколько самых свежих строк чекпоинт не видел")
    ap.add_argument("--variants", nargs="+", default=["base"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1.6e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")

    t0 = time.time()
    import buffer_io
    D = buffer_io.load_arrays(a.buffer)     # файл или каталог кусков, поля уже в памяти
    order, how = age_order(D)
    hold, train_rows = order[-a.holdout:], order[:-a.holdout]
    rng = np.random.default_rng(a.seed)
    hold_eval = slice_rows(D, np.sort(rng.choice(hold, min(a.eval_n, len(hold)), replace=False)))
    print(f"буфер {len(order):,} ({how}), загрузка {time.time() - t0:.0f} с")
    print(f"обучение на {len(train_rows):,} старых строках, отложено {len(hold):,} свежих; "
          f"оценка по {a.eval_n} позиций\n")

    final = {}
    for v in a.variants:
        cfg = Config()
        cfg.batch_size = 512
        cfg.learning_rate = a.lr
        cfg.log_every = 10 ** 9
        rows = train_rows
        if v.startswith("vw"):
            cfg.value_loss_weight = float(v[2:])
        elif v.startswith("wd"):
            cfg.weight_decay = float(v[2:])
        elif v.startswith("k"):
            rows = per_game_subsample(train_rows, D["mlhs"][train_rows].astype(np.float32),
                                      int(v[1:]), np.random.default_rng(a.seed))
        elif v != "base":
            raise SystemExit(f"неизвестный вариант {v}")

        vrng = np.random.default_rng(a.seed + 1)
        seen_eval = slice_rows(D, np.sort(vrng.choice(rows, min(a.eval_n, len(rows)), replace=False)))
        buf = rows_to_buffer(D, rows)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        net = load_net(a.checkpoint, dev)
        opt = build_optimizer(net, cfg, dev)

        print(f"── {v}: {len(rows):,} строк, value×{cfg.value_loss_weight:g}, wd {cfg.weight_decay:g}")
        print(f"   {'шаг':>5} {'value отлож.':>13} {'value видел':>12} {'разрыв':>7} {'policy отлож.':>14}")

        def report(step):
            net.eval()
            ph, vh = losses(net, hold_eval, dev)
            _, vs = losses(net, seen_eval, dev)
            print(f"   {step:>5} {vh:>13.3f} {vs:>12.3f} {vh / max(vs, 1e-6):>6.2f}x {ph:>14.3f}", flush=True)
            return vh, vs, ph

        last = report(0)
        cfg.train_steps = a.eval_every
        cfg.min_train_steps = a.eval_every
        for s in range(a.eval_every, a.steps + 1, a.eval_every):
            train_epoch(net, opt, buf, cfg, dev, 0)
            last = report(s)
        final[v] = last
        del net, opt, buf
        torch.cuda.empty_cache()
        print()

    print("ИТОГ после", a.steps, "шагов (меньше value отлож. — лучше):")
    for v, (vh, vs, ph) in final.items():
        print(f"   {v:<8} value отлож. {vh:.3f}   разрыв {vh / max(vs, 1e-6):.2f}x   policy отлож. {ph:.3f}")


if __name__ == "__main__":
    main()
