#!/usr/bin/env python3
"""game_stats.py — Статистика партий без обучения и без буфера.

Пример:
    python game_stats.py checkpoints/latest.pth --games 256 --simulations 128
    python game_stats.py checkpoints/model_iter00025.pth --games 128 --simulations 64 --no-resign
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from model import CapablancaNet
from mcts import UltraFastMCTS


def detect_arch(sd: dict) -> tuple[int, int]:
    """Определяет (num_channels, num_res_blocks) из state_dict без загрузки модели."""
    stem = sd.get("input_conv.net.0.weight")
    if stem is None:
        stem = sd.get("input_block.0.weight")
    if stem is None:
        raise ValueError("Не удалось найти input_conv в state_dict — неизвестный формат чекпоинта")
    ch = stem.shape[0]
    bl = sum(1 for k in sd if "res_blocks" in k and k.endswith("conv1.weight"))
    return ch, bl

try:
    from capablanca_engine import CapablancaEngine
    from capablanca_engine import RustMCTS as _RustMCTS
except ImportError:
    raise ImportError("capablanca_engine not found. Build: maturin develop --release")


# ── Game runner ───────────────────────────────────────────────────────────────

def run_games(net: nn.Module, device: torch.device, args: argparse.Namespace):
    """Прогоняет args.games партий, возвращает list[dict] с per-game статистикой.

    Каждый dict:
        moves   : int   — сколько полуходов сделано
        reason  : str   — "white" | "black" | "draw" | "adj" | "resign" | "timeout"
        result  : float — 1=белые победили, -1=чёрные, 0=ничья
    """
    mcts = UltraFastMCTS(net, device, c_puct=args.c_puct,
                          batch_size=args.mcts_batch,
                          parallel_sims=args.parallel_sims)

    records = []
    batch_sz  = args.mcts_batch
    num_batches = (args.games + batch_sz - 1) // batch_sz

    for b in range(num_batches):
        n = min(batch_sz, args.games - b * batch_sz)
        engines       = [CapablancaEngine() for _ in range(n)]
        move_counts   = [0] * n
        # Счётчики на КАЖДУЮ сторону — v чередует знак ply-to-ply.
        resign_counts = [[0, 0] for _ in range(n)]
        resigned      = [False] * n
        adjudicated   = [None] * n

        active   = list(range(n))
        move_num = 0

        rust_mcts = _RustMCTS(engines, args.parallel_sims)

        while active and move_num < args.max_game_length:
            steps = max(1, (args.simulations + args.parallel_sims - 1) // args.parallel_sims)
            for _ in range(steps):
                lm = rust_mcts.collect_leaves(args.simulations)
                if lm.shape[0] == 0:
                    break
                lh = rust_mcts.get_leaf_hashes() if mcts.nn_cache_enabled else None
                rp, rv, rd, rm = mcts._infer(lm, hashes=lh)
                rust_mcts.apply_inference_buffered(
                    np.ascontiguousarray(rp, dtype=np.float32),
                    np.ascontiguousarray(rv, dtype=np.float32),
                    np.ascontiguousarray(rd, dtype=np.float32),
                    np.ascontiguousarray(rm, dtype=np.float32),
                    rust_mcts.get_current_batch_counts(),
                )

            raw_pols  = rust_mcts.get_policies()
            raw_vals  = rust_mcts.get_values()
            policies  = [np.array(p, dtype=np.float32) for p in raw_pols]
            values_np = np.array(raw_vals, dtype=np.float32)

            new_active = []
            for j, game_idx in enumerate(active):
                eng   = engines[game_idx]
                legal = eng.get_legal_moves_int()
                if not legal:
                    continue

                # get_policies/get_values возвращают по записи на КАЖДУЮ игру,
                # индексировать по game_idx (не j), иначе после первого завершения
                # игры в батче остальные получают чужие policy/value.
                pol = policies[game_idx]
                side = eng.side_to_move()

                # Выбор хода с температурой
                raw = np.array([pol[eng.move_int_to_policy_idx(m) or 0]
                                 for m in legal], dtype=np.float64)
                if args.temperature < 0.01:
                    move = int(legal[int(np.argmax(raw))])
                else:
                    raw   = np.power(np.maximum(raw, 1e-8), 1.0 / args.temperature)
                    probs = raw / raw.sum()
                    move  = int(np.random.choice(legal, p=probs))

                eng.make_move_int(move)
                rust_mcts.make_move(game_idx, move)
                move_counts[game_idx] += 1

                if eng.is_game_over():
                    continue

                # Сдача (если не отключена)
                if not args.no_resign and move_num >= args.resign_min_move:
                    v = float(values_np[game_idx]) if game_idx < len(values_np) else 0.0
                    if v < args.resign_threshold:
                        resign_counts[game_idx][side] += 1
                    else:
                        resign_counts[game_idx][side] = 0
                    if resign_counts[game_idx][side] >= args.resign_consec:
                        resigned[game_idx] = True
                        continue

                adj = eng.adjudication_result()
                if adj is not None:
                    adjudicated[game_idx] = adj
                else:
                    new_active.append(game_idx)

            active   = new_active
            move_num += 1

        # Собираем результаты батча
        for i, eng in enumerate(engines):
            mc = move_counts[i]
            if resigned[i]:
                last_side = eng.side_to_move()
                # last_side = сторона ПОСЛЕ последнего хода (противник сдавшегося).
                # last_side==0 (белые ходят) → сдались чёрные → white wins.
                result = 1.0 if last_side == 0 else -1.0
                reason = "resign"
            elif adjudicated[i] is not None:
                result = float(adjudicated[i])
                reason = "adj"
            elif eng.is_game_over():
                result = float(eng.game_result())
                if result > 0.5:    reason = "white"
                elif result < -0.5: reason = "black"
                else:               reason = "draw"
            else:
                result = 0.0 if args.timeout_as_draw else float(eng.material_result())
                reason = "timeout"

            records.append({"moves": mc, "reason": reason, "result": result})

        done = len(records)
        pct = done / args.games * 100
        print(f"  [{pct:5.1f}%] batch {b+1}/{num_batches}: "
              f"{n} игр | всего {done}/{args.games}", flush=True)

    return records


# ── Вывод ─────────────────────────────────────────────────────────────────────

def bar(frac: float, width: int = 30) -> str:
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def print_stats(records: list, max_game_length: int):
    n     = len(records)
    moves = np.array([r["moves"] for r in records], dtype=np.int32)

    reason_counts = defaultdict(int)
    for r in records:
        reason_counts[r["reason"]] += 1

    print(f"\n{'═'*62}")
    print(f"  СТАТИСТИКА: {n} партий")
    print(f"{'═'*62}")

    # ── Исходы ────────────────────────────────────────────────────────────────
    print("\n  ИСХОДЫ:")
    order  = ["white", "black", "draw", "adj", "resign", "timeout"]
    labels = {
        "white":   "Победа белых  ",
        "black":   "Победа чёрных ",
        "draw":    "Ничья         ",
        "adj":     "Adjudication  ",
        "resign":  "Сдача         ",
        "timeout": "Таймаут       ",
    }
    for key in order:
        cnt = reason_counts[key]
        if cnt == 0:
            continue
        pct = cnt / n * 100
        print(f"    {labels[key]}: {cnt:>4} ({pct:5.1f}%)  {bar(pct/100, 25)}")

    # ── Гистограмма длин ──────────────────────────────────────────────────────
    print(f"\n  ДЛИНА ПАРТИЙ (полуходов, N={n}):")
    step       = 10
    max_bin    = ((max_game_length + step) // step) * step
    bin_edges  = list(range(0, max_bin + step, step))
    counts, _  = np.histogram(moves, bins=bin_edges)
    max_cnt    = max(counts) if counts.max() > 0 else 1

    for lo, hi, cnt in zip(bin_edges[:-1], bin_edges[1:], counts):
        pct     = cnt / n * 100
        bw      = int(cnt / max_cnt * 32)
        suffix  = "  ← timeout" if lo >= max_game_length else ""
        print(f"    {lo:>3}–{hi:<4} [{cnt:>4}]  "
              f"{'█' * bw + '░' * (32 - bw)}  {pct:5.1f}%{suffix}")

    # ── Перцентили ────────────────────────────────────────────────────────────
    p10, p25, p50, p75, p90 = (int(np.percentile(moves, p)) for p in (10, 25, 50, 75, 90))
    print(f"\n  Мин={moves.min()}  "
          f"P10={p10}  P25={p25}  "
          f"Медиана={p50}  "
          f"P75={p75}  P90={p90}  "
          f"Макс={moves.max()}")
    print(f"  Среднее={moves.mean():.1f}  Станд.откл={moves.std():.1f}")

    # ── Средняя длина по исходу ───────────────────────────────────────────────
    print(f"\n  СРЕДНЯЯ ДЛИНА ПО ИСХОДУ:")
    for key in order:
        subset = [r["moves"] for r in records if r["reason"] == key]
        if subset:
            arr = np.array(subset)
            print(f"    {labels[key]}: {arr.mean():6.1f} ± {arr.std():4.1f}  "
                  f"(мин={arr.min()} макс={arr.max()})")

    print(f"\n{'═'*62}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Статистика партий Capablanca Chess без обучения",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("checkpoint",
                        help="Путь к .pth чекпоинту модели (или 'latest' — "
                             "ищет latest.pth в --checkpoint-dir)")

    # Модель (по умолчанию авто-определение из чекпоинта)
    parser.add_argument("--channels",       type=int,   default=None,
                        help="Каналы модели (авто если не указано)")
    parser.add_argument("--res-blocks",     type=int,   default=None,
                        help="Residual блоков (авто если не указано)")
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints",
                        help="Директория чекпоинтов (для 'latest')")
    parser.add_argument("--use-ema",        action="store_true", default=True,
                        help="Использовать EMA веса если есть (default: True)")
    parser.add_argument("--no-ema",         dest="use_ema", action="store_false")

    # Игровые параметры
    parser.add_argument("--games",           type=int,   default=256)
    parser.add_argument("--simulations",     type=int,   default=128)
    parser.add_argument("--mcts-batch",      type=int,   default=64,
                        help="Параллельных партий в батче")
    parser.add_argument("--parallel-sims",   type=int,   default=64,
                        help="Листьев MCTS за один шаг")
    parser.add_argument("--max-game-length", type=int,   default=110)
    parser.add_argument("--temperature",     type=float, default=0.5,
                        help="Температура выбора хода (0=жадный argmax)")
    parser.add_argument("--c-puct",          type=float, default=1.25)

    # Resign
    parser.add_argument("--no-resign",         action="store_true",
                        help="Отключить сдачу (партии всегда до конца/таймаута)")
    parser.add_argument("--resign-threshold",  type=float, default=-0.95)
    parser.add_argument("--resign-consec",     type=int,   default=3)
    parser.add_argument("--resign-min-move",   type=int,   default=20)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--timeout-as-draw", action="store_true", default=False,
                        help="Таймаут = ничья (0.0) вместо оценки по материалу")
    args = parser.parse_args()

    # Резолвим путь к чекпоинту
    ckpt_path = args.checkpoint
    if ckpt_path == "latest":
        ckpt_path = os.path.join(args.checkpoint_dir, "latest.pth")
    if not os.path.exists(ckpt_path):
        print(f"❌ Чекпоинт не найден: {ckpt_path}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and args.device == "cuda":
        print("⚠️  CUDA не найдена, используется CPU")

    # Загрузка чекпоинта
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    iteration = ckpt.get("iteration", "?")

    raw_sd = ckpt.get("model", ckpt)
    raw_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
              for k, v in raw_sd.items()}

    # Авто-определение архитектуры (если не задано явно через --channels/--res-blocks)
    try:
        auto_ch, auto_bl = detect_arch(raw_sd)
    except ValueError as e:
        print(f"⚠️  {e}")
        auto_ch, auto_bl = 128, 10

    ch = args.channels  if args.channels   is not None else auto_ch
    bl = args.res_blocks if args.res_blocks is not None else auto_bl

    if args.channels is None and args.res_blocks is None:
        print(f"   Архитектура определена из чекпоинта: {ch}ch × {bl}bl")
    elif args.channels is None or args.res_blocks is None:
        print(f"   Архитектура (авто+ручная): {ch}ch × {bl}bl")

    # Создаём модель с правильной архитектурой
    net = CapablancaNet(ch, bl).to(device)
    net = net.to(memory_format=torch.channels_last)

    if hasattr(torch, "compile"):
        try:
            net = torch.compile(net, dynamic=True)
        except Exception:
            pass

    net_target = net._orig_mod if hasattr(net, "_orig_mod") else net

    # Фильтруем несовместимые слои (напр. value_head при scalar→WDL переходе)
    target_sd = net_target.state_dict()
    incompatible = [k for k in raw_sd if k in target_sd and raw_sd[k].shape != target_sd[k].shape]
    for k in incompatible:
        del raw_sd[k]
    if incompatible:
        print(f"   Пропущено несовместимых слоёв: {incompatible}")

    net_target.load_state_dict(raw_sd, strict=False)

    ema_loaded = False
    if args.use_ema and "ema" in ckpt:
        ema_sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["ema"].items()}
        net_target.load_state_dict(ema_sd, strict=False)
        ema_loaded = True

    print(f"✅ Модель: {ch}ch × {bl}bl"
          f"  iter={iteration}"
          f"{'  [EMA]' if ema_loaded else ''}")
    print(f"   Чекпоинт: {ckpt_path}")
    print(f"🎮 {args.games} партий × {args.simulations} sims/ход"
          f"  batch={args.mcts_batch}  par={args.parallel_sims}"
          f"  max_len={args.max_game_length}"
          f"  resign={'off' if args.no_resign else args.resign_threshold}\n")

    net.eval()
    with torch.inference_mode():
        records = run_games(net, device, args)

    print_stats(records, args.max_game_length)


if __name__ == "__main__":
    main()
