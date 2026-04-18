# eval.py — Сравнение моделей через self-play (round-robin турнир)
#
# Примеры запуска:
#   # Два конкретных чекпоинта
#   python eval.py checkpoints/model_iter00010.pth checkpoints/model_iter00025.pth --games 100
#
#   # Все чекпоинты в папке (round-robin каждый против каждого)
#   python eval.py checkpoints/ --games 50 --simulations 200
#
#   # Только последние N чекпоинтов из папки
#   python eval.py checkpoints/ --games 50 --last 4
#
#   # Быстрый тест без GPU
#   python eval.py checkpoints/ --games 20 --simulations 50 --device cpu

import os
import sys
import time
import argparse
import itertools
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import torch

from model import CapablancaNet
from mcts import UltraFastMCTS

try:
    from capablanca_engine import CapablancaEngine
except ImportError:
    raise ImportError("capablanca_engine not found. Build with: maturin develop --release")


# ── Загрузка модели ────────────────────────────────────────────────────────────

def load_model(path: str, device: torch.device) -> Tuple[CapablancaNet, str]:
    """Загружает чекпоинт, автоматически определяя архитектуру."""
    ckpt = torch.load(path, map_location=device)
    sd = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    # Убираем prefix от torch.compile / DataParallel
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}

    ch = sd["input_conv.net.0.weight"].shape[0]
    bl = len([k for k in sd if "res_blocks" in k and "conv1.weight" in k])

    net = CapablancaNet(num_channels=ch, num_res_blocks=bl)
    net.load_state_dict(sd)
    net.to(device).eval()
    net = net.to(memory_format=torch.channels_last)

    # Короткое имя для вывода: "iter00025" или имя файла без расширения
    name = os.path.splitext(os.path.basename(path))[0]
    if "iter" in name:
        # model_iter00025 → iter25
        num = name.split("iter")[-1].lstrip("0") or "0"
        name = f"iter{num}"

    return net, name


def collect_checkpoints(paths: List[str], last: int = 0) -> List[str]:
    """Из списка путей/папок собирает .pth файлы."""
    result = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(
                [os.path.join(p, f) for f in os.listdir(p) if f.endswith(".pth")]
            )
            result.extend(found)
        elif os.path.isfile(p) and p.endswith(".pth"):
            result.append(p)
        else:
            print(f"⚠️  Пропускаю: {p}")
    if last > 0:
        result = result[-last:]
    return result


# ── Игра ──────────────────────────────────────────────────────────────────────

def play_batch(
    net_white: CapablancaNet,
    net_black: CapablancaNet,
    device: torch.device,
    num_games: int,
    simulations: int,
    max_moves: int,
    temperature_moves: int,
    mcts_batch: int,
) -> List[float]:
    """
    Играет num_games партий: net_white — белые, net_black — чёрные.
    Возвращает список результатов с точки зрения белых:
        +1.0 = победа белых, -1.0 = победа чёрных, 0.0 = ничья, 0.5/-0.5 = по материалу
    """
    mcts_w = UltraFastMCTS(net_white, device, c_puct=1.25,
                            batch_size=mcts_batch, add_dirichlet=False)
    mcts_b = UltraFastMCTS(net_black, device, c_puct=1.25,
                            batch_size=mcts_batch, add_dirichlet=False)

    engines = [CapablancaEngine() for _ in range(num_games)]
    active  = list(range(num_games))
    results = [None] * num_games
    move_counts = [0] * num_games

    while active:
        # Разбиваем активные игры по чьему ходу
        white_games = [i for i in active if engines[i].side_to_move() == 0]
        black_games = [i for i in active if engines[i].side_to_move() == 1]

        # MCTS для белых
        if white_games:
            w_engines = [engines[i] for i in white_games]
            w_policies = mcts_w.search_games(w_engines, simulations)
            for j, gi in enumerate(white_games):
                _apply_policy_move(engines[gi], w_policies[j],
                                   move_counts[gi], temperature_moves)
                move_counts[gi] += 1

        # MCTS для чёрных
        if black_games:
            b_engines = [engines[i] for i in black_games]
            b_policies = mcts_b.search_games(b_engines, simulations)
            for j, gi in enumerate(black_games):
                _apply_policy_move(engines[gi], b_policies[j],
                                   move_counts[gi], temperature_moves)
                move_counts[gi] += 1

        # Проверяем завершение
        new_active = []
        for gi in active:
            eng = engines[gi]
            if eng.is_game_over():
                results[gi] = eng.game_result()
            elif move_counts[gi] >= max_moves:
                results[gi] = eng.material_result()
            else:
                new_active.append(gi)
        active = new_active

    return results


def _apply_policy_move(engine: CapablancaEngine, policy: np.ndarray,
                        move_num: int, temperature_moves: int):
    """Выбирает и применяет ход согласно policy."""
    legal = engine.get_legal_moves_int()
    if not legal:
        return

    probs = np.array([
        policy[engine.move_int_to_policy_idx(m) or 0] for m in legal
    ], dtype=np.float64)

    s = probs.sum()
    if s < 1e-10:
        probs = np.ones(len(legal)) / len(legal)
    else:
        probs /= s

    if move_num < temperature_moves:
        move = int(np.random.choice(legal, p=probs))
    else:
        move = int(legal[np.argmax(probs)])

    engine.make_move_int(move)


# ── Матч двух моделей ─────────────────────────────────────────────────────────

def run_match(
    name_a: str, net_a: CapablancaNet,
    name_b: str, net_b: CapablancaNet,
    device: torch.device,
    games: int,
    simulations: int,
    max_moves: int,
    temperature_moves: int,
    mcts_batch: int,
) -> Dict:
    """
    Играет games партий между A и B (половина — A белые, половина — B белые).
    Возвращает словарь со статистикой.
    """
    half = games // 2
    remainder = games % 2

    print(f"\n  {'─'*50}")
    print(f"  {name_a}  vs  {name_b}  ({games} партий, {simulations} симуляций)")
    print(f"  {'─'*50}")

    wins_a = draws = wins_b = 0
    result_detail = []

    # --- Блок 1: A = белые ---
    n1 = half + remainder
    print(f"  [{name_a} белые] {n1} партий...", end="", flush=True)
    t0 = time.time()
    res1 = play_batch(net_a, net_b, device, n1, simulations,
                      max_moves, temperature_moves, mcts_batch)
    for r in res1:
        if r > 0:   wins_a += 1
        elif r < 0: wins_b += 1
        else:       draws  += 1
        result_detail.append(("A_white", r))
    print(f" {time.time()-t0:.0f}s  →  +{sum(1 for r in res1 if r>0)}/"
          f"={sum(1 for r in res1 if r==0)}/"
          f"-{sum(1 for r in res1 if r<0)}")

    # --- Блок 2: B = белые ---
    print(f"  [{name_b} белые] {half} партий...", end="", flush=True)
    t0 = time.time()
    res2 = play_batch(net_b, net_a, device, half, simulations,
                      max_moves, temperature_moves, mcts_batch)
    for r in res2:
        # r — результат белых (= B), конвертируем в результат A
        r_a = -r
        if r_a > 0:   wins_a += 1
        elif r_a < 0: wins_b += 1
        else:         draws  += 1
        result_detail.append(("B_white", r))
    print(f" {time.time()-t0:.0f}s  →  +{sum(1 for r in res2 if r<0)}/"   # r<0 = B проиграл = A выиграл
          f"={sum(1 for r in res2 if r==0)}/"
          f"-{sum(1 for r in res2 if r>0)}")

    total = wins_a + wins_b + draws
    wr_a = (wins_a + 0.5 * draws) / total if total > 0 else 0.0

    # Доверительный интервал Уилсона (95%)
    ci = _wilson_ci(wins_a + 0.5 * draws, total)

    print(f"\n  Итог: {name_a} {wins_a}W / {draws}D / {wins_b}L  "
          f"винрейт {wr_a:.1%}  CI [{ci[0]:.1%}, {ci[1]:.1%}]")

    return {
        "name_a": name_a, "name_b": name_b,
        "wins_a": wins_a, "draws": draws, "wins_b": wins_b,
        "winrate_a": wr_a, "ci_lo": ci[0], "ci_hi": ci[1],
        "total": total,
    }


def _wilson_ci(score: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Интервал Уилсона для пропорции (95% по умолчанию)."""
    if n == 0:
        return 0.0, 1.0
    p = score / n
    denom = 1 + z*z / n
    centre = (p + z*z / (2*n)) / denom
    margin = (z * (p*(1-p)/n + z*z/(4*n*n))**0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ── Таблица результатов ────────────────────────────────────────────────────────

def print_leaderboard(models: List[Tuple[str, CapablancaNet]],
                      match_results: List[Dict]):
    """Печатает турнирную таблицу и рейтинг по очкам."""
    names = [m[0] for m in models]
    n = len(names)

    # Матрица винрейтов [i][j] = винрейт i против j
    wr_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
    points: Dict[str, float] = defaultdict(float)

    for r in match_results:
        a, b = r["name_a"], r["name_b"]
        wr_matrix[a][b] = r["winrate_a"]
        wr_matrix[b][a] = 1.0 - r["winrate_a"]
        points[a] += r["wins_a"] + 0.5 * r["draws"]
        points[b] += r["wins_b"] + 0.5 * r["draws"]

    print("\n" + "═"*70)
    print("  ТУРНИРНАЯ ТАБЛИЦА")
    print("═"*70)

    # Заголовок
    col = 14
    header = f"  {'Модель':<{col}}"
    for name in names:
        short = name[-col:] if len(name) > col else name
        header += f"  {short:>{col}}"
    header += f"  {'Очки':>8}  {'Место':>5}"
    print(header)
    print("  " + "─"*68)

    # Строки
    sorted_names = sorted(names, key=lambda x: points[x], reverse=True)
    for rank, name in enumerate(sorted_names, 1):
        row = f"  {name[-col:] if len(name)>col else name:<{col}}"
        for opp in names:
            if opp == name:
                row += f"  {'───':>{col}}"
            elif opp in wr_matrix[name]:
                wr = wr_matrix[name][opp]
                marker = "✓" if wr > 0.5 else ("~" if wr == 0.5 else "✗")
                row += f"  {f'{wr:.1%} {marker}':>{col}}"
            else:
                row += f"  {'—':>{col}}"
        row += f"  {points[name]:>8.1f}  {rank:>5}"
        print(row)

    print("═"*70)

    # Итоговый рейтинг
    print("\n  РЕЙТИНГ:")
    for rank, name in enumerate(sorted_names, 1):
        matches_played = sum(
            r["total"] for r in match_results
            if r["name_a"] == name or r["name_b"] == name
        )
        print(f"  {rank}. {name:<20} {points[name]:.1f} очков  "
              f"({matches_played} партий)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Eval: сравнение чекпоинтов Capablanca Chess",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Два конкретных чекпоинта
  python eval.py checkpoints/model_iter00010.pth checkpoints/model_iter00025.pth

  # Все чекпоинты из папки
  python eval.py checkpoints/ --games 100

  # Только последние 4 чекпоинта
  python eval.py checkpoints/ --last 4 --games 50

  # Без GPU, быстро
  python eval.py checkpoints/ --last 2 --games 20 --simulations 50 --device cpu
        """
    )
    parser.add_argument("paths", nargs="+",
                        help="Пути к .pth файлам или папкам с чекпоинтами")
    parser.add_argument("--games",       type=int, default=100,
                        help="Партий на каждую пару моделей (default: 100)")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS симуляций на ход (default: 100)")
    parser.add_argument("--max-moves",   type=int, default=150,
                        help="Лимит ходов на партию (default: 150)")
    parser.add_argument("--temperature-moves", type=int, default=10,
                        help="Ходов с температурной выборкой (default: 10)")
    parser.add_argument("--mcts-batch",  type=int, default=64,
                        help="Размер батча для MCTS inference (default: 64)")
    parser.add_argument("--last",        type=int, default=0,
                        help="Взять только последние N чекпоинтов из папки")
    parser.add_argument("--device",      type=str, default="auto",
                        help="cuda / cpu / auto (default: auto)")
    args = parser.parse_args()

    # Устройство
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"🖥️  Устройство: {device}")

    # Собираем чекпоинты
    ckpt_paths = collect_checkpoints(args.paths, last=args.last)
    if len(ckpt_paths) < 2:
        print(f"❌ Нужно минимум 2 модели, найдено: {len(ckpt_paths)}")
        sys.exit(1)

    print(f"\n📦 Загружаю {len(ckpt_paths)} моделей...")
    models: List[Tuple[str, CapablancaNet]] = []
    for path in ckpt_paths:
        try:
            net, name = load_model(path, device)
            models.append((name, net))
            ch = net.num_channels
            bl = len(net.res_blocks)
            print(f"  ✓ {name:<20} ({ch}ch × {bl} blocks)  [{path}]")
        except Exception as e:
            print(f"  ✗ Ошибка загрузки {path}: {e}")

    if len(models) < 2:
        print("❌ Не удалось загрузить минимум 2 модели")
        sys.exit(1)

    # Параметры
    pairs = list(itertools.combinations(models, 2))
    total_games = len(pairs) * args.games
    print(f"\n🏆 Турнир: {len(models)} моделей, {len(pairs)} пар, "
          f"{total_games} партий всего")
    print(f"   Симуляций/ход: {args.simulations} | "
          f"Макс ходов: {args.max_moves} | "
          f"Температура: первые {args.temperature_moves} ходов")

    # Round-robin турнир
    match_results = []
    t_total = time.time()

    for i, ((name_a, net_a), (name_b, net_b)) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}]", end="")
        result = run_match(
            name_a, net_a, name_b, net_b, device,
            games=args.games,
            simulations=args.simulations,
            max_moves=args.max_moves,
            temperature_moves=args.temperature_moves,
            mcts_batch=args.mcts_batch,
        )
        match_results.append(result)

    # Таблица
    print_leaderboard(models, match_results)
    print(f"\n⏱️  Общее время: {(time.time()-t_total)/60:.1f} мин")


if __name__ == "__main__":
    main()
