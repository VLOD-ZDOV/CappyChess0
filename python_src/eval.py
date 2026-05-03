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
import datetime
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


# ── UCI / PGN helpers ─────────────────────────────────────────────────────────

_PROMO_FROM_VAL = [None, None, 'n', 'b', 'r', 'q', 'a', 'c']

def move_to_uci(m_int: int) -> str:
    p_val   = m_int & 0b111
    to_sq   = (m_int >> 3) & 0x7F
    from_sq = (m_int >> 10) & 0x7F
    def sq(s): return f"{chr(ord('a') + s % 10)}{s // 10 + 1}"
    promo = _PROMO_FROM_VAL[p_val] if 0 < p_val < len(_PROMO_FROM_VAL) else None
    return sq(from_sq) + sq(to_sq) + (promo or "")


def save_pgn(moves: list, result_str: str, pgn_path: str,
             white_name: str = "White", black_name: str = "Black"):
    """Сохраняет партию в PGN файл (дописывает если файл существует)."""
    os.makedirs(os.path.dirname(os.path.abspath(pgn_path)), exist_ok=True)
    now = datetime.datetime.now().strftime("%Y.%m.%d")
    uci = [move_to_uci(m) for m in moves]
    body = ""
    for i in range(0, len(uci), 2):
        body += f"{i//2+1}. {uci[i]}"
        if i + 1 < len(uci): body += f" {uci[i+1]}"
        body += " "
    with open(pgn_path, "a", encoding="utf-8") as f:
        f.write(f'[Event "Capablanca Eval"]\n')
        f.write(f'[Date "{now}"]\n')
        f.write(f'[White "{white_name}"]\n')
        f.write(f'[Black "{black_name}"]\n')
        f.write(f'[Result "{result_str}"]\n')
        f.write(f'[Variant "capablanca"]\n')
        f.write(f'[FEN "rnabqkcbnr/pppppppppp/10/10/10/10/PPPPPPPPPP/RNABQKCBNR w KQkq - 0 1"]\n')
        f.write(f'[SetUp "1"]\n\n')
        f.write(body + result_str + "\n\n")


# ── Загрузка модели ────────────────────────────────────────────────────────────

def load_model(path: str, device: torch.device) -> Tuple[CapablancaNet, str]:
    """
    Загружает чекпоинт, автоматически определяя архитектуру.
    Поддерживает старую (scalar value, policy=6880) и новую (WDL, policy=7000).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_sd = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in raw_sd.items()}

    # Архитектура из весов
    stem_key = next((k for k in sd if ("input_conv" in k or "input_block" in k)
                     and k.endswith(".weight") and "bn" not in k and "bias" not in k), None)
    ch = sd[stem_key].shape[0] if stem_key else 128
    bl = sum(1 for k in sd if "res_blocks" in k and k.endswith("conv1.weight"))

    net = CapablancaNet(num_channels=ch, num_res_blocks=bl)

    # Фильтруем слои с несовместимой формой (scalar↔WDL переход)
    target = net.state_dict()
    skipped = [k for k in sd if k in target and sd[k].shape != target[k].shape]
    for k in skipped: del sd[k]

    net.load_state_dict(sd, strict=False)
    net.to(device).eval()
    net = net.to(memory_format=torch.channels_last)

    # Определяем тип value head для информации
    vkey = next((k for k in sd if "value_head" in k and k.endswith(".weight")
                 and "6." in k), None)
    arch_tag = ""
    if vkey and vkey in sd:
        arch_tag = " [WDL]" if sd[vkey].shape[0] == 3 else " [scalar]"
    if skipped:
        arch_tag += f" ⚠{len(skipped)}skip"

    # Короткое имя: model_iter00025 → iter25
    name = os.path.splitext(os.path.basename(path))[0]
    if "iter" in name:
        num = name.split("iter")[-1].lstrip("0") or "0"
        name = f"iter{num}"

    return net, name, arch_tag


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
    verbose: bool = False,
    pgn_path: str = None,
    white_name: str = "White",
    black_name: str = "Black",
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
    histories = [[] for _ in range(num_games)]  # для verbose и PGN

    while active:
        # Разбиваем активные игры по чьему ходу
        white_games = [i for i in active if engines[i].side_to_move() == 0]
        black_games = [i for i in active if engines[i].side_to_move() == 1]

        # MCTS для белых
        if white_games:
            w_engines = [engines[i] for i in white_games]
            w_policies = mcts_w.search_games(w_engines, simulations)
            for j, gi in enumerate(white_games):
                m = _apply_policy_move(engines[gi], w_policies[j],
                                       move_counts[gi], temperature_moves)
                if m is not None: histories[gi].append(m)
                move_counts[gi] += 1

        # MCTS для чёрных
        if black_games:
            b_engines = [engines[i] for i in black_games]
            b_policies = mcts_b.search_games(b_engines, simulations)
            for j, gi in enumerate(black_games):
                m = _apply_policy_move(engines[gi], b_policies[j],
                                       move_counts[gi], temperature_moves)
                if m is not None: histories[gi].append(m)
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
                continue
            # Игра завершилась — verbose и PGN
            r = results[gi]
            if verbose and move_counts[gi] <= 10:
                print(f"  Партия {gi+1}: {move_counts[gi]} ходов, "
                      f"результат={'1-0' if r>0 else ('0-1' if r<0 else '½-½')}")
                for mn, mv in enumerate(histories[gi][:10]):
                    side = "Бел" if mn % 2 == 0 else "Чёрн"
                    print(f"    {mn+1:2d}. {side}: {move_to_uci(mv)}")
            if pgn_path is not None:
                res_str = "1-0" if r > 0.5 else ("0-1" if r < -0.5 else "1/2-1/2")
                save_pgn(histories[gi], res_str, pgn_path, white_name, black_name)
        active = new_active

    return results


def _apply_policy_move(engine: CapablancaEngine, policy: np.ndarray,
                        move_num: int, temperature_moves: int) -> int:
    """Выбирает и применяет ход согласно policy. Возвращает выбранный ход."""
    legal = engine.get_legal_moves_int()
    if not legal:
        return None

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
    return move


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
    verbose: bool = False,
    pgn_dir: str = None,
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
    pgn1 = os.path.join(pgn_dir, f"{name_a}_vs_{name_b}.pgn") if pgn_dir else None
    res1 = play_batch(net_a, net_b, device, n1, simulations,
                      max_moves, temperature_moves, mcts_batch,
                      verbose=verbose, pgn_path=pgn1,
                      white_name=name_a, black_name=name_b)
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
    pgn2 = os.path.join(pgn_dir, f"{name_b}_vs_{name_a}.pgn") if pgn_dir else None
    res2 = play_batch(net_b, net_a, device, half, simulations,
                      max_moves, temperature_moves, mcts_batch,
                      verbose=verbose, pgn_path=pgn2,
                      white_name=name_b, black_name=name_a)
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
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Показывать первые 10 ходов каждой партии")
    parser.add_argument("--pgn-dir",      type=str, default=None,
                        help="Папка для сохранения PGN партий (напр. games/)")
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
            net, name, arch_tag = load_model(path, device)
            models.append((name, net))
            ch = net.num_channels
            bl = len(net.res_blocks)
            print(f"  ✓ {name:<20} ({ch}ch × {bl} blocks){arch_tag}  [{path}]")
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
            verbose=args.verbose,
            pgn_dir=args.pgn_dir,
        )
        match_results.append(result)

    # Таблица
    print_leaderboard(models, match_results)
    print(f"\n⏱️  Общее время: {(time.time()-t_total)/60:.1f} мин")


if __name__ == "__main__":
    main()
