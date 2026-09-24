# eval.py — Model comparison via self-play (round-robin tournament)
#
# Usage examples:
#   # Two specific checkpoints
#   python eval.py checkpoints/model_iter00010.pth checkpoints/model_iter00025.pth --games 100
#
#   # All checkpoints in a folder (round-robin every pair)
#   python eval.py checkpoints/ --games 50 --simulations 200
#
#   # Only the last N checkpoints from a folder
#   python eval.py checkpoints/ --games 50 --last 4
#
#   # Quick test without GPU
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
import archive as archive_mod
import torch

# PyTorch's intra-op pool defaults to half the core count and busy-waits
# between GPU calls. The CPU side of this workload is tree descent in Rust plus
# sparse-policy assembly, not matrix work, so those threads buy nothing: measured
# throughput flat across every thread count tried, while CPU use goes
# up several times over. Freeing the cores also stops a co-running eval or GUI from
# costing throughput (measured -30% under contention).
# See docs/experiments/cpu_threads.md. Override with OMP_NUM_THREADS.
if "OMP_NUM_THREADS" not in os.environ:
    torch.set_num_threads(2)

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
    """Saves the game to a PGN file (appends if file exists)."""
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


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(path: str, device: torch.device) -> Tuple[CapablancaNet, str, str, str]:
    """
    Loads a checkpoint, automatically detecting the architecture.
    `path:ema` loads the EMA shadow instead of the live weights.
    Supports old (scalar value, policy=6880) and new (WDL, policy=7000) formats.
    """
    from model import split_weights, pick_state_dict
    path, which = split_weights(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_sd = pick_state_dict(ckpt, which)

    # Use the canonical helper from model.py — it knows about ALL architecture
    # knobs (channels, res blocks, transformer, heads, mlh, future, AND the
    # BT5 trim: qkv_bias / use_rmsnorm / piece_embed_dim). Inline logic here
    # used to ignore the BT5 flags → checkpoints trained with them loaded into
    # a default net with random qkv.bias / ln.bias / no piece_embed, ruining
    # the transformer blocks and giving meaningless eval results.
    from model import build_net_from_state_dict, describe_arch
    net, sd = build_net_from_state_dict(raw_sd)
    # Track shape-mismatch drops for the arch_tag warning below.
    full_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
               for k, v in raw_sd.items()}
    skipped = [k for k in full_sd if k not in sd]

    result = net.load_state_dict(sd, strict=False)
    net.to(device).eval()
    # NCHW on purpose: this net is GroupNorm-heavy — channels_last measured
    # ~10-15% SLOWER (see experiments/perf/ and the notes in mcts.py/train.py).

    # Detect value head type for informational purposes
    vkey = next((k for k in sd if "value_head" in k and k.endswith(".weight")
                 and "6." in k), None)
    arch_tag = ""
    if vkey and vkey in sd:
        arch_tag = " [WDL]" if sd[vkey].shape[0] == 3 else " [scalar]"
    if skipped:
        arch_tag += f" ⚠{len(skipped)}skip"

    # Short name: model_iter00025 → iter25
    name = os.path.splitext(os.path.basename(path))[0]
    if "iter" in name:
        num = name.split("iter")[-1].lstrip("0") or "0"
        name = f"iter{num}"
    if which == "ema":
        name += "-ema"      # иначе живые и EMA одного чекпоинта неразличимы в таблице

    return net, name, arch_tag, describe_arch(net)


def collect_checkpoints(paths: List[str], last: int = 0) -> List[str]:
    """Collects .pth files from a list of paths/directories."""
    from model import split_weights
    result = []
    for p in paths:
        # Суффикс `:ema` отделяем ДО проверки файла и возвращаем после: иначе
        # `x.pth:ema` не проходит ни isfile, ни endswith(".pth") и молча
        # выпадает из турнира. Для каталога суффикс применяется ко всем файлам.
        base, _ = split_weights(p)
        tag = p[len(base):]
        if os.path.isdir(base):
            found = sorted(
                [os.path.join(base, f) + tag for f in os.listdir(base)
                 if f.endswith(".pth")]
            )
            result.extend(found)
        elif os.path.isfile(base) and base.endswith(".pth"):
            result.append(p)
        else:
            print(f"⚠️  Пропускаю: {p}")
    if last > 0:
        result = result[-last:]
    return result


# ── Game ──────────────────────────────────────────────────────────────────────

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
    timeout_as_draw: bool = False,
    white_name: str = "White",
    black_name: str = "Black",
    compile_mode: str = None,
    kld_threshold: float = 0.0,
    parallel_sims: int = None,
    parallel_sims_black: int = None,
    c_puct_white: float = None,
    c_puct_black: float = None,
    fpu_white: float = None,
    fpu_black: float = None,
    ptemp_white: float = 1.0,
    ptemp_black: float = None,
    policy_only_after_white: int = None,
    policy_only_after_black: int = None,
    adjudicate_q: float = 0.0,
    adjudicate_plies: int = 4,
    arch=None,
) -> List[float]:
    """
    Plays num_games games: net_white as white, net_black as black.
    Returns a list of results from white's perspective:
        +1.0 = white wins, -1.0 = black wins, 0.0 = draw, 0.5/-0.5 = by material
    """
    mcts_w = UltraFastMCTS(net_white, device, c_puct=1.25,
                            batch_size=mcts_batch, add_dirichlet=False,
                            compile_mode=compile_mode, kld_threshold=kld_threshold,
                            parallel_sims=parallel_sims,
                            rust_c_puct=c_puct_white,
                            rust_fpu=fpu_white, policy_temp=ptemp_white)
    mcts_b = UltraFastMCTS(net_black, device, c_puct=1.25,
                            batch_size=mcts_batch, add_dirichlet=False,
                            compile_mode=compile_mode, kld_threshold=kld_threshold,
                            parallel_sims=(parallel_sims_black
                                           if parallel_sims_black is not None
                                           else parallel_sims),
                            rust_c_puct=(c_puct_black if c_puct_black is not None
                                         else c_puct_white),
                            rust_fpu=(fpu_black if fpu_black is not None else fpu_white),
                            policy_temp=(ptemp_black if ptemp_black is not None
                                         else ptemp_white))

    engines = [CapablancaEngine() for _ in range(num_games)]
    active  = list(range(num_games))
    results = [None] * num_games
    move_counts = [0] * num_games
    histories = [[] for _ in range(num_games)]  # for verbose and PGN
    arch_rows = [[] for _ in range(num_games)]  # позиции для архива
    adj_streak = [0] * num_games      # знак = кто выигрывает, модуль = длина серии
    adjudicated = [False] * num_games

    # One persistent tree per network, both tracking the real game. Every move
    # is applied to both, so each search resumes from the sub-tree the previous
    # one already built (tree reuse) instead of starting from scratch. The old
    # loop rebuilt the tree every ply, which measures the network playing WITHOUT
    # the reuse it gets in self-play — i.e. weaker than it actually is.
    tree_w = mcts_w.new_tree(engines)
    tree_b = mcts_b.new_tree(engines)

    # Progress on the same line as the block header: a match is minutes long and
    # the games run in lockstep, so without this there is nothing between "N
    # партий..." and the final score.
    report_step = max(1, num_games // 10)
    reported = 0

    while active:
        # All games advance one ply per pass, so they stay in lockstep and the
        # side to move is the same for every active game.
        side = engines[active[0]].side_to_move()
        searcher, tree = (mcts_w, tree_w) if side == 0 else (mcts_b, tree_b)
        cut = policy_only_after_white if side == 0 else policy_only_after_black
        # Сторона с заданным порогом после него ходит ГОЛОЙ политикой, без
        # поиска. Так меряется вклад поиска именно в середине партии: до порога
        # обе стороны играют одинаково, значит дебютный эффект в замер не лезет.
        bare = cut is not None and move_counts[active[0]] >= cut
        if bare:
            raw = searcher.raw_policies([engines[gi] for gi in active])
            raw_by_game = {gi: raw[k] for k, gi in enumerate(active)}
        else:
            searcher.run_search(tree, simulations)
            # Sparse root policy: one entry per visited legal move instead of a dense
            # 7000-float vector per game per ply.
            sparse_pols = tree.get_policies_sparse()
            # Доказанный границами исход; -1 = ничего не доказано.
            proven_moves = tree.get_best_moves()

        # Архив: доска и ПОЛНОЕ распределение визитов на 800 симуляциях. Поиск
        # их уже посчитал ради выбора хода — оставалось только не выбрасывать.
        if (arch is not None or adjudicate_q > 0) and not bare:
            vals = np.asarray(tree.get_values(), dtype=np.float32)
            draws_np = np.asarray(tree.get_draws(), dtype=np.float32)
        if adjudicate_q > 0 and not bare:
            # Оценка корня приводится к взгляду БЕЛЫХ, поэтому серия из N
            # полуходов означает, что обе сети подряд согласны с приговором.
            for gi in active:
                q = float(vals[gi]) if gi < len(vals) else 0.0
                qw = q if side == 0 else -q
                if qw >= adjudicate_q:
                    adj_streak[gi] = adj_streak[gi] + 1 if adj_streak[gi] > 0 else 1
                elif qw <= -adjudicate_q:
                    adj_streak[gi] = adj_streak[gi] - 1 if adj_streak[gi] < 0 else -1
                else:
                    adj_streak[gi] = 0

        for gi in active:
            if bare:
                dense = raw_by_game[gi]
                lookup = {int(i): float(dense[i])
                          for i in (engines[gi].move_int_to_policy_idx(m)
                                    for m in engines[gi].get_legal_moves_int())
                          if i is not None}
            else:
                pol_idx, pol_val = sparse_pols[gi]
                lookup = {int(i): float(v) for i, v in zip(pol_idx, pol_val)}
                # Доказанный мат бьёт число визитов (правило lc0): ему хватает
                # трёх посещений, и по счётчику он проигрывал ходу с двумя
                # сотнями. В архив ниже идёт честное распределение визитов.
                _pm = int(proven_moves[gi]) if gi < len(proven_moves) else -1
                if _pm >= 0:
                    lookup = {_pm: 1.0}
            if arch is not None and not bare:
                pi, pv = sparse_pols[gi]
                # Индексы ходов считаются ДО хода: move_int_to_policy_idx
                # переворачивает доску по стороне, которая ходит СЕЙЧАС, а
                # _apply_policy_move ход уже применяет.
                idx_of = {mv: engines[gi].move_int_to_policy_idx(mv)
                          for mv in engines[gi].get_legal_moves_int()}
                arch_rows[gi].append([
                    np.asarray(engines[gi].get_board_tensor(), dtype=np.float32),
                    np.asarray(pi, dtype=np.int16), np.asarray(pv, dtype=np.float16),
                    int(move_counts[gi]), int(side),
                    float(vals[gi]) if gi < len(vals) else 0.0,
                    float(draws_np[gi]) if gi < len(draws_np) else -1.0,
                    -1, -1,
                ])
            else:
                idx_of = None
            m = _apply_policy_move(engines[gi], lookup,
                                   move_counts[gi], temperature_moves)
            if m is not None:
                histories[gi].append(m)
                if idx_of is not None:
                    got = idx_of.get(m)
                    arch_rows[gi][-1][7] = -1 if got is None else int(got)
                    arch_rows[gi][-1][8] = int(m)
                tree_w.make_move(gi, m)
                tree_b.make_move(gi, m)
            move_counts[gi] += 1

        # Check for game completion
        new_active = []
        for gi in active:
            eng = engines[gi]
            if eng.is_game_over():
                results[gi] = eng.game_result()
            elif adjudicate_q > 0 and abs(adj_streak[gi]) >= adjudicate_plies:
                results[gi] = 1.0 if adj_streak[gi] > 0 else -1.0
                adjudicated[gi] = True
            elif move_counts[gi] >= max_moves:
                results[gi] = 0.0 if timeout_as_draw else eng.material_result()
            else:
                new_active.append(gi)
                continue
            tree_w.set_game_finished(gi)
            tree_b.set_game_finished(gi)
            # Game over — verbose and PGN
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
            if arch is not None and arch_rows[gi]:
                if eng.is_game_over():
                    term = (archive_mod.TERM_MATE if abs(r) > 0.5
                            else archive_mod.DRAW_REASON_TO_TERM.get(
                                eng.draw_reason(), archive_mod.TERM_DRAW_RULE))
                elif adjudicated[gi]:
                    term = archive_mod.TERM_ADJUDICATED
                else:
                    term = archive_mod.TERM_LIMIT
                g_id = arch.add_game(
                    result=int(round(r)) if abs(r) > 0.5 else 0,
                    plies=move_counts[gi], term=term, playthrough=0,
                    resign_ply=-1, resign_side=-1,
                    source=archive_mod.SOURCE_MATCH,
                    white_id=arch.name_id(white_name),
                    black_id=arch.name_id(black_name))
                for row in arch_rows[gi]:
                    arch.add_position(row[0], row[1], row[2], game=g_id,
                                      ply=row[3], side=row[4], full=True,
                                      root_q=row[5], root_d=row[6], move=row[7],
                                      move_raw=row[8])
                arch_rows[gi] = []
        done = num_games - len(new_active)
        if done - reported >= report_step or (not new_active and done > reported):
            reported = done
            print(f" {done * 100 // num_games}%", end="", flush=True)
        active = new_active

    return results, histories


def _apply_policy_move(engine: CapablancaEngine, policy: dict,
                        move_num: int, temperature_moves: int) -> int:
    """Selects and applies a move from a sparse root policy {policy_idx: prob}.
    Returns the chosen move, or None if the position has no legal moves."""
    legal = engine.get_legal_moves_int()
    if not legal:
        return None

    probs = np.array([
        policy.get(engine.move_int_to_policy_idx(m), 0.0) for m in legal
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


# ── Match between two models ───────────────────────────────────────────────────

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
    timeout_as_draw: bool = False,
    compile_mode: str = None,
    kld_threshold: float = 0.0,
    parallel_sims: int = None,
    parallel_sims_b: int = None,
    c_puct: float = None,
    c_puct_b: float = None,
    fpu: float = None,
    fpu_b: float = None,
    ptemp: float = 1.0,
    ptemp_b: float = None,
    policy_only_after: int = None,
    policy_only_after_b: int = None,
    adjudicate_q: float = 0.0,
    adjudicate_plies: int = 4,
    arch=None,
) -> Dict:
    """
    Plays `games` games between A and B (half with A as white, half with B as white).
    Returns a dict with statistics.
    """
    half = games // 2
    remainder = games % 2

    print(f"\n  {'─'*50}")
    print(f"  {name_a}  vs  {name_b}  ({games} партий, {simulations} симуляций)")
    print(f"  {'─'*50}")

    wins_a = draws = wins_b = 0
    result_detail = []

    # --- Block 1: A = white ---
    n1 = half + remainder
    print(f"  [{name_a} белые] {n1} партий...", end="", flush=True)
    t0 = time.time()
    pgn1 = os.path.join(pgn_dir, f"{name_a}_vs_{name_b}.pgn") if pgn_dir else None
    res1, hist1 = play_batch(net_a, net_b, device, n1, simulations,
                      max_moves, temperature_moves, mcts_batch,
                      verbose=verbose, pgn_path=pgn1,
                      white_name=name_a, black_name=name_b,
                      timeout_as_draw=timeout_as_draw,
                      compile_mode=compile_mode, kld_threshold=kld_threshold,
                      parallel_sims=parallel_sims,
                      parallel_sims_black=parallel_sims_b,
                      c_puct_white=c_puct, c_puct_black=c_puct_b,
                      fpu_white=fpu, fpu_black=fpu_b,
                      ptemp_white=ptemp, ptemp_black=ptemp_b,
                      policy_only_after_white=policy_only_after,
                      policy_only_after_black=policy_only_after_b,
                      adjudicate_q=adjudicate_q, adjudicate_plies=adjudicate_plies,
                      arch=arch)
    for r in res1:
        if r > 0:   wins_a += 1
        elif r < 0: wins_b += 1
        else:       draws  += 1
        result_detail.append(("A_white", r))
    print(f" {time.time()-t0:.0f}s  →  +{sum(1 for r in res1 if r>0)}/"
          f"={sum(1 for r in res1 if r==0)}/"
          f"-{sum(1 for r in res1 if r<0)}")

    # --- Block 2: B = white ---
    print(f"  [{name_b} белые] {half} партий...", end="", flush=True)
    t0 = time.time()
    pgn2 = os.path.join(pgn_dir, f"{name_b}_vs_{name_a}.pgn") if pgn_dir else None
    res2, hist2 = play_batch(net_b, net_a, device, half, simulations,
                      max_moves, temperature_moves, mcts_batch,
                      verbose=verbose, pgn_path=pgn2,
                      white_name=name_b, black_name=name_a,
                      timeout_as_draw=timeout_as_draw,
                      compile_mode=compile_mode, kld_threshold=kld_threshold,
                      parallel_sims=(parallel_sims_b if parallel_sims_b is not None
                                     else parallel_sims),
                      parallel_sims_black=parallel_sims,
                      c_puct_white=c_puct_b, c_puct_black=c_puct,
                      fpu_white=fpu_b, fpu_black=fpu,
                      ptemp_white=(ptemp_b if ptemp_b is not None else 1.0),
                      ptemp_black=ptemp,
                      policy_only_after_white=policy_only_after_b,
                      policy_only_after_black=policy_only_after,
                      adjudicate_q=adjudicate_q, adjudicate_plies=adjudicate_plies,
                      arch=arch)
    for r in res2:
        # r — result for white (= B), convert to result for A
        r_a = -r
        if r_a > 0:   wins_a += 1
        elif r_a < 0: wins_b += 1
        else:         draws  += 1
        result_detail.append(("B_white", r))
    print(f" {time.time()-t0:.0f}s  →  +{sum(1 for r in res2 if r<0)}/"   # r<0 = B lost = A won
          f"={sum(1 for r in res2 if r==0)}/"
          f"-{sum(1 for r in res2 if r>0)}")

    total = wins_a + wins_b + draws
    wr_a = (wins_a + 0.5 * draws) / total if total > 0 else 0.0

    # Сколько партий на самом деле РАЗНЫЕ. Шум в матче вносит только температура
    # первых ходов, а чем острее становится политика, тем меньше разных дебютов
    # она выдаёт — и часть партий повторяется ход в ход. Повтор не несёт новых
    # сведений: он всегда даёт тот же результат, только утяжеляет его в счёте.
    # Поэтому интервал, посчитанный по номинальному числу партий, оказывается
    # уже настоящего, и матч выглядит точнее, чем он есть.
    dup_note = ""
    seqs = [tuple(h) for h in (list(hist1) + list(hist2)) if h]
    if seqs:
        from collections import Counter
        clusters = Counter(seqs)
        distinct = len(clusters)
        biggest = max(clusters.values())
        # Поправка Киша на кластеризацию: N_eff = N^2 / sum(m_i^2).
        n_eff = len(seqs) ** 2 / sum(m * m for m in clusters.values())
        if distinct < len(seqs):
            dup_note = (f"  Разных партий: {distinct} из {len(seqs)}"
                        f" (самый крупный повтор — {biggest}),"
                        f" действующий объём выборки {n_eff:.0f}")
            if n_eff < 0.8 * len(seqs):
                ci_eff = _wilson_ci((wins_a + 0.5 * draws) * n_eff / total, n_eff)
                dup_note += (f"\n  ⚠️  Повторов много: честный интервал шире —"
                             f" [{ci_eff[0]*100:.1f}%, {ci_eff[1]*100:.1f}%]."
                             f" Поднять --temperature-moves или число партий.")

    # Wilson confidence interval (95%)
    ci = _wilson_ci(wins_a + 0.5 * draws, total)

    print(f"\n  Итог: {name_a} {wins_a}W / {draws}D / {wins_b}L  "
          f"винрейт {wr_a:.1%}  CI [{ci[0]:.1%}, {ci[1]:.1%}]")
    if dup_note:
        print(dup_note)

    return {
        "name_a": name_a, "name_b": name_b,
        "wins_a": wins_a, "draws": draws, "wins_b": wins_b,
        "winrate_a": wr_a, "ci_lo": ci[0], "ci_hi": ci[1],
        "total": total,
    }


def _wilson_ci(score: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion (95% by default)."""
    if n == 0:
        return 0.0, 1.0
    p = score / n
    denom = 1 + z*z / n
    centre = (p + z*z / (2*n)) / denom
    margin = (z * (p*(1-p)/n + z*z/(4*n*n))**0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ── Leaderboard ────────────────────────────────────────────────────────────────

def print_leaderboard(models: List[Tuple[str, CapablancaNet]],
                      match_results: List[Dict]):
    """Prints a tournament table and ranking by score."""
    names = [m[0] for m in models]
    n = len(names)

    # Win-rate matrix [i][j] = win-rate of i against j
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

    # Header
    col = 14
    header = f"  {'Модель':<{col}}"
    for name in names:
        short = name[-col:] if len(name) > col else name
        header += f"  {short:>{col}}"
    header += f"  {'Очки':>8}  {'Место':>5}"
    print(header)
    print("  " + "─"*68)

    # Rows
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

    # Final ranking
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
    parser.add_argument("--adjudicate-q", type=float, default=0.0,
                        help="Присудить победу, если оценка корня держится выше "
                             "этого порога подряд --adjudicate-plies полуходов "
                             "(0 = выключено). В матче две трети партий иначе "
                             "упираются в лимит и съедают 72%% времени")
    parser.add_argument("--adjudicate-plies", type=int, default=4,
                        help="Сколько полуходов подряд держится приговор "
                             "(default: 4 — по два хода каждой стороны)")
    parser.add_argument("--timeout-as-draw", action="store_true", default=False,
                        help="Таймаут = ничья (0.0) вместо оценки по материалу")
    parser.add_argument("--compile-inference", type=str, default="default",
                        choices=["none", "default", "reduce-overhead", "max-autotune"],
                        help="torch.compile режим инференса (~1.5x на форварде). "
                             "default — безопасно; none — отключить (default: default)")
    parser.add_argument("--mcts-parallel-sims", type=int, default=None,
                        help="Листьев на вызов NN. ceil(simulations/это) = число "
                             "последовательных раундов PUCT; ниже ~12 раундов "
                             "поиск вырождается в равномерный "
                             "(см. docs/experiments/policy_target_collapse.md)")
    parser.add_argument("--c-puct", type=float, default=None,
                        help="База CPuct для поиска в Rust (по умолчанию 1.745 — "
                             "значение lc0 для доски 8x8). Задаёт первую сеть")
    parser.add_argument("--c-puct-b", type=float, default=None,
                        help="То же для второй сети — так меряется A/B по c_puct")
    parser.add_argument("--policy-only-after", type=int, default=None,
                        help="С этого полухода ПЕРВАЯ сеть ходит голой политикой "
                             "без поиска. Так меряется вклад поиска в середине "
                             "партии: до порога обе стороны играют одинаково")
    parser.add_argument("--policy-only-after-b", type=int, default=None,
                        help="То же для второй сети")
    parser.add_argument("--fpu", type=float, default=None,
                        help="Насколько пессимистичен непосещённый ход: "
                             "fpu = q_родителя − значение × sqrt(уже просмотренной "
                             "политики). Умолчание движка 0.33 размазывает посещения; "
                             "больше — поиск сильнее доверяет политике")
    parser.add_argument("--fpu-b", type=float, default=None,
                        help="То же для второй сети — так меряется A/B по FPU")
    parser.add_argument("--policy-temp", type=float, default=1.0,
                        help="Температура приоритета перед подачей в дерево. "
                             "Меньше 1 — заострить политику, больше — размыть")
    parser.add_argument("--policy-temp-b", type=float, default=None,
                        help="То же для второй сети")
    parser.add_argument("--mcts-parallel-sims-b", type=int, default=None,
                        help="parallel_sims для ВТОРОЙ модели. Позволяет столкнуть "
                             "одни и те же веса с разными настройками поиска — "
                             "измерить настройку поиска без переобучения.")
    parser.add_argument("--archive-dir", type=str, default="",
                        help="Писать партии матча в архив. Поиск в матче идёт на "
                             "800 симуляциях и сдачи нет вовсе, поэтому такие "
                             "партии чище самоигровых — но они OFF-POLICY: годятся "
                             "для засева новой линии, не для живого буфера")
    parser.add_argument("--kld", type=float, default=0.0,
                        help="Порог KLD early-exit: MCTS останавливается раньше, когда "
                             "распределение визитов стабилизировалось. 0 = выкл. "
                             "Типично 1e-4..3e-4 для заметного ускорения (default: 0)")
    args = parser.parse_args()

    compile_mode = None if args.compile_inference == "none" else args.compile_inference

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"🖥️  Устройство: {device}")

    # Collect checkpoints
    ckpt_paths = collect_checkpoints(args.paths, last=args.last)
    if len(ckpt_paths) < 2:
        print(f"❌ Нужно минимум 2 модели, найдено: {len(ckpt_paths)}")
        sys.exit(1)

    print(f"\n📦 Загружаю {len(ckpt_paths)} моделей...")
    models: List[Tuple[str, CapablancaNet]] = []
    for path in ckpt_paths:
        try:
            net, name, arch_tag, arch = load_model(path, device)
            models.append((name, net))
            print(f"  ✓ {name:<14} {arch}{arch_tag}")
        except Exception as e:
            print(f"  ✗ Ошибка загрузки {path}: {e}")

    if len(models) < 2:
        print("❌ Не удалось загрузить минимум 2 модели")
        sys.exit(1)

    # Parameters
    pairs = list(itertools.combinations(models, 2))
    total_games = len(pairs) * args.games
    print(f"\n🏆 Турнир: {len(models)} моделей, {len(pairs)} пар, "
          f"{total_games} партий всего")
    print(f"   Симуляций/ход: {args.simulations} | "
          f"Макс ходов: {args.max_moves} | "
          f"Температура: первые {args.temperature_moves} ходов")
    print(f"   compile={compile_mode or 'none'} | "
          f"KLD early-exit={'off' if args.kld <= 0 else args.kld}")

    # Round-robin tournament
    match_results = []
    t_total = time.time()

    # Писатель архива один на весь турнир: номер «итерации» берём из времени,
    # чтобы куски разных турниров не перетирали друг друга.
    archive_writer = None
    if args.archive_dir:
        archive_writer = archive_mod.ArchiveWriter(
            args.archive_dir, int(time.time()) % 1000000,
            int(np.asarray(CapablancaEngine().get_board_tensor()).size))

    for i, ((name_a, net_a), (name_b, net_b)) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}]", end="")
        result = run_match(
            name_a, net_a, name_b, net_b, device,
            games=args.games,
            simulations=args.simulations,
            max_moves=args.max_moves,
            temperature_moves=args.temperature_moves,
            mcts_batch=args.mcts_batch,
            parallel_sims=args.mcts_parallel_sims,
            parallel_sims_b=args.mcts_parallel_sims_b,
            c_puct=args.c_puct,
            c_puct_b=args.c_puct_b,
            fpu=args.fpu,
            fpu_b=args.fpu_b,
            policy_only_after=args.policy_only_after,
            policy_only_after_b=args.policy_only_after_b,
            ptemp=args.policy_temp,
            ptemp_b=args.policy_temp_b,
            verbose=args.verbose,
            pgn_dir=args.pgn_dir,
            timeout_as_draw=args.timeout_as_draw,
            compile_mode=compile_mode,
            kld_threshold=args.kld,
            adjudicate_q=args.adjudicate_q,
            adjudicate_plies=args.adjudicate_plies,
            arch=archive_writer,
        )
        match_results.append(result)

    if archive_writer is not None:
        n = archive_writer.close()
        print(f"\n  📚 В архив записано {n:,} позиций "
              f"({len(archive_writer.games['result']):,} партий матчей)".replace(",", " "))

    # Leaderboard
    print_leaderboard(models, match_results)
    print(f"\n⏱️  Общее время: {(time.time()-t_total)/60:.1f} мин")


if __name__ == "__main__":
    import signal as _signal

    def _save_archive_and_exit(signum, frame):
        files, rows = archive_mod.close_open_writers()
        if files:
            print(f"\n  💾 {_signal.Signals(signum).name}: архив дописан, "
                  f"{rows:,} позиций", flush=True)
        os._exit(128 + signum)
    _signal.signal(_signal.SIGTERM, _save_archive_and_exit)
    _signal.signal(_signal.SIGINT, _save_archive_and_exit)
    main()
