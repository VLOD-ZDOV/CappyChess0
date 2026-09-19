#!/usr/bin/env python
"""Партия по часам: сеть против Fairy-Stockfish, с настоящим контролем времени.

Время считает СУДЬЯ, а не участники: он замеряет по настенным часам, сколько
прошло от просьбы «ходи» до полученного хода, вычитает это из часов той стороны
и добавляет прибавку. Так устроены все турнирные менеджеры движков — иначе
подсудимый ведёт протокол сам.

Сеть ходит тем же Rust-поиском, что и во всех наших матчах (UltraFastMCTS),
меняется только условие остановки: вместо «сделай N симуляций» — «думай до
срока». Сколько симуляций влезет в бюджет, оценивается по фактической скорости
предыдущих ходов и уточняется по ходу партии.

    python play_timed.py <weights.pth> --tc 180+2 --nn-side white

Оговорка о честности: у нашей стороны есть постоянная плата за ход (поднять
дерево, собрать первую пачку, дождаться видеокарты), которой нет у движка на
C++. В блице она весит больше, чем в рапиде.
"""
import argparse
import os
import sys
import time
from typing import List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capablanca_engine import CapablancaEngine                          # noqa: E402
from mcts import UltraFastMCTS                                          # noqa: E402
from model import build_net_from_state_dict                             # noqa: E402
from play_fsf import FairyStockfishWrapper, int_to_uci, uci_to_int      # noqa: E402

FSF_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fairy-stockfish-largeboard_x86-64-bmi2")


class TimedFSF(FairyStockfishWrapper):
    """Тот же движок, но по часам: получает остатки на обоих и сам решает,
    сколько думать. Распределение бюджета — его родной режим, он это умеет."""

    def __init__(self, path: str, threads: int = 4, hash_mb: int = 256):
        super().__init__(path)
        # Сколько ядер даём сопернику — часть условий матча, её надо объявлять
        # вместе с результатом: наша сторона считает на видеокарте, его — на
        # процессоре, и «равное железо» тут вопрос договорённости, а не факта.
        self.send(f"setoption name Threads value {threads}")
        self.send(f"setoption name Hash value {hash_mb}")
        self.send("isready")
        self._wait_for("readyok")

    def best_move_timed(self, history: List[str], wtime_ms: int, btime_ms: int,
                        winc_ms: int, binc_ms: int) -> str:
        self.send(f"position startpos moves {' '.join(history)}")
        self.send(f"go wtime {wtime_ms} btime {btime_ms} "
                  f"winc {winc_ms} binc {binc_ms}")
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                return line.split()[1]


def budget_seconds(remaining: float, inc: float, moves_made: int) -> float:
    """Сколько думать над ходом. Классическое правило: доля остатка плюс почти
    вся прибавка. В дебюте чуть скромнее — там цена ошибки ниже, а позиций,
    которые сеть уже видела в self-play, больше."""
    share = 30.0 if moves_made > 12 else 40.0
    t = remaining / share + inc * 0.8
    return max(0.05, min(t, remaining * 0.4))


def save_pgn(path, history, result, nn_side, tc, threads):
    """PGN с ходами в UCI: его читает и наш gui.py (кнопка «Открыть партию»), и
    разбор партий на pychess.org, если указать вариант капабланка."""
    import datetime
    white = "CapablancaNet" if nn_side == 0 else "Fairy-Stockfish"
    black = "Fairy-Stockfish" if nn_side == 0 else "CapablancaNet"
    tag = {"победа белых": "1-0", "победа чёрных": "0-1",
           "ничья": "1/2-1/2"}.get(result, "*")
    body = ""
    for i in range(0, len(history), 2):
        body += f"{i // 2 + 1}. {history[i]} "
        if i + 1 < len(history):
            body += f"{history[i + 1]} "
    with open(path, "w", encoding="utf-8") as f:
        f.write('[Event "Матч по часам"]\n')
        f.write(f'[Date "{datetime.datetime.now():%Y.%m.%d}"]\n')
        f.write(f'[White "{white}"]\n[Black "{black}"]\n')
        f.write(f'[Result "{tag}"]\n[Variant "capablanca"]\n')
        f.write(f'[TimeControl "{tc}"]\n')
        f.write(f'[Termination "{result}"]\n')
        f.write(f'[Annotator "движок на {threads} ядрах, сеть на видеокарте"]\n')
        f.write('[FEN "rnabqkcbnr/pppppppppp/10/10/10/10/PPPPPPPPPP/RNABQKCBNR w KQkq - 0 1"]\n')
        f.write('[SetUp "1"]\n\n')
        f.write(body + tag + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tc", default="180+2", help="контроль в секундах: база+прибавка")
    ap.add_argument("--nn-side", default="white", choices=["white", "black"])
    ap.add_argument("--parallel-sims", type=int, default=8)
    ap.add_argument("--mcts-batch", type=int, default=8)
    ap.add_argument("--sim-cap", type=int, default=400_000)
    ap.add_argument("--fsf-path", default=FSF_DEFAULT)
    ap.add_argument("--max-moves", type=int, default=400)
    ap.add_argument("--fpu", type=float, default=None,
                    help="Пессимизм поиска к непосещённым ходам; в игре "
                         "1.0 доказан матчем (+72 Elo к умолчанию 0.33)")
    ap.add_argument("--fsf-threads", type=int, default=4)
    ap.add_argument("--fsf-hash", type=int, default=256)
    ap.add_argument("--archive-dir", default="",
                    help="писать партию в архив")
    ap.add_argument("--pgn", default=None, help="куда записать партию (PGN)")
    a = ap.parse_args()

    base, inc = (float(x) for x in a.tc.replace("+", " ").split())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    net = build_net_from_state_dict(sd)
    net = net[0] if isinstance(net, tuple) else net
    # build_net_from_state_dict только СТРОИТ архитектуру, веса грузит вызывающий.
    # Пропустить этот шаг — играть случайной сетью, причём молча: ошибок не будет,
    # просто ходы окажутся бессмысленными.
    res = net.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        sys.exit(f"веса не сошлись: пропущено {len(res.missing_keys)}, "
                 f"лишних {len(res.unexpected_keys)}")
    net = net.to(dev).eval()
    mcts = UltraFastMCTS(net, dev, c_puct=1.745, batch_size=a.mcts_batch,
                         parallel_sims=a.parallel_sims, add_dirichlet=False,
                         compile_mode=None, nn_cache=True, rust_fpu=a.fpu)
    fsf = TimedFSF(a.fsf_path, threads=a.fsf_threads, hash_mb=a.fsf_hash)

    engine = CapablancaEngine()
    nn_side = 0 if a.nn_side == "white" else 1
    clock = [base, base]                       # часы белых и чёрных
    history: List[str] = []
    rate = 300.0                               # симуляций в секунду, уточняется по ходу
    print(f"Контроль {base:.0f}+{inc:.0f}, сеть играет "
          f"{'белыми' if nn_side == 0 else 'чёрными'}, "
          f"соперник — Fairy-Stockfish на {a.fsf_threads} ядрах")
    print(f"{'ход':>4} {'сторона':>8} {'время':>7} {'остаток':>8} {'симуляций':>10}")

    move_num = 0
    result = None
    while not engine.is_game_over() and move_num < a.max_moves:
        side = engine.side_to_move()
        t0 = time.perf_counter()

        if side == nn_side:
            budget = budget_seconds(clock[side], inc, move_num)
            deadline = t0 + budget
            sims = max(a.parallel_sims, min(a.sim_cap, int(rate * budget)))
            pol = mcts.search_games([engine], sims, deadline=deadline)[0]
            legal = engine.get_legal_moves_int()
            idx = [engine.move_int_to_policy_idx(m) for m in legal]
            scores = np.array([pol[i] if i is not None else -1.0 for i in idx])
            mv = int(legal[int(scores.argmax())])
            uci = int_to_uci(mv)
            spent = time.perf_counter() - t0
            done = getattr(mcts, "last_sims_done", sims)
            rate = 0.7 * rate + 0.3 * (done / max(spent, 1e-3))
            shown = done
        else:
            uci = fsf.best_move_timed(history,
                                      int(clock[0] * 1000), int(clock[1] * 1000),
                                      int(inc * 1000), int(inc * 1000))
            if uci == "(none)":
                break
            mv = uci_to_int(uci, engine)
            if mv is None:
                print(f"рассинхронизация: движок предложил {uci}")
                break
            spent = time.perf_counter() - t0
            shown = "—"

        clock[side] -= spent
        if clock[side] <= 0:
            result = "проигрыш по времени у " + ("сети" if side == nn_side else "движка")
            print(f"{move_num + 1:>4} {'сеть' if side == nn_side else 'движок':>8} "
                  f"{spent:>7.2f} {'флаг':>8}")
            break
        clock[side] += inc

        print(f"{move_num + 1:>4} {'сеть' if side == nn_side else 'движок':>8} "
              f"{spent:>7.2f} {clock[side]:>8.1f} {shown:>10}", flush=True)
        engine.make_move_int(mv)
        history.append(uci)
        move_num += 1

    if result is None:
        r = engine.game_result()
        result = {1: "победа белых", -1: "победа чёрных", 0: "ничья"}.get(
            r, f"партия оборвана на {move_num} полуходах")
    print(f"\nИтог: {result}")
    print(f"Часы: белые {clock[0]:.1f} с, чёрные {clock[1]:.1f} с")
    if a.pgn:
        save_pgn(a.pgn, history, result, nn_side, a.tc, a.fsf_threads)
    if a.archive_dir:
        import archive as A
        n = A.archive_moves(a.archive_dir, history,
                            None if not engine.is_game_over() else int(r),
                            A.SOURCE_FSF,
                            white="сеть" if nn_side == 0 else "движок",
                            black="движок" if nn_side == 0 else "сеть")
        print(f"в архив записано {n} позиций")
        print(f"Партия записана в {a.pgn} — открывается кнопкой «Открыть партию» в gui.py")
    fsf.close()


if __name__ == "__main__":
    main()
