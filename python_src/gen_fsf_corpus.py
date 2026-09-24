#!/usr/bin/env python
"""Генератор партий движка против себя — корпус для архива.

    python gen_fsf_corpus.py <архив> [--threads 6] [--hours 0]

Зачем. Матом кончается лишь 8% наших самоигровых партий: сеть получает перевес и
не умеет его довести. Движок умеет. Эти партии НЕ идут в обучающий буфер (доза
8% уже утаскивала value в пессимизм за четыре итерации) — они копятся в архиве
как корпус: эндшпильная техника, книга, материал для засева следующей линии.

Узлы разные по стадиям. Скорость движка от стадии почти не зависит (750k узлов/с
в начале, 890k в эндшпиле), поэтому дело не в экономии, а в том, где точность
нужна: дебют мы всё равно берём из книги, а ценность корпуса — в окончаниях.

    первые 30 полуходов   100k узлов (глубина 11)
    дальше                300k узлов (глубина 13)

Разнообразие даёт книга: каждая партия начинается со своей линии из 784, иначе
движок против себя играл бы одну и ту же партию снова и снова.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archive as A                                   # noqa: E402
from capablanca_engine import CapablancaEngine        # noqa: E402

FSF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fairy-stockfish-largeboard_x86-64-bmi2")
BOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "experiments", "book2.json")
# Сильная и слабая стороны. РАВНЫЕ движки почти всегда делают ничью: в пробе
# 11 партий дали 0 матов и 64% упора в лимит ходов. Корпус нужен ради техники
# реализации, поэтому перевес создаётся искусственно, а сторона перевеса
# чередуется. Слабому хватает глубины 9, чтобы не зевать сразу.
EARLY_PLIES = 30
STRONG_EARLY, STRONG_LATE = 100_000, 300_000
WEAK_NODES = 20_000
MAX_PLIES = 300
FLUSH_EVERY = 25          # партий на кусок архива


class Engine:
    def __init__(self):
        self.p = subprocess.Popen([FSF], universal_newlines=True, bufsize=1,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._cmd("uci", "uciok")
        self.send("setoption name UCI_Variant value capablanca")
        # Без EvalFile движок считает КЛАССИЧЕСКОЙ оценкой — это минус 417 Elo.
        # Все остальные вызовы (gen_book4, train, play_fsf) сеть подключают;
        # здесь её забыли, и корпус до 21.09 сгенерирован слабым движком.
        _net = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "capablanca-bb644ef32758.nnue")
        if os.path.exists(_net):
            self.send(f"setoption name EvalFile value {_net}")
        self._cmd("isready", "readyok")

    def send(self, c):
        self.p.stdin.write(c + "\n"); self.p.stdin.flush()

    def _cmd(self, c, wait):
        self.send(c)
        while wait not in self.p.stdout.readline():
            pass

    def best(self, moves, nodes):
        self.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))
        self.send(f"go nodes {nodes}")
        while True:
            l = self.p.stdout.readline()
            if not l:
                return None
            if l.startswith("bestmove"):
                mv = l.split()[1]
                return None if mv in ("(none)", "0000") else mv

    def close(self):
        try:
            self.send("quit"); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


def play(eng, opening, strong_side):
    """Партия движка против себя от книжной позиции. → (ходы, исход) или None."""
    board = CapablancaEngine()
    moves = []
    for uci in opening:
        mv = A._uci_to_move(board, uci)
        if mv is None:
            return None
        board.make_move_int(mv); moves.append(uci)
    while len(moves) < MAX_PLIES and not board.is_game_over():
        if board.side_to_move() == strong_side:
            nodes = STRONG_EARLY if len(moves) < EARLY_PLIES else STRONG_LATE
        else:
            nodes = WEAK_NODES
        uci = eng.best(moves, nodes)
        if uci is None:
            break
        mv = A._uci_to_move(board, uci)
        if mv is None:
            break
        board.make_move_int(mv); moves.append(uci)
    if board.is_game_over():
        r = int(board.game_result())
        term = (A.TERM_MATE if abs(r) > 0.5
                else A.DRAW_REASON_TO_TERM.get(board.draw_reason(), A.TERM_DRAW_RULE))
    else:
        m = board.material_result()
        r = 1 if m > 0.5 else (-1 if m < -0.5 else 0)
        term = A.TERM_LIMIT
    return moves, r, term


def worker(wid, adir, lines, deadline):
    eng = Engine()
    rng = random.Random(1000 + wid)
    board_len = int(np.asarray(CapablancaEngine().get_board_tensor()).size)
    w = None
    n_games = n_pos = n_mate = 0
    batch = 0
    shard = int(time.time()) % 90000
    while deadline == 0 or time.time() < deadline:
        L = rng.choice(lines)
        strong = rng.randint(0, 1)
        out = play(eng, L["moves"], strong)
        if out is None:
            continue
        moves, r, term = out
        if w is None:
            # Номер куска должен быть уникален и между перезапусками, иначе
            # новый кусок молча затрёт старый.
            w = A.ArchiveWriter(adir, 500000 + wid * 100000 + shard, board_len)
            shard += 1
        board = CapablancaEngine()
        rows = []
        ok = True
        for uci in moves:
            mv = A._uci_to_move(board, uci)
            if mv is None:
                ok = False; break
            rows.append((np.asarray(board.get_board_tensor(), dtype=np.float32),
                         board.side_to_move(), mv))
            board.make_move_int(mv)
        if not ok:
            continue
        g = w.add_game(result=r, plies=len(rows), term=term, playthrough=0,
                       resign_ply=-1, source=A.SOURCE_FSF,
                       white_id=w.name_id("движок+" if strong == 0 else "движок-"),
                       black_id=w.name_id("движок-" if strong == 0 else "движок+"))
        for ply, (b, side, mv) in enumerate(rows):
            w.add_position(b, [], [], game=g, ply=ply, side=side, full=False,
                           root_q=0.0, root_d=-1.0, move=-1, move_raw=int(mv))
        n_games += 1; n_pos += len(rows); batch += 1
        n_mate += (term == A.TERM_MATE)
        if batch >= FLUSH_EVERY:
            w.close(); w = None; batch = 0
            print(f"[поток {wid}] {n_games} партий, {n_pos} позиций, "
                  f"матом {n_mate} ({100*n_mate/n_games:.0f}%)", flush=True)
    if w is not None:
        w.close()
    eng.close()
    print(f"[поток {wid}] всего {n_games} партий, {n_pos} позиций", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--hours", type=float, default=0, help="0 = без ограничения")
    a = ap.parse_args()
    lines = json.load(open(BOOK))["lines"]
    deadline = time.time() + a.hours * 3600 if a.hours else 0
    print(f"{a.threads} потоков, книга {len(lines)} линий, "
          f"сильный {STRONG_EARLY//1000}k/{STRONG_LATE//1000}k, слабый {WEAK_NODES//1000}k", flush=True)
    import multiprocessing as mp
    ps = [mp.Process(target=worker, args=(i, a.archive, lines, deadline))
          for i in range(a.threads)]
    for p in ps: p.start()
    try:
        for p in ps: p.join()
    except KeyboardInterrupt:
        for p in ps: p.terminate()


if __name__ == "__main__":
    main()
