#!/usr/bin/env python
"""Загрузка партий из PGN (или простого списка ходов) в архив.

    python import_pgn.py checkpoints_v11/archive мои_партии.pgn [ещё.pgn ...]

Зачем отдельным путём: партии, сыгранные человеком в GUI или взятые со стороны,
приходят как список ходов — без распределения визитов и без оценок поиска. Их
нельзя использовать как цель для политики, зато они полноценны как ЗАПИСЬ
ПАРТИИ: позиция, сделанный ход, исход. Этого хватает и для книги, и для
статистики, и для отбора позиций.

Поэтому они помечаются `source = человек`, а политика у них пустая — читатель
архива видит это сразу и не спутает их с самоигрой.

Формат разбирается тот же, что понимает GUI: берутся токены, похожие на ход UCI
(`e2e4`, `f7f8q`), а номера ходов, заголовки в квадратных скобках, комментарии и
знаки результата пропускаются.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archive as A                                   # noqa: E402
from capablanca_engine import CapablancaEngine        # noqa: E402

MOVE_RE = re.compile(r"^[a-j][1-8][a-j][1-8][nbrqac]?$")
RESULT = {"1-0": 1, "0-1": -1, "1/2-1/2": 0, "½-½": 0, "*": None}


def parse(text):
    """→ список (ходы, результат). Партии разделяются строкой результата."""
    games, moves, result = [], [], None
    text = re.sub(r"\{[^}]*\}", " ", text)          # комментарии
    for tok in text.split():
        if tok.startswith("["):
            continue
        if tok in RESULT:
            result = RESULT[tok]
            if moves:
                games.append((moves, result))
                moves, result = [], None
            continue
        t = tok.rstrip(".").lower()
        t = t.split(".")[-1]                         # «12.e2e4»
        if MOVE_RE.match(t):
            moves.append(t)
    if moves:
        games.append((moves, result))
    return games


def uci_to_move(eng, uci):
    """UCI → ход движка, сверяясь со списком легальных в ЭТОЙ позиции."""
    for m in eng.get_legal_moves_int():
        f, t, p = (m >> 10) & 0x7F, (m >> 3) & 0x7F, m & 0b111
        s = (f"{chr(ord('a') + f % 10)}{f // 10 + 1}"
             f"{chr(ord('a') + t % 10)}{t // 10 + 1}")
        if p:
            s += " nbrqac"[p]
        if s == uci:
            return m
    return None


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    adir, files = sys.argv[1], sys.argv[2:]
    board_len = int(np.asarray(CapablancaEngine().get_board_tensor()).size)
    import time
    w = A.ArchiveWriter(adir, 800000 + (int(time.time()) % 90000), board_len)
    total_g = total_p = skipped = 0

    for path in files:
        games = parse(open(path, errors="ignore").read())
        name = os.path.splitext(os.path.basename(path))[0][:60]
        for moves, result in games:
            eng = CapablancaEngine()
            rows, ok = [], True
            for uci in moves:
                mv = uci_to_move(eng, uci)
                if mv is None:
                    ok = False
                    break
                rows.append((np.asarray(eng.get_board_tensor(), dtype=np.float32),
                             eng.side_to_move(), mv))
                eng.make_move_int(mv)
            if not ok or not rows:
                skipped += 1
                continue
            # Результат из PGN, а если его нет — доиграно ли до конца по правилам.
            if result is None:
                result = int(eng.game_result()) if eng.is_game_over() else 0
            term = (A.TERM_MATE if eng.is_game_over() and abs(result) > 0.5
                    else A.DRAW_REASON_TO_TERM.get(eng.draw_reason(), A.TERM_DRAW_RULE)
                    if eng.is_game_over() else A.TERM_LIMIT)
            g_id = w.add_game(result=result, plies=len(rows), term=term,
                              playthrough=0, resign_ply=-1,
                              source=A.SOURCE_HUMAN,
                              white_id=w.name_id(name), black_id=w.name_id(name))
            for ply, (board, side, mv) in enumerate(rows):
                # Политики нет — пустой список, а не выдуманное распределение.
                w.add_position(board, [], [], game=g_id, ply=ply, side=side,
                               full=False, root_q=0.0, root_d=-1.0,
                               move=-1, move_raw=int(mv))
            total_g += 1
            total_p += len(rows)

    n = w.close()
    print(f"загружено {total_g} партий, {total_p} позиций"
          + (f"; пропущено {skipped} (ход не разобран)" if skipped else ""))
    print(f"в архиве {n} новых строк · помечены как «человек», политики у них нет")


if __name__ == "__main__":
    main()
