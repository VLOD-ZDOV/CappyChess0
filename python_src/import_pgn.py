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


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    adir, files = sys.argv[1], sys.argv[2:]
    total_g = total_p = skipped = 0
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0][:60]
        for moves, result in parse(open(path, errors="ignore").read()):
            n = A.archive_moves(adir, moves, result, A.SOURCE_HUMAN,
                                white=name, black=name)
            if n:
                total_g += 1
                total_p += n
            else:
                skipped += 1
    print(f"загружено {total_g} партий, {total_p} позиций"
          + (f"; пропущено {skipped} (ход не разобран)" if skipped else ""))
    return 0


if __name__ == "__main__":
    main()
