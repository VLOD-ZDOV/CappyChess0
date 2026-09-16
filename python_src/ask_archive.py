#!/usr/bin/env python
"""Запросы к архиву самоигры из командной строки.

    python ask_archive.py checkpoints_v11/archive
    python ask_archive.py <архив> --result -1 --moves 34
    python ask_archive.py <архив> --term мат --max-plies 40 --pgn wins.pgn
    python ask_archive.py <архив> --outcome loss --full --export loss.npz

Отбор идёт в два шага: сначала партии по их свойствам, потом позиции внутри
отобранных партий. Доски читаются только у того, что действительно попросили,
поэтому запрос к архиву в сотни гигабайт стоит столько же, сколько к маленькому.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archive as A     # noqa: E402

TERMS = {"мат": A.TERM_MATE, "сдача": A.TERM_RESIGN, "ничья": A.TERM_DRAW_RULE,
         "лимит": A.TERM_LIMIT, "судья": A.TERM_ADJUDICATED}


def pair(s):
    """`34` или `20-40` — одиночное значение либо диапазон."""
    if "-" in s.lstrip("-"):
        a, b = s.split("-", 1) if not s.startswith("-") else (s, s)
        return (int(a), int(b))
    return int(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="каталог архива")
    g = ap.add_argument_group("отбор партий")
    g.add_argument("--result", type=int, choices=(-1, 0, 1),
                   help="+1 победа белых, 0 ничья, -1 победа чёрных")
    g.add_argument("--decisive", action="store_true", help="только результативные")
    g.add_argument("--term", choices=sorted(TERMS), help="чем кончилась партия")
    g.add_argument("--moves", type=pair, help="на каком ПОЛНОМ ходу кончилась (34 или 20-40)")
    g.add_argument("--min-plies", type=int)
    g.add_argument("--max-plies", type=int)
    g.add_argument("--iters", type=pair, help="итерации обучения")
    g.add_argument("--playthrough", action="store_true", help="только доигранные партии")
    g.add_argument("--false-resign", action="store_true",
                   help="доигранные, где приговор о сдаче оказался неверным")
    p = ap.add_argument_group("отбор позиций")
    p.add_argument("--outcome", choices=("win", "loss", "draw"),
                   help="исход глазами того, кто ходит")
    p.add_argument("--side", type=int, choices=(0, 1), help="0 белые, 1 чёрные")
    p.add_argument("--full", action="store_true", help="только полный поиск")
    p.add_argument("--fast", action="store_true", help="только быстрый поиск")
    p.add_argument("--min-ply", type=int)
    p.add_argument("--max-ply", type=int)
    o = ap.add_argument_group("вывод")
    o.add_argument("--book", type=int, metavar="N", nargs="?", const=2,
                   help="вывести книгу: позиции, встреченные хотя бы N раз (по "
                        "умолчанию 2), с ходами и их результатами")
    o.add_argument("--export", metavar="ФАЙЛ.npz", help="выгрузить доски и политику")
    o.add_argument("--limit", type=int, help="не больше стольких позиций")
    args = ap.parse_args()

    ar = A.Archive(args.dir)
    print(ar.summary(), "\n")

    iters = args.iters
    if isinstance(iters, tuple):
        iters = list(range(iters[0], iters[1] + 1))
    gm = ar.games_where(
        result=args.result, decisive=args.decisive,
        term=TERMS[args.term] if args.term else None,
        moves=args.moves, min_plies=args.min_plies, max_plies=args.max_plies,
        iters=iters, playthrough=True if args.playthrough else None,
        false_resign=args.false_resign)
    full = True if args.full else (False if args.fast else None)
    pm = ar.positions_where(gm, full=full, side=args.side,
                            min_ply=args.min_ply, max_ply=args.max_ply,
                            outcome=args.outcome)
    if args.limit:
        idx = np.flatnonzero(pm)[args.limit:]
        pm[idx] = False

    n_g, n_p = int(gm.sum()), int(pm.sum())
    print(f"отобрано: {n_g} партий, {n_p} позиций")
    if n_g:
        pl = ar.g["plies"][gm]
        res = ar.g["result"][gm]
        print(f"  длина партий: {pl.min()}–{pl.max()} полуходов, в среднем {pl.mean():.0f}")
        print(f"  исходы: белые {int((res>0).sum())}, чёрные {int((res<0).sum())}, "
              f"ничьи {int((res==0).sum())}")
        for t, name in A.TERM_NAMES.items():
            c = int((ar.g["term"][gm] == t).sum())
            if c:
                print(f"  {name}: {c}")
    if not n_p:
        return 0

    if args.book:
        book = ar.book_from(pm, min_games=args.book)
        print(f"\nкнига: {len(book)} позиций, встреченных ≥{args.book} раз")
        top = sorted(book.items(), key=lambda kv: -kv[1]["n"])[:15]
        for k, e in top:
            moves = sorted(e["moves"].items(), key=lambda kv: -kv[1][0])[:4]
            mv_s = ", ".join(f"ход {m}: {c}× ({100*sc/c:.0f}%)" for m, (c, sc) in moves)
            print(f"  позиция встречена {e['n']:5d}× · очки ходящего "
                  f"{100*e['score']/e['n']:.0f}% · {mv_s}")

    if args.export:
        b = ar.boards(pm)
        pol = ar.policy(pm)
        k = max((len(i) for i, _ in pol), default=0)
        pi = np.full((len(pol), k), -1, dtype=np.int16)
        pv = np.zeros((len(pol), k), dtype=np.float16)
        for i, (a, v) in enumerate(pol):
            pi[i, :len(a)] = a; pv[i, :len(v)] = v
        cols = {c: ar.p[c][pm] for c in A.POS_COLS}
        cols.update({"g_" + c: ar.g[c][ar.p["game"][pm]] for c in A.GAME_COLS})
        np.savez(args.export, boards=b, pol_idx=pi, pol_val=pv, **cols)
        print(f"\nвыгружено в {args.export}: {b.shape[0]} позиций, "
              f"{os.path.getsize(args.export)/1e6:.1f} МБ на диске")
    return 0


if __name__ == "__main__":
    sys.exit(main())
