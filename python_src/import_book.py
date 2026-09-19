#!/usr/bin/env python
"""Перенос перебора дебюта из jsonl в архив.

    python import_book.py <архив> experiments/book4.jsonl

Возобновляемо: уже перенесённые линии пропускаются, так что можно запускать
хоть каждый час, пока генератор ещё считает.

Оценка движка кладётся в root_q последней позиции линии, переведённая в шкалу
[-1, 1] логистикой с масштабом 300 сантипешек. Исхода у этих записей нет —
они помечены `перебор дебюта` и из статистики книги исключаются.
"""
import argparse, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archive as A
from capablanca_engine import CapablancaEngine

BATCH = 5000


def cp_to_v(cp, mate):
    if mate is not None:
        return 0.99 if mate > 0 else -0.99
    if cp is None:
        return 0.0
    return max(-0.999, min(0.999, 2.0 / (1.0 + math.exp(-cp / 300.0)) - 1.0))


def already(adir):
    """Линии, уже лежащие в архиве: ключ — ходы в координатах доски."""
    try:
        ar = A.Archive(adir)
    except FileNotFoundError:
        return set()
    m = ar.g["source"] == A.SOURCE_BOOK
    if not m.any():
        return set()
    out = set()
    for g in np.flatnonzero(m):
        sel = ar.p["game"] == g
        mv = ar.p["move_raw"][sel][np.argsort(ar.p["ply"][sel])]
        out.add(tuple(int(x) for x in mv))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive"); ap.add_argument("jsonl")
    a = ap.parse_args()
    seen = already(a.archive)
    print(f"в архиве уже {len(seen):,} линий перебора".replace(",", " "))

    board_len = int(np.asarray(CapablancaEngine().get_board_tensor()).size)
    w = None; n_g = n_p = skipped = 0; shard = 400000
    for line in open(a.jsonl, errors="ignore"):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        eng = CapablancaEngine(); rows = []; ok = True
        for u in rec["moves"]:
            mv = A._uci_to_move(eng, u)
            if mv is None:
                ok = False; break
            rows.append((np.asarray(eng.get_board_tensor(), dtype=np.float32),
                         eng.side_to_move(), mv))
            eng.make_move_int(mv)
        if not ok or not rows:
            skipped += 1; continue
        key = tuple(int(m) for _, _, m in rows)
        if key in seen:
            continue
        seen.add(key)
        if w is None:
            w = A.ArchiveWriter(a.archive, shard, board_len); shard += 1
        v = cp_to_v(rec.get("cp"), rec.get("mate"))
        g = w.add_game(result=0, plies=len(rows), term=A.TERM_LIMIT,
                       playthrough=0, resign_ply=-1, source=A.SOURCE_BOOK,
                       white_id=w.name_id("перебор"), black_id=w.name_id("перебор"))
        for ply, (b, side, mv) in enumerate(rows):
            last = ply == len(rows) - 1
            w.add_position(b, [], [], game=g, ply=ply, side=side, full=False,
                           root_q=(v if side == 0 else -v) if last else 0.0,
                           root_d=-1.0, move=-1, move_raw=int(mv))
        n_g += 1; n_p += len(rows)
        if n_g % BATCH == 0:
            w.close(); w = None
            print(f"перенесено {n_g:,} линий".replace(",", " "), flush=True)
    if w is not None:
        w.close()
    print(f"итого перенесено {n_g:,} линий, {n_p:,} позиций"
          .replace(",", " ") + (f"; пропущено {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
