"""Пересборка обучающих образцов из архива.

Архив хранит ФАКТЫ (исход партии, оценка корня, сделанный ход), а не готовые
цели. Здесь факты превращаются обратно в то, чем кормят сеть, — по тем же
формулам, что и самоигра. Смысл в том, что цель ПЕРЕСЧИТЫВАЕТСЯ: можно взять
другой `value_q_weight`, другую нормировку MLH, другой набор партий, и всё это
не требует переигрывать ни одной партии.

Формулы обязаны совпадать с `_finish` в train.py. Проверяется это не чтением, а
сверкой: `--verify` берёт настоящий кусок буфера за ту же итерацию и сравнивает
побайтово. Разошлись — значит формулы разъехались.

Осторожно с отбором. Буфер обучения должен быть примерно ON-POLICY, то есть
состоять из партий, сыгранных ТЕКУЩЕЙ сетью. Набор, отобранный из архива, —
это уже обучение с учителем на неподвижных данных: сеть сойдётся к тому, кто
эти партии играл, и перестанет расти. И отдельно опасен отбор ПО ИСХОДУ: если
оставить только победы, цель value везде +1, и голова разучится отличать
выигранную позицию от проигранной. Безопасно отбирать по признакам, не
связанным с меткой (итерация, полный ли поиск, доигранная ли партия) — а брать
только победы или только маты можно лишь как ДОБАВКУ к обычному буферу.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archive as A         # noqa: E402
import buffer_io            # noqa: E402

MLH_PLY_NORM = 200.0        # как в train.py


def build(ar, pos_mask, value_q_weight=0.25, mlh_norm=MLH_PLY_NORM):
    """Образцы в формате буфера: (доска, (idx, val), value, mlh, future, draw)."""
    idx = np.flatnonzero(pos_mask)
    boards = ar.boards(pos_mask)
    pol = ar.policy(pos_mask)
    game = ar.p["game"][idx]
    ply = ar.p["ply"][idx].astype(np.int32)
    side = ar.p["side"][idx]
    root_q = ar.p["root_q"][idx].astype(np.float32)
    root_d = ar.p["root_d"][idx].astype(np.float32)
    move = ar.p["move"][idx]
    result = ar.g["result"][game].astype(np.float32)
    plies = ar.g["plies"][game].astype(np.int32)

    # z — исход глазами того, кто ходит.
    z = np.where(side == 0, result, -result)
    v = z.copy()
    draw = np.full(z.shape, -1.0, dtype=np.float32)
    if value_q_weight > 0.0:
        w = value_q_weight
        v = (1.0 - w) * z + w * root_q
        has_d = root_d >= 0.0
        z_is_draw = (np.abs(result) < 1e-6).astype(np.float32)
        draw = np.where(has_d, (1.0 - w) * z_is_draw + w * root_d, -1.0)

    mlh = np.minimum(1.0, np.maximum(0, plies - ply) / mlh_norm)

    # future — ход через два полухода В ТОЙ ЖЕ партии. Позиции идут по порядку,
    # поэтому сосед через две строки годится, только если он из той же партии и
    # действительно на два полухода дальше: быстрые позиции могли быть отсеяны.
    future = np.full(idx.shape, -1, dtype=np.int64)
    key = {(int(g), int(p)): i for i, (g, p) in enumerate(zip(game, ply))}
    for i, (g, p) in enumerate(zip(game, ply)):
        j = key.get((int(g), int(p) + 2))
        if j is not None:
            future[i] = move[j]

    return [(boards[i], (pol[i][0], pol[i][1]), float(v[i]), float(mlh[i]),
             int(future[i]), float(draw[i])) for i in range(idx.size)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="каталог архива")
    ap.add_argument("--iters", type=int, nargs="+", help="только эти итерации")
    ap.add_argument("--value-q-weight", type=float, default=0.25)
    ap.add_argument("--out", help="куда записать кусок буфера (.npz)")
    ap.add_argument("--verify", metavar="КУСОК.npz",
                    help="сверить с настоящим куском буфера за ту же итерацию")
    args = ap.parse_args()

    ar = A.Archive(args.dir)
    gm = ar.games_where(iters=args.iters)
    # В буфер обучения идут ТОЛЬКО позиции полного поиска — как в самоигре.
    pm = ar.positions_where(gm, full=True)
    print(f"{int(gm.sum())} партий, {int(pm.sum())} позиций полного поиска")
    samples = build(ar, pm, args.value_q_weight)

    if args.verify:
        a = buffer_io.read_archive(args.verify)
        ref = buffer_io.arrays_to_samples(a)
        bad = []
        if len(ref) != len(samples):
            bad.append(f"число позиций: {len(samples)} против {len(ref)}")
        for i, (x, y) in enumerate(zip(samples, ref)):
            if not np.array_equal(np.asarray(x[0], dtype=np.float16),
                                  np.asarray(y[0], dtype=np.float16)):
                bad.append(f"доска {i}"); break
            if not np.array_equal(x[1][0], y[1][0]) or \
               not np.array_equal(np.asarray(x[1][1], dtype=np.float16),
                                  np.asarray(y[1][1], dtype=np.float16)):
                bad.append(f"политика {i}"); break
            for k, name in ((2, "value"), (3, "mlh"), (5, "draw")):
                if abs(np.float16(x[k]) - np.float16(y[k])) > 1e-3:
                    bad.append(f"{name} {i}: {x[k]} против {y[k]}"); break
            if int(x[4]) != int(y[4]):
                bad.append(f"future {i}: {int(x[4])} против {int(y[4])}")
            if bad:
                break
        if not bad:
            print("✅ пересобранное сходится с настоящим буфером по всем полям")
        else:
            print(f"❌ расхождение: {bad[0]}")
            if "future" in bad[0]:
                print("   (ожидаемо для кусков архива до 16.09 вечера: столбец "
                      "со сделанным ходом появился позже, и цель future из них "
                      "не восстановить)")
        if bad:
            return 1

    if args.out:
        buffer_io.write_samples(args.out, samples, [len(samples), 0])
        print(f"записано {len(samples)} позиций в {args.out} "
              f"({os.path.getsize(args.out)/1e6:.0f} МБ логически)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
