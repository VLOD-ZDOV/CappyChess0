#!/usr/bin/env python3
"""Quick look at a replay buffer's value-target distribution.

    python check_buffer.py [checkpoints/buffer.npz]

Reads the numpy archive train.py writes (`save_npz`), falling back to the
legacy pickle if only that is present.
"""
import os
import pickle
import sys

import numpy as np


def load_values(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        with np.load(path) as z:
            return z["values"].astype(np.float32)
    with open(path, "rb") as f:
        data, _ptr, _full = pickle.load(f)
    return np.array([s[2] for s in data], dtype=np.float32)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/buffer.npz"
    if not os.path.exists(path) and path.endswith(".npz"):
        legacy = path[:-4] + ".pkl"
        if os.path.exists(legacy):
            path = legacy
    if not os.path.exists(path):
        print(f"❌ Буфер не найден: {path}")
        sys.exit(1)

    values = load_values(path)
    n = len(values)
    print(f"Буфер: {path}")
    print(f"Всего позиций: {n:,}")
    if n == 0:
        return

    print("\nРаспределение value-таргетов:")
    # Q-blended targets are continuous, so bucket by sign instead of testing
    # for exact ±1.0 / 0.0 (which only ever matched pure game-outcome targets).
    for label, mask in (
        ("победа белых (v >  0.15)", values > 0.15),
        ("ничья       (|v| ≤ 0.15)", np.abs(values) <= 0.15),
        ("победа чёрн (v < -0.15)", values < -0.15),
    ):
        cnt = int(mask.sum())
        print(f"  {label}: {cnt:>9,}  ({100 * cnt / n:5.1f}%)")

    exact = int(np.isin(values, (-1.0, 0.0, 1.0)).sum())
    print(f"\nРовно ±1/0 (чистый game-outcome): {exact:,} ({100 * exact / n:.1f}%)")
    print(f"MSE если всегда предсказывать 0: {(values ** 2).mean():.6f}")
    print(f"Среднее={values.mean():+.6f}  Std={values.std():.6f}")


if __name__ == "__main__":
    main()
