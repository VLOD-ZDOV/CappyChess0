#!/usr/bin/env python
"""Cheap test #1: footprint of --rep-search-perslot. No training, no GPU.
The buffer was encoded by get_board_tensor (per-slot rep flags). Measure on real
positions how often the HISTORICAL slots (1..7) carry a repetition flag — i.e.
how often the per-slot search encoding would differ from the legacy slot-0-only.
If ~0%, the lever changes almost nothing and isn't worth an iteration.
Plane layout: history slot h ∈0..8 → planes h*17+0..16, where +16 is the rep flag.
"""
import sys, numpy as np

PLANES_PER_BOARD = 17
HISTORY_LEN = 8
BOARD_SQ = 80
rep_plane = lambda h: h * PLANES_PER_BOARD + 16  # 16,33,50,67,84,101,118,135

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_big/buffer.npz"
    z = np.load(path)
    boards = z["boards"]                       # (N, 139*80) f16 or (N,139,8,10)
    n = boards.shape[0]
    sample = min(n, 300_000)
    idx = np.random.default_rng(0).choice(n, sample, replace=False)
    b = boards[idx].reshape(sample, -1, BOARD_SQ)   # (S, 139, 80)
    # A rep flag plane is all-ones across the 80 squares when set → check square 0.
    slot0 = b[:, rep_plane(0), 0] > 0.5
    hist = np.zeros(sample, dtype=bool)
    per_slot_counts = []
    for h in range(1, HISTORY_LEN):
        on = b[:, rep_plane(h), 0] > 0.5
        per_slot_counts.append(float(on.mean()))
        hist |= on
    print(f"buffer: {path}  positions sampled: {sample:,} / {n:,}")
    print(f"  slot-0 rep set (current position is a repetition): {slot0.mean()*100:.2f}%")
    print(f"  ANY historical slot (1..7) rep set                : {hist.mean()*100:.2f}%")
    print(f"    → fraction of positions where per-slot encoding DIFFERS from slot-0-only")
    print(f"  per-slot rep rate (slots 1..7): "
          f"{', '.join(f'{c*100:.2f}%' for c in per_slot_counts)}")
    diff = hist.mean()
    print(f"\nVERDICT: --rep-search-perslot changes ~{diff*100:.1f}% of positions.")
    if diff < 0.01:
        print("  → negligible footprint; expected near-zero effect. Low priority.")
    elif diff < 0.05:
        print("  → small footprint; minor effect at best.")
    else:
        print("  → non-trivial footprint; worth measuring output delta / A/B.")


if __name__ == "__main__":
    main()
