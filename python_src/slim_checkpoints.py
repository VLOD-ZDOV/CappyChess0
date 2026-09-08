#!/usr/bin/env python
"""Shrink the checkpoint/buffer tree losslessly-for-play (variant 2).

Writes ONLY into sibling `<dir>_slim/` folders; never touches originals.
Rules:
  - buffer.npz  -> savez_compressed  (np.load reads it transparently)
  - buffer.pkl  -> gzip copy (.pkl.gz), lossless
  - latest*.pth -> copied verbatim (full model+ema+optimizer, resumable)
  - other *.pth -> keep `model` (fp32, bit-identical play weights) + tiny
                   metadata; drop ema/optimizer/scheduler.

Run from python_src/ with the project venv.
"""
import os, gzip, glob, shutil, time
import numpy as np
import torch

DROP = {"ema", "optimizer", "scheduler"}
KEEP_FULL = lambda name: os.path.basename(name).startswith("latest")


def human(n):
    for u in "B KB MB GB TB".split():
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def slim_dir(d):
    return (d.rstrip("/") + "_slim") if d not in ("", ".") else "_root_slim"


def process_pth(path):
    run = os.path.dirname(path)
    out_dir = slim_dir(run)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(path))
    if os.path.exists(out):
        return 0, 0, "skip(exists)"
    before = os.path.getsize(path)
    if KEEP_FULL(path):
        shutil.copy2(path, out)            # verbatim, fully resumable
        return before, os.path.getsize(out), "full"
    d = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(d, dict) or "model" not in d:
        shutil.copy2(path, out)            # already weights-only
        return before, os.path.getsize(out), "copy(no-model-key)"
    lean = {k: v for k, v in d.items() if k not in DROP}
    torch.save(lean, out)
    return before, os.path.getsize(out), "model-only"


def process_npz(path):
    out_dir = slim_dir(os.path.dirname(path))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(path))
    if os.path.exists(out):
        return 0, 0, "skip(exists)"
    before = os.path.getsize(path)
    z = np.load(path, allow_pickle=True)
    np.savez_compressed(out, **{k: z[k] for k in z.files})
    if not out.endswith(".npz"):
        os.rename(out + ".npz", out)
    return before, os.path.getsize(out), "npz-compressed"


def process_pkl(path):
    out_dir = slim_dir(os.path.dirname(path))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(path) + ".gz")
    if os.path.exists(out):
        return 0, 0, "skip(exists)"
    before = os.path.getsize(path)
    with open(path, "rb") as fi, gzip.open(out, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, length=16 << 20)
    return before, os.path.getsize(out), "pkl-gzip"


def main():
    t0 = time.time()
    tot_b = tot_a = 0
    print("=== buffers ===", flush=True)
    for p in sorted(glob.glob("**/buffer.npz", recursive=True)):
        b, a, tag = process_npz(p)
        tot_b += b; tot_a += a
        print(f"  {p:48s} {human(b):>9} -> {human(a):>9}  [{tag}]", flush=True)
    for p in sorted(glob.glob("**/buffer.pkl", recursive=True)):
        b, a, tag = process_pkl(p)
        tot_b += b; tot_a += a
        print(f"  {p:48s} {human(b):>9} -> {human(a):>9}  [{tag}]", flush=True)

    print("=== checkpoints ===", flush=True)
    for p in sorted(glob.glob("**/*.pth", recursive=True)):
        if "_slim/" in p:
            continue
        b, a, tag = process_pth(p)
        tot_b += b; tot_a += a
        print(f"  {p:48s} {human(b):>9} -> {human(a):>9}  [{tag}]", flush=True)

    print(f"\nTOTAL  {human(tot_b)} -> {human(tot_a)}   "
          f"({tot_a/max(tot_b,1)*100:.1f}%, saved {human(tot_b-tot_a)})  "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
