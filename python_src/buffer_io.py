"""Replay buffer on disk: one archive, or a directory of per-iteration chunks.

Chunks are what Leela does: each iteration appends only its own positions as
`buffer_chunks/iter_000045.npz`, and a chunk is deleted once it has fallen
wholly out of the buffer window. Rewriting the whole buffer every iteration
cost 4.5 GB of writes at 200k positions and 22 GB at 1M — about 5 TB a day at
six minutes an iteration. A chunk is ~145 MB whatever the buffer size, and a
crash loses at most the iteration in flight.

Every file, chunk or single archive, has the same layout — an uncompressed zip
of .npy members, exactly what np.savez writes:

    boards   (N, ...) f16      pol_idx (N, K) i16, -1 padding
    values   (N,)     f16      pol_val (N, K) f16,  0 padding
    mlhs     (N,)     f16      futures (N,)   i32
    draws    (N,)     f16      meta    [ptr, full]

Tools that are handed `…/buffer.npz` keep working after a run has moved to
chunks: `resolve` falls back to the sibling `buffer_chunks/` directory, and the
training loop renames the old archive when it migrates, so nobody silently
reads a stale snapshot.
"""
import os
import re
import zipfile

import numpy as np

CHUNK_DIR = "buffer_chunks"
_CHUNK_RE = re.compile(r"^iter_(\d+)(_base)?\.npz$")
_ROWS = ("boards", "pol_idx", "pol_val", "values", "mlhs", "futures", "draws")


# ── one archive ───────────────────────────────────────────────────────────────

def write_samples(path, samples, meta):
    """Write samples to `path` atomically (temp file + rename).

    Boards are streamed into the archive 4096 at a time instead of stacked:
    np.stack over a whole buffer was a transient copy the size of the buffer,
    and that allocation once ended in a SIGSEGV inside libc. Returns False and
    writes nothing for an empty list."""
    n = len(samples)
    if n == 0:
        return False
    # Sparse policy is ragged (one entry per legal move); pad to the longest.
    max_k = max((len(s[1][0]) for s in samples), default=0)
    pol_idx = np.full((n, max_k), -1, dtype=np.int16)
    pol_val = np.zeros((n, max_k), dtype=np.float16)
    for i, s in enumerate(samples):
        idxs, vals = s[1]
        k = len(idxs)
        if k:
            pol_idx[i, :k] = idxs
            pol_val[i, :k] = vals
    values = np.fromiter((float(s[2]) for s in samples), dtype=np.float16, count=n)
    mlhs = np.fromiter((float(s[3]) if len(s) > 3 else 0.0 for s in samples),
                       dtype=np.float16, count=n)
    futures = np.fromiter((int(s[4]) if len(s) > 4 else -1 for s in samples),
                          dtype=np.int32, count=n)
    # Full-WDL Q-blend draw target (-1 = none → reconstruct from value).
    draws = np.fromiter((float(s[5]) if len(s) > 5 else -1.0 for s in samples),
                        dtype=np.float16, count=n)

    def member(zf, name, arr):
        with zf.open(name + ".npy", "w", force_zip64=True) as f:
            np.lib.format.write_array(f, np.asanyarray(arr), allow_pickle=False)

    board_shape = np.shape(samples[0][0])
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as zf:
        with zf.open("boards.npy", "w", force_zip64=True) as f:
            np.lib.format.write_array_header_1_0(f, {
                "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
                "fortran_order": False,
                "shape": (n,) + tuple(board_shape),
            })
            for a in range(0, n, 4096):
                f.write(np.stack([np.asarray(s[0], dtype=np.float16)
                                  for s in samples[a:a + 4096]]).tobytes())
        member(zf, "pol_idx", pol_idx)
        member(zf, "pol_val", pol_val)
        member(zf, "values", values)
        member(zf, "mlhs", mlhs)
        member(zf, "futures", futures)
        member(zf, "draws", draws)
        member(zf, "meta", np.asarray(meta, dtype=np.int64))
    os.replace(tmp, path)
    return True


def read_archive(path):
    """Every member of one archive as a plain dict.

    NpzFile re-reads a member from disk on every access; a dict reads each
    once, which matters when a caller indexes boards row by row."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def arrays_to_samples(a):
    """Rows of an archive dict → the buffer's list-of-tuples representation.
    Rows come back in file order; unrolling a saved ring is the caller's job."""
    boards, pol_idx, pol_val = a["boards"], a["pol_idx"], a["pol_val"]
    values, mlhs, futures = a["values"], a["mlhs"], a["futures"]
    draws = a.get("draws")      # older archives predate the Q-blend draw axis
    out = []
    for i in range(boards.shape[0]):
        mask = pol_idx[i] >= 0
        out.append((
            boards[i],
            (pol_idx[i][mask].astype(np.int16, copy=False),
             pol_val[i][mask].astype(np.float16, copy=False)),
            float(values[i]),
            float(mlhs[i]),
            int(futures[i]),
            float(draws[i]) if draws is not None else -1.0,
        ))
    return out


# ── chunks ────────────────────────────────────────────────────────────────────

def chunk_dir(checkpoint_dir):
    return os.path.join(checkpoint_dir, CHUNK_DIR)


def list_chunks(d):
    """[(iteration, path), ...] oldest first; empty if the directory is absent.

    A base chunk (`iter_N_base.npz`, a migrated whole buffer) sorts before the
    ordinary chunk of the same iteration: it holds everything older than the
    run that wrote it."""
    if not os.path.isdir(d):
        return []
    found = []
    for name in os.listdir(d):
        m = _CHUNK_RE.match(name)
        if m:
            found.append((int(m.group(1)), 0 if m.group(2) else 1,
                          os.path.join(d, name)))
    return [(it, path) for it, _, path in sorted(found)]


def chunk_rows(path):
    with np.load(path) as z:
        return int(z["values"].shape[0])       # reads only the small member


def write_chunk(d, iteration, samples, base=False):
    """`base=True` names it `iter_N_base.npz`. A migrated buffer must not be
    `iter_000000.npz`: a run with no checkpoint starts at iteration 0, and its
    first chunk would overwrite the whole migrated buffer."""
    os.makedirs(d, exist_ok=True)
    name = f"iter_{iteration:06d}{'_base' if base else ''}.npz"
    return write_samples(os.path.join(d, name), samples, [len(samples), 0])


def evict(d, keep_rows):
    """Delete the oldest chunks that lie wholly outside the newest `keep_rows`
    positions. A chunk only partly inside the window stays. Returns how many
    were deleted."""
    chunks = list_chunks(d)
    sizes = [chunk_rows(p) for _, p in chunks]
    total = sum(sizes)
    removed = 0
    for (_, p), n in zip(chunks, sizes):
        if total - n < keep_rows:
            break
        os.remove(p)
        total -= n
        removed += 1
    return removed


# ── reading for tools ─────────────────────────────────────────────────────────

def resolve(path):
    """(source, kind) for a buffer path: a chunk directory, an archive, or an
    archive path whose run has migrated to chunks next to it."""
    if os.path.isdir(path):
        return path, "chunks"
    if os.path.exists(path):
        return path, "file"
    d = os.path.join(os.path.dirname(path) or ".", CHUNK_DIR)
    if list_chunks(d):
        return d, "chunks"
    raise FileNotFoundError(f"нет ни {path}, ни кусков в {d}")


def load_arrays(path):
    """The whole buffer as one dict, oldest row first.

    For chunks, meta is [rows, 0]: the rows are already in age order, so a
    reader that unrolls a saved ring leaves them alone. Chunks are copied into
    preallocated arrays one at a time, so peak memory is the result plus one
    chunk, not twice the buffer."""
    src, kind = resolve(path)
    if kind == "file":
        return read_archive(src)
    chunks = [p for _, p in list_chunks(src)]
    if not chunks:
        raise FileNotFoundError(f"в {src} нет кусков буфера")
    heads = []
    for p in chunks:
        with np.load(p) as z:
            heads.append((int(z["values"].shape[0]), int(z["pol_idx"].shape[1])))
    n, k = sum(h[0] for h in heads), max(h[1] for h in heads)
    out: dict[str, np.ndarray] = {}
    at = 0
    for p, (rows, kk) in zip(chunks, heads):
        a = read_archive(p)
        if not out:
            for f in _ROWS:
                if f == "pol_idx":
                    out[f] = np.full((n, k), -1, dtype=np.int16)
                elif f == "pol_val":
                    out[f] = np.zeros((n, k), dtype=np.float16)
                else:
                    out[f] = np.empty((n,) + a[f].shape[1:], dtype=a[f].dtype)
        for f in _ROWS:
            if f in ("pol_idx", "pol_val"):
                out[f][at:at + rows, :kk] = a[f]
            else:
                out[f][at:at + rows] = a[f]
        at += rows
    out["meta"] = np.array([n, 0], dtype=np.int64)
    return out


def read_field(path, name):
    """One field across the whole buffer, oldest first, without loading the
    rest — `values` for a histogram shouldn't pull 22 GB of boards along."""
    src, kind = resolve(path)
    if kind == "file":
        with np.load(src) as z:
            return z[name]
    parts = []
    for _, p in list_chunks(src):
        with np.load(p) as z:
            parts.append(z[name])
    if not parts:
        raise FileNotFoundError(f"в {src} нет кусков буфера")
    return np.concatenate(parts)


def sample_boards(path, n):
    """A few boards without loading the whole buffer: the newest chunk, or the
    head of a single archive."""
    src, kind = resolve(path)
    if kind == "chunks":
        src = list_chunks(src)[-1][1]
    with np.load(src) as z:
        return z["boards"][:n]
