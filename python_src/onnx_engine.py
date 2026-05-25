"""ONNX-runtime inference backend for the analysis GUI.

`gui.py` only ever needs the batched network evaluation (`_infer`) plus the
two scalars `c_puct` / `batch_size` — never the training-time Rust MCTS. So in
the GUI the heavyweight `UltraFastMCTS` (PyTorch) is replaced by `OnnxEngine`,
which runs a self-contained `.onnx` graph through onnxruntime.

Result: the distributed GUI carries no PyTorch dependency. With an NVIDIA GPU
present the CUDA execution provider is used automatically; otherwise the engine
falls back to CPU. See `export_onnx.py` for producing the `.onnx` file and the
README packaging section for bundling.
"""

from collections import OrderedDict

import numpy as np
import onnxruntime as ort

# onnxruntime's CUDA provider needs the CUDA 12 / cuDNN 9 runtime libraries.
# preload_dlls() loads them from the nvidia-*-cu12 pip packages (which PyTorch
# also installs as dependencies), so the GPU provider works without a
# system-wide CUDA install. Guarded — absent on older onnxruntime.
if hasattr(ort, "preload_dlls"):
    try:
        ort.preload_dlls()
    except Exception:
        pass

# Mirrors the Rust engine constant (VIRTUAL_LOSS_V) — re-exported so gui.py can
# import it from here instead of from the PyTorch-side mcts module.
VIRTUAL_LOSS = 3

POLICY_SIZE = 7000
INPUT_PLANES = 139
BOARD_H, BOARD_W = 8, 10


class OnnxEngine:
    """Drop-in network provider for the GUI's SearchThread.

    Exposes the same surface the search relies on:
      * ``_infer(tensors)`` -> (policy, q, d, m) numpy arrays
      * ``c_puct``, ``batch_size`` attributes
    plus a transposition NN-cache so repeated positions skip the GPU.
    """

    def __init__(self, onnx_path, c_puct=1.745, batch_size=96,
                 nn_cache=True, nn_cache_max=600_000):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # CUDA first, CPU as a graceful fallback when no NVIDIA GPU is present.
        self.session = ort.InferenceSession(
            onnx_path, sess_options=so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        active = self.session.get_providers()
        self.gpu = "CUDAExecutionProvider" in active
        self.provider = active[0] if active else "CPUExecutionProvider"
        self.input_name = self.session.get_inputs()[0].name

        self.c_puct = c_puct
        self.batch_size = batch_size

        self.nn_cache_enabled = bool(nn_cache)
        self.nn_cache_max = nn_cache_max
        self._cache = OrderedDict() if nn_cache else None
        self._cache_hits = 0
        self._cache_misses = 0

    # ---- raw GPU/CPU call ----
    def _run(self, batch):
        """batch: contiguous float32 (N, 139, 8, 10) -> (policy, q, d, m)."""
        p, q, d, m = self.session.run(None, {self.input_name: batch})
        return (p.astype(np.float32, copy=False),
                q.astype(np.float32, copy=False).reshape(-1),
                d.astype(np.float32, copy=False).reshape(-1),
                m.astype(np.float32, copy=False).reshape(-1))

    @staticmethod
    def _empty():
        e = np.empty((0,), np.float32)
        return (np.empty((0, POLICY_SIZE), np.float32), e, e.copy(), e.copy())

    def _infer(self, tensors, hashes=None):
        """Batched inference with an optional transposition NN-cache.

        ``tensors`` — a list of per-board arrays (139,8,10) or flat — exactly
        what the GUI's SearchThread already collects. Returns four arrays:
        softmaxed policy (N,7000), value Q (N,), draw prob D (N,), moves-left M.
        """
        if isinstance(tensors, list):
            if not tensors:
                return self._empty()
            batch = np.stack(tensors, axis=0)
        else:
            batch = np.asarray(tensors)
        batch = np.ascontiguousarray(
            batch.reshape(-1, INPUT_PLANES, BOARD_H, BOARD_W), dtype=np.float32)
        n = batch.shape[0]
        if n == 0:
            return self._empty()

        if not self.nn_cache_enabled:
            return self._run(batch)

        # Cache key: caller-supplied position hash if available, else raw bytes.
        if hashes is not None and len(hashes) == n:
            keys = list(hashes)
        else:
            keys = [batch[i].tobytes() for i in range(n)]

        pol = np.empty((n, POLICY_SIZE), np.float32)
        qv = np.empty(n, np.float32)
        dv = np.empty(n, np.float32)
        mv = np.empty(n, np.float32)
        miss = []
        for i, k in enumerate(keys):
            hit = self._cache.get(k)
            if hit is None:
                miss.append(i)
            else:
                pol[i], qv[i], dv[i], mv[i] = hit
                self._cache.move_to_end(k)

        self._cache_hits += n - len(miss)
        self._cache_misses += len(miss)

        if miss:
            sub = np.ascontiguousarray(batch[miss])
            p, q, d, m = self._run(sub)
            for j, i in enumerate(miss):
                pol[i], qv[i], dv[i], mv[i] = p[j], q[j], d[j], m[j]
                self._cache[keys[i]] = (p[j].copy(), float(q[j]),
                                        float(d[j]), float(m[j]))
                if len(self._cache) > self.nn_cache_max:
                    self._cache.popitem(last=False)
        return pol, qv, dv, mv

    def clear_nn_cache(self):
        if self._cache is not None:
            self._cache.clear()
        self._cache_hits = self._cache_misses = 0
