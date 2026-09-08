# mcts.py — thin Python driver for the Rust MCTS.
#
# The search tree lives in `capablanca_engine.RustMCTS`; this module only
# feeds it batched GPU inference (BF16 weights, pinned staging buffer,
# optional transposition cache) and owns the KLD early-exit bookkeeping.

import numpy as np
import torch
from collections import OrderedDict
from typing import List

# NVIDIA optimizations (mirrors train.py settings — ensures they're active
# when imported from gui.py / game_stats.py, when train.py is not loaded).
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        pass
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    except AttributeError:
        pass

# torch.compile cache: bucket-padding intentionally keeps the number of unique
# batch sizes bounded. Default cache_size_limit=8 triggers fallback to eager —
# raise it so every bucket keeps its compiled CUDA graph.
try:
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
except (ImportError, AttributeError):
    pass


def _bucket_size(n: int, step: int, pow2: bool = True) -> int:
    """Round n up to a stable batch shape.

    pow2=True  (torch.compile path): coarse buckets instead of strict powers
               of two. Power-of-two padding kept Dynamo happy, but on large
               partially-finished self-play batches it could nearly double the
               GPU work (e.g. 2100 leaves → 4096). Coarse 512-ish buckets keep
               compile shapes bounded while wasting much less compute.
    pow2=False (eager path): next multiple of `step` only. Measured GPU
               throughput is *linear* in batch size (≈0.115 ms/leaf for the
               384×15+4 net), so power-of-two padding is pure wasted
               compute — up to ~1.8x on an under-filled call (e.g. 2100→4096).
               A fine grid keeps padding waste ≤ step/n while cudnn.benchmark
               autotune of each new shape is a one-time ~1-forward cost that
               amortizes to seconds over a full iteration."""
    if pow2:
        if n <= step:
            return step
        small_grid = step
        coarse_grid = max(step * 16, 512)
        grid = small_grid if n <= coarse_grid else coarse_grid
        return ((n + grid - 1) // grid) * grid
    if n <= step:
        return step
    return ((n + step - 1) // step) * step

try:
    from capablanca_engine import RustMCTS as _RustMCTS
except ImportError as exc:  # the search tree lives in Rust — there is no fallback
    raise ImportError(
        "capablanca_engine not found. Build it with: maturin develop --release"
    ) from exc

# Import network constants — all sizes/reshapes must derive from these,
# so the wrapper can never drift from the architecture it is feeding.
from model import CapablancaNet, POLICY_SIZE
INPUT_PLANES = CapablancaNet.INPUT_PLANES   # 139 (8 history × 17 + 3 meta)
BOARD_H = CapablancaNet.BOARD_H              # 8
BOARD_W = CapablancaNet.BOARD_W              # 10
FLAT_SIZE = INPUT_PLANES * BOARD_H * BOARD_W # 11120

# parallel_sims is the number of leaves collected per NN call. The binding
# constraint is NOT this number on its own but the count of *sequential* PUCT
# rounds a search gets: ceil(simulations / parallel_sims). Virtual loss pushes
# the leaves of one round onto different children, so with too few rounds the
# visits spread flat across the root and the policy target degenerates to
# uniform. On a 10x8 board (~41 legal moves) `simulations 100 / parallel 32` is
# 4 rounds — about 2.3 visits per root move, i.e. no target at all;
# `--fast-simulations 50` at parallel 32 is 2 rounds and is *literally* uniform.
#
# Keep simulations / parallel_sims >= 12.
#
# The old comment here claimed "low (<16) under-utilizes GPU". Measured on this
# machine that is false: 128 games x 8 parallel = 1024 leaves per call already
# saturates the card and runs marginally FASTER than 4096 (the total leaf count
# per search is identical either way — only the batch size changes). See
# docs/experiments/policy_target_collapse.md.
#
# Lowered 32 -> 8 on 2026-09-08 after a 40-iteration A/B from a common
# checkpoint: 300 games, parallel 8 scored 69.8% at a parallel-32 match search
# and 76.0% at a parallel-8 one (+146 / +200 Elo).
PARALLEL_SIMS = 8

# LC0 transposition cache: same position is evaluated by NN only once.
# Useful for: transpositions in MCTS, repeated roots between games, tree reuse.
# Key — bytes view of tensor (FLAT_SIZE float32 ≈ 44KB with history planes). Hash bytes is fast (cityhash in CPython).
# 50K entries × ~30KB per entry (tensor+policy+q+d) ≈ 1.5GB — set higher if needed.
NN_CACHE_DEFAULT_MAX = 50_000


class UltraFastMCTS:
    """
    Batched MCTS. Tree lives in Rust, Python handles only GPU inference.
    """

    def __init__(self, net: torch.nn.Module, device: torch.device,
                 c_puct: float = 1.25, batch_size: int = 256,
                 add_dirichlet: bool = True, parallel_sims: int = None,
                 nn_cache: bool = False, nn_cache_max: int = NN_CACHE_DEFAULT_MAX,
                 compile_mode: str = None, bf16_weights: bool = True,
                 kld_threshold: float = 0.0, kld_check_every: int = 4,
                 kld_min_sims_frac: float = 0.25,
                 contempt: float = 0.0, rep_search_perslot: bool = False):
        # compile_mode: None (no compile), 'default', 'reduce-overhead', 'max-autotune'.
        # 'default' — safest, ~15-25% speedup, minimal warmup.
        # 'reduce-overhead' — uses CUDA graphs, up to 50% speedup, but recompiles on shape change.
        # 'max-autotune' — best speedup, but 1-2 minute warmup.
        #
        # bf16_weights: True → creates BF16 copy of weights for inference. Training net stays FP32.
        # ~1.5-2x speedup vs autocast(bf16) — no FP32→BF16 cast on every weight read.
        # VRAM: extra BF16 copy (~½ of FP32 size; for 128ch×10 this is ~6 MB).
        self._bf16_weights = bf16_weights and torch.cuda.is_available()
        if self._bf16_weights:
            import copy as _copy
            # Unwrap _orig_mod if net is already under torch.compile.
            src = net._orig_mod if hasattr(net, '_orig_mod') else net
            inference_net = _copy.deepcopy(src).to(torch.bfloat16).eval()
            self.net = inference_net
        else:
            self.net = net
        self.device = device
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.add_dirichlet = add_dirichlet
        # Contempt: 0.0 = standard play. > 0 → avoid draws, < 0 → welcome them.
        # Applied by RustMCTS.set_contempt on every per-game tree right after creation.
        self.contempt = float(contempt)
        # Per-slot repetition planes during search (match training encoding).
        self._rep_search_perslot = bool(rep_search_perslot)
        self._parallel_sims = parallel_sims if parallel_sims is not None else PARALLEL_SIMS
        if self._parallel_sims > 64:
            print(f"⚠️  parallel_sims={self._parallel_sims} > 64: PUCT exploration "
                  f"может ломаться из-за насыщения virtual_loss. Рекомендуется 16-64.")

        # torch.compile: compiles forward into an optimized CUDA graph.
        # On Blackwell with BF16 gives +15-50% raw inference speedup. Applied on top of
        # BF16 copy (if bf16_weights=True) or the original net.
        self._compile_mode = compile_mode
        if compile_mode is not None and hasattr(torch, 'compile'):
            try:
                # dynamic=False: shapes are bucketed, so compile sees a bounded
                # set of static shapes and can optimize them aggressively.
                self.net = torch.compile(self.net, mode=compile_mode, dynamic=False)
                print(f"🔥 torch.compile(mode={compile_mode!r}) — первый inference будет медленнее (warmup).")
            except Exception as e:
                print(f"⚠️  torch.compile failed: {e}. Откат на eager mode.")

        # Pinned memory + BF16 cast both make sense only when there's a GPU to
        # ship the data to. On CPU pin_memory=True raises RuntimeError, and
        # passing BF16 input into FP32 weights raises a dtype-mismatch — that
        # used to make `--device cpu` unusable. With CUDA: BF16 pinned buffer,
        # H2D copies are half the bytes. Without CUDA: plain FP32 buffer.
        self._has_cuda = torch.cuda.is_available()
        MAX_LEAVES = max(8192, batch_size * self._parallel_sims * 2)
        if self._has_cuda:
            # NCHW (contiguous), NOT channels_last: the inference net is
            # GroupNorm-heavy and runs fastest with NCHW weights (measured —
            # channels_last is ~10-15% slower, see experiments/perf/). A NCHW
            # pinned buffer also makes the per-step copy a plain contiguous memcpy
            # instead of a strided layout-convert (measured 1.1-1.3x faster H2D).
            self.pinned_buf = torch.empty(
                (MAX_LEAVES, INPUT_PLANES, BOARD_H, BOARD_W),
                pin_memory=True, dtype=torch.bfloat16)
        else:
            self.pinned_buf = torch.empty(
                (MAX_LEAVES, INPUT_PLANES, BOARD_H, BOARD_W),
                pin_memory=False, dtype=torch.float32)
            print("⚠️  CUDA не найдена — inference на CPU. Очень медленно, "
                  "training/eval скорее иллюстративные. Установите GPU + драйвер.")
        self.pinned_size = MAX_LEAVES
        self.net.eval()

        # NN transposition cache. Enabled ONLY for inference (gui/play),
        # usually disabled for self-play — branches are more unique + cache grows.
        self.nn_cache_enabled = nn_cache
        self.nn_cache_max = nn_cache_max
        self.nn_cache: "OrderedDict[bytes, tuple]" = OrderedDict() if nn_cache else None
        self._cache_hits = 0
        self._cache_misses = 0

        # KLD-early-exit (Lc0 smart pruning).
        # After every kld_check_every parallel steps, check KL(prev_visits || curr_visits)
        # for all live games. If max KL < threshold AND visits >= min_frac*total → break.
        # threshold=0 disables the feature.
        self.kld_threshold = float(kld_threshold)
        self.kld_check_every = max(1, int(kld_check_every))
        self.kld_min_sims_frac = float(kld_min_sims_frac)
        self._kld_early_exits = 0       # how many times early-exit fired
        self._kld_total_calls = 0       # total search calls
        self._kld_sims_saved = 0        # total sims saved
        self._kld_sims_requested = 0    # total sims requested

    def clear_nn_cache(self) -> None:
        """Clears the cache — must be called after network weights are updated."""
        if self.nn_cache is not None:
            self.nn_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def kld_stats(self) -> dict:
        """KLD-early-exit statistics. Useful for logging."""
        if self._kld_total_calls == 0:
            return {"exit_rate": 0.0, "savings": 0.0, "calls": 0}
        return {
            "exit_rate": self._kld_early_exits / self._kld_total_calls,
            "savings": (self._kld_sims_saved / self._kld_sims_requested
                        if self._kld_sims_requested > 0 else 0.0),
            "calls": self._kld_total_calls,
        }

    @torch.no_grad()
    def _infer(self, tensors, hashes=None):
        """
        Batched GPU inference with optional NN cache (transposition table).

        tensors: np.ndarray (N, FLAT_SIZE) or List.
        hashes:  optional List[int] u64 — board hashes as cache keys.

        Returns (policies, q_values, d_values, mlh_values).
        """
        if isinstance(tensors, list):
            if len(tensors) == 0:
                empty_p = np.empty((0, POLICY_SIZE), dtype=np.float32)
                empty_v = np.empty((0,), dtype=np.float32)
                return empty_p, empty_v, empty_v.copy(), empty_v.copy()
            tensors = np.stack(tensors, axis=0).reshape(len(tensors), FLAT_SIZE)

        n = tensors.shape[0]
        if n == 0:
            empty_p = np.empty((0, POLICY_SIZE), dtype=np.float32)
            empty_v = np.empty((0,), dtype=np.float32)
            return empty_p, empty_v, empty_v.copy(), empty_v.copy()

        # === NN cache fast-path ===
        if self.nn_cache_enabled and self.nn_cache is not None:
            if hashes is not None and len(hashes) == n:
                keys: list = list(hashes) if not isinstance(hashes, list) else hashes
            else:
                arr_for_keys = np.ascontiguousarray(tensors.reshape(n, -1), dtype=np.float32)
                keys = [bytes(arr_for_keys[i].data) for i in range(n)]
            uncached_idx = []
            policies = np.empty((n, POLICY_SIZE), dtype=np.float32)
            q_values = np.empty(n, dtype=np.float32)
            d_values = np.empty(n, dtype=np.float32)
            m_values = np.empty(n, dtype=np.float32)
            for i, k in enumerate(keys):
                cached = self.nn_cache.get(k)
                if cached is None:
                    uncached_idx.append(i)
                else:
                    p, q, d, m = cached
                    policies[i] = p
                    q_values[i] = q
                    d_values[i] = d
                    m_values[i] = m
                    self.nn_cache.move_to_end(k)
            self._cache_hits   += (n - len(uncached_idx))
            self._cache_misses += len(uncached_idx)
            if not uncached_idx:
                return policies, q_values, d_values, m_values
            uncached_tensors = tensors[uncached_idx]
            p_un, q_un, d_un, m_un = self._infer_raw_nn(uncached_tensors)
            for j, i in enumerate(uncached_idx):
                policies[i] = p_un[j]
                q_values[i] = q_un[j]
                d_values[i] = d_un[j]
                m_values[i] = m_un[j]
                self.nn_cache[keys[i]] = (p_un[j].copy(), float(q_un[j]),
                                          float(d_un[j]), float(m_un[j]))
                if len(self.nn_cache) > self.nn_cache_max:
                    self.nn_cache.popitem(last=False)
            return policies, q_values, d_values, m_values

        return self._infer_raw_nn(tensors)

    @torch.no_grad()
    def _infer_raw_nn(self, tensors):
        """Direct batched NN call without cache. tensors: ndarray (N, FLAT_SIZE).
        Returns (policies, q, d, m)."""
        n = tensors.shape[0]
        if n == 0:
            empty_p = np.empty((0, POLICY_SIZE), dtype=np.float32)
            empty_v = np.empty((0,), dtype=np.float32)
            return empty_p, empty_v, empty_v.copy(), empty_v.copy()

        # Pad to a stable bucket so torch.compile sees a bounded set of input
        # shapes. Strict power-of-two buckets waste too much compute on
        # partially-finished game batches, so _bucket_size uses a coarse grid.
        ps = self._parallel_sims
        # Power-of-two buckets only matter when torch.compile/CUDA-graphs need
        # shape stability. In eager mode (the only stable path on Blackwell) the
        # GPU is compute-bound and throughput is linear in batch size, so we pad
        # to the next multiple of `ps` instead — up to ~1.8x less wasted compute
        # on under-filled calls.
        target = _bucket_size(n, ps, pow2=self._compile_mode is not None)
        n_pad = target - n

        arr = np.ascontiguousarray(
            tensors.reshape(n, INPUT_PLANES, BOARD_H, BOARD_W)
        )
        n_total = target

        if n_total <= self.pinned_size:
            buf = self.pinned_buf[:n_total]
            # copy_(fp32) casts to buf's dtype (BF16 on CUDA, FP32 on CPU).
            buf[:n].copy_(torch.from_numpy(arr))
            if n_pad > 0:
                # Padding rows are discarded after inference. Duplicate existing
                # rows in-place instead of building a new numpy array with
                # np.concatenate on every MCTS step.
                filled = n
                while filled < n_total:
                    take = min(filled, n_total - filled)
                    buf[filled:filled + take].copy_(buf[:take])
                    filled += take
            x = buf.to(self.device, non_blocking=True)
        else:
            # Fallback when pinned buffer is exceeded. Match the buffer dtype
            # so we never feed BF16 into FP32 weights on CPU.
            if n_pad > 0:
                pad_idx = np.arange(n_pad) % n
                arr = np.concatenate([arr, arr[pad_idx]], axis=0)
            target_dtype = torch.bfloat16 if self._has_cuda else torch.float32
            cpu_t = torch.from_numpy(arr).to(target_dtype)
            x = cpu_t.to(self.device, non_blocking=True)

        # NCHW throughout — the net runs NCHW (channels_last is slower here), so
        # no layout conversion; x stays contiguous from the pinned buffer.
        if self._bf16_weights:
            # Both weights and input in BF16 — no autocast needed.
            out = self.net(x)
        elif self._has_cuda:
            # FP32 weights on CUDA: autocast runs matmul/conv in BF16 on the fly.
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = self.net(x)
        else:
            # Pure CPU path: everything stays in FP32.
            out = self.net(x)

        # Support multiple network output variants:
        #   - 4 outputs: (policy, wdl, mlh, future) — model with future head
        #   - 3 outputs: (policy, wdl, mlh)          — model with MLH
        #   - 2 outputs: (policy, wdl)               — old checkpoints
        # future head is not needed at inference → ignored.
        if isinstance(out, tuple) and len(out) == 4:
            logits, values, mlh_raw, _ = out
        elif isinstance(out, tuple) and len(out) == 3:
            logits, values, mlh_raw = out
        else:
            logits, values = out
            mlh_raw = None

        # nan_to_num scrubs NaN/inf on the GPU (pointwise kernel, no host sync).
        # The old `.isnan().any()` guards each forced a device→host sync on every
        # inference batch — pure latency in the self-play hot loop. zeros → uniform
        # softmax for logits and a neutral value, the same fallback as before.
        logits_f = torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
        values_f = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)

        policies = torch.softmax(logits_f, dim=1).cpu().numpy()

        if values_f.shape[-1] == 3:
            wdl_probs = torch.softmax(values_f, dim=1)
            q_values  = (wdl_probs[:, 0] - wdl_probs[:, 2]).cpu().numpy()
            d_values  = wdl_probs[:, 1].cpu().numpy()
        else:
            q_values = values_f.view(-1).cpu().numpy()
            d_values = np.zeros_like(q_values)

        # MLH: sigmoid raw → ∈ [0, 1] (normalized "fraction of game remaining").
        if mlh_raw is not None:
            mlh_f = torch.nan_to_num(mlh_raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
            m_values = torch.sigmoid(mlh_f).view(-1).cpu().numpy()
        else:
            m_values = np.zeros_like(q_values)

        return policies[:n], q_values[:n], d_values[:n], m_values[:n]

    def new_tree(self, engines: List):
        """Fresh RustMCTS over `engines`, configured from this instance's flags."""
        rust_mcts = _RustMCTS(engines, self._parallel_sims)
        if self.contempt != 0.0:
            rust_mcts.set_contempt(self.contempt)
        # add_dirichlet=False (eval / FSF / lagged) must actually disable root
        # noise in Rust — the flag used to be silently ignored.
        if not self.add_dirichlet:
            rust_mcts.set_add_dirichlet(False)
        if self._rep_search_perslot:
            rust_mcts.set_rep_search_perslot(True)
        return rust_mcts

    def run_search(self, rust_mcts, simulations: int) -> None:
        """Run one search on an existing tree: collect → infer → apply, with
        Lc0-style KLD early exit. Leaves the tree in place, so callers doing
        tree reuse (self-play, game_stats) drive the same loop as one-shot
        callers instead of each keeping their own copy of it.

        Inference is applied *immediately* after each collect_leaves, in the
        same step. The old "double-buffered" variant deferred the apply by one
        step, but collect_leaves clears Rust's `pending`, so the deferred apply
        wrote the previous batch's NN outputs onto the new leaves.
        """
        parallel = self._parallel_sims
        steps = max(1, (simulations + parallel - 1) // parallel)
        kld_enabled = self.kld_threshold > 0.0
        kld_min_steps = int(np.ceil(steps * self.kld_min_sims_frac))
        if kld_enabled:
            rust_mcts.kld_reset_all()
        self._kld_total_calls += 1
        self._kld_sims_requested += simulations

        for step in range(steps):
            leaf_matrix = rust_mcts.collect_leaves(simulations)
            if leaf_matrix.shape[0] == 0:
                break
            curr_counts = rust_mcts.get_current_batch_counts()
            curr_hashes = rust_mcts.get_leaf_hashes() if self.nn_cache_enabled else None
            p, v, d, m = self._infer(leaf_matrix, hashes=curr_hashes)
            rust_mcts.apply_inference_buffered(
                np.ascontiguousarray(p, dtype=np.float32),
                np.ascontiguousarray(v, dtype=np.float32),
                np.ascontiguousarray(d, dtype=np.float32),
                np.ascontiguousarray(m, dtype=np.float32),
                curr_counts,
            )

            if (kld_enabled and step >= kld_min_steps
                    and (step + 1) % self.kld_check_every == 0
                    and step + 1 < steps):
                max_kl = rust_mcts.kld_snapshot_and_check()
                if max_kl != float('inf'):
                    kl_gain = max_kl / max(1, self.kld_check_every * parallel)
                    if kl_gain < self.kld_threshold:
                        self._kld_early_exits += 1
                        self._kld_sims_saved += (steps - step - 1) * parallel
                        break

    def search_games(self, engines: List, simulations: int = 80) -> List[np.ndarray]:
        return self.search_games_with_values(engines, simulations)[0]

    def search_games_with_values(self, engines: List, simulations: int = 80):
        """Returns (policies, values). values are needed for resign logic.

        If self.kld_threshold > 0, MCTS may stop early when the
        visit distribution stops changing (Lc0 smart pruning).
        """
        rust_mcts = self.new_tree(engines)
        self.run_search(rust_mcts, simulations)
        raw_policies = rust_mcts.get_policies()
        raw_values   = rust_mcts.get_values()
        policies = [np.array(p, dtype=np.float32) for p in raw_policies]
        return policies, np.array(raw_values, dtype=np.float32)

