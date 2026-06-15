# mcts.py — Python wrapper for RustMCTS
#
# OPTIMIZATIONS v2:
#   - _infer now accepts np.ndarray (N, FLAT_SIZE) directly — list comprehension removed
#   - PARALLEL_SIMS raised to 32: fewer round-trips Python↔Rust↔GPU per game
#   - _infer: intermediate np.stack removed (leaf_matrix is already a ready matrix)
#   - cuda.synchronize removed — non_blocking=True + autocast is sufficient
#   - pinned memory reused without extra copies

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
               384×15+4 net on a 5080), so power-of-two padding is pure wasted
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
    RUST_MCTS_AVAILABLE = True
except ImportError:
    RUST_MCTS_AVAILABLE = False
    print("⚠️  RustMCTS не найден — нужно пересобрать: maturin develop --release")

# Import input constants — all sizes/reshapes must derive from these
from model import CapablancaNet
INPUT_PLANES = CapablancaNet.INPUT_PLANES   # 139 (8 history × 17 + 3 meta)
BOARD_H = CapablancaNet.BOARD_H              # 8
BOARD_W = CapablancaNet.BOARD_W              # 10
FLAT_SIZE = INPUT_PLANES * BOARD_H * BOARD_W # 11120

POLICY_SIZE   = 7000
VIRTUAL_LOSS  = 3
# High parallel_sims (>64) saturates virtual_loss → exploration breaks,
# PUCT starts selecting weak branches. Low (<16) under-utilizes GPU.
# 32 — sweet spot for most networks.
PARALLEL_SIMS = 32

# LC0 transposition cache: same position is evaluated by NN only once.
# Useful for: transpositions in MCTS, repeated roots between games, tree reuse.
# Key — bytes view of tensor (FLAT_SIZE float32 ≈ 44KB with history planes). Hash bytes is fast (cityhash in CPython).
# 50K entries × ~30KB per entry (tensor+policy+q+d) ≈ 1.5GB — set higher if needed.
NN_CACHE_DEFAULT_MAX = 50_000


class MCTSNode:
    """Stub for backward compatibility with gui.py."""
    __slots__ = ("parent", "move", "prior", "children", "visits",
                 "value_sum", "virtual_loss", "is_expanded", "is_terminal")

    def __init__(self, parent, move, prior):
        self.parent = parent; self.move = move; self.prior = float(prior)
        self.children = {}; self.visits = 0; self.value_sum = 0.0
        self.virtual_loss = 0; self.is_expanded = False; self.is_terminal = False

    def q(self):
        d = self.visits + self.virtual_loss
        return self.value_sum / d if d > 0 else 0.0


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
                 contempt: float = 0.0):
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
            # Keep reference to the original FP32 net to update the BF16 copy
            # via update_inference_weights() after each train epoch.
            self._fp32_src = src
        else:
            self.net = net
            self._fp32_src = None
        self.device = device
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.add_dirichlet = add_dirichlet
        # Contempt: 0.0 = standard play. > 0 → avoid draws, < 0 → welcome them.
        # Applied by RustMCTS.set_contempt on every per-game tree right after creation.
        self.contempt = float(contempt)
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
            self.pinned_buf = torch.empty(
                (MAX_LEAVES, INPUT_PLANES, BOARD_H, BOARD_W),
                pin_memory=True, dtype=torch.bfloat16,
                memory_format=torch.channels_last)
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

    def update_inference_weights(self, fp32_net: torch.nn.Module = None) -> None:
        """Synchronizes BF16 inference network weights with the current FP32 net.

        Call after every train_epoch / EMA apply so self-play uses fresh weights.
        Does nothing if bf16_weights=False.

        fp32_net: optionally pass a different FP32 net (e.g., EMA shadow).
                  Defaults to self._fp32_src (the net passed at init).
        """
        if not self._bf16_weights:
            return
        src = fp32_net if fp32_net is not None else self._fp32_src
        if src is None:
            return
        # Unwrap compiled wrapper if present
        src_inner = src._orig_mod if hasattr(src, '_orig_mod') else src
        # Unwrap target if it's under torch.compile
        target = self.net._orig_mod if hasattr(self.net, '_orig_mod') else self.net
        with torch.no_grad():
            target_state = target.state_dict()
            for k, v in src_inner.state_dict().items():
                if k not in target_state:
                    continue
                if target_state[k].dtype.is_floating_point:
                    target_state[k].copy_(v.detach().to(target_state[k].dtype))
                else:
                    target_state[k].copy_(v.detach())
        self.clear_nn_cache()

    def nn_cache_stats(self) -> dict:
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": (self._cache_hits / total) if total > 0 else 0.0,
            "size": len(self.nn_cache) if self.nn_cache is not None else 0,
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
            x = cpu_t.to(self.device, non_blocking=True,
                         memory_format=torch.channels_last)

        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.to(memory_format=torch.channels_last)
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

    def search_games(self, engines: List, simulations: int = 80) -> List[np.ndarray]:
        if RUST_MCTS_AVAILABLE:
            return self._search_rust(engines, simulations)
        return self._search_python(engines, simulations)

    def search_games_with_values(self, engines: List, simulations: int = 80):
        """Returns (policies, values). values are needed for resign logic.

        If self.kld_threshold > 0, MCTS may stop early when the
        visit distribution stops changing (Lc0 smart pruning).
        """
        if RUST_MCTS_AVAILABLE:
            rust_mcts = _RustMCTS(engines, self._parallel_sims)
            if self.contempt != 0.0:
                rust_mcts.set_contempt(self.contempt)
            # add_dirichlet=False (eval/FSF/lagged) must actually disable noise
            # in Rust — the flag was silently ignored before.
            if not self.add_dirichlet:
                rust_mcts.set_add_dirichlet(False)
            steps = max(1, (simulations + self._parallel_sims - 1) // self._parallel_sims)

            # NOTE: this path applies inference *immediately* after each
            # collect_leaves, in the same step. The previous "double-buffered"
            # variant overlapped GPU work with the next collect by deferring
            # apply_inference_buffered by one step — but collect_leaves clears
            # `game.pending` on the Rust side, so the deferred apply wrote the
            # previous batch's NN outputs onto the *new* leaves. That was a
            # silent correctness bug in every search_games_with_values caller
            # (eval.py, FSF games, lagged games, play_fsf). We trade a tiny
            # CPU/GPU overlap window for a guaranteed-correct alignment.
            kld_enabled = self.kld_threshold > 0.0
            kld_min_steps = int(np.ceil(steps * self.kld_min_sims_frac))
            if kld_enabled:
                rust_mcts.kld_reset_all()
            self._kld_total_calls += 1
            self._kld_sims_requested += simulations
            early_exit_step = None

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
                        sims_added = self.kld_check_every * self._parallel_sims
                        kl_gain = max_kl / max(1, sims_added)
                        if kl_gain < self.kld_threshold:
                            self._kld_early_exits += 1
                            self._kld_sims_saved += (steps - step - 1) * self._parallel_sims
                            early_exit_step = step
                            break

            raw_policies = rust_mcts.get_policies()
            raw_values   = rust_mcts.get_values()
            policies = [np.array(p, dtype=np.float32) for p in raw_policies]
            return policies, np.array(raw_values, dtype=np.float32)

        policies = self._search_python(engines, simulations)
        return policies, np.zeros(len(engines), dtype=np.float32)

    def _search_rust(self, engines: List, simulations: int) -> List[np.ndarray]:
        policies, _ = self.search_games_with_values(engines, simulations)
        return policies

    def _search_rust_full(self, engines: List, simulations: int) -> List[np.ndarray]:
        rust_mcts = _RustMCTS(engines, self._parallel_sims)
        if self.contempt != 0.0:
            rust_mcts.set_contempt(self.contempt)
        if not self.add_dirichlet:
            rust_mcts.set_add_dirichlet(False)
        steps = max(1, (simulations + self._parallel_sims - 1) // self._parallel_sims)

        # Correct double buffering:
        # GPU processes batch N while Rust collects batch N+1.
        # Inference applied in-step (no double-buffering). The old "collect
        # next, apply previous" scheme silently wrote previous-batch NN outputs
        # onto fresh pending leaves: collect_leaves clears Rust's `pending`,
        # so by the time apply_inference_buffered ran, the leaves it indexed
        # were already from the next step. Same fix as search_games_with_values.
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

        raw = rust_mcts.get_policies()
        return [np.array(p, dtype=np.float32) for p in raw]

    # ── Python fallback ────────────────────────────────────────────────────────

    def _search_python(self, engines: List, simulations: int) -> List[np.ndarray]:
        """Old Python MCTS — only used if RustMCTS is not compiled."""
        import math
        num_games = len(engines)
        roots = [MCTSNode(None, -1, 1.0) for _ in range(num_games)]
        root_tensors = np.stack([
            np.array(e.get_board_tensor(), dtype=np.float32) for e in engines
        ]).reshape(num_games, FLAT_SIZE)
        policies, _, _, _ = self._infer(root_tensors)
        for i in range(num_games):
            self._expand_node_py(roots[i], engines[i], policies[i],
                                 add_noise=self.add_dirichlet)

        steps = max(1, (simulations + PARALLEL_SIMS - 1) // PARALLEL_SIMS)
        for _ in range(steps):
            all_tensors, all_meta = [], []
            for g in range(num_games):
                if engines[g].is_game_over(): continue
                for _ in range(PARALLEL_SIMS):
                    node, stack = self._select_py(roots[g])
                    if node.is_terminal:
                        sim = engines[g].copy()
                        for m in stack: sim.make_move_int(m)
                        r = sim.game_result()
                        v = r if sim.side_to_move() == 0 else -r
                        self._backup_py(node, v)
                        continue
                    sim = engines[g].copy()
                    for m in stack: sim.make_move_int(m)
                    all_tensors.append(np.array(sim.get_board_tensor(), dtype=np.float32))
                    all_meta.append((g, node, stack))
                    self._vloss_py(node, VIRTUAL_LOSS)
            if not all_tensors: continue
            tensor_matrix = np.stack(all_tensors).reshape(len(all_tensors), FLAT_SIZE)
            pols, vals, _, _ = self._infer(tensor_matrix)
            for i, (g, node, stack) in enumerate(all_meta):
                sim = engines[g].copy()
                for m in stack: sim.make_move_int(m)
                if not node.is_expanded:
                    self._expand_node_py(node, sim, pols[i], add_noise=False)
                self._vloss_py(node, -VIRTUAL_LOSS)
                self._backup_py(node, float(vals[i]))

        result = []
        for g, root in enumerate(roots):
            pol = np.zeros(POLICY_SIZE, dtype=np.float32)
            total = sum(c.visits for c in root.children.values())
            if total > 0:
                for m, child in root.children.items():
                    idx = engines[g].move_int_to_policy_idx(m)
                    if idx is not None: pol[idx] = child.visits / total
            result.append(pol)
        return result

    # ── Helpers for gui.py ─────────────────────────────────────────────────────

    def _select_py(self, root):
        import math
        FPU_REDUCTION = 0.330
        node, stack = root, []
        while node.is_expanded and node.children and not node.is_terminal:
            parent_q = node.q()
            sqrt_n = math.sqrt(max(node.visits + node.virtual_loss, 1))
            # Relative FPU: fpu = parent_q - 0.330 * sqrt(sum_of_visited_priors)
            visited_pol = sum(c.prior for c in node.children.values()
                              if c.visits > 0 or c.virtual_loss > 0)
            fpu = max(parent_q - FPU_REDUCTION * math.sqrt(visited_pol), -1.0)
            best, best_s = None, -1e18
            for child in node.children.values():
                started = child.visits + child.virtual_loss
                q_in_parent = -child.q() if started > 0 else fpu
                s = q_in_parent + self.c_puct * child.prior * sqrt_n / (1 + started)
                if s > best_s: best_s = s; best = child
            node = best; stack.append(node.move)
        return node, stack

    def _expand_node_py(self, node, engine, policy_vec, add_noise=False):
        if node.is_expanded: return
        legal = engine.get_legal_moves_int()
        if not legal:
            node.is_terminal = True; node.is_expanded = True; return
        n = len(legal)
        priors = np.array([
            float(policy_vec[idx]) if (idx := engine.move_int_to_policy_idx(m)) is not None
                                       and 0 <= idx < len(policy_vec) else 1e-8
            for m in legal
        ], dtype=np.float64)
        s = priors.sum()
        if s > 1e-12:
            priors = priors / s
        else:
            # dead policy: all priors ≈ 0 → uniform fallback
            # Frequent occurrences = policy collapse
            priors = np.ones(n) / n
            if not hasattr(self, '_dead_policy_count'):
                self._dead_policy_count = 0
            self._dead_policy_count += 1
        if add_noise and n > 0:
            priors = 0.75 * priors + 0.25 * np.random.dirichlet([0.3] * n)
        for j, m in enumerate(legal):
            node.children[m] = MCTSNode(node, m, float(priors[j]))
        node.is_expanded = True

    def _vloss_py(self, node, delta):
        cur = node
        while cur:
            cur.virtual_loss = max(0, cur.virtual_loss + delta) if delta < 0 else cur.virtual_loss + delta
            cur = cur.parent

    def _backup_py(self, leaf, value):
        cur, sign = leaf, 1.0
        while cur:
            cur.visits += 1; cur.value_sum += value * sign; sign *= -1.0; cur = cur.parent

    def _select(self, root): return self._select_py(root)
    def _expand_node(self, node, engine, policy_vec, add_noise=False):
        return self._expand_node_py(node, engine, policy_vec, add_noise)
    def _apply_virtual_loss(self, node, delta): return self._vloss_py(node, delta)
    def _backup(self, leaf, move_stack, value, apply_vloss=False):
        return self._backup_py(leaf, value)


def mcts_policy_vector(engine, net, simulations=80, c_puct=1.25, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcts = UltraFastMCTS(net, device, c_puct, batch_size=256, add_dirichlet=True)
    return mcts.search_games([engine], simulations)[0]
