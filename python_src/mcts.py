# mcts.py — thin Python driver for the Rust MCTS.
#
# The search tree lives in `capablanca_engine.RustMCTS`; this module only
# feeds it batched GPU inference (BF16 weights, pinned staging buffer,
# optional transposition cache) and owns the KLD early-exit bookkeeping.

import time
import numpy as np
import torch
from collections import OrderedDict
from typing import List, Optional

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
                 contempt: float = 0.0, rep_search_perslot: bool = False,
                 rust_c_puct: float = None, trt_inference: bool = False,
                 rust_fpu: float = None, policy_temp: float = 1.0):
        # compile_mode: None (no compile), 'default', 'reduce-overhead', 'max-autotune'.
        # 'default' — safest, ~15-25% speedup, minimal warmup.
        # 'reduce-overhead' — uses CUDA graphs, up to 50% speedup, but recompiles on shape change.
        # 'max-autotune' — best speedup, but 1-2 minute warmup.
        #
        # bf16_weights: True → creates BF16 copy of weights for inference. Training net stays FP32.
        # ~1.5-2x speedup vs autocast(bf16) — no FP32→BF16 cast on every weight read.
        # VRAM: extra BF16 copy (~½ of FP32 size; for 128ch×10 this is ~6 MB).
        # trt_inference: инференс через TensorRT вместо torch.compile. Замер
        # 14.09 на 47M/384ch — 1.40x на батче 96, лучший ход совпадает в 100%
        # позиций, оценка расходится на 0.0008. Движок собирается под свежие
        # веса здесь же, потому что generate_games создаёт MCTS каждую итерацию.
        self._trt = bool(trt_inference) and torch.cuda.is_available()
        self._bf16_weights = bf16_weights and torch.cuda.is_available() and not self._trt
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
        # Base CPuct for the Rust selection. None = leave the engine's own
        # default (lc0's 1.745). `self.c_puct` above is a different thing — it
        # belongs to the Python-side search the GUI uses, and never reached Rust.
        self._rust_c_puct = None if rust_c_puct is None else float(rust_c_puct)
        # Насколько поиск пессимистичен к непосещённым ходам. Умолчание движка
        # 0.33 размазывает посещения: замер 15.09 — после 400 симуляций верхний
        # ход держит 14% из 28, и поиск уходит от ходов, которые политика
        # (корреляция +0.71 с глубокой оценкой движка) считает лучшими.
        self._rust_fpu = None if rust_fpu is None else float(rust_fpu)
        # Температура приоритета перед подачей в дерево. У lc0 её поднимают,
        # чтобы РАЗМЫТЬ слишком резкую политику; у нас случай обратный —
        # политика размазана, значит нужна температура МЕНЬШЕ единицы.
        self._policy_temp = float(policy_temp)
        self._parallel_sims = parallel_sims if parallel_sims is not None else PARALLEL_SIMS
        if self._parallel_sims > 64:
            print(f"⚠️  parallel_sims={self._parallel_sims} > 64: PUCT exploration "
                  f"может ломаться из-за насыщения virtual_loss. Рекомендуется 16-64.")

        # torch.compile: compiles forward into an optimized CUDA graph.
        # On Blackwell with BF16 gives +15-50% raw inference speedup. Applied on top of
        # BF16 copy (if bf16_weights=True) or the original net.
        self._compile_mode = None if self._trt else compile_mode
        # При TensorRT компиляция бессмысленна: граф всё равно будет заменён
        # готовым движком, а прогрев торча стоит минуту на итерацию.
        if self._compile_mode is not None and hasattr(torch, 'compile'):
            try:
                # dynamic=False: shapes are bucketed, so compile sees a bounded
                # set of static shapes and can optimize them aggressively.
                self.net = torch.compile(self.net, mode=self._compile_mode, dynamic=False)
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
                pin_memory=True,
                dtype=torch.float16 if self._trt else torch.bfloat16)
        else:
            self.pinned_buf = torch.empty(
                (MAX_LEAVES, INPUT_PLANES, BOARD_H, BOARD_W),
                pin_memory=False, dtype=torch.float32)
            print("⚠️  CUDA не найдена — inference на CPU. Очень медленно, "
                  "training/eval скорее иллюстративные. Установите GPU + драйвер.")
        self.pinned_size = MAX_LEAVES
        if self._trt:
            from trt_engine import TRTNet
            # Потолок движка — по РЕАЛЬНОМУ верхнему пределу батча (игры в
            # пачке × параллельных симуляций, с запасом), а не по размеру
            # закреплённого буфера: тот берётся с большим избытком (8192), и
            # профиль на такую ширину TensorRT собрать не может.
            trt_max = min(MAX_LEAVES, batch_size * self._parallel_sims * 2)
            self.net = TRTNet(self.net, self.device, max_batch=trt_max)
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
        self._slot_bufs: dict = {}      # slot -> (in, pol_out, qdm_out, event), см. _launch

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

    def _stage_input(self, tensors, buf=None):
        """Leaves → padded device tensor. buf — pinned staging buffer to use
        (default: the shared one). Returns (x, n)."""
        n = tensors.shape[0]
        # Pad to a stable bucket so torch.compile sees a bounded set of input
        # shapes. In eager mode the GPU is compute-bound and throughput is
        # linear in batch size, so pad only to the next multiple of `ps`.
        ps = self._parallel_sims
        target = _bucket_size(n, ps, pow2=self._compile_mode is not None)
        n_pad = target - n
        arr = np.ascontiguousarray(
            tensors.reshape(n, INPUT_PLANES, BOARD_H, BOARD_W)
        )
        if buf is None:
            buf = self.pinned_buf
        if target <= buf.shape[0]:
            b = buf[:target]
            # copy_(fp32) casts to buf's dtype (BF16 on CUDA, FP32 on CPU).
            b[:n].copy_(torch.from_numpy(arr))
            if n_pad > 0:
                # Padding rows are discarded after inference: duplicate existing
                # rows in place instead of np.concatenate on every step.
                filled = n
                while filled < target:
                    take = min(filled, target - filled)
                    b[filled:filled + take].copy_(b[:take])
                    filled += take
            return b.to(self.device, non_blocking=True), n
        # Pinned buffer exceeded. Match the buffer dtype so we never feed BF16
        # into FP32 weights on CPU.
        if n_pad > 0:
            pad_idx = np.arange(n_pad) % n
            arr = np.concatenate([arr, arr[pad_idx]], axis=0)
        target_dtype = (torch.float16 if self._trt else
                        torch.bfloat16) if self._has_cuda else torch.float32
        cpu_t = torch.from_numpy(arr).to(target_dtype)
        return cpu_t.to(self.device, non_blocking=True), n

    def _forward_post(self, x):
        """Forward + postprocessing on the device. Returns float32 device
        tensors (policies, q, d, m) over the padded batch; m/d may be None."""
        if self._trt or self._bf16_weights or not self._has_cuda:
            # TRT: FP16 engine; BF16 weights + BF16 input; CPU: all FP32.
            out = self.net(x)
        else:
            # FP32 weights on CUDA: autocast runs matmul/conv in BF16 on the fly.
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = self.net(x)

        # Output variants: (policy, wdl, mlh, future) / (policy, wdl, mlh) /
        # (policy, wdl) for old checkpoints. future is not needed at inference.
        if isinstance(out, tuple) and len(out) == 4:
            logits, values, mlh_raw, _ = out
        elif isinstance(out, tuple) and len(out) == 3:
            logits, values, mlh_raw = out
        else:
            logits, values = out
            mlh_raw = None

        # nan_to_num on the GPU instead of `.isnan().any()` guards: those forced
        # a device→host sync on every batch. zeros → uniform softmax / neutral value.
        logits_f = torch.nan_to_num(logits.float(), nan=0.0, posinf=0.0, neginf=0.0)
        values_f = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)

        if self._policy_temp != 1.0:
            # Температура на логитах до softmax, а не степень вероятностей.
            logits_f = logits_f / self._policy_temp
        policies = torch.softmax(logits_f, dim=1)

        if values_f.shape[-1] == 3:
            wdl_probs = torch.softmax(values_f, dim=1)
            q = wdl_probs[:, 0] - wdl_probs[:, 2]
            d = wdl_probs[:, 1]
        else:
            q = values_f.view(-1)
            d = None
        m = None
        if mlh_raw is not None:
            # sigmoid raw → ∈ [0, 1] (доля оставшейся партии).
            mlh_f = torch.nan_to_num(mlh_raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
            m = torch.sigmoid(mlh_f).view(-1)
        return policies, q, d, m

    @torch.no_grad()
    def _infer_raw_nn(self, tensors):
        """Direct batched NN call without cache. tensors: ndarray (N, FLAT_SIZE).
        Returns (policies, q, d, m)."""
        n = tensors.shape[0]
        if n == 0:
            empty_p = np.empty((0, POLICY_SIZE), dtype=np.float32)
            empty_v = np.empty((0,), dtype=np.float32)
            return empty_p, empty_v, empty_v.copy(), empty_v.copy()
        x, n = self._stage_input(tensors)
        pol, q, d, m = self._forward_post(x)
        policies = pol.cpu().numpy()
        q_values = q.cpu().numpy()
        d_values = d.cpu().numpy() if d is not None else np.zeros_like(q_values)
        m_values = m.cpu().numpy() if m is not None else np.zeros_like(q_values)
        return policies[:n], q_values[:n], d_values[:n], m_values[:n]

    def _can_async(self) -> bool:
        # Кэш позиций и TensorRT идут прежним синхронным путём: кэшу нужен
        # ответ сразу, а про поток исполнения движка TRT мы ничего не знаем.
        return self._has_cuda and not self._trt and not self.nn_cache_enabled

    @torch.no_grad()
    def _launch(self, tensors, slot: int):
        """Поставить пачку в очередь карты и вернуться, не дожидаясь ответа.

        slot — номер собственного набора закреплённых буферов (вход и выходы).
        Грабли: буфер слота переиспользуется только после _wait этого же
        слота. Самоигра так и ходит — группа ждёт свой ответ, разносит его и
        лишь потом ставит следующую пачку; два слота = две пачки в полёте."""
        n = tensors.shape[0]
        target = _bucket_size(n, self._parallel_sims,
                              pow2=self._compile_mode is not None)
        bufs = self._slot_bufs.get(slot)
        if bufs is None or bufs[0].shape[0] < target:
            cap = max(target, self.batch_size * self._parallel_sims)
            cap = _bucket_size(cap, self._parallel_sims, pow2=True)
            # Свой вход, а не общий pinned_buf: его берёт синхронный _infer.
            bufs = (torch.empty((cap, INPUT_PLANES, BOARD_H, BOARD_W),
                                pin_memory=True, dtype=torch.bfloat16),
                    torch.empty((cap, POLICY_SIZE), dtype=torch.float32, pin_memory=True),
                    torch.empty((cap, 3), dtype=torch.float32, pin_memory=True),
                    torch.cuda.Event())
            self._slot_bufs[slot] = bufs
        in_buf, pol_out, qdm_out, ev = bufs
        x, n = self._stage_input(tensors, in_buf)
        pol, q, d, m = self._forward_post(x)
        zeros = torch.zeros_like(q)
        qdm = torch.stack([q, d if d is not None else zeros,
                           m if m is not None else zeros], dim=1)
        pol_out[:target].copy_(pol, non_blocking=True)
        qdm_out[:target].copy_(qdm, non_blocking=True)
        ev.record()
        return slot, n

    def _wait(self, handle):
        """Дождаться пачки из _launch. Массивы — виды на закреплённые буферы
        слота: действительны до следующего _launch в этот слот."""
        slot, n = handle
        _, pol_out, qdm_out, ev = self._slot_bufs[slot]
        ev.synchronize()
        qdm = qdm_out[:n].numpy()
        return (pol_out[:n].numpy(), np.ascontiguousarray(qdm[:, 0]),
                np.ascontiguousarray(qdm[:, 1]), np.ascontiguousarray(qdm[:, 2]))

    def raw_policies(self, engines: List) -> np.ndarray:
        """Политика сети БЕЗ поиска — один прогон на позицию.

        Нужна, чтобы столкнуть «ход интуиции» с «ходом поиска» в матче: вопрос
        «улучшает ли поиск политику в середине партии» нельзя решить внешним
        арбитром (у движка в этом варианте классическая оценка, слабая как раз
        в тихих позициях), зато его можно решить очками."""
        arr = np.ascontiguousarray(
            np.stack([np.asarray(e.get_board_tensor(), dtype=np.float32)
                      for e in engines]))
        policies, _, _, _ = self._infer(arr)
        return policies

    def new_tree(self, engines: List, seed: Optional[int] = None):
        """Fresh RustMCTS over `engines`, configured from this instance's flags.

        `seed` fixes the Rust-side RNG (root Dirichlet noise). Left None the
        engine seeds itself from the clock, which is what self-play wants; tests
        and A/B runs pass a seed so two runs can be compared directly."""
        rust_mcts = _RustMCTS(engines, self._parallel_sims,
                              None if seed is None else int(seed))
        if self.contempt != 0.0:
            rust_mcts.set_contempt(self.contempt)
        # add_dirichlet=False (eval / FSF / lagged) must actually disable root
        # noise in Rust — the flag used to be silently ignored.
        if not self.add_dirichlet:
            rust_mcts.set_add_dirichlet(False)
        if self._rep_search_perslot:
            rust_mcts.set_rep_search_perslot(True)
        if self._rust_c_puct is not None:
            rust_mcts.set_c_puct(self._rust_c_puct)
        if self._rust_fpu is not None:
            rust_mcts.set_fpu_reduction(self._rust_fpu)
        return rust_mcts

    def run_search(self, rust_mcts, simulations: int, deadline: float = None) -> None:
        """Run one search on an existing tree: collect → infer → apply, with
        Lc0-style KLD early exit. Leaves the tree in place, so callers doing
        tree reuse (self-play, game_stats) drive the same loop as one-shot
        callers instead of each keeping their own copy of it."""
        for _ in self.search_steps(rust_mcts, simulations, deadline):
            pass

    def search_steps(self, rust_mcts, simulations: int, deadline: float = None,
                     slot: int = 0):
        """Тот же поиск, что run_search, но генератором: отдаёт управление
        сразу после того, как пачка листьев ушла на карту. Пока карта считает,
        вызывающий может собрать листья другого дерева (самоигра группами);
        следующий next() дождётся ответа, разнесёт его и поставит новую пачку.
        У каждого одновременно живого генератора должен быть свой slot.

        Inference is applied *immediately* after its own collect_leaves: the
        old "double-buffered" variant deferred the apply by one step, but
        collect_leaves clears Rust's `pending`, so the deferred apply wrote the
        previous batch's NN outputs onto the new leaves."""
        parallel = self._parallel_sims
        steps = max(1, (simulations + parallel - 1) // parallel)
        kld_enabled = self.kld_threshold > 0.0
        kld_min_steps = int(np.ceil(steps * self.kld_min_sims_frac))
        async_ok = self._can_async()
        # Сколько симуляций реально сделано: заказ не равен факту, если сработал
        # срок или ранний выход по сходимости. Нужно для честного протокола партии.
        sims_done = 0
        self.last_sims_done = 0
        if kld_enabled:
            rust_mcts.kld_reset_all()
        self._kld_total_calls += 1
        self._kld_sims_requested += simulations

        for step in range(steps):
            # Жёсткий срок для игры по часам: проверяем между пачками — прерывать
            # пачку на полпути нельзя, Rust ждёт выводы сети на свои листья.
            if deadline is not None and step > 0 and time.perf_counter() >= deadline:
                break
            leaf_matrix = rust_mcts.collect_leaves(simulations)
            if leaf_matrix.shape[0] == 0:
                # Пустая пачка = все выборы пришли в ТЕРМИНАЛЫ, select() их уже
                # откатил, и следующий заход пойдёт в другую ветку. break здесь
                # убивал поиск целиком. Цикл ограничен steps.
                continue
            sims_done += int(leaf_matrix.shape[0])
            self.last_sims_done = sims_done
            curr_counts = rust_mcts.get_current_batch_counts()
            if async_ok:
                handle = self._launch(leaf_matrix, slot)
                del leaf_matrix
                yield
                p, v, d, m = self._wait(handle)
            else:
                curr_hashes = rust_mcts.get_leaf_hashes() if self.nn_cache_enabled else None
                p, v, d, m = self._infer(leaf_matrix, hashes=curr_hashes)
                yield
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
        self.last_sims_done = sims_done

    def search_games(self, engines: List, simulations: int = 80,
                     deadline: float = None) -> List[np.ndarray]:
        return self.search_games_with_values(engines, simulations, deadline)[0]

    def search_games_with_values(self, engines: List, simulations: int = 80,
                                 deadline: float = None):
        """Returns (policies, values). values are needed for resign logic.

        If self.kld_threshold > 0, MCTS may stop early when the
        visit distribution stops changing (Lc0 smart pruning).
        """
        rust_mcts = self.new_tree(engines)
        self.run_search(rust_mcts, simulations, deadline)
        raw_policies = rust_mcts.get_policies()
        raw_values   = rust_mcts.get_values()
        policies = [np.array(p, dtype=np.float32) for p in raw_policies]
        return policies, np.array(raw_values, dtype=np.float32)

