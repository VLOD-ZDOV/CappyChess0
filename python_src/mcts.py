# mcts.py — Python-обёртка над RustMCTS
#
# ОПТИМИЗАЦИИ v2:
#   - _infer теперь принимает np.ndarray (N, FLAT_SIZE) напрямую — убран list comprehension
#   - PARALLEL_SIMS поднят до 32: меньше round-trips Python↔Rust↔GPU за игру
#   - _infer: убран промежуточный np.stack (leaf_matrix уже готовая матрица)
#   - cuda.synchronize убран — non_blocking=True + autocast достаточно
#   - pinned memory переиспользуется без лишних копий

import numpy as np
import torch
from collections import OrderedDict
from typing import List

# NVIDIA оптимизации (повторяет настройки train.py — гарантирует включение
# при импорте из gui.py / game_stats.py, когда train.py не загружается).
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

try:
    from capablanca_engine import RustMCTS as _RustMCTS
    RUST_MCTS_AVAILABLE = True
except ImportError:
    RUST_MCTS_AVAILABLE = False
    print("⚠️  RustMCTS не найден — нужно пересобрать: maturin develop --release")

# Импортируем константы инпута — все размеры/reshape должны идти от них
from model import CapablancaNet
INPUT_PLANES = CapablancaNet.INPUT_PLANES   # 139 (8 history × 17 + 3 meta)
BOARD_H = CapablancaNet.BOARD_H              # 8
BOARD_W = CapablancaNet.BOARD_W              # 10
FLAT_SIZE = INPUT_PLANES * BOARD_H * BOARD_W # 11120

POLICY_SIZE   = 7000
VIRTUAL_LOSS  = 3
# Высокий parallel_sims (>64) насыщает virtual_loss → exploration ломается,
# PUCT начинает выбирать слабые ветки. Низкий (<16) недогружает GPU.
# 32 — sweet spot для большинства сетей.
PARALLEL_SIMS = 32

# LC0 transposition cache: одна и та же позиция оценивается NN один раз.
# Полезно: транспозиции в MCTS, повторяющиеся корни между играми, tree reuse.
# Ключ — bytes view тензора (FLAT_SIZE float32 ≈ 44KB при history planes). Хеш bytes быстрый (cityhash в CPython).
# 50K записей × ~30KB на запись (тензор+policy+q+d) ≈ 1.5GB — устанавливаем выше при необходимости.
NN_CACHE_DEFAULT_MAX = 50_000


class MCTSNode:
    """Заглушка для обратной совместимости с gui.py."""
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
    Батчевый MCTS. Дерево живёт в Rust, Python — только GPU inference.
    """

    def __init__(self, net: torch.nn.Module, device: torch.device,
                 c_puct: float = 1.25, batch_size: int = 256,
                 add_dirichlet: bool = True, parallel_sims: int = None,
                 nn_cache: bool = False, nn_cache_max: int = NN_CACHE_DEFAULT_MAX,
                 compile_mode: str = None, bf16_weights: bool = True,
                 kld_threshold: float = 0.0, kld_check_every: int = 4,
                 kld_min_sims_frac: float = 0.25):
        # compile_mode: None (без compile), 'default', 'reduce-overhead', 'max-autotune'.
        # 'default' — самый безопасный, ~15-25% speedup, минимальный warmup.
        # 'reduce-overhead' — использует CUDA graphs, до 50% speedup, но recompile при смене shape.
        # 'max-autotune' — лучший speedup, но warmup 1-2 минуты.
        #
        # bf16_weights: True → создаёт BF16-копию весов для inference. Training-net остаётся FP32.
        # ~1.5-2x speedup vs autocast(bf16) — нет cast FP32→BF16 на каждом read of weights.
        # VRAM: дополнительная BF16-копия (~½ от FP32 размера; для 128ch×10 это ~6 MB).
        self._bf16_weights = bf16_weights and torch.cuda.is_available()
        if self._bf16_weights:
            import copy as _copy
            # Распаковываем _orig_mod если net уже под torch.compile.
            src = net._orig_mod if hasattr(net, '_orig_mod') else net
            inference_net = _copy.deepcopy(src).to(torch.bfloat16).eval()
            self.net = inference_net
            # Сохраняем ссылку на оригинальный FP32 net чтобы потом обновлять BF16-копию
            # через update_inference_weights() после train epoch.
            self._fp32_src = src
        else:
            self.net = net
            self._fp32_src = None
        self.device = device
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.add_dirichlet = add_dirichlet
        self._parallel_sims = parallel_sims if parallel_sims is not None else PARALLEL_SIMS
        if self._parallel_sims > 64:
            print(f"⚠️  parallel_sims={self._parallel_sims} > 64: PUCT exploration "
                  f"может ломаться из-за насыщения virtual_loss. Рекомендуется 16-64.")

        # torch.compile: компилирует forward в оптимизированный CUDA graph.
        # На Blackwell с BF16 даёт +15-50% к raw inference. Применяется поверх
        # BF16-копии (если bf16_weights=True) либо оригинального net.
        self._compile_mode = compile_mode
        if compile_mode is not None and hasattr(torch, 'compile'):
            try:
                # dynamic=False: shapes фиксированы (padding к multiple of parallel_sims),
                # позволяет агрессивные оптимизации. Если shape всё-таки меняется,
                # torch.compile сам сделает recapture.
                self.net = torch.compile(self.net, mode=compile_mode, dynamic=False)
                print(f"🔥 torch.compile(mode={compile_mode!r}) — первый inference будет медленнее (warmup).")
            except Exception as e:
                print(f"⚠️  torch.compile failed: {e}. Откат на eager mode.")

        # Pinned memory: batch_size × parallel_sims листьев максимум.
        # BF16 pinned: tensor.copy_(fp32_tensor) делает CPU-side cast FP32→BF16 за O(N),
        # затем H2D copy уже идёт в BF16 — в 2 раза меньше PCIe bandwidth (44KB → 22KB на сэмпл).
        # На больших батчах (8192 leaves × 22KB = 176MB вместо 352MB) — заметный win.
        MAX_LEAVES = max(8192, batch_size * self._parallel_sims * 2)
        self.pinned_buf = torch.empty(MAX_LEAVES, INPUT_PLANES, BOARD_H, BOARD_W,
                                      pin_memory=True, dtype=torch.bfloat16)
        self.pinned_size = MAX_LEAVES
        self.net.eval()

        # NN transposition cache. Включается ТОЛЬКО для inference (gui/play),
        # для self-play обычно выключен — ветки чаще уникальны + кеш растёт.
        self.nn_cache_enabled = nn_cache
        self.nn_cache_max = nn_cache_max
        self.nn_cache: "OrderedDict[bytes, tuple]" = OrderedDict() if nn_cache else None
        self._cache_hits = 0
        self._cache_misses = 0

        # KLD-early-exit (Lc0 smart pruning).
        # После каждых kld_check_every parallel-шагов проверяем KL(prev_visits || curr_visits)
        # для всех живых игр. Если max KL < threshold AND visits >= min_frac*total → break.
        # threshold=0 отключает фичу.
        self.kld_threshold = float(kld_threshold)
        self.kld_check_every = max(1, int(kld_check_every))
        self.kld_min_sims_frac = float(kld_min_sims_frac)
        self._kld_early_exits = 0       # сколько раз сработал early-exit
        self._kld_total_calls = 0       # всего поисковых вызовов
        self._kld_sims_saved = 0        # суммарно сэкономлено sims
        self._kld_sims_requested = 0    # суммарно запрошено sims

    def clear_nn_cache(self) -> None:
        """Сбрасывает кеш — нужно вызывать после обновления весов сети."""
        if self.nn_cache is not None:
            self.nn_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    @staticmethod
    def _max_kl_divergence(prev: np.ndarray, curr: np.ndarray, eps: float = 1e-8) -> float:
        """Max KL(prev || curr) по играм. prev/curr: (N_games, POLICY_SIZE), стохастические.

        KL(P||Q) = Σ P * log(P/Q). Используется L0-style smart-pruning gate.
        Возвращает +inf если массивы пусты.
        """
        if prev.shape != curr.shape or prev.size == 0:
            return float('inf')
        p = prev + eps
        q = curr + eps
        kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)  # (N_games,)
        return float(np.max(kl))

    def kld_stats(self) -> dict:
        """Статистика KLD-early-exit. Полезно для логирования."""
        if self._kld_total_calls == 0:
            return {"exit_rate": 0.0, "savings": 0.0, "calls": 0}
        return {
            "exit_rate": self._kld_early_exits / self._kld_total_calls,
            "savings": (self._kld_sims_saved / self._kld_sims_requested
                        if self._kld_sims_requested > 0 else 0.0),
            "calls": self._kld_total_calls,
        }

    def update_inference_weights(self, fp32_net: torch.nn.Module = None) -> None:
        """Синхронизирует BF16-копию весов inference-сети с актуальным FP32 net.

        Вызывать после каждого train_epoch / EMA apply, чтобы selfplay использовал
        свежие веса. Если bf16_weights=False — ничего не делает.

        fp32_net: опционально передать другой FP32 net (например, EMA shadow).
                  По умолчанию использует self._fp32_src (тот net что был при init).
        """
        if not self._bf16_weights:
            return
        src = fp32_net if fp32_net is not None else self._fp32_src
        if src is None:
            return
        # Распаковываем компилированную обёртку если есть
        src_inner = src._orig_mod if hasattr(src, '_orig_mod') else src
        # Распаковываем target если он под torch.compile
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
        Батчевый GPU inference с опциональным NN-кешем (transposition table).

        tensors: np.ndarray (N, FLAT_SIZE) или List.
        hashes:  опциональный List[int] u64 — board hashes как ключи кеша.

        Возвращает (policies, q_values, d_values, mlh_values).
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
        """Прямой батчевый вызов NN без кеша. tensors: ndarray (N, FLAT_SIZE).
        Возвращает (policies, q, d, m)."""
        n = tensors.shape[0]
        if n == 0:
            empty_p = np.empty((0, POLICY_SIZE), dtype=np.float32)
            empty_v = np.empty((0,), dtype=np.float32)
            return empty_p, empty_v, empty_v.copy(), empty_v.copy()

        # Паддинг до кратного parallel_sims: маленькие батчи плохо утилизируют GPU.
        # Паддируем дублированием случайных позиций, обрезаем результат до n в конце.
        ps = self._parallel_sims
        target = ((n + ps - 1) // ps) * ps
        n_pad = target - n

        arr = tensors.reshape(n, INPUT_PLANES, BOARD_H, BOARD_W)
        if n_pad > 0:
            pad_idx = np.random.randint(0, n, n_pad)
            arr = np.concatenate([arr, arr[pad_idx]], axis=0)
        n_total = arr.shape[0]

        if n_total <= self.pinned_size:
            buf = self.pinned_buf[:n_total]
            # copy_(fp32) → CPU-side cast в BF16 (pinned_buf — bfloat16).
            buf.copy_(torch.from_numpy(arr))
            x = buf.to(self.device, non_blocking=True)
        else:
            # Fallback при превышении pinned-буфера: всё равно кастим в BF16 до H2D.
            cpu_t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16)
            x = cpu_t.to(self.device, non_blocking=True)

        x = x.to(memory_format=torch.channels_last)
        if self._bf16_weights:
            # И веса и input в BF16 → autocast не нужен (избегаем overhead context manager).
            out = self.net(x)
        else:
            # FP32 веса: autocast делает все matmul/conv в BF16 на лету.
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = self.net(x)

        # Поддерживаем варианты сети:
        #   - 4 выхода: (policy, wdl, mlh, future) — модель с future-головой
        #   - 3 выхода: (policy, wdl, mlh)         — модель с MLH
        #   - 2 выхода: (policy, wdl)              — старые чекпоинты
        # future-голова в inference не нужна → игнорируем.
        if isinstance(out, tuple) and len(out) == 4:
            logits, values, mlh_raw, _ = out
        elif isinstance(out, tuple) and len(out) == 3:
            logits, values, mlh_raw = out
        else:
            logits, values = out
            mlh_raw = None

        logits_f = logits.float()
        values_f = values.float()

        if torch.isnan(logits_f).any() or torch.isinf(logits_f).any():
            print(f"⚠️  _infer: NaN/inf в logits! Возвращаем равномерное распределение.")
            logits_f = torch.zeros_like(logits_f)
        if torch.isnan(values_f).any() or torch.isinf(values_f).any():
            print(f"⚠️  _infer: NaN/inf в values! Возвращаем нули.")
            values_f = torch.zeros_like(values_f)

        policies = torch.softmax(logits_f, dim=1).cpu().numpy()

        if values_f.shape[-1] == 3:
            wdl_probs = torch.softmax(values_f, dim=1)
            q_values  = (wdl_probs[:, 0] - wdl_probs[:, 2]).cpu().numpy()
            d_values  = wdl_probs[:, 1].cpu().numpy()
        else:
            q_values = values_f.view(-1).cpu().numpy()
            d_values = np.zeros_like(q_values)

        # MLH: sigmoid raw → ∈ [0, 1] (нормализованная "доля до конца игры").
        if mlh_raw is not None:
            mlh_f = mlh_raw.float()
            if torch.isnan(mlh_f).any() or torch.isinf(mlh_f).any():
                mlh_f = torch.zeros_like(mlh_f)
            m_values = torch.sigmoid(mlh_f).view(-1).cpu().numpy()
        else:
            m_values = np.zeros_like(q_values)

        return policies[:n], q_values[:n], d_values[:n], m_values[:n]

    def search_games(self, engines: List, simulations: int = 80) -> List[np.ndarray]:
        if RUST_MCTS_AVAILABLE:
            return self._search_rust(engines, simulations)
        return self._search_python(engines, simulations)

    def search_games_with_values(self, engines: List, simulations: int = 80):
        """Возвращает (policies, values). values нужны для resign логики.

        Если self.kld_threshold > 0, MCTS может остановиться досрочно когда
        визит-распределение перестало меняться (Lc0 smart pruning).
        """
        if RUST_MCTS_AVAILABLE:
            rust_mcts = _RustMCTS(engines, self._parallel_sims)
            steps = max(1, (simulations + self._parallel_sims - 1) // self._parallel_sims)

            prev_policies = None
            prev_values   = None
            prev_draws    = None
            prev_mlhs     = None
            prev_counts   = None

            # KLD-early-exit (Rust-side compute, минимум marshalling).
            kld_enabled = self.kld_threshold > 0.0
            kld_min_steps = int(np.ceil(steps * self.kld_min_sims_frac))
            if kld_enabled:
                rust_mcts.kld_reset_all()
            self._kld_total_calls += 1
            self._kld_sims_requested += simulations
            early_exit_step = None  # для отчёта

            for step in range(steps + 1):
                if step < steps:
                    leaf_matrix = rust_mcts.collect_leaves(simulations)
                    has_leaves  = leaf_matrix.shape[0] > 0
                    if has_leaves:
                        curr_counts = rust_mcts.get_current_batch_counts()
                        # Получаем хеши ДО apply_inference, т.к. apply_inference очищает pending.
                        curr_hashes = rust_mcts.get_leaf_hashes() if self.nn_cache_enabled else None
                else:
                    has_leaves = False

                if prev_policies is not None and prev_counts is not None:
                    rust_mcts.apply_inference_buffered(
                        prev_policies, prev_values, prev_draws, prev_mlhs, prev_counts
                    )
                    prev_policies = prev_values = prev_draws = prev_mlhs = prev_counts = None

                if has_leaves:
                    p, v, d, m = self._infer(leaf_matrix, hashes=curr_hashes)
                    prev_policies = np.ascontiguousarray(p, dtype=np.float32)
                    prev_values   = np.ascontiguousarray(v, dtype=np.float32)
                    prev_draws    = np.ascontiguousarray(d, dtype=np.float32)
                    prev_mlhs     = np.ascontiguousarray(m, dtype=np.float32)
                    prev_counts   = curr_counts

                # === KLD-early-exit check (Lc0-style smart pruning) ===
                # Rust считает max KL gain — Python получает 1 float.
                if (kld_enabled and step >= kld_min_steps and step < steps
                        and (step + 1) % self.kld_check_every == 0):
                    # Применяем pending inference чтобы KL отражал свежие визиты.
                    if prev_policies is not None and prev_counts is not None:
                        rust_mcts.apply_inference_buffered(
                            prev_policies, prev_values, prev_draws, prev_mlhs, prev_counts
                        )
                        prev_policies = prev_values = prev_draws = prev_mlhs = prev_counts = None
                    max_kl = rust_mcts.kld_snapshot_and_check()
                    if max_kl != float('inf'):
                        sims_added = self.kld_check_every * self._parallel_sims
                        kl_gain = max_kl / max(1, sims_added)
                        if kl_gain < self.kld_threshold:
                            self._kld_early_exits += 1
                            self._kld_sims_saved += (steps - step - 1) * self._parallel_sims
                            early_exit_step = step
                            break

            # Финальный pending — если break не произошёл, может остаться один apply.
            if prev_policies is not None and prev_counts is not None:
                rust_mcts.apply_inference_buffered(
                    prev_policies, prev_values, prev_draws, prev_mlhs, prev_counts
                )

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
        steps = max(1, (simulations + self._parallel_sims - 1) // self._parallel_sims)

        # Правильная двойная буферизация:
        # GPU считает батч N пока Rust собирает батч N+1.
        # Ключ: batch_counts сохраняется ВМЕСТЕ с данными батча N
        # и передаётся в apply_inference_buffered — не перезаписывается батчем N+1.
        #
        # Схема:
        #   шаг 0: collect(0) → counts_0 = get_counts() → infer_start(0)
        #   шаг 1: collect(1) [пока GPU считает 0] → counts_1
        #          apply_buffered(result_0, counts_0) → infer_start(1)
        #   шаг 2: collect(2) → counts_2
        #          apply_buffered(result_1, counts_1) → infer_start(2)
        #   ...
        #   финал: apply_buffered(result_last, counts_last)

        prev_policies  = None
        prev_values    = None
        prev_draws     = None
        prev_mlhs      = None
        prev_counts    = None

        for step in range(steps + 1):
            # Собираем следующий батч (если ещё есть шаги)
            if step < steps:
                leaf_matrix = rust_mcts.collect_leaves(simulations)
                has_leaves  = leaf_matrix.shape[0] > 0
                if has_leaves:
                    # Сохраняем counts ЭТОГО батча — он не изменится при следующем collect
                    curr_counts = rust_mcts.get_current_batch_counts()
                    curr_hashes = rust_mcts.get_leaf_hashes() if self.nn_cache_enabled else None
            else:
                has_leaves = False

            # Применяем результаты предыдущего inference с ПРАВИЛЬНЫМИ counts
            if prev_policies is not None and prev_counts is not None:
                rust_mcts.apply_inference_buffered(
                    prev_policies, prev_values, prev_draws, prev_mlhs, prev_counts
                )

            # Запускаем inference для текущего батча
            if has_leaves:
                p, v, d, m = self._infer(leaf_matrix, hashes=curr_hashes)
                prev_policies = np.ascontiguousarray(p, dtype=np.float32)
                prev_values   = np.ascontiguousarray(v, dtype=np.float32)
                prev_draws    = np.ascontiguousarray(d, dtype=np.float32)
                prev_mlhs     = np.ascontiguousarray(m, dtype=np.float32)
                prev_counts   = curr_counts
            else:
                prev_policies = None
                prev_values   = None
                prev_draws    = None
                prev_mlhs     = None
                prev_counts   = None

        raw = rust_mcts.get_policies()
        return [np.array(p, dtype=np.float32) for p in raw]

    # ── Python fallback ────────────────────────────────────────────────────────

    def _search_python(self, engines: List, simulations: int) -> List[np.ndarray]:
        """Старый Python MCTS — используется только если RustMCTS не скомпилирован."""
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

    # ── Helpers для gui.py ─────────────────────────────────────────────────────

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
            # dead policy: все priors ≈ 0 → uniform fallback
            # Частые такие случаи = policy collapse
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
