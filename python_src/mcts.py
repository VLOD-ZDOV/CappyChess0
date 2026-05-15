# mcts.py — Python-обёртка над RustMCTS
#
# ОПТИМИЗАЦИИ v2:
#   - _infer теперь принимает np.ndarray (N,1600) напрямую — убран list comprehension
#   - PARALLEL_SIMS поднят до 32: меньше round-trips Python↔Rust↔GPU за игру
#   - _infer: убран промежуточный np.stack (leaf_matrix уже готовая матрица)
#   - cuda.synchronize убран — non_blocking=True + autocast достаточно
#   - pinned memory переиспользуется без лишних копий

import numpy as np
import torch
from typing import List

try:
    from capablanca_engine import RustMCTS as _RustMCTS
    RUST_MCTS_AVAILABLE = True
except ImportError:
    RUST_MCTS_AVAILABLE = False
    print("⚠️  RustMCTS не найден — нужно пересобрать: maturin develop --release")

POLICY_SIZE   = 7000
VIRTUAL_LOSS  = 3
PARALLEL_SIMS = 64   # 64: оптимальный баланс GPU-утилизации vs качества поиска для 256x15 сети


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
                 add_dirichlet: bool = True, parallel_sims: int = None):
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.add_dirichlet = add_dirichlet
        self._parallel_sims = parallel_sims if parallel_sims is not None else PARALLEL_SIMS

        # Pinned memory: batch_size × parallel_sims листьев максимум
        # Больше parallel_sims = меньше GPU вызовов = быстрее, но хуже качество MCTS
        MAX_LEAVES = max(8192, batch_size * self._parallel_sims * 2)
        self.pinned_buf = torch.empty(MAX_LEAVES, 20, 8, 10,
                                      pin_memory=True, dtype=torch.float32)
        self.pinned_size = MAX_LEAVES
        self.net.eval()

    @torch.no_grad()
    def _infer(self, tensors):
        """
        Батчевый GPU inference.
        tensors: np.ndarray (N, 1600) из Rust,
                 ИЛИ List[np.ndarray (1600,)] из gui.py / python fallback.
        Оба формата поддерживаются.
        """
        # Нормализуем: list → matrix
        if isinstance(tensors, list):
            if len(tensors) == 0:
                return np.empty((0, POLICY_SIZE), dtype=np.float32), np.empty((0,), dtype=np.float32)
            tensors = np.stack(tensors, axis=0).reshape(len(tensors), 1600)

        n = tensors.shape[0]
        if n == 0:
            return np.empty((0, POLICY_SIZE), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Паддинг до кратного parallel_sims: маленькие батчи плохо утилизируют GPU.
        # Паддируем дублированием последней позиции, обрезаем результат до n в конце.
        ps = self._parallel_sims
        target = ((n + ps - 1) // ps) * ps
        n_pad = target - n

        arr = tensors.reshape(n, 20, 8, 10)
        if n_pad > 0:
            # Случайный паддинг из батча вместо repeat-last:
            # repeat-last перекашивает BatchNorm/GroupNorm статистики к одной позиции.
            pad_idx = np.random.randint(0, n, n_pad)
            arr = np.concatenate([arr, arr[pad_idx]], axis=0)
        n_total = arr.shape[0]

        if n_total <= self.pinned_size:
            buf = self.pinned_buf[:n_total]
            buf.copy_(torch.from_numpy(arr))
            x = buf.to(self.device, non_blocking=True)
        else:
            x = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device, non_blocking=True)

        x = x.to(memory_format=torch.channels_last)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, values = self.net(x)

        logits_f = logits.float()
        values_f = values.float()

        # Проверка на NaN/inf — может возникнуть при несовместимости весов
        if torch.isnan(logits_f).any() or torch.isinf(logits_f).any():
            print(f"⚠️  _infer: NaN/inf в logits! Возвращаем равномерное распределение.")
            logits_f = torch.zeros_like(logits_f)
        if torch.isnan(values_f).any() or torch.isinf(values_f).any():
            print(f"⚠️  _infer: NaN/inf в values! Возвращаем нули.")
            values_f = torch.zeros_like(values_f)

        policies = torch.softmax(logits_f, dim=1).cpu().numpy()

        # WDL → Q: сеть возвращает (N,3) logits [Win,Draw,Loss]
        # Q = P(Win) - P(Loss) ∈ [-1, 1] — скалярная оценка для MCTS backup
        # Если values (N,1) — старая модель, конвертируем напрямую
        if values_f.shape[-1] == 3:
            wdl_probs = torch.softmax(values_f, dim=1)
            q_values  = (wdl_probs[:, 0] - wdl_probs[:, 2]).cpu().numpy()
        else:
            # Совместимость со старыми скалярными чекпоинтами
            q_values = values_f.view(-1).cpu().numpy()

        # Убираем паддинг — возвращаем только реальные n позиций
        return policies[:n], q_values[:n]

    def search_games(self, engines: List, simulations: int = 80) -> List[np.ndarray]:
        if RUST_MCTS_AVAILABLE:
            return self._search_rust(engines, simulations)
        return self._search_python(engines, simulations)

    def search_games_with_values(self, engines: List, simulations: int = 80):
        """Возвращает (policies, values). values нужны для resign логики."""
        if RUST_MCTS_AVAILABLE:
            rust_mcts = _RustMCTS(engines, self._parallel_sims)
            steps = max(1, (simulations + self._parallel_sims - 1) // self._parallel_sims)

            prev_policies = None
            prev_values   = None
            prev_counts   = None

            for step in range(steps + 1):
                if step < steps:
                    leaf_matrix = rust_mcts.collect_leaves()
                    has_leaves  = leaf_matrix.shape[0] > 0
                    if has_leaves:
                        curr_counts = rust_mcts.get_current_batch_counts()
                else:
                    has_leaves = False

                if prev_policies is not None and prev_counts is not None:
                    rust_mcts.apply_inference_buffered(
                        prev_policies, prev_values, prev_counts
                    )

                if has_leaves:
                    p, v = self._infer(leaf_matrix)
                    prev_policies = np.ascontiguousarray(p, dtype=np.float32)
                    prev_values   = np.ascontiguousarray(v, dtype=np.float32)
                    prev_counts   = curr_counts
                else:
                    prev_policies = prev_values = prev_counts = None

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
        prev_counts    = None

        for step in range(steps + 1):
            # Собираем следующий батч (если ещё есть шаги)
            if step < steps:
                leaf_matrix = rust_mcts.collect_leaves()
                has_leaves  = leaf_matrix.shape[0] > 0
                if has_leaves:
                    # Сохраняем counts ЭТОГО батча — он не изменится при следующем collect
                    curr_counts = rust_mcts.get_current_batch_counts()
            else:
                has_leaves = False

            # Применяем результаты предыдущего inference с ПРАВИЛЬНЫМИ counts
            if prev_policies is not None and prev_counts is not None:
                rust_mcts.apply_inference_buffered(
                    prev_policies, prev_values, prev_counts
                )

            # Запускаем inference для текущего батча
            if has_leaves:
                p, v = self._infer(leaf_matrix)
                prev_policies = np.ascontiguousarray(p, dtype=np.float32)
                prev_values   = np.ascontiguousarray(v, dtype=np.float32)
                prev_counts   = curr_counts
            else:
                prev_policies = None
                prev_values   = None
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
        ]).reshape(num_games, 1600)
        policies, _ = self._infer(root_tensors)
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
            tensor_matrix = np.stack(all_tensors).reshape(len(all_tensors), 1600)
            pols, vals = self._infer(tensor_matrix)
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
        node, stack = root, []
        while node.is_expanded and node.children and not node.is_terminal:
            sqrt_n = math.sqrt(max(node.visits + node.virtual_loss, 1))
            best, best_s = None, -1e18
            for child in node.children.values():
                s = child.q() + self.c_puct * child.prior * sqrt_n / (1 + child.visits + child.virtual_loss)
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
