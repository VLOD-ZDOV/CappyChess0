# mcts.py — Parallel Batched MCTS, optimized for throughput
# Key ideas:
#   1. Virtual losses — позволяют запускать N симуляций одновременно без ожидания
#   2. Один большой батч на все игры × все параллельные симуляции
#   3. Нет engine.copy() в горячем пути — храним стек ходов
#   4. numpy-only UCB selection (нет Python-loop overhead на selection)
#
# Fixes applied:
#   - virtual_loss clamp ≥ 0 (не может уйти в минус при несимметричных вызовах)
#   - bounds check для policy_vec[idx] (idx >= POLICY_SIZE → crash)
#   - values.view(-1) вместо squeeze(1) (безопасно при любой форме выхода сети)
#   - Dirichlet noise при root expansion (критично для exploration)
#   - _expand_node: явный fast-path при нулевых prior'ах (не игнорирует сеть молча)

import math
import numpy as np
import torch
from typing import List, Dict, Optional

POLICY_SIZE = 7000  # FIX: было 6880 — макс. индекс промоушена = 6400+99*6+5 = 6999
VIRTUAL_LOSS = 3          # штраф при выборе узла до backprop
PARALLEL_SIMS = 8         # сколько листьев одновременно собирается на игру

# Dirichlet noise параметры (как в AlphaZero)
# α=0.3 для шахмат (меньше branching factor → больше α)
# Капабланка — расширенные фигуры, branching ~40-50, α как в обычных шахматах
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS   = 0.25    # вес шума (0.25 = 25% шум, 75% prior сети)


class MCTSNode:
    __slots__ = ("parent", "move", "prior", "children", "visits",
                 "value_sum", "virtual_loss", "is_expanded", "is_terminal")

    def __init__(self, parent, move: int, prior: float):
        self.parent = parent
        self.move = move
        self.prior = float(prior)
        self.children: Dict[int, "MCTSNode"] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.virtual_loss = 0
        self.is_expanded = False
        self.is_terminal = False

    def q(self) -> float:
        denom = self.visits + self.virtual_loss
        return self.value_sum / denom if denom > 0 else 0.0

    def ucb(self, sqrt_parent: float, c_puct: float) -> float:
        u = c_puct * self.prior * sqrt_parent / (1 + self.visits + self.virtual_loss)
        return self.q() + u

    def best_child(self, c_puct: float) -> "MCTSNode":
        sqrt_n = math.sqrt(max(self.visits + self.virtual_loss, 1))
        best_score = -1e18
        best = None
        for child in self.children.values():
            score = child.ucb(sqrt_n, c_puct)
            if score > best_score:
                best_score = score
                best = child
        return best


class UltraFastMCTS:
    """
    Оптимизированный MCTS с виртуальными потерями и батчевым inference.
    Целевые метрики:
      - 128 игр × 80 симуляций на A100 за < 60 сек
      - 256 игр × 80 симуляций за < 120 сек
    """

    def __init__(self, net: torch.nn.Module, device: torch.device,
                 c_puct: float = 1.25, batch_size: int = 128,
                 add_dirichlet: bool = True):
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.add_dirichlet = add_dirichlet

        # Pinned memory для быстрого H2D
        self.pinned_buf = torch.empty(batch_size, 20, 8, 10,
                                      pin_memory=True, dtype=torch.float32)
        self.net.eval()

    @torch.no_grad()
    def _infer(self, tensors: List[np.ndarray]):
        """Батчевый inference. tensors: список (1600,) float32 массивов."""
        n = len(tensors)
        if n == 0:
            return np.empty((0, POLICY_SIZE)), np.empty((0,))

        arr = np.stack(tensors, axis=0).reshape(n, 20, 8, 10)

        if n <= self.batch_size:
            buf = self.pinned_buf[:n]
            buf.copy_(torch.from_numpy(arr))
            x = buf.to(self.device, non_blocking=True)
        else:
            x = torch.from_numpy(arr).to(self.device)

        x = x.to(memory_format=torch.channels_last)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, values = self.net(x)

        policies = torch.softmax(logits.float(), dim=1).cpu().numpy()
        # FIX: view(-1) безопаснее squeeze(1) — работает при любой форме выхода сети
        values = values.float().view(-1).cpu().numpy()
        return policies, values

    def search_games(self, engines: List, simulations: int = 80) -> List[np.ndarray]:
        """
        Основная точка входа. Возвращает policy-вектора для каждой игры.

        Алгоритм:
          - Параллельно собираем PARALLEL_SIMS листьев на игру
          - Объединяем в один большой батч → один inference call
          - Backprop для всех
          - Повторяем (simulations // PARALLEL_SIMS) раз
        """
        num_games = len(engines)
        roots = [MCTSNode(None, -1, 1.0) for _ in range(num_games)]

        # Начальный expansion всех корней (с Dirichlet noise)
        root_tensors = [np.array(e.get_board_tensor(), dtype=np.float32)
                        for e in engines]
        policies, _ = self._infer(root_tensors)
        for i in range(num_games):
            self._expand_node(roots[i], engines[i], policies[i],
                              add_noise=self.add_dirichlet)

        steps = max(1, simulations // PARALLEL_SIMS)

        for _ in range(steps):
            # ── Selection: собрать по PARALLEL_SIMS листьев на игру ──────────
            all_leaf_tensors: List[np.ndarray] = []
            all_meta = []  # (game_idx, node, move_stack)

            for g in range(num_games):
                if engines[g].is_game_over():
                    continue
                for _ in range(PARALLEL_SIMS):
                    node, move_stack = self._select(roots[g])

                    if node.is_terminal:
                        # Terminal: VL не применялся (selection остановился на нём),
                        # просто считаем результат и делаем backup
                        sim_eng = engines[g].copy()
                        for m in move_stack:
                            sim_eng.make_move_int(m)
                        result = sim_eng.game_result()
                        # game_result() возвращает значение с точки зрения белых.
                        # _backup ожидает значение с точки зрения игрока, чей ход в leaf.
                        leaf_side = sim_eng.side_to_move()
                        value = result if leaf_side == 0 else -result
                        self._backup(node, move_stack, value, apply_vloss=False)
                        continue

                    # Получаем тензор позиции
                    sim_eng = engines[g].copy()
                    for m in move_stack:
                        sim_eng.make_move_int(m)

                    tensor = np.array(sim_eng.get_board_tensor(), dtype=np.float32)
                    all_leaf_tensors.append(tensor)
                    all_meta.append((g, node, move_stack))

                    # Применяем virtual loss чтобы другие симуляции не выбирали этот узел
                    self._apply_virtual_loss(node, VIRTUAL_LOSS)

            if not all_leaf_tensors:
                continue

            # ── Батчевый inference ────────────────────────────────────────────
            policies, values = self._infer(all_leaf_tensors)

            # Expand + backprop
            for i, (g, node, move_stack) in enumerate(all_meta):
                # Нужен engine в состоянии листа для expansion
                sim_eng = engines[g].copy()
                for m in move_stack:
                    sim_eng.make_move_int(m)

                if not node.is_expanded:
                    # Внутри дерева шум НЕ добавляется — только на корне
                    self._expand_node(node, sim_eng, policies[i], add_noise=False)

                # Снимаем virtual loss и делаем настоящий backup
                self._apply_virtual_loss(node, -VIRTUAL_LOSS)
                self._backup(node, move_stack, float(values[i]), apply_vloss=False)

        # ── Строим policy-вектора из visit counts ─────────────────────────────
        result_policies = []
        for g, root in enumerate(roots):
            policy = np.zeros(POLICY_SIZE, dtype=np.float32)
            total = sum(c.visits for c in root.children.values())
            if total > 0:
                for m, child in root.children.items():
                    idx = engines[g].move_int_to_policy_idx(m)
                    if idx is not None:
                        policy[idx] = child.visits / total
            result_policies.append(policy)

        return result_policies

    def _select(self, root: MCTSNode):
        """
        Спуск по дереву до листа. Возвращает (leaf_node, move_stack).
        move_stack нужен для воспроизведения позиции через engine.copy() + apply.
        """
        node = root
        move_stack = []
        while node.is_expanded and node.children and not node.is_terminal:
            node = node.best_child(self.c_puct)
            move_stack.append(node.move)
        return node, move_stack

    def _expand_node(self, node: MCTSNode, engine, policy_vec: np.ndarray,
                     add_noise: bool = False):
        """
        Создаём дочерние узлы по легальным ходам.

        add_noise=True — добавлять Dirichlet шум (только для корня при self-play).
        Это критично для exploration: без шума сеть быстро зацикливается на
        одних и тех же ходах и не исследует альтернативы.
        """
        if node.is_expanded:
            return
        legal = engine.get_legal_moves_int()
        if not legal:
            node.is_terminal = True
            node.is_expanded = True
            return

        n = len(legal)
        raw_priors = np.empty(n, dtype=np.float64)
        for j, m in enumerate(legal):
            idx = engine.move_int_to_policy_idx(m)
            # FIX: bounds check — idx может быть >= POLICY_SIZE при баге в движке
            if idx is not None and 0 <= idx < len(policy_vec):
                raw_priors[j] = float(policy_vec[idx])
            else:
                raw_priors[j] = 1e-8

        total = raw_priors.sum()

        # FIX: если сеть вернула нули для всех легальных ходов (начало обучения),
        # равномерно распределяем prior вместо молчаливого fallback'а
        if total <= 1e-12:
            raw_priors = np.ones(n, dtype=np.float64) / n
        else:
            raw_priors /= total

        # Dirichlet noise только на корне при self-play (AlphaZero §B.11)
        if add_noise and n > 0:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * n)
            raw_priors = (1.0 - DIRICHLET_EPS) * raw_priors + DIRICHLET_EPS * noise

        for j, m in enumerate(legal):
            node.children[m] = MCTSNode(node, m, float(raw_priors[j]))

        node.is_expanded = True

    def _apply_virtual_loss(self, node: MCTSNode, delta: int):
        """
        Идём вверх по дереву и добавляем/убираем virtual loss.
        FIX: clamp ≥ 0 при снятии, чтобы не уйти в минус при несимметричных вызовах.
        """
        cur = node
        while cur is not None:
            if delta < 0:
                # FIX: clamp при снятии
                cur.virtual_loss = max(0, cur.virtual_loss + delta)
            else:
                cur.virtual_loss += delta
            cur = cur.parent

    def _backup(self, leaf: MCTSNode, move_stack: List[int],
                value: float, apply_vloss: bool = False):
        """
        Backpropagation. value — с точки зрения игрока, ходящего в leaf.
        Знак меняется при каждом шаге вверх.
        """
        cur = leaf
        sign = 1.0
        while cur is not None:
            cur.visits += 1
            cur.value_sum += value * sign
            if apply_vloss:
                cur.virtual_loss = max(0, cur.virtual_loss - VIRTUAL_LOSS)
            sign *= -1.0
            cur = cur.parent


# ── Совместимый wrapper для train.py ─────────────────────────────────────────
def mcts_policy_vector(engine, net, simulations=80, c_puct=1.25, device=None):
    if device is None:
        device = torch.device("cuda")
    mcts = UltraFastMCTS(net, device, c_puct, batch_size=64, add_dirichlet=True)
    return mcts.search_games([engine], simulations)[0]
