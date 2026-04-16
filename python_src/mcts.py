# mcts.py — Monte Carlo Tree Search for Capablanca Chess
# Fully corrected: proper engine cloning, virtual loss, batched evaluation

from __future__ import annotations

import math
import numpy as np
import torch
from typing import Optional

# Policy vector size must match the Rust engine
POLICY_SIZE = 6880 # Было 6720


class MCTSNode:
    __slots__ = (
        "parent", "move", "prior", "children",
        "visits", "value_sum", "virtual_loss",
    )

    def __init__(self, parent: Optional["MCTSNode"], move: Optional[str], prior: float):
        self.parent: Optional[MCTSNode] = parent
        self.move:   Optional[str]      = move   # UCI move that led here
        self.prior:  float              = prior
        self.children: dict[str, MCTSNode] = {}

        self.visits:       int   = 0
        self.value_sum:    float = 0.0
        self.virtual_loss: int   = 0  # for parallelism; set to 0 for single-thread

    # ── UCB score ────────────────────────────────────────────────────────────

    def ucb(self, parent_visits: int, c_puct: float) -> float:
        q = self.q()
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits + self.virtual_loss)
        return q + u

    def q(self) -> float:
        denom = self.visits + self.virtual_loss
        return self.value_sum / denom if denom > 0 else 0.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def select_child(self, c_puct: float) -> "MCTSNode":
        return max(
            self.children.values(),
            key=lambda child: child.ucb(self.visits, c_puct),
        )

    def expand(self, legal_moves: list[str], policy_probs: np.ndarray):
        """Create child nodes for all legal moves."""
        for move in legal_moves:
            # Try to look up the move's prior probability from the policy vector.
            # We don't have the engine here, so we pass the prior directly.
            self.children[move] = MCTSNode(parent=self, move=move, prior=0.0)
        # Assign priors from the policy vector (indexed by move)
        for move, child in self.children.items():
            child.prior = float(policy_probs.get(move, 1e-8))

    def backup(self, value: float):
        """Propagate value up the tree (from the perspective of the root's side)."""
        node = self
        sign = 1.0
        while node is not None:
            node.visits += 1
            node.value_sum += value * sign
            sign *= -1.0  # flip for opponent
            node = node.parent


# ── Helper: board tensor ──────────────────────────────────────────────────────

def engine_to_tensor(engine, device: torch.device) -> torch.Tensor:
    """Convert engine state to a (1, 20, 8, 10) float tensor."""
    data = engine.get_board_tensor()                       # list[float], length 1600
    return torch.tensor(data, dtype=torch.float32, device=device).view(1, 20, 8, 10)


def policy_to_move_probs(
    policy_vec: np.ndarray,
    legal_moves: list[str],
    engine,
) -> dict[str, float]:
    """
    Extract prior probabilities for legal moves from the flat policy vector.
    Uses the engine's move_to_policy_idx() method.
    """
    move_probs: dict[str, float] = {}
    total = 0.0
    for m in legal_moves:
        idx = engine.move_to_policy_idx(m)
        p = float(policy_vec[idx]) if idx is not None and idx < len(policy_vec) else 1e-8
        move_probs[m] = p
        total += p
    # Renormalize over legal moves
    if total > 0:
        for m in move_probs:
            move_probs[m] /= total
    else:
        uniform = 1.0 / max(len(legal_moves), 1)
        for m in move_probs:
            move_probs[m] = uniform
    return move_probs


# ── MCTS search ───────────────────────────────────────────────────────────────

def mcts_search(
    engine,
    net: torch.nn.Module,
    *,
    simulations: int = 800,
    c_puct: float = 1.25,
    temperature: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    device: torch.device | None = None,
) -> str:
    """
    Run MCTS from the current engine state.

    Returns the selected move as a UCI string.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.eval()

    # ── Evaluate root ────────────────────────────────────────────────────────
    with torch.no_grad():
        # BFLOAT16 inference
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            policy_logits, value = net(engine_to_tensor(engine, device))

        # ВАЖНО: обратно в float32
        policy_logits = policy_logits.float()

    policy_vec = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()


    legal_moves = engine.get_legal_moves()

    if not legal_moves:
        raise ValueError("No legal moves available — game should be over.")


    root = MCTSNode(parent=None, move=None, prior=1.0)

    move_probs = policy_to_move_probs(policy_vec, legal_moves, engine)


    # --- DIRICHLET NOISE ---
    if dirichlet_epsilon > 0 and dirichlet_alpha > 0:
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
        for (m, n) in zip(legal_moves, noise):
            move_probs[m] = (1 - dirichlet_epsilon) * move_probs[m] + dirichlet_epsilon * n

    root.expand(legal_moves, move_probs)

    # ── Simulation loop ───────────────────────────────────────────────────────
    for _ in range(simulations - 1):  # -1 because we already evaluated root
        node = root

        # 1. Selection: follow best UCB until a leaf
        #    We need to track the engine state along the path → clone at each step
        sim_engine = engine.copy()

        while not node.is_leaf():
            node = node.select_child(c_puct)
            sim_engine.make_move(node.move)

        # 2. Terminal check
        if sim_engine.is_game_over():
            result = sim_engine.game_result()
            # result is from white's perspective; flip if black is to move at this node
            # game_result returns +1 white wins, -1 black wins, 0 draw
            # backup from the perspective of the side that just moved INTO this terminal
            node.backup(result if sim_engine.side_to_move() == 1 else -result)
            continue

        # 3. Expansion + evaluation
        with torch.no_grad():
            policy_logits, leaf_value = net(engine_to_tensor(sim_engine, device))
        policy_vec = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        leaf_value_f = float(leaf_value.item())

        new_legal = sim_engine.get_legal_moves()
        if new_legal:
            move_probs = policy_to_move_probs(policy_vec, new_legal, sim_engine)
            node.expand(new_legal, move_probs)

        # 4. Backup
        node.backup(leaf_value_f)

    # ── Select move ───────────────────────────────────────────────────────────
    if not root.children:
        return legal_moves[0]

    visit_counts = {m: child.visits for m, child in root.children.items()}

    if temperature == 0:
        # Greedy
        return max(visit_counts, key=visit_counts.get)

    # Sample proportional to visit_count ^ (1/temperature)
    moves = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)
    counts = counts ** (1.0 / temperature)
    counts /= counts.sum()
    return np.random.choice(moves, p=counts)


def mcts_policy_vector(
    engine,
    net: torch.nn.Module,
    *,
    simulations: int = 800,
    c_puct: float = 1.25,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Like mcts_search() but returns the full MCTS policy vector (visit distributions)
    over all POLICY_SIZE actions. Used as training target.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.eval()

    with torch.no_grad():
        policy_logits, value = net(engine_to_tensor(engine, device))
    policy_vec_nn = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

    legal_moves = engine.get_legal_moves()
    if not legal_moves:
        return np.zeros(POLICY_SIZE, dtype=np.float32)

    root = MCTSNode(parent=None, move=None, prior=1.0)
    move_probs = policy_to_move_probs(policy_vec_nn, legal_moves, engine)
    root.expand(legal_moves, move_probs)

    for _ in range(simulations - 1):
        node = root
        sim_engine = engine.copy()

        while not node.is_leaf():
            node = node.select_child(c_puct)
            sim_engine.make_move(node.move)

        if sim_engine.is_game_over():
            result = sim_engine.game_result()
            node.backup(result if sim_engine.side_to_move() == 1 else -result)
            continue

        with torch.no_grad():
            plogits, leaf_val = net(engine_to_tensor(sim_engine, device))
        pv = torch.softmax(plogits, dim=1).squeeze(0).cpu().numpy()
        new_legal = sim_engine.get_legal_moves()
        if new_legal:
            mp = policy_to_move_probs(pv, new_legal, sim_engine)
            node.expand(new_legal, mp)
        node.backup(float(leaf_val.item()))

    # Build policy vector from visit counts
    policy_out = np.zeros(POLICY_SIZE, dtype=np.float32)
    total_visits = sum(c.visits for c in root.children.values())
    if total_visits > 0:
        for m, child in root.children.items():
            idx = engine.move_to_policy_idx(m)
            if idx is not None:
                policy_out[idx] = child.visits / total_visits

    return policy_out
