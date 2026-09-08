# train.py — AlphaZero training loop for Capablanca Chess.
#
# One iteration = self-play (Rust MCTS + GPU inference) → replay buffer →
# supervised steps on sampled batches → checkpoint. Optional extras, all
# opt-in from the CLI: Fairy-Stockfish curriculum, a lagged-checkpoint
# opponent, distillation from a teacher buffer.
#
# Everything the loop needs lives in `Config`; `train(cfg)` is importable and
# does not read argparse state.

import os
import time
import pickle
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch's intra-op pool defaults to half the core count and busy-waits
# between GPU calls. The CPU side of this workload is tree descent in Rust plus
# sparse-policy assembly, not matrix work, so those threads buy nothing: measured
# throughput flat across every thread count tried, while CPU use goes
# up several times over. Freeing the cores also stops a co-running eval or GUI from
# costing throughput (measured -30% under contention).
# See docs/experiments/cpu_threads.md. Override with OMP_NUM_THREADS.
if "OMP_NUM_THREADS" not in os.environ:
    torch.set_num_threads(2)
from dataclasses import dataclass
from typing import List, Optional, Tuple


class ModelEMA:
    """Exponential Moving Average of model weights.

    AlphaZero uses an EMA copy for self-play instead of the current weights.
    Without this, fresh weights may overfit to the last batch,
    self-play produces inconsistent data → unstable training.

    decay=0.999 → weights updated by 0.1% per train step.
    """
    def __init__(self, model, decay=0.999):
        import torch
        self.decay = decay
        src = model._orig_mod if hasattr(model, '_orig_mod') else model
        self.shadow = {k: v.clone().detach() for k, v in src.state_dict().items()}

    def update(self, model):
        import torch
        src = model._orig_mod if hasattr(model, '_orig_mod') else model
        with torch.no_grad():
            for k, v in src.state_dict().items():
                if k not in self.shadow: continue
                if self.shadow[k].dtype.is_floating_point:
                    self.shadow[k].lerp_(v.detach().to(self.shadow[k].dtype), 1.0 - self.decay)
                else:
                    self.shadow[k].copy_(v.detach())

    def apply_to(self, model):
        src = model._orig_mod if hasattr(model, '_orig_mod') else model
        src.load_state_dict(self.shadow, strict=False)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        for k in self.shadow:
            if k in sd and sd[k].shape == self.shadow[k].shape:
                self.shadow[k].copy_(sd[k])


class LaggedOpponentPool:
    """Stores model weight snapshots from past iterations as weaker opponents.

    Creates a natural curriculum between Random and FSF-1:
      iter 0-4  → Random only (pool is empty)
      iter 5+   → lagged iter 0 (~=Random+ with some training)
      iter 10+  → lagged iter 5 (~= noticeably weaker than current, but not Random)
    Older snapshots are weighted higher — a weaker opponent is more useful.
    """

    def __init__(self, max_snapshots: int = 5):
        self.max_snapshots = max_snapshots
        self.snapshots: List[Tuple[int, dict]] = []  # (iter, cpu_state_dict)

    def maybe_save(self, net, iteration: int, interval: int):
        if interval <= 0 or iteration % interval != 0:
            return
        src = net._orig_mod if hasattr(net, '_orig_mod') else net
        sd = {k: v.clone().cpu() for k, v in src.state_dict().items()}
        self.snapshots.append((iteration, sd))
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        print(f"  📸 Lagged pool: iter {iteration} сохранён ({len(self.snapshots)}/{self.max_snapshots})")

    def sample(self):
        """Returns (iteration, state_dict), weighting older snapshots more heavily."""
        if not self.snapshots:
            return None
        n = len(self.snapshots)
        weights = np.array([n - i for i in range(n)], dtype=np.float64)
        weights /= weights.sum()
        idx = int(np.random.choice(n, p=weights))
        return self.snapshots[idx]

    def __len__(self):
        return len(self.snapshots)


# ─────────────────────────────────────────────────────────────────────────────
# Fairy-Stockfish integration (optional, activated via --fsf-path)
# ─────────────────────────────────────────────────────────────────────────────

def get_fsf_schedule(iteration: int, base_self_play: int):
    """Returns (self_games, fsf_games_this_iter, fsf_every) for the given iteration.

    Schedule:
      iter 0-29:  1152 self + 2688 FSF (every 3 iterations ~269 games)
      iter 30-49: 2304 self + 1152 FSF (every 5 iterations ~288 games)
      iter 50+:   base_self_play + 0 FSF
    """
    if iteration < 30:
        return 1152, (269 if iteration % 3 == 0 else 0), 3
    elif iteration < 50:
        return 2304, (288 if iteration % 5 == 0 else 0), 5
    else:
        return base_self_play, 0, 0


_PROMO_CHARS = {2: 'n', 3: 'b', 4: 'r', 5: 'q', 6: 'a', 7: 'c'}

def _int_to_uci(m: int) -> str:
    p_val = m & 0b111
    t = (m >> 3) & 0x7F
    f = (m >> 10) & 0x7F
    uci = f"{chr(ord('a') + f%10)}{f//10+1}{chr(ord('a') + t%10)}{t//10+1}"
    if p_val in _PROMO_CHARS:
        uci += _PROMO_CHARS[p_val]
    return uci


def _uci_to_int(uci: str, engine):
    """Resolve a UCI string against the engine's legal moves. None if illegal."""
    for m in engine.get_legal_moves_int():
        if _int_to_uci(m) == uci:
            return m
    return None


class FairyStockfishWrapper:
    """UCI wrapper around Fairy-Stockfish for the Capablanca variant."""
    def __init__(self, path: str, skill: int = None, elo: int = None):
        """skill / elo — настоящие ручки силы движка.

        `go nodes N` силу почти не снижает: даже при N=1 движок отвечает
        статической оценкой с qsearch и обыгрывает всё, что мы обучали, всухую
        (0 из 64 при 96% против Random). Ступеней между Random и FSF там нет.
        `Skill Level` (-20..20) и `UCI_LimitStrength`+`UCI_Elo` (500..2850) дают
        нормальную лестницу. См. docs/experiments/fsf_ladder.md.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fairy-Stockfish не найден: {path}")
        self.proc = subprocess.Popen(
            [path], universal_newlines=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1,
        )
        self._send("uci");          self._wait("uciok")
        self._send("setoption name UCI_Variant value capablanca")
        if skill is not None:
            self._send(f"setoption name Skill Level value {int(skill)}")
        if elo is not None:
            self._send("setoption name UCI_LimitStrength value true")
            self._send(f"setoption name UCI_Elo value {int(elo)}")
        self._send("isready");      self._wait("readyok")

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n"); self.proc.stdin.flush()

    def _wait(self, target: str) -> str:
        while True:
            raw = self.proc.stdout.readline()
            if not raw:  # EOF — FSF process died; don't spin forever
                raise RuntimeError("Fairy-Stockfish process closed unexpectedly")
            line = raw.strip()
            if target in line: return line

    def best_move(self, uci_history, nodes: int) -> tuple:
        """Returns (move_uci, score_cp) where score_cp is from side-to-move perspective.
        score_cp > 0 means the side to move (FSF) is winning."""
        moves = " ".join(uci_history) if uci_history else ""
        self._send("position startpos" + (f" moves {moves}" if moves else ""))
        self._send(f"go nodes {nodes}")
        score_cp = 0
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("info"):
                parts = line.split()
                if "score" in parts:
                    si = parts.index("score")
                    if si + 2 < len(parts):
                        if parts[si + 1] == "cp":
                            try: score_cp = int(parts[si + 2])
                            except ValueError: pass
                        elif parts[si + 1] == "mate":
                            try: score_cp = 9999 if int(parts[si + 2]) > 0 else -9999
                            except ValueError: pass
            elif line.startswith("bestmove"):
                return line.split()[1], score_cp

    def close(self):
        try: self._send("quit"); self.proc.wait(timeout=3)
        except: self.proc.kill()


def generate_fsf_games(net, device, cfg, num_games: int, fsf_path: str,
                       fsf_nodes: int, mcts_sims: int = 100):
    """Generates num_games games against an external opponent.

    fsf_nodes == 0 : random mover — first curriculum level.
                     Opponent positions are NOT saved. Value = game result.
    fsf_nodes >= 1 : Fairy-Stockfish. Opponent positions are NOT saved.
                     Value = fsf_value_alpha * fsf_eval + (1-alpha) * game_result
                     where fsf_eval = -tanh(score_cp/400) after each NN move.
                     This gives a dense, position-specific value signal
                     instead of a single delayed game result for the whole game.

    Games run in batches with tree reuse. Fairy-Stockfish is a single serial
    UCI process, so opponent plies are still queried game by game — but the NN
    plies (the expensive half) now go to the GPU as one batch, and the search
    tree survives from move to move instead of being rebuilt every ply.
    All games in a batch share `nn_side` so the whole batch has the same side
    to move each ply; the side alternates between batches.
    """
    fsf_value_alpha = getattr(cfg, 'fsf_value_alpha', 0.7)
    fsf_noise_prob  = getattr(cfg, 'fsf_noise_prob', 0.0)
    use_random = (fsf_nodes == 0)
    fsf = None
    if not use_random:
        try:
            fsf = FairyStockfishWrapper(fsf_path,
                                        skill=getattr(cfg, 'fsf_skill', None),
                                        elo=getattr(cfg, 'fsf_elo', None))
        except Exception as e:
            print(f"  ❌ FSF: {e}")
            return [], 0, 0, 0

    # Contempt is most useful here — vs an external opponent the NN benefits
    # from biasing away from draws to convert positional advantage into wins.
    # >=2 batches so `nn_side = b % 2` still splits the colours ~50/50;
    # one giant batch would play every game as the same side.
    batch_sz = max(1, min(cfg.mcts_batch, (num_games + 1) // 2))
    mcts = UltraFastMCTS(net, device, c_puct=1.745, batch_size=batch_sz,
                         add_dirichlet=False, parallel_sims=cfg.mcts_parallel_sims,
                         contempt=cfg.contempt,
                         rep_search_perslot=cfg.rep_search_perslot)

    all_samples = []
    wins = draws = losses = errors = 0
    nn_wins = nn_draws = nn_losses = 0
    num_batches = (num_games + batch_sz - 1) // batch_sz

    for b in range(num_batches):
        n = min(batch_sz, num_games - b * batch_sz)
        nn_side = b % 2
        engines = [CapablancaEngine() for _ in range(n)]
        tree = mcts.new_tree(engines)

        # NN positions only: (board, sparse_pol, side, ply, root_q, root_d, [fsf_eval])
        # fsf_eval is filled when the opponent replies (its eval of the position
        # right after our move); score_cp > 0 means the mover (FSF) is winning,
        # so the NN's view is -tanh(cp/400).
        positions = [[] for _ in range(n)]
        uci_history = [[] for _ in range(n)]
        game_plies = [0] * n
        adjudicated = [None] * n
        broken = [False] * n
        active = list(range(n))
        move_num = 0

        while active and move_num < cfg.max_game_length:
            is_nn = (move_num % 2) == nn_side
            new_active = []

            if is_nn:
                mcts.run_search(tree, mcts_sims)
                sparse_pols = tree.get_policies_sparse()
                vals = np.asarray(tree.get_values(), dtype=np.float32)
                draw_ps = np.asarray(tree.get_draws(), dtype=np.float32)

            for g in active:
                eng = engines[g]
                legal = eng.get_legal_moves_int()
                if not legal:
                    tree.set_game_finished(g)
                    continue

                if is_nn:
                    side = eng.side_to_move()
                    pol_idx, pol_val = sparse_pols[g]
                    pol_lookup = {int(i): float(v) for i, v in zip(pol_idx, pol_val)}
                    positions[g].append([
                        np.asarray(eng.get_board_tensor(), dtype=np.float32),
                        (np.asarray(pol_idx, dtype=np.int16),
                         np.asarray(pol_val, dtype=np.float16)),
                        side, move_num,
                        float(vals[g]) if g < len(vals) else None,
                        float(draw_ps[g]) if g < len(draw_ps) else None,
                        None,
                    ])
                    move = _sample_move_from_policy(eng, pol_lookup, legal)
                elif use_random:
                    move = int(np.random.choice(legal))
                else:
                    # fsf_noise_prob > 0: FSF occasionally plays at random →
                    # a softened opponent. No eval recorded for a random move.
                    if fsf_noise_prob > 0.0 and np.random.random() < fsf_noise_prob:
                        move = int(np.random.choice(legal))
                    else:
                        uci, score_cp = fsf.best_move(uci_history[g], nodes=fsf_nodes)
                        if uci == "(none)":
                            broken[g] = True
                            tree.set_game_finished(g)
                            continue
                        move = _uci_to_int(uci, eng)
                        if move is None:
                            errors += 1
                            broken[g] = True
                            tree.set_game_finished(g)
                            continue
                        if positions[g] and positions[g][-1][6] is None:
                            positions[g][-1][6] = -float(np.tanh(score_cp / 400.0))

                eng.make_move_int(move)
                uci_history[g].append(_int_to_uci(move))
                tree.make_move(g, move)
                game_plies[g] += 1
                if eng.is_game_over():
                    tree.set_game_finished(g)
                    continue
                adj = eng.adjudication_result() if cfg.adjudicate else None
                if adj is not None:
                    adjudicated[g] = adj
                    tree.set_game_finished(g)
                else:
                    new_active.append(g)

            active = new_active
            move_num += 1

        for g, eng in enumerate(engines):
            if broken[g]:
                continue
            result = _finish_result(cfg, eng, adjudicated[g])
            if result > 0.5:    wins   += 1
            elif result < -0.5: losses += 1
            else:               draws  += 1
            nn_result = result if nn_side == 0 else -result
            if nn_result > 0.5:    nn_wins   += 1
            elif nn_result < -0.5: nn_losses += 1
            else:                  nn_draws  += 1

            total_plies = game_plies[g]
            for board_np, pol_sparse, side, ply, root_q, root_d, fsf_eval in positions[g]:
                z = result if side == 0 else -result
                if not use_random and fsf_eval is not None:
                    # Mix FSF's dense positional eval with the game result.
                    z = fsf_value_alpha * fsf_eval + (1.0 - fsf_value_alpha) * z
                v, draw_target = _blend_value(cfg, z, root_q, root_d, result)
                remaining = max(0, total_plies - 1 - ply)
                mlh_norm = min(1.0, remaining / MLH_PLY_NORM)
                all_samples.append(pack_sample_sparse(
                    board_np, pol_sparse, float(v), float(mlh_norm),
                    -1, float(draw_target)))

    if fsf is not None:
        fsf.close()
    opp_label = "Random" if use_random else f"FSF-{fsf_nodes}"
    print(f"  {opp_label}: {num_games-errors} партий | бел={wins} чёрн={losses} ничьи={draws} "
          f"ошибки={errors} | NN: +{nn_wins}/={nn_draws}/-{nn_losses} "
          f"| {len(all_samples)} позиций")
    return all_samples, nn_wins, nn_draws, nn_losses


def _sample_move_from_policy(eng, pol_lookup, legal, tau: float = 0.8):
    """Pick a move from a sparse root policy with temperature `tau`."""
    raw = np.array([pol_lookup.get(eng.move_int_to_policy_idx(m), 0.0)
                    for m in legal], dtype=np.float64)
    if not np.isfinite(raw).all() or raw.max(initial=0.0) <= 0.0:
        return int(np.random.choice(legal))
    raw = np.power(np.maximum(raw, 1e-8), 1.0 / tau)
    s = raw.sum()
    probs = raw / s if s > 0 else np.ones(len(legal)) / len(legal)
    return int(np.random.choice(legal, p=probs))


def _blend_value(cfg, z: float, root_q, root_d, result: float):
    """KataGo/LC0 value target: (1-w)*z + w*root_Q, plus the WDL draw axis.

    Returns (value, draw_target) where draw_target < 0 means "reconstruct the
    WDL from the value" (legacy). Same formula generate_games uses — the
    opponent generators used to emit pure-z targets, so half the curriculum
    data missed the variance reduction entirely.
    """
    if cfg.value_q_weight <= 0.0 or root_q is None:
        return z, -1.0
    w = cfg.value_q_weight
    v = (1.0 - w) * z + w * float(root_q)
    if root_d is None:
        return v, -1.0
    z_is_draw = 1.0 if abs(result) < 1e-6 else 0.0
    return v, (1.0 - w) * z_is_draw + w * float(root_d)


def _finish_result(cfg, eng, adjudicated):
    if adjudicated is not None:
        return float(adjudicated)
    if eng.is_game_over():
        return float(eng.game_result())
    return 0.0 if cfg.timeout_as_draw else float(eng.material_result())


def generate_lagged_games(net, lagged_sd: dict, cfg, device: "torch.device",
                           num_games: int, mcts_sims: int = 50):
    """Plays the current model against an old checkpoint (lagged_sd).

    Current network (NN): with Dirichlet, full exploratory mode.
    Lagged network (OPP): without Dirichlet, deterministic mode.
    Only positions played by the CURRENT network are saved.

    Games run in batches with tree reuse, exactly like `generate_games`: both
    sides are networks, so there is nothing serial to wait on. The old
    one-game-at-a-time / fresh-tree-every-ply loop measured ~4.7x slower per
    game (37.8 vs 8.0 ms per half-move on 256ch×15+4tb at 100 sims) and threw
    the whole sub-tree away on every move.

    All games inside one batch share the same `nn_side` so the whole batch has
    the same side to move on every ply and one GPU call serves all of them;
    the side alternates between batches to keep colours balanced.

    Returns: (samples, cur_wins, cur_draws, cur_losses)
    """
    from model import CapablancaNet

    lagged_net = CapablancaNet(
        cfg.num_channels, cfg.num_res_blocks,
        enable_mlh=cfg.enable_mlh,
        num_transformer_blocks=cfg.num_transformer_blocks,
        transformer_heads=cfg.transformer_heads,
        enable_future=cfg.enable_future,
        qkv_bias=cfg.qkv_bias,
        use_rmsnorm=cfg.use_rmsnorm,
        piece_embed_dim=cfg.piece_embed_dim,
        qk_norm=cfg.qk_norm,
        swiglu=cfg.swiglu,
        attn_policy=cfg.attn_policy,
        ffn_mult=cfg.ffn_mult,
        restricted_policy=cfg.restricted_policy,
        abs_pos_embed=cfg.abs_pos_embed,
        wide_value=cfg.wide_value,
        num_registers=cfg.num_registers,
        value_residual=cfg.value_residual,
        hyper_streams=cfg.hyper_streams,
    ).to(device)
    lagged_net.load_state_dict(lagged_sd, strict=False)
    lagged_net.eval()

    # >=2 batches so `nn_side = b % 2` still splits the colours ~50/50;
    # one giant batch would play every game as the same side.
    batch_sz = max(1, min(cfg.mcts_batch, (num_games + 1) // 2))
    mcts_cur = UltraFastMCTS(net, device, c_puct=1.745, batch_size=batch_sz,
                             add_dirichlet=True, parallel_sims=cfg.mcts_parallel_sims,
                             rep_search_perslot=cfg.rep_search_perslot)
    mcts_lag = UltraFastMCTS(lagged_net, device, c_puct=1.745, batch_size=batch_sz,
                             add_dirichlet=False, parallel_sims=cfg.mcts_parallel_sims,
                             rep_search_perslot=cfg.rep_search_perslot)

    all_samples = []
    cur_wins = cur_draws = cur_losses = 0
    num_batches = (num_games + batch_sz - 1) // batch_sz

    for b in range(num_batches):
        n = min(batch_sz, num_games - b * batch_sz)
        nn_side = b % 2                      # whole batch plays the same colour
        engines = [CapablancaEngine() for _ in range(n)]
        # Two trees over the same games: one per network. Both get every move
        # applied so their roots stay on the real game position.
        tree_cur = mcts_cur.new_tree(engines)
        tree_lag = mcts_lag.new_tree(engines)

        positions = [[] for _ in range(n)]   # (board, sparse_pol, side, ply, q, d)
        game_plies = [0] * n                 # per-game length; batches end ragged
        adjudicated = [None] * n
        active = list(range(n))
        move_num = 0

        while active and move_num < cfg.max_game_length:
            is_current = (move_num % 2) == nn_side
            searcher = mcts_cur if is_current else mcts_lag
            tree = tree_cur if is_current else tree_lag
            searcher.run_search(tree, mcts_sims)
            sparse_pols = tree.get_policies_sparse()
            vals = np.asarray(tree.get_values(), dtype=np.float32)
            draws = np.asarray(tree.get_draws(), dtype=np.float32)

            new_active = []
            for g in active:
                eng = engines[g]
                legal = eng.get_legal_moves_int()
                if not legal:
                    tree_cur.set_game_finished(g); tree_lag.set_game_finished(g)
                    continue
                side = eng.side_to_move()
                pol_idx, pol_val = sparse_pols[g]
                pol_lookup = {int(i): float(v) for i, v in zip(pol_idx, pol_val)}

                if is_current:
                    positions[g].append((
                        np.asarray(eng.get_board_tensor(), dtype=np.float32),
                        (np.asarray(pol_idx, dtype=np.int16),
                         np.asarray(pol_val, dtype=np.float16)),
                        side, move_num,
                        float(vals[g]) if g < len(vals) else None,
                        float(draws[g]) if g < len(draws) else None,
                    ))

                move = _sample_move_from_policy(eng, pol_lookup, legal)
                eng.make_move_int(move)
                game_plies[g] += 1
                tree_cur.make_move(g, move)   # keep BOTH trees on the real position
                tree_lag.make_move(g, move)
                if eng.is_game_over():
                    tree_cur.set_game_finished(g); tree_lag.set_game_finished(g)
                    continue
                adj = eng.adjudication_result() if cfg.adjudicate else None
                if adj is not None:
                    adjudicated[g] = adj
                    tree_cur.set_game_finished(g); tree_lag.set_game_finished(g)
                else:
                    new_active.append(g)

            active = new_active
            move_num += 1

        for g, eng in enumerate(engines):
            result = _finish_result(cfg, eng, adjudicated[g])
            cur_result = result if nn_side == 0 else -result
            if cur_result > 0.5:    cur_wins   += 1
            elif cur_result < -0.5: cur_losses += 1
            else:                   cur_draws  += 1

            total_plies = game_plies[g]
            for board_np, pol_sparse, side, ply, root_q, root_d in positions[g]:
                z = result if side == 0 else -result
                v, draw_target = _blend_value(cfg, z, root_q, root_d, result)
                remaining = max(0, total_plies - 1 - ply)
                mlh_norm = min(1.0, remaining / MLH_PLY_NORM)
                all_samples.append(pack_sample_sparse(
                    board_np, pol_sparse, float(v), float(mlh_norm),
                    -1, float(draw_target)))

    del lagged_net
    total = cur_wins + cur_draws + cur_losses
    wr = (cur_wins + 0.5 * cur_draws) / total if total > 0 else 0.0
    print(f"  Lagged NN: {num_games} партий | NN: +{cur_wins}/={cur_draws}/-{cur_losses} "
          f"| wr={wr:.1%} | {len(all_samples)} позиций")
    return all_samples, cur_wins, cur_draws, cur_losses

from model import CapablancaNet
from mcts import UltraFastMCTS

try:
    from capablanca_engine import CapablancaEngine
except ImportError:
    raise ImportError("capablanca_engine not found. Build with: maturin develop --release")

# NVIDIA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
# PyTorch 2.0+: sets FP32 matmul precision = TF32 for any FP32 GEMM
# not covered by autocast (fallback paths, gradient ops).
torch.set_float32_matmul_precision('high')
# Blackwell+: allows BF16 matmul to use BF16 accumulator (instead of FP32).
# Inside autocast(bfloat16) gives ~10-15% speedup on attention/FFN. Safe with batch_norm/GroupNorm.
try:
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
except AttributeError:
    pass  # older PyTorch version

# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model
    num_channels: int = 64
    num_res_blocks: int = 5
    # LC0 BT3+ inspired: transformer blocks with RPB (Relative Position Bias) after ResNet tower.
    # Provides global "positional understanding" — connects any two squares in 1 step.
    # 2 blocks is usually sufficient. Set =0 to fall back to pure ResNet.
    num_transformer_blocks: int = 2
    transformer_heads: int = 8
    enable_mlh: bool = True
    # Future move head (LC0 BT4-inspired): predicts our move 2 half-moves ahead.
    # Auxiliary task — improves planning representations in trunk. Not used at inference.
    enable_future: bool = True

    # BT5-style trim. Defaults preserve the legacy architecture so existing
    # checkpoints keep loading unchanged. Flip via CLI when starting a fresh
    # run (or a distillation into a new architecture) to pick up the upgrade.
    qkv_bias: bool = True            # False → drop QKV bias (~5% faster inference)
    use_rmsnorm: bool = False        # True  → RMSNorm instead of LayerNorm
    piece_embed_dim: int = 0         # >0    → per-square Linear(16 → N) added to input

    # Modern-transformer stack, borrowed from LLM practice. All default-off so
    # existing checkpoints keep loading; they change the architecture, so they
    # only make sense for a fresh run or a distillation into a new net.
    qk_norm: bool = False        # RMSNorm on Q and K before the dot product
    swiglu: bool = False         # gated FFN instead of Linear→Mish→Linear
    attn_policy: bool = False    # bilinear from→to policy head instead of Linear(C*80, 7000)
    ffn_mult: int = 2                # FFN multiplier in transformer blocks
    restricted_policy: bool = False  # policy head only on the 2672 reachable indices
    abs_pos_embed: bool = False      # absolute 80xC position on the token stream
    wide_value: bool = False         # value head 32ch->512 instead of 8ch->256
    num_registers: int = 0       # extra learnable non-square tokens for the transformer
    value_residual: bool = False # ResFormer: mix layer-0 values into later layers
    hyper_streams: int = 0       # mHC: widen the residual into N streams (0 = plain residual)
    # Optimizer: "adamw" or "muon". Muon orthogonalises the momentum of 2D
    # weights (Newton-Schulz) and keeps AdamW for everything else — norms,
    # biases, and any 1D/4D tensor.
    optimizer: str = "adamw"
    # Muon's update is unit-scale by construction, so its LR lives on a different
    # scale from AdamW's (~1e-2 vs ~1e-3) and is annealed on the same cosine.
    muon_lr: float = 2e-2

    # Self-play
    simulations: int = 100

    # Playout Cap Randomization (AlphaZero, lc0):
    # On fast_sim_fraction moves use fast_simulations instead of simulations.
    # Only full searches (simulations, MORE) enter the training buffer
    # when playout_cap_train_only_full=True — standard PCR.
    fast_simulations: int = 80
    fast_sim_fraction: float = 0.75
    playout_cap_train_only_full: bool = True
    c_puct: float = 1.25
    temperature_moves: int = 50       # more exploration early in the game
    temperature: float = 1.0          # tau for first temperature_moves moves (1.0 = proportional to visit counts)
    temperature_late: float = 0.0     # tau after temperature_moves: 0.0 = hard argmax (better mating)
    games_per_iter: int = 128
    max_game_length: int = 300  # ply (LC0=450 for chess, Capablanca ~1.5x longer but be careful)
    mcts_batch: int = 128
    mcts_parallel_sims: int = 32  # leaves per MCTS step (more = fewer round-trips Python↔GPU)
    # torch.compile mode for inference net: None / 'default' / 'reduce-overhead' / 'max-autotune'.
    # None = no compilation (fast start). 'reduce-overhead' = CUDA graphs, up to 50% speedup on Blackwell.
    compile_inference: Optional[str] = None

    # KLD-early-exit (Lc0 style smart pruning).
    # Per-visit KLD gain = max KL(prev || curr) / Δsims. If for all games gain < threshold
    # AND visits >= kld_min_sims_frac * total → MCTS stops.
    # Idea: if visit distribution stopped changing, more sims yield no new information.
    # Measurements on 64-game batches: per-visit gain drops from ~1.4e-2 (step 7/13) to ~4-7e-3 (step 11/13).
    # threshold=5e-3 — compromise: saves 15-20% sims on simple positions, leaves hard ones untouched.
    # NOTE: with simulations<200 the feature rarely triggers (few snapshots between check_every).
    # 0 = disable.
    kld_threshold: float = 5e-3
    kld_check_every: int = 2             # check KL every N parallel-steps (2*32=64 sims between snapshots)
    kld_min_sims_frac: float = 0.30      # minimum 30% of full sims are mandatory
    kld_enabled: bool = True

    # Training
    batch_size: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    train_steps: int = 200
    min_train_steps: int = 20
    value_loss_weight: float = 1.0
    # Value target = (1-w)*game_outcome_z + w*MCTS_root_Q  (KataGo/LC0).
    # 0.0 = legacy pure-z (high variance). ~0.5 = KataGo-style blend: the search's
    # own evaluation of each position is a much lower-variance target than the
    # single game result, improving value calibration and sample efficiency.
    # Applied at self-play time in generate_games (the root-Q is already computed).
    value_q_weight: float = 0.0
    # Rebalance each training batch to ~33/33/33 W/D/L (sample_balanced). True =
    # legacy behavior. False = sample on the NATURAL outcome distribution (KataGo/
    # LC0 style) — avoids distorting the value-head prior and the (shared) policy
    # sampling. Worth A/B-ing together with value_q_weight, since balancing
    # partially undoes the calibration the Q-blend buys.
    value_balance: bool = True
    # Policy target: "visits" (legacy AlphaZero visit-count distribution) or
    # "gumbel" (Gumbel completed-Q improved policy, Danihelka 2022). Gumbel is a
    # lower-variance, stronger target at low sim counts — only the stored TARGET
    # changes; move selection stays on visit counts, so search/KLD/resign are
    # untouched. Default "visits" = byte-identical legacy behavior.
    policy_target_mode: str = "visits"
    # Adjudicate clearly-decided games early (material ≥8, ≥20 ply quiet, ≥15
    # moves) → saves timeout-tail compute, cleaner value targets. Default False
    # (legacy: games run to natural end / resign / timeout). Changes training data.
    adjudicate: bool = False
    # Q-gate for adjudication (lc0-style): only adjudicate if the network's own
    # root value agrees with the material verdict and is confident — guards
    # against false adjudication of fortresses / compensation where material
    # leads but the position isn't won. Require root_Q (white-POV) · adj ≥ gate.
    # 0 = material-only (no net confirmation). Free: root_Q is already computed.
    adjudicate_q_gate: float = 0.5
    # Per-slot repetition planes during search (align with training encoding).
    # Default False = legacy slot-0-only. Changes search-time NN inputs.
    rep_search_perslot: bool = False
    # LC0 MLH loss weight. Too large → MLH dominates over policy/value.
    # 0.1 — standard value in LC0.
    mlh_loss_weight: float = 0.1
    # Future move loss weight. Auxiliary — keep small so it doesn't dominate.
    future_loss_weight: float = 0.15

    # Distillation: N epochs of supervised training on loaded buffer BEFORE self-play.
    # Buffer from a strong old network = teacher data (policy = MCTS visits). A new larger
    # network quickly reaches ≈ teacher strength without expensive self-play. 0 = off.
    pretrain_epochs: int = 0
    pretrain_only: bool = False  # exit immediately after distillation (skip self-play loop)

    # Buffer
    buffer_max: int = 1_000_000
    buffer_min_to_train: int = 10_000

    # Resignation: if V < resign_threshold for
    # resign_consec consecutive moves after resign_min_move — game ends in loss.
    # Eliminates 30-40% of timeouts: network won't play 80 moves with a bare king.
    # On early iterations (< resign_warmup_iters) threshold is strict (-0.99),
    # then transitions to resign_threshold (-0.95).
    # This protects against a weak network's evaluation errors.
    resign_threshold: float = -0.95   # Q threshold (final) — fallback if no WDL
    resign_threshold_early: float = -0.99  # Q threshold on early iterations
    resign_warmup_iters: int = 30     # iterations before switching to resign_threshold
    resign_consec: int = 3            # consecutive moves below threshold
    resign_min_move: int = 20         # don't resign before this move
    # WDL-based resign (LC0-style): P(L) > resign_wdl_threshold.
    # Q-based threshold confuses "hard draw" with loss (q=-0.95 may be P(D)=0.95,
    # P(L)=0.05 — that's not a loss!). WDL separates unambiguously.
    # 0.85 = "probability of losing > 85%" → resign.
    resign_wdl_threshold: float = 0.85
    resign_wdl_early: float = 0.95    # stricter on early iterations
    # Resign playthrough (LC0 tournament.cc:388): fraction of games where resign is DISABLED.
    # Without this:
    #   1) No threshold calibration — false-positives (winning/drawn position, resigned) go undetected
    #   2) Network doesn't learn to defend in tough positions (resigns cut off data)
    # 0.10 = 10% of games played to completion, rest with resign.
    resign_playthrough: float = 0.10

    # Infrastructure
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    # Stop after this many iterations (counting from start_iter). 0 = run forever.
    # Needed for controlled experiments: two arms must do the SAME number of
    # iterations, not the same wall-clock time.
    max_iters: int = 0
    # Seed for torch/numpy. -1 = leave both alone (previous behaviour). Set it
    # for ablations: otherwise arms differ by weight init and self-play RNG on
    # top of the thing you are actually measuring.
    seed: int = -1
    save_every: int = 5
    log_every: int = 50

    # When changing --lr on restart, pass --reset-scheduler so
    # scheduler starts a new cycle rather than continuing mid-cycle
    reset_scheduler: bool = False

    # policy_loss below this threshold = collapse — checkpoint not saved
    collapse_threshold: float = 0.01

    # EMA of model weights (AlphaZero self-play stabilization)
    use_ema: bool = True
    ema_decay: float = 0.9999  # per-step: window ~10K steps ≈ 10 iterations (LC0 selfplay)
    # EMA not applied to self-play before this iteration: early EMA weights = average of
    # random weights → worse than live NN. Always updated, only used from ema_start_iter.
    ema_start_iter: int = 10
    force_save: bool = False  # if True — save checkpoint even at low loss

    # Fairy-Stockfish opponent. Empty path = pure self-play (FSF branches off).
    # These live in Config (not on the argparse namespace) so train(cfg) is
    # callable as a library, not only from __main__.
    fsf_path: Optional[str] = None
    fsf_nodes: int = 500              # node limit for FSF in the non-curriculum schedule
    fsf_mcts_sims: int = 100          # MCTS sims for our net when playing FSF
    reset_ema: bool = False           # re-seed EMA from the live net on checkpoint load

    # FSF eval as value target: mix FSF position evaluation with game result.
    # fsf_eval = -tanh(score_cp/400) — position evaluation after NN move from NN's perspective.
    # 0.7 = 70% FSF eval (dense signal) + 30% game result (long-term).
    fsf_value_alpha: float = 0.7

    # Curriculum training: FSF as adaptive teacher
    # --curriculum --fsf-path ./binary --fsf-nodes-start 1 --fsf-nodes-max 10000
    curriculum_mode: bool = False
    fsf_nodes_current: int = 1        # current FSF level (auto-adapts)
    curriculum_nodes_min: int = 0  # 0 = Random mover (lower bound)
    curriculum_nodes_max: int = 10000
    curriculum_self_play_ratio: float = 0.0   # 0.0 = FSF only, 0.2 = 20% self-play
    curriculum_promote_threshold: float = 0.55  # avg winrate above → increase nodes
    curriculum_demote_threshold: float = 0.35   # avg winrate below → decrease nodes
    curriculum_window: int = 3        # iterations for winrate averaging

    # Lagged opponent: play against checkpoint from N iterations ago
    # Fills the gap between Random and FSF-1, no external engine required.
    # lag_opponent_interval=5 → save a weight snapshot every 5 iterations.
    # lag_opponent_ratio=0.3 → 30% of self-play games vs lagged, 70% vs current.
    lag_opponent_interval: int = 0    # 0 = disabled
    lag_opponent_ratio: float = 0.0   # fraction of self-play vs lagged model
    lag_opponent_pool_size: int = 5   # how many snapshots to keep
    lag_opponent_sims: int = 50       # MCTS sims for lagged model

    # Softened FSF: FSF occasionally makes random moves → intermediate difficulty
    # 0.0 = deterministic FSF, 0.4 = 40% random moves
    fsf_noise_prob: float = 0.0

    # On timeout (game reached max_game_length): True = draw, False = material evaluation
    timeout_as_draw: bool = False

    # Contempt: PUCT Q-bias on the network's draw probability.
    # 0.0 = standard play. +N → MCTS avoids draws (useful vs weaker opponents
    # like FSF), -N → accepts draws. In pure self-play both sides apply it
    # symmetrically — keep at 0.0 to avoid skewing value targets.
    contempt: float = 0.0


# Tuple: (board_f16, sparse_policy, value, mlh_norm)
# mlh_norm = remaining_plies / MLH_PLY_NORM ∈ [0, 1], 0.0 for old samples without MLH
# (board, (pol_idx, pol_val), value, mlh_norm, future_idx, draw_target)
# draw_target = -1.0 → WDL reconstructed from value (legacy); ≥0 → full-WDL blend.
CompactSample = Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], float, float, int, float]
Sample = CompactSample
MLH_PLY_NORM = 200.0   # must match CapablancaNet.MLH_PLY_NORM


def pack_sample(board: np.ndarray, policy: np.ndarray, value: float,
                mlh_norm: float = 0.0, future_idx: int = -1,
                draw: float = -1.0) -> CompactSample:
    """future_idx: policy index of our move 2 half-moves ahead (-1 = unknown/none).
    draw: explicit WDL draw target ∈[0,1] for the full-WDL Q-blend; -1 = none
    (WDL reconstructed from value)."""
    board_f16 = board.astype(np.float16)
    nz = np.nonzero(policy)[0]
    pol_idx = nz.astype(np.int16)
    pol_val = policy[nz].astype(np.float16)
    return (board_f16, (pol_idx, pol_val), np.float32(value),
            np.float32(mlh_norm), np.int32(future_idx), np.float32(draw))


def pack_sample_sparse(board: np.ndarray, pol_sparse: Tuple[np.ndarray, np.ndarray],
                       value: float, mlh_norm: float = 0.0,
                       future_idx: int = -1, draw: float = -1.0) -> CompactSample:
    """Pack a sample when MCTS already returned sparse visit probabilities.
    draw: explicit WDL draw target ∈[0,1] (full-WDL Q-blend); -1 = none."""
    board_f16 = board.astype(np.float16)
    pol_idx, pol_val = pol_sparse
    return (
        board_f16,
        (pol_idx.astype(np.int16, copy=False),
         pol_val.astype(np.float16, copy=False)),
        np.float32(value),
        np.float32(mlh_norm),
        np.int32(future_idx),
        np.float32(draw),
    )


def unpack_policy(pol_sparse: Tuple[np.ndarray, np.ndarray],
                  size: int = 7000) -> np.ndarray:
    pol = np.zeros(size, dtype=np.float32)
    idx, val = pol_sparse
    pol[idx.astype(np.int32)] = val.astype(np.float32)
    return pol


def gumbel_improved_policy(idxs: np.ndarray, priors: np.ndarray,
                           visits: np.ndarray, qs: np.ndarray, root_v: float,
                           maxvisit_init: float = 50.0, value_scale: float = 0.1
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Gumbel completed-Q improved policy target (Danihelka et al. 2022).

    Matches DeepMind mctx `qtransform_completed_by_mix_value`:
      completedQ(a) = Q(a) if N(a)>0 else v_mix
      v_mix = (v_root + ΣN · Σ_vis P·Q / Σ_vis P) / (1 + ΣN)
      normQ = (completedQ - lo) / (hi - lo)   # lo/hi over completedQ ∪ {v_mix}
      σ(a)  = (maxvisit_init + max_N) · value_scale · normQ
      π'(a) = softmax( log P(a) + σ(a) )

    The min-max normalize + value_scale=0.1 are essential: without them σ scales
    with the raw Q magnitude and saturates the target to one-hot. A lower-variance,
    stronger target than raw visit counts, especially at low sim counts. All inputs
    are side-to-move POV (Q = -child.q() from Rust, root_v from get_values).
    Falls back to the prior when there is no search signal. Returns (idxs, probs).
    """
    idxs = np.asarray(idxs, dtype=np.int64)
    priors = np.asarray(priors, dtype=np.float64)
    visits = np.asarray(visits, dtype=np.float64)
    qs = np.asarray(qs, dtype=np.float64)
    n = idxs.shape[0]
    sum_n = float(visits.sum())
    if n == 0 or sum_n <= 0.0:
        s = priors.sum()
        probs = (priors / s) if s > 0 else np.full(n, 1.0 / max(n, 1))
        return idxs, probs.astype(np.float32)

    vis_mask = visits > 0
    sum_pi_vis = float(priors[vis_mask].sum())
    if sum_pi_vis > 1e-9:
        weighted_q = float((priors[vis_mask] * qs[vis_mask]).sum()) / sum_pi_vis
    else:
        weighted_q = float(root_v)
    v_mix = (float(root_v) + sum_n * weighted_q) / (1.0 + sum_n)

    completed_q = np.where(vis_mask, qs, v_mix)
    # Min-max rescale to [0,1], range taken over completedQ and v_mix (mctx).
    lo = min(v_mix, float(completed_q.min()))
    hi = max(v_mix, float(completed_q.max()))
    norm_q = (completed_q - lo) / max(hi - lo, 1e-8)

    max_n = float(visits.max())
    sigma = (maxvisit_init + max_n) * value_scale * norm_q
    logits = np.log(np.maximum(priors, 1e-9)) + sigma
    logits -= logits.max()  # softmax stabilization
    e = np.exp(logits)
    probs = e / e.sum()
    return idxs, probs.astype(np.float32)


# ── Optimizer ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def _newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of G (Muon, Jordan et al. 2024).

    Runs a quintic Newton-Schulz iteration on the normalised matrix. The
    coefficients are the ones tuned in the reference implementation: they do not
    converge to an exact orthogonalisation, they push the singular values into a
    band around 1 fast, which is all the update needs. Runs in bfloat16 — the
    iteration is numerically forgiving and this is per-step overhead on every 2D
    weight, so precision here is not worth the bandwidth.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum-orthogonalised SGD for 2D weight matrices.

    Standard momentum takes a step along a matrix whose singular values can be
    wildly uneven, so a few directions dominate the update. Muon replaces the
    momentum buffer with its orthogonal factor before stepping, which equalises
    them. It only applies to genuinely matrix-shaped parameters — norms, biases
    and anything 1D or 4D belong in AdamW, and `build_param_groups` splits them.

    lr is not comparable to AdamW's: the update is unit-scale by construction,
    so this wants something in the 1e-2 range where AdamW wants 1e-3.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        super().__init__(list(params), dict(lr=lr, momentum=momentum,
                                            nesterov=nesterov, ns_steps=ns_steps,
                                            weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            mom, nesterov = group["momentum"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.lerp_(g, 1.0 - mom)
                upd = g.lerp_(buf, mom) if nesterov else buf
                upd = _newton_schulz_orthogonalize(upd, group["ns_steps"])
                # Scale by the aspect ratio so wide and tall matrices move by a
                # comparable amount (reference implementation's rule).
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if group["weight_decay"] > 0.0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(upd, alpha=-group["lr"] * scale)
        return loss


class _MultiOptimizer:
    """Two optimizers behind one optimizer-shaped surface.

    `param_groups` is the concatenation, which is what makes the LR scheduler
    work unchanged: CosineAnnealingWarmRestarts records each group's own
    `base_lr` and scales it, so Muon's much larger LR is annealed on the same
    cosine without being overwritten by AdamW's value.
    """

    def __init__(self, *opts):
        self.opts = [o for o in opts if o is not None]

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return {"multi": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        if "multi" not in sd:
            raise ValueError("checkpoint holds a single-optimizer state")
        for o, s in zip(self.opts, sd["multi"]):
            o.load_state_dict(s)


def build_optimizer(net, cfg, device):
    """AdamW, or Muon on the 2D weights + AdamW on everything else."""
    fused = (device.type == "cuda")   # fused AdamW is CUDA-only
    if cfg.optimizer != "muon":
        return torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay, fused=fused)
    matrices, others = [], []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        # 2D and matrix-shaped: Linear weights, the RPB table, piece embeddings.
        # Conv kernels are 4D and stay with AdamW — orthogonalising a reshaped
        # conv is not what the method is about.
        (matrices if p.ndim == 2 and min(p.shape) > 1 else others).append(p)
    adamw = torch.optim.AdamW(others, lr=cfg.learning_rate,
                              weight_decay=cfg.weight_decay, fused=fused)
    muon = Muon(matrices, lr=cfg.muon_lr, weight_decay=cfg.weight_decay)
    print(f"   Оптимайзер:    Muon на {len(matrices)} матрицах "
          f"({sum(p.numel() for p in matrices)/1e6:.1f}M) + AdamW на "
          f"{len(others)} прочих ({sum(p.numel() for p in others)/1e6:.1f}M)")
    return _MultiOptimizer(adamw, muon)


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data: List[Sample] = []
        # Parallel float32 array for fast stratification by value.
        # Updated incrementally in push() — O(1) per element, not O(N) at sampling.
        self._val_arr = np.zeros(max_size, dtype=np.float32)
        self._ptr = 0
        self._full = False

    def push(self, samples: List[Sample]):
        for s in samples:
            self._ptr = self._ptr % self.max_size  # guard against loading with different max_size
            if not self._full:
                self.data.append(s)
                if len(self.data) == self.max_size:
                    self._full = True
            else:
                self.data[self._ptr] = s
            self._val_arr[self._ptr] = float(s[2])
            self._ptr = (self._ptr + 1) % self.max_size

    def rebuild_val_arr(self):
        """Rebuild _val_arr from data after loading. Cap to max_size: a buffer
        file saved with a larger --buffer-max would otherwise index past _val_arr
        (and break the ring-buffer invariant)."""
        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size:]
            self._ptr = 0
            self._full = True
        n = min(len(self.data), self.max_size)
        for i in range(n):
            self._val_arr[i] = float(self.data[i][2])

    def save_npz(self, path: str):
        """Save the buffer as a numpy archive. ~5-10× faster than pickle on 1M
        positions and produces a smaller file (float16 storage). The expensive
        bit is the one-time np.stack across 1M boards; after that np.savez
        writes contiguous blocks via a single OS write per array.

        Layout: boards (N,planes,H,W) f16, sparse policy packed into a
        rectangular (N, K) pair of (indices, values) with -1 / 0 padding,
        plus three scalar arrays for value / mlh / future_idx and a tiny
        meta vector with [ptr, full]. Saved atomically via a .tmp file
        rename — a crash mid-write can't leave a half-written buffer."""
        if not self.data:
            return
        n = len(self.data)
        # Stack boards in one shot; numpy iterates through the list internally
        # — still O(N) but in C, ~10× faster than pickle on the same data.
        boards = np.stack([s[0] for s in self.data]).astype(np.float16, copy=False)
        # Sparse policy is ragged (one entry per legal move). Pad to the max
        # length seen in this buffer so the result is a single dense array.
        pol_idxs = [s[1][0] for s in self.data]
        pol_vals = [s[1][1] for s in self.data]
        max_k = max((len(x) for x in pol_idxs), default=0)
        pol_idx_arr = np.full((n, max_k), -1, dtype=np.int16)
        pol_val_arr = np.zeros((n, max_k), dtype=np.float16)
        for i, (idxs, vals) in enumerate(zip(pol_idxs, pol_vals)):
            k = len(idxs)
            if k:
                pol_idx_arr[i, :k] = idxs
                pol_val_arr[i, :k] = vals
        values  = np.fromiter((float(s[2]) for s in self.data),
                              dtype=np.float16, count=n)
        mlhs    = np.fromiter((float(s[3]) if len(s) > 3 else 0.0 for s in self.data),
                              dtype=np.float16, count=n)
        futures = np.fromiter((int(s[4]) if len(s) > 4 else -1 for s in self.data),
                              dtype=np.int32, count=n)
        # Full-WDL Q-blend draw target (-1 = none → reconstruct from value).
        draws = np.fromiter((float(s[5]) if len(s) > 5 else -1.0 for s in self.data),
                            dtype=np.float16, count=n)
        meta = np.array([self._ptr, int(self._full)], dtype=np.int64)

        # np.savez auto-appends `.npz` to the path it gets. Pass a base name
        # without extension and rename the real file to the target path.
        tmp_base = path + ".tmp"
        np.savez(tmp_base, boards=boards,
                 pol_idx=pol_idx_arr, pol_val=pol_val_arr,
                 values=values, mlhs=mlhs, futures=futures, draws=draws, meta=meta)
        os.replace(tmp_base + ".npz", path)

    def load_npz(self, path: str):
        """Reverse of save_npz — rebuilds the list-of-tuples representation."""
        z = np.load(path)
        boards   = z["boards"]
        pol_idx  = z["pol_idx"]
        pol_val  = z["pol_val"]
        values   = z["values"]
        mlhs     = z["mlhs"]
        futures  = z["futures"]
        # draws: optional (added for full-WDL Q-blend). Old buffers lack it → -1.
        draws    = z["draws"] if "draws" in z.files else None
        meta     = z["meta"]
        self._ptr  = int(meta[0]) % self.max_size
        self._full = bool(meta[1])
        n = boards.shape[0]
        self.data = []
        for i in range(n):
            mask = pol_idx[i] >= 0
            sp_idx = pol_idx[i][mask].astype(np.int16, copy=False)
            sp_val = pol_val[i][mask].astype(np.float16, copy=False)
            self.data.append((
                boards[i],
                (sp_idx, sp_val),
                float(values[i]),
                float(mlhs[i]),
                int(futures[i]),
                float(draws[i]) if draws is not None else -1.0,
            ))
        self.rebuild_val_arr()

    def sample(self, batch_size: int) -> List[Sample]:
        n = len(self.data)
        if n == 0:
            return []
        indices = np.random.choice(n, batch_size, replace=True)
        return [self.data[i] for i in indices]

    def sample_balanced(self, batch_size: int) -> List[Sample]:
        """Sample with win/draw/loss balancing (approximately 33/33/33).

        Uses _val_arr for O(N) vector search without a Python loop over data.
        If one class is missing — balances over available classes.
        Overfit protection: if a bin is small (rare class), cap duplicates —
        each element at most ~3 times per batch. Otherwise 10 wins out of 50K positions
        would get replicated thousands of times per epoch.
        """
        n = len(self.data)
        if n == 0:
            return []
        vals = self._val_arr[:n]
        win_idx  = np.where(vals > 0.15)[0]
        draw_idx = np.where((vals >= -0.15) & (vals <= 0.15))[0]
        loss_idx = np.where(vals < -0.15)[0]

        bins = [b for b in [win_idx, draw_idx, loss_idx] if len(b) > 0]
        if len(bins) < 2:
            return self.sample(batch_size)

        # If the smallest bin is too small — balancing is useless, fall back to plain sample
        min_bin = min(len(b) for b in bins)
        if min_bin < 50:
            return self.sample(batch_size)

        per_bin = batch_size // len(bins)
        result = []
        for b in bins:
            # Cap: no more than len(b)*3 duplicates from one bin
            take = min(per_bin, len(b) * 3)
            local_idx = np.random.randint(0, len(b), take)
            result.extend([self.data[int(b[i])] for i in local_idx])
        # Top up from the large bin (usually draw) if small bins were capped
        big_bin = max(bins, key=len)
        while len(result) < batch_size:
            result.append(self.data[int(big_bin[np.random.randint(len(big_bin))])])
        np.random.shuffle(result)
        return result[:batch_size]

    def __len__(self):
        return len(self.data)


# ── Policy diversity diagnostics ──────────────────────────────────────────────

def policy_diversity_stats(samples: List[Sample], n: int = 200) -> dict:
    """
    Computes diversity metrics over a random sample.
    entropy_mean  — mean policy entropy (normal ~1.5-4.0, collapse < 0.3)
    top1_mean     — mean probability of best move (collapse > 0.95)
    nonzero_mean  — mean number of non-zero moves
    value_std     — standard deviation of value (collapse < 0.05)
    """
    if not samples:
        return {}
    idx = np.random.choice(len(samples), min(n, len(samples)), replace=False)
    entropies, top1s, nonzeros, values = [], [], [], []
    for i in idx:
        pol = unpack_policy(samples[i][1])
        pol_nz = pol[pol > 0]
        if len(pol_nz) > 0:
            ent = float(-np.sum(pol_nz * np.log(pol_nz + 1e-12)))
            entropies.append(ent)
            top1s.append(float(pol_nz.max()))
            nonzeros.append(len(pol_nz))
        values.append(float(samples[i][2]))
    return {
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "top1_mean":    float(np.mean(top1s))     if top1s    else 0.0,
        "nonzero_mean": float(np.mean(nonzeros))  if nonzeros else 0.0,
        "value_std":    float(np.std(values))     if values   else 0.0,
        "value_mean":   float(np.mean(values))    if values   else 0.0,
    }


def print_diversity(stats: dict, prefix: str = "  Diversity"):
    if not stats:
        return
    warn = ""
    if stats.get("value_std", 1.0) < 0.05:
        warn += " ⚠️ value_std критически мало!"
    if stats.get("entropy_mean", 1.0) < 0.3:
        warn += " ⚠️ entropy критически мала!"
    if stats.get("top1_mean", 0.0) > 0.95:
        warn += " ⚠️ top1 почти 1 — policy схлопнулась!"
    print(f"{prefix}: entropy={stats['entropy_mean']:.3f}  "
          f"top1={stats['top1_mean']:.3f}  "
          f"nonzero={stats['nonzero_mean']:.1f}  "
          f"value_std={stats['value_std']:.3f}  "
          f"value_mean={stats['value_mean']:.3f}{warn}")


# ── Self-play ─────────────────────────────────────────────────────────────────

def generate_games(net: nn.Module, cfg: Config, device: torch.device, iteration: int = 0) -> List[Sample]:
    kld_thr = cfg.kld_threshold if cfg.kld_enabled else 0.0
    mcts = UltraFastMCTS(net, device, cfg.c_puct, batch_size=cfg.mcts_batch,
                         parallel_sims=cfg.mcts_parallel_sims,
                         compile_mode=cfg.compile_inference,
                         kld_threshold=kld_thr,
                         kld_check_every=cfg.kld_check_every,
                         kld_min_sims_frac=cfg.kld_min_sims_frac,
                         contempt=cfg.contempt)
    all_samples: List[Sample] = []

    batch_sz = cfg.mcts_batch
    num_batches = (cfg.games_per_iter + batch_sz - 1) // batch_sz

    for b in range(num_batches):
        start = b * batch_sz
        n = min(batch_sz, cfg.games_per_iter - start)
        engines = [CapablancaEngine() for _ in range(n)]
        histories: List[List] = [[] for _ in range(n)]
        # Per-side counters: v alternates sign ply-to-ply
        # (root from side-to-move perspective), a shared counter would reset every other half-move.
        resign_counts = [[0, 0] for _ in range(n)]
        resigned = [False] * n
        # LC0 resign playthrough: with probability resign_playthrough play WITHOUT resign
        # (to calibrate the threshold and gather data about "tough positions").
        enable_resign = [np.random.random() >= cfg.resign_playthrough for _ in range(n)]

        active = list(range(n))
        move_num = 0
        adjudicated = [None] * n

        # Tree reuse: one RustMCTS for the entire game batch. After each move
        # make_move(game_idx, move) shifts the root to the chosen child and
        # keeps the sub-tree — 2-3x better quality at the same inference cost.
        rust_mcts_reuse = mcts.new_tree(engines)

        while active and move_num < cfg.max_game_length:
            # Playout Cap Randomization: on fast_sim_fraction moves use fast_simulations
            # Decision applied to the whole batch simultaneously (shared MCTS object)
            use_full_search = np.random.random() >= cfg.fast_sim_fraction
            current_sims = cfg.simulations if use_full_search else cfg.fast_simulations

            # collect → infer → apply (+ KLD early exit) lives in UltraFastMCTS
            # so self-play, eval and game_stats all drive the same loop.
            mcts.run_search(rust_mcts_reuse, current_sims)
            sparse_pols = rust_mcts_reuse.get_policies_sparse()
            raw_vals  = rust_mcts_reuse.get_values()
            raw_draws = rust_mcts_reuse.get_draws()
            # Gumbel completed-Q policy target: fetch per-child root stats. Only
            # the STORED target uses it; move selection below stays on visit counts.
            gumbel = cfg.policy_target_mode == "gumbel"
            child_stats = rust_mcts_reuse.get_root_children_stats() if gumbel else None
            # get_policies_sparse()/get_values() return one entry per EACH game
            # in rust_mcts_reuse.games (length = n, not len(active)).
            # Index by game_idx; otherwise after the first game in the batch finishes
            # all remaining games get wrong policy/value.
            values_np = np.array(raw_vals,  dtype=np.float32)
            draws_np  = np.array(raw_draws, dtype=np.float32)

            new_active = []
            for j, game_idx in enumerate(active):
                eng = engines[game_idx]
                legal = eng.get_legal_moves_int()
                if not legal:
                    rust_mcts_reuse.set_game_finished(game_idx)
                    continue

                side = eng.side_to_move()
                pol_idx_raw, pol_val_raw = sparse_pols[game_idx]
                pol_lookup = {
                    int(idx): float(val)
                    for idx, val in zip(pol_idx_raw, pol_val_raw)
                }
                root_v_raw = float(values_np[game_idx]) if game_idx < len(values_np) else 0.0
                root_d_raw = float(draws_np[game_idx]) if game_idx < len(draws_np) else 0.0
                keep_position = (
                    (not cfg.playout_cap_train_only_full) or use_full_search
                )
                if keep_position:
                    # Board encoding is expensive. For PCR fast-search moves the
                    # sample is discarded later, so do not compute/store it.
                    board_np = np.asarray(eng.get_board_tensor(), dtype=np.float32)
                    if gumbel and game_idx < len(child_stats):
                        c_idx, c_pri, c_vis, c_q = child_stats[game_idx]
                        g_idx, g_val = gumbel_improved_policy(
                            c_idx, c_pri, c_vis, c_q, root_v_raw)
                        pol_sparse = (
                            np.asarray(g_idx, dtype=np.int16),
                            np.asarray(g_val, dtype=np.float16),
                        )
                    else:
                        pol_sparse = (
                            np.asarray(pol_idx_raw, dtype=np.int16),
                            np.asarray(pol_val_raw, dtype=np.float16),
                        )
                else:
                    board_np = None
                    pol_sparse = None
                # idx 5 (move_idx) — policy index of selected move, patched below
                # after sampling. idx 6 (root_d_raw) — search draw prob for the
                # full-WDL Q-blend. List (not tuple) so move_idx can be mutated.
                histories[game_idx].append(
                    [board_np, pol_sparse, side, root_v_raw, use_full_search, -1,
                     root_d_raw])

                # Temperature decay (argmax branch below catches tau ≈ 0)
                if move_num < cfg.temperature_moves:
                    tau = cfg.temperature
                elif move_num < cfg.temperature_moves + 20:
                    progress = (move_num - cfg.temperature_moves) / 20.0
                    tau = cfg.temperature * (1 - progress) + cfg.temperature_late * progress
                else:
                    tau = cfg.temperature_late

                raw = np.array([
                    pol_lookup.get(eng.move_int_to_policy_idx(m), 0.0)
                    for m in legal
                ], dtype=np.float64)

                if tau < 0.01:
                    # tau≈0 = hard argmax (avoid 1/0 = inf → NaN)
                    if np.isfinite(raw).all() and raw.max(initial=0.0) > 0.0:
                        move = int(legal[int(np.argmax(raw))])
                    else:
                        move = int(np.random.choice(legal))
                else:
                    if np.isfinite(raw).all() and raw.max(initial=0.0) > 0.0:
                        raw = np.power(np.maximum(raw, 1e-8), 1.0 / tau)
                        s = raw.sum()
                        probs = raw / s if s > 0 else np.ones(len(legal)) / len(legal)
                    else:
                        probs = np.ones(len(legal)) / len(legal)
                    move = int(np.random.choice(legal, p=probs))

                # Record canonical policy index of the selected move in history —
                # this is the future target for the future head of neighboring positions.
                _mpidx = eng.move_int_to_policy_idx(move)
                histories[game_idx][-1][5] = _mpidx if _mpidx is not None else -1

                eng.make_move_int(move)
                rust_mcts_reuse.make_move(game_idx, move)  # tree reuse

                if eng.is_game_over():
                    rust_mcts_reuse.set_game_finished(game_idx)
                    continue

                # Resign: WDL-based (LC0-style), with playthrough probability.
                # P(L) = (1 - Q - D) / 2 — exact probability of losing.
                # enable_resign[g]=False → play to completion (for calibration + data).
                if move_num >= cfg.resign_min_move and enable_resign[game_idx]:
                    q = float(values_np[game_idx]) if game_idx < len(values_np) else 0.0
                    d = float(draws_np[game_idx])  if game_idx < len(draws_np)  else 0.0
                    p_loss = max(0.0, min(1.0, (1.0 - q - d) / 2.0))
                    _wdl_thr = (cfg.resign_wdl_early
                                if iteration < cfg.resign_warmup_iters
                                else cfg.resign_wdl_threshold)
                    _q_thr = (cfg.resign_threshold_early
                              if iteration < cfg.resign_warmup_iters
                              else cfg.resign_threshold)
                    # Triggers if EITHER P(L) is high OR Q is low (old checkpoints have D=0)
                    should_resign = p_loss > _wdl_thr or q < _q_thr
                    if should_resign:
                        resign_counts[game_idx][side] += 1
                    else:
                        resign_counts[game_idx][side] = 0

                    if resign_counts[game_idx][side] >= cfg.resign_consec:
                        resigned[game_idx] = True
                        # Rust cannot see a resignation — tell it, or the dead
                        # game keeps eating leaves out of every GPU batch.
                        rust_mcts_reuse.set_game_finished(game_idx)
                        continue

                # Adjudication:
                # If after the move there is a decisive material advantage
                # (≥8 points, ≥10 moves without captures, ≥15 full moves) —
                # end the game without waiting for checkmate or timeout.
                adj = eng.adjudication_result() if cfg.adjudicate else None
                if adj is not None and cfg.adjudicate_q_gate > 0.0:
                    # lc0-style net confirmation: root_v_raw is the side-to-move
                    # (mover) POV value of the pre-move position; convert to white
                    # POV and require it to agree with the material verdict (adj is
                    # white POV ±1). Rejects fortress/compensation false positives.
                    root_v_white = root_v_raw if side == 0 else -root_v_raw
                    if root_v_white * adj < cfg.adjudicate_q_gate:
                        adj = None
                if adj is not None:
                    adjudicated[game_idx] = adj
                    rust_mcts_reuse.set_game_finished(game_idx)
                else:
                    new_active.append(game_idx)

            active = new_active
            move_num += 1

        batch_positions = 0
        # Game categories. Win/loss counts split BY OUTCOME TYPE so a colour
        # imbalance isn't hidden inside the resign/timeout buckets — that used
        # to look like "black wins more" when in reality most black wins came
        # from white resigning. resigns/timeouts are still also reported as
        # totals for backward-compat sanity.
        mate_w = mate_b = draws = adjudications = 0
        resign_w = resign_b = 0     # split by who actually won
        timeout_w = timeout_b = timeout_d = 0

        for i, eng in enumerate(engines):
            if resigned[i]:
                # Resignation: the side that moved last lost
                # Determine who resigned from side_to_move (opponent to move → previous side resigned)
                last_side = histories[i][-1][2] if histories[i] else 0
                result = -1.0 if last_side == 0 else 1.0
                if result > 0: resign_w += 1
                else:          resign_b += 1
            elif adjudicated[i] is not None:
                # Adjudication — decisive material advantage
                result = adjudicated[i]
                adjudications += 1
                if result > 0: mate_w += 1
                else:          mate_b += 1
            elif eng.is_game_over():
                result = eng.game_result()
                if result == 1.0:   mate_w += 1
                elif result == -1.0: mate_b += 1
                else:                draws += 1
            else:
                result = 0.0 if cfg.timeout_as_draw else eng.material_result()
                if result > 0.5:    timeout_w += 1
                elif result < -0.5: timeout_b += 1
                else:                timeout_d += 1

            total_plies = len(histories[i])
            for k, entry in enumerate(histories[i]):
                board_np, pol_sparse, side = entry[0], entry[1], entry[2]
                # Playout cap: skip fast-search positions during training
                if cfg.playout_cap_train_only_full and len(entry) > 4 and not entry[4]:
                    continue
                if board_np is None or pol_sparse is None:
                    continue
                v = result if side == 0 else -result
                draw_target = -1.0   # sentinel → WDL reconstructed from v (legacy)
                # Value target = blend of game outcome z and the MCTS root value
                # (KataGo/LC0). Pure z is high-variance: one late blunder flips the
                # label on every earlier position. The search already evaluated THIS
                # position (entry[3]=root_Q, entry[6]=root_D, side-to-move POV, same
                # sign as z), a far lower-variance estimate. Full WDL: blend the
                # win-loss axis (Q) AND the draw axis (D). value_q_weight=0 → pure-z.
                if cfg.value_q_weight > 0.0:
                    root_q = entry[3]
                    root_d = entry[6] if len(entry) > 6 else None
                    if root_q is not None:
                        w = cfg.value_q_weight
                        v = (1.0 - w) * v + w * float(root_q)
                        if root_d is not None:
                            z_is_draw = 1.0 if abs(result) < 1e-6 else 0.0
                            draw_target = (1.0 - w) * z_is_draw + w * float(root_d)
                # MLH target: how many half-moves REMAIN from this position to game end.
                # Normalized to [0, 1] by dividing by MLH_PLY_NORM.
                # (On timeout the final position is unknown → use the game "tail" as-is.)
                remaining = max(0, total_plies - 1 - k)
                mlh_norm = min(1.0, remaining / MLH_PLY_NORM)
                # Future move target: move at k+2 (our next move — same side,
                # same canonical policy-index orientation). -1 if game ended.
                future_idx = histories[i][k + 2][5] if k + 2 < total_plies else -1
                all_samples.append(
                    pack_sample_sparse(board_np, pol_sparse, float(v),
                                       float(mlh_norm), int(future_idx),
                                       float(draw_target)))
                batch_positions += 1

        # Aggregate totals per colour across all outcome types. Adjudications
        # are already counted inside mate_w/mate_b above, so they aren't added.
        total_w = mate_w + resign_w + timeout_w
        total_b = mate_b + resign_b + timeout_b
        total_d = draws + timeout_d
        total_counted = total_w + total_b + total_d
        sanity = "" if total_counted == n else f" ⚠️ sanity {total_counted}/{n}"
        print(f"  Batch {b+1}/{num_batches}: {n} games, "
              f"{batch_positions} positions, {move_num} ходов | "
              f"W={total_w} (мат {mate_w}, resign {resign_w}, timeout {timeout_w}) · "
              f"B={total_b} (мат {mate_b}, resign {resign_b}, timeout {timeout_b}) · "
              f"D={total_d} (пат {draws}, timeout {timeout_d}){sanity}")

    # KLD-early-exit statistics
    if cfg.kld_enabled and cfg.kld_threshold > 0.0:
        _ks = mcts.kld_stats()
        if _ks['calls'] > 0:
            print(f"  KLD: exit_rate={_ks['exit_rate']*100:.0f}% "
                  f"savings={_ks['savings']*100:.0f}% (за {_ks['calls']} MCTS вызовов)")

    return all_samples


# ── Dataset ───────────────────────────────────────────────────────────────────

def value_to_wdl(v: float) -> np.ndarray:
    """
    Convert v∈[-1,1] → [P(Win), P(Draw), P(Loss)].

    Linear mapping that EXACTLY preserves Q = P(Win) - P(Loss) = v:
      p_win = max(0, v), p_loss = max(0, -v), p_draw = 1 - |v|
      v=+1.0 → [1.0, 0.0, 0.0]
      v=+0.5 → [0.5, 0.5, 0.0]   (Q recovers to exactly 0.5)
      v= 0.0 → [0.0, 1.0, 0.0]
    The old sqrt variant gave v=0.5 → [0.71, 0.29, 0] → Q=0.71 ≠ 0.5: the WDL head
    was trained on an inconsistent target. For integer game outcomes
    (v ∈ {-1,0,+1}, pure self-play) both variants are identical.
    """
    v = float(np.clip(v, -1.0, 1.0))
    p_win  = max(0.0, v)
    p_loss = max(0.0, -v)
    p_draw = max(0.0, 1.0 - abs(v))
    return np.array([p_win, p_draw, p_loss], dtype=np.float32)


def value_draw_to_wdl(v: float, d: float) -> np.ndarray:
    """Build [W, D, L] from an explicit value v=W−L and draw prob d (the search's
    own D). Used by the full-WDL Q-blend: keeps the search's draw estimate instead
    of reconstructing it as 1−|v| (value_to_wdl). W=(1−d+v)/2, L=(1−d−v)/2.
    Falls back to value_to_wdl when d<0 (sentinel = no explicit draw)."""
    if d < 0.0:
        return value_to_wdl(v)
    v = float(np.clip(v, -1.0, 1.0))
    d = float(np.clip(d, 0.0, 1.0))
    p_win  = max(0.0, (1.0 - d + v) / 2.0)
    p_loss = max(0.0, (1.0 - d - v) / 2.0)
    p_draw = max(0.0, d)
    s = p_win + p_draw + p_loss
    if s <= 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([p_win / s, p_draw / s, p_loss / s], dtype=np.float32)


def _collate_batch(samples: List[CompactSample]):
    """Pack a small list of samples into batched tensors.

    Called once per training step on a freshly drawn 256-ish batch — no
    pre-stacking of the entire training run. Used to be a SelfPlayDataset
    that pre-stacked `train_steps * batch_size` samples up front; on the
    1M buffer that pre-stack peaked at ~70 GB RAM and silently dropped
    the process into swap. The lazy variant keeps RAM at a few MB."""
    boards_np = np.stack([s[0].astype(np.float32) for s in samples]).reshape(
        -1, CapablancaNet.INPUT_PLANES, CapablancaNet.BOARD_H, CapablancaNet.BOARD_W
    )
    policies_np = np.stack([unpack_policy(s[1]) for s in samples])
    # s[5] = explicit draw target (full-WDL Q-blend). Absent / <0 → reconstruct
    # via value_to_wdl (legacy one-hot / scalar path).
    wdl_np = np.stack([
        value_draw_to_wdl(float(s[2]), float(s[5])) if len(s) > 5
        else value_to_wdl(float(s[2]))
        for s in samples
    ])
    mlh_np = np.array(
        [float(s[3]) if len(s) > 3 else 0.0 for s in samples],
        dtype=np.float32,
    )
    future_np = np.array(
        [int(s[4]) if len(s) > 4 else -1 for s in samples],
        dtype=np.int64,
    )
    return (
        torch.from_numpy(boards_np),
        torch.from_numpy(policies_np),
        torch.from_numpy(wdl_np),
        torch.from_numpy(mlh_np),
        torch.from_numpy(future_np),
    )


# ── Training step ─────────────────────────────────────────────────────────────

def train_epoch(net: nn.Module, optimizer: torch.optim.Optimizer,
                buffer: ReplayBuffer, cfg: Config, device: torch.device,
                iteration: int, ema=None):
    """ema: ModelEMA or None. If given — updated AFTER EACH step (LC0/AlphaZero).
    Previously EMA was updated once per iteration → with decay=0.999 and 900 steps/iter
    by switchover (iter 10) EMA ≈ 99% random weights → self-play played "blind",
    all games timed out, resigns never triggered.
    """
    net.train()

    # With lazy sampling we can do as many steps as requested — each step draws
    # an independent random batch from the buffer. The "1 epoch" cap that used
    # to apply when we pre-stacked everything is no longer meaningful here.
    effective_steps = max(cfg.min_train_steps, cfg.train_steps)
    # Give a heads-up if requested steps × batch_size exceeds the buffer many
    # times over — that's a sign of likely overfit on a small buffer.
    coverage = (effective_steps * cfg.batch_size) / max(1, len(buffer))
    if coverage > 1.0:
        print(f"  ℹ️  Буфер {len(buffer):,} поз × ~{coverage:.1f} проходов "
              f"({effective_steps} шагов × batch {cfg.batch_size})")

    # Lazy per-step sampling: at each training step draw a fresh balanced batch
    # of batch_size samples and collate to GPU tensors. With a 1M buffer and
    # 4000 steps this keeps the RAM footprint at ~10 MB per step instead of
    # the ~70 GB pre-stack that used to push the process into swap.
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_mlh_loss = 0.0
    total_future_loss = 0.0
    steps = 0

    for step_idx in range(effective_steps):
        samples = (buffer.sample_balanced(cfg.batch_size) if cfg.value_balance
                   else buffer.sample(cfg.batch_size))
        boards, policies, values, mlh_targets, future_targets = _collate_batch(samples)

        # NCHW: the net runs NCHW (channels_last is slower here — see model build
        # site above), so feed plain contiguous input.
        boards = boards.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)  # WDL: (batch, 3)
        mlh_targets = mlh_targets.to(device, non_blocking=True)  # (batch,) in [0,1]
        future_targets = future_targets.to(device, non_blocking=True)  # (batch,) int64

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, wdl_logits, mlh_raw, future_logits = net(boards)
            logits     = logits.float()
            wdl_logits = wdl_logits.float()

            # Policy: cross-entropy with visit counts as soft labels
            log_probs   = F.log_softmax(logits, dim=1)
            policy_loss = -(policies * log_probs).sum(dim=1).mean()

            # WDL: cross-entropy with soft one-hot [Win, Draw, Loss]
            log_wdl    = F.log_softmax(wdl_logits, dim=1)
            value_loss = -(values * log_wdl).sum(dim=1).mean()

            # MLH: MSE between sigmoid(predict) and target ∈ [0, 1]
            # Weight 0.1 (LC0 calibration) — small influence so it doesn't dominate.
            if mlh_raw is not None:
                mlh_pred = torch.sigmoid(mlh_raw.float().squeeze(-1))
                mlh_loss = F.mse_loss(mlh_pred, mlh_targets)
            else:
                mlh_loss = torch.zeros((), device=device)

            # Future move: cross-entropy with hard target (policy idx of move k+2).
            # Mask samples with future_idx < 0 (game end / old samples without target).
            if future_logits is not None:
                fmask = future_targets >= 0
                if fmask.any():
                    future_loss = F.cross_entropy(
                        future_logits.float()[fmask], future_targets[fmask]
                    )
                else:
                    future_loss = torch.zeros((), device=device)
            else:
                future_loss = torch.zeros((), device=device)

            loss = (policy_loss
                    + cfg.value_loss_weight * value_loss
                    + cfg.mlh_loss_weight * mlh_loss
                    + cfg.future_loss_weight * future_loss)

        # bfloat16 has the same exponent range as fp32 — GradScaler is not needed
        # and dangerous (the continue below would break scaler.update() state machine).
        loss.backward()
        # Gradient-explosion guard. clip_grad_norm_ already walks every parameter
        # in C++ and returns the global L2 grad norm; if any gradient is NaN/Inf
        # that norm is non-finite. Checking the returned scalar is ONE device→host
        # sync. The old per-parameter `.any()` loop forced one sync per tensor —
        # hundreds of syncs per step on an 84M-param net, a real throughput tax.
        total_norm = nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if not torch.isfinite(total_norm):
            print("  ⚠️  NaN/Inf в градиентах — пропускаем шаг")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        # EMA per-step (LC0/AlphaZero-style). With decay=0.999 and ~900 steps/iter
        # by end of iteration 10 EMA will have 9000 updates → decay^9000 ≈ 1e-4 →
        # virtually matches the current net (model is already "warmed up").
        if ema is not None:
            ema.update(net)

        # Accumulate on-GPU (detached) — no per-step sync. The running averages
        # are only materialised to the host every `log_every` steps (below) and
        # once at the end, instead of 5 `.item()` syncs on every single step.
        total_loss += loss.detach()
        total_policy_loss += policy_loss.detach()
        total_value_loss += value_loss.detach()
        total_mlh_loss += mlh_loss.detach()
        total_future_loss += future_loss.detach()
        steps += 1

        if steps % cfg.log_every == 0:
            avg_p = float(total_policy_loss) / steps
            avg_v = float(total_value_loss) / steps
            avg_m = float(total_mlh_loss) / steps
            avg_f = float(total_future_loss) / steps
            avg_t = float(total_loss) / steps
            print(f"    step {steps:4d}/{effective_steps} | "
                  f"policy_loss={avg_p:.4f}  value_loss={avg_v:.4f}  "
                  f"mlh_loss={avg_m:.4f}  future_loss={avg_f:.4f}  total={avg_t:.4f}")

    n = max(steps, 1)
    # float() materialises the on-GPU accumulators to the host exactly once here.
    return {
        "loss": float(total_loss) / n,
        "policy_loss": float(total_policy_loss) / n,
        "value_loss": float(total_value_loss) / n,
        "mlh_loss": float(total_mlh_loss) / n,
        "future_loss": float(total_future_loss) / n,
        "steps": steps,
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

def train(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    if cfg.seed >= 0:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        print(f"🎲 seed={cfg.seed} (Rust MCTS сеет себя от времени — полной "
              f"детерминированности не даёт)")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("⚠️  CUDA не найдена, используется CPU — будет медленно")

    print(f"🚀 Тренировка на {device}")
    print(f"   Модель:        {cfg.num_channels}ch × {cfg.num_res_blocks} blocks")
    print(f"   Self-play:     {cfg.games_per_iter} игр/итер, {cfg.simulations} симуляций/ход")
    print(f"   MCTS batch:    {cfg.mcts_batch}  parallel_sims={cfg.mcts_parallel_sims}")
    print(f"   Train batch:   {cfg.batch_size} × до {cfg.train_steps} шагов (≤1 эпохи буфера)")
    print(f"   LR:            {cfg.learning_rate:.2e}  weight_decay={cfg.weight_decay}")
    print(f"   Precision:     BF16 + TF32\n")

    net = CapablancaNet(
        cfg.num_channels, cfg.num_res_blocks,
        enable_mlh=cfg.enable_mlh,
        num_transformer_blocks=cfg.num_transformer_blocks,
        transformer_heads=cfg.transformer_heads,
        enable_future=cfg.enable_future,
        qkv_bias=cfg.qkv_bias,
        use_rmsnorm=cfg.use_rmsnorm,
        piece_embed_dim=cfg.piece_embed_dim,
        qk_norm=cfg.qk_norm,
        swiglu=cfg.swiglu,
        attn_policy=cfg.attn_policy,
        ffn_mult=cfg.ffn_mult,
        restricted_policy=cfg.restricted_policy,
        abs_pos_embed=cfg.abs_pos_embed,
        wide_value=cfg.wide_value,
        num_registers=cfg.num_registers,
        value_residual=cfg.value_residual,
        hyper_streams=cfg.hyper_streams,
    ).to(device)
    # NCHW (default), NOT channels_last: this net is GroupNorm-heavy (~32 GN ops).
    # Tensor-core convs want NHWC but GroupNorm wants NCHW, so channels_last forces
    # NHWC<->NCHW transposes around every norm. Measured ~10-11% SLOWER for both
    # training (fwd+bwd, bs 64/128/512: 0.88-0.91x) and inference (bs 256-4096:
    # 0.86-0.91x). See experiments/perf/. NCHW is also what lc0 uses for fp16
    # conv nets <=384 filters.

    if hasattr(torch, "compile"):
        try:
            net = torch.compile(net, dynamic=True)  # dynamic=True — no recompilation on different MCTS batch sizes
            print("✅ torch.compile() применён\n")
        except Exception as e:
            print(f"⚠️  torch.compile() недоступен: {e}\n")

    optimizer = build_optimizer(net, cfg, device)

    # EMA copy of weights for self-play (AlphaZero-style stabilization)
    ema = ModelEMA(net, decay=cfg.ema_decay) if cfg.use_ema else None

    # Linear warmup + CosineAnnealingWarmRestarts.
    # First warmup_iters iterations LR grows linearly from 0 to cfg.learning_rate,
    # then cosine annealing. This stabilizes the start of training.
    WARMUP_ITERS = 5

    class WarmupCosineScheduler:
        """Linear warmup, then CosineAnnealingWarmRestarts.

        With Muon there are two optimizers and two very different LR scales, and
        torch's scheduler only accepts a real `Optimizer`. So the cosine is
        attached to the AdamW half (`cosine_on`) and every group — including
        Muon's — is then set to `its own base LR × the same factor`. With a plain
        AdamW setup `cosine_on is optimizer` and the behaviour is unchanged.
        """

        def __init__(self, optimizer, warmup_iters, T_0, T_mult, eta_min, base_lr,
                     cosine_on=None):
            self.warmup_iters = warmup_iters
            self.base_lr = base_lr
            self.optimizer = optimizer
            self._base_lrs = [pg['lr'] for pg in optimizer.param_groups]
            cosine_on = cosine_on if cosine_on is not None else optimizer
            self._cosine_base = cosine_on.param_groups[0]['lr']
            self.cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                cosine_on, T_0=T_0, T_mult=T_mult, eta_min=eta_min
            )
            self._last_lr = [base_lr]
            self._iter = 0

        def _apply(self, factor):
            # Each group keeps its own scale — a single shared LR would clobber
            # Muon's (which lives ~10x above AdamW's).
            for pg, base in zip(self.optimizer.param_groups, self._base_lrs):
                pg['lr'] = base * factor

        def step(self):
            self._iter += 1
            if self._iter <= self.warmup_iters:
                self._apply(self._iter / self.warmup_iters)
            else:
                self.cosine.step()
                self._apply(self.cosine.get_last_lr()[0] / self._cosine_base)
            self._last_lr = [self.optimizer.param_groups[0]['lr']]

        def get_last_lr(self):
            return self._last_lr

        def state_dict(self):
            return {"cosine": self.cosine.state_dict(), "_iter": self._iter}

        def load_state_dict(self, sd):
            self.cosine.load_state_dict(sd["cosine"])
            self._iter = sd.get("_iter", 0)

    def make_scheduler(opt):
        # torch's scheduler type-checks for a real Optimizer, so drive the cosine
        # from the AdamW half when Muon is in play.
        inner = opt.opts[0] if isinstance(opt, _MultiOptimizer) else opt
        return WarmupCosineScheduler(
            opt, warmup_iters=WARMUP_ITERS,
            T_0=50, T_mult=2,
            eta_min=cfg.learning_rate * 0.05,
            base_lr=cfg.learning_rate,
            cosine_on=inner,
        )

    scheduler = make_scheduler(optimizer)
    buffer = ReplayBuffer(cfg.buffer_max)

    curriculum_winrate_history: List[float] = []
    lagged_pool = LaggedOpponentPool(max_snapshots=cfg.lag_opponent_pool_size)

    # Prefer the new numpy archive (fast); fall back to legacy pickle only if
    # no npz exists. Old .pkl files are migrated implicitly on the next save.
    buffer_path     = os.path.join(cfg.checkpoint_dir, "buffer.npz")
    buffer_path_pkl = os.path.join(cfg.checkpoint_dir, "buffer.pkl")
    load_path = buffer_path if os.path.exists(buffer_path) else (
        buffer_path_pkl if os.path.exists(buffer_path_pkl) else None)
    if load_path:
        try:
            t_load = time.time()
            if load_path.endswith(".npz"):
                buffer.load_npz(load_path)
            else:
                with open(load_path, "rb") as f:
                    buffer.data, buffer._ptr, buffer._full = pickle.load(f)
                buffer.rebuild_val_arr()
            # Sanitize after load: _ptr / _full may have been saved with a
            # different max_size. Trim oldest if buffer overshot.
            if len(buffer.data) > buffer.max_size:
                buffer.data = list(buffer.data[-buffer.max_size:])
                buffer._full = True
                buffer._ptr  = 0
            elif len(buffer.data) == buffer.max_size:
                buffer._full = True
                buffer._ptr  = buffer._ptr % buffer.max_size
            else:
                buffer._full = False
                buffer._ptr  = len(buffer.data) % buffer.max_size
            elapsed = time.time() - t_load
            kind = "npz" if load_path.endswith(".npz") else "pkl (legacy)"
            print(f"📦 Загружен буфер: {len(buffer):,} позиций "
                  f"({kind}, {elapsed:.1f}s)\n")
        except Exception as e:
            print(f"⚠️  Не удалось загрузить буфер: {e}\n")

    start_iter = 0
    ckpts = sorted([f for f in os.listdir(cfg.checkpoint_dir) if f.endswith(".pth")])
    # Prefer latest.pth (saved every iteration)
    _latest = os.path.join(cfg.checkpoint_dir, "latest.pth")
    ckpts = [f for f in ckpts if not f.startswith("latest")]
    if os.path.exists(_latest) or ckpts:
        path = _latest if os.path.exists(_latest) else os.path.join(cfg.checkpoint_dir, ckpts[-1])
        ckpt = torch.load(path, map_location=device, weights_only=False)

        raw_sd = ckpt["model"]
        # Strip torch.compile / DataParallel prefixes
        raw_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
                  for k, v in raw_sd.items()}

        # Filter layers incompatible with WDL (old Linear(256,1) → new Linear(256,3))
        # value_head.6.weight shape: old (1,256), new (3,256)
        incompatible_keys = []
        target_sd = net._orig_mod.state_dict() if hasattr(net, "_orig_mod") else net.state_dict()
        for k, v in raw_sd.items():
            if k in target_sd and v.shape != target_sd[k].shape:
                incompatible_keys.append(k)
        for k in incompatible_keys:
            del raw_sd[k]

        if hasattr(net, "_orig_mod"):
            missing, unexpected = net._orig_mod.load_state_dict(raw_sd, strict=False)
        else:
            missing, unexpected = net.load_state_dict(raw_sd, strict=False)

        if incompatible_keys:
            print(f"✅ Веса загружены (слои {incompatible_keys} пропущены для адаптации)")
        if missing:
            print(f"   Инициализированы заново: {missing}")

        # Load optimizer only if architecture is fully compatible.
        # On incompatibility (WDL transition) — create optimizer from scratch,
        # otherwise dtype mismatch between old fp16 states and new fp32 layers.
        if not incompatible_keys and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("✅ Оптимайзер загружен из чекпоинта")
            except Exception as e:
                print(f"⚠️  Оптимайзер не загружен ({e}), начинаем заново")
        else:
            print("ℹ️  Оптимайзер инициализирован заново (несовместимая архитектура)")

        # LR override only if scheduler is reset or architecture is incompatible.
        # Otherwise we break the cosine cycle phase that expects the current LR.
        if cfg.reset_scheduler or incompatible_keys or "scheduler" not in ckpt:
            for pg in optimizer.param_groups:
                pg['lr'] = cfg.learning_rate
            print("🔄 Scheduler сброшен (начинается новый косинусный цикл)\n")
        else:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"⚠️  Scheduler не загружен ({e}), используем свежий\n")
                for pg in optimizer.param_groups:
                    pg['lr'] = cfg.learning_rate

        # Load EMA if present, OR reset if --reset-ema flag is set
        if ema is not None and "ema" in ckpt:
            if cfg.reset_ema:
                # Re-initialize EMA from the current network — needed when a checkpoint was saved
                # by old code with per-iter EMA updates (decay=0.999 → 99% random weights).
                src = net._orig_mod if hasattr(net, '_orig_mod') else net
                ema.shadow = {k: v.clone().detach() for k, v in src.state_dict().items()}
                print("🔄 EMA сброшен на текущие веса (--reset-ema)")
            else:
                try:
                    ema.load_state_dict(ckpt["ema"])
                    print("✅ EMA загружен")
                except Exception as e:
                    print(f"⚠️  EMA не загружен: {e}")
        start_iter = ckpt.get("iteration", 0) + 1
        print(f"📂 Загружен чекпоинт: {path} (итерация {start_iter})")
        if cfg.curriculum_mode:
            cfg.fsf_nodes_current = ckpt.get("curriculum_fsf_nodes", cfg.fsf_nodes_current)
            curriculum_winrate_history = list(ckpt.get("curriculum_winrate_history", []))
            print(f"📚 Curriculum восстановлен: FSF nodes={cfg.fsf_nodes_current}  "
                  f"история={[f'{w:.0%}' for w in curriculum_winrate_history]}")

        # Buffer diagnostics at startup — immediately visible if buffer is compromised
        if len(buffer) > 0:
            stats = policy_diversity_stats(buffer.data)
            print_diversity(stats, prefix="   Буфер diversity")
            if stats.get('value_std', 1.0) < 0.05:
                print("   ⚠️  value_std < 0.05 — почти все value одинаковые!")
                print("   ⚠️  Рассмотри перезапуск с --reset-buffer\n")
            else:
                print()

    # ── Distillation / pretrain on teacher buffer ─────────────────────────────
    # Buffer is filled with data from a strong old network: policy = MCTS visits (search-improved
    # policy), value = game outcome. By training a new larger network purely
    # supervised on this buffer, we transfer teacher knowledge: student raw policy
    # learns to reproduce teacher searched policy → fast start ≈ teacher strength.
    if cfg.pretrain_epochs > 0:
        if len(buffer) < cfg.batch_size * 10:
            print(f"⚠️  Буфер мал ({len(buffer):,}) для дистилляции — пропускаем pretrain.\n")
        else:
            print(f"\n🎓 Дистилляция: {cfg.pretrain_epochs} эпох supervised на буфере "
                  f"({len(buffer):,} позиций), без self-play")
            for pe in range(1, cfg.pretrain_epochs + 1):
                net.train()
                t0 = time.time()
                metrics = train_epoch(net, optimizer, buffer, cfg, device, 0, ema=ema)
                scheduler.step()
                lr = scheduler.get_last_lr()[0]
                print(f"  🎓 Эпоха {pe}/{cfg.pretrain_epochs} за {time.time()-t0:.1f}s | "
                      f"policy={metrics['policy_loss']:.4f} value={metrics['value_loss']:.4f} "
                      f"mlh={metrics['mlh_loss']:.4f} future={metrics.get('future_loss',0.0):.4f} "
                      f"total={metrics['loss']:.4f} lr={lr:.2e}")
            # Re-sync EMA to current weights after distillation — otherwise self-play
            # would start with a mix (random_init ⊕ distilled) due to decay lag.
            if ema is not None:
                ema = ModelEMA(net, decay=cfg.ema_decay)
                print("  🔄 EMA ресинхронизирован на дистиллированные веса")
            # Save the distilled network
            model_to_save = net._orig_mod if hasattr(net, "_orig_mod") else net
            distill_ckpt = {
                "iteration": start_iter,
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": metrics,
                "curriculum_fsf_nodes": cfg.fsf_nodes_current,
                "curriculum_winrate_history": list(curriculum_winrate_history),
            }
            if ema is not None:
                distill_ckpt["ema"] = ema.state_dict()
            torch.save(distill_ckpt, os.path.join(cfg.checkpoint_dir, "distilled.pth"))
            torch.save(distill_ckpt, os.path.join(cfg.checkpoint_dir, "latest.pth"))
            print(f"  💾 distilled.pth + latest.pth — дистилляция завершена.\n")
            if cfg.pretrain_only:
                print("✅ pretrain_only: выходим. Перезапусти без --pretrain-epochs "
                      "для перехода в self-play.\n")
                return

    # Baseline temperature the entropy controller relaxes back toward (see below).
    base_temperature = cfg.temperature
    last_iter = (start_iter + cfg.max_iters) if cfg.max_iters > 0 else 100_000
    for iteration in range(start_iter, last_iter):
        iter_start = time.time()

        # ── Self-play / Curriculum ────────────────────────────────────────────
        sp_start = time.time()
        # EMA not used for self-play for the first ema_start_iter iterations:
        # early EMA weights = mixture of random weights → worse than live NN for data generation.
        # EMA always updated (so that by iter=ema_start_iter it already reflects the trained net).
        use_ema_now = ema is not None and iteration >= cfg.ema_start_iter
        if use_ema_now:
            saved_state = {k: v.clone() for k, v in (net._orig_mod if hasattr(net, '_orig_mod') else net).state_dict().items()
                           if v.dtype.is_floating_point}
            ema.apply_to(net)
        elif ema is None:
            pass  # no EMA configured
        else:
            print(f"  ℹ️  EMA отложен до iter {cfg.ema_start_iter} (сейчас {iteration}), используется live NN")

        fsf_path = cfg.fsf_path
        fsf_enabled = bool(fsf_path and os.path.exists(fsf_path))

        if cfg.curriculum_mode and not fsf_enabled:
            print(f"[Iter {iteration}] ⚠️  --curriculum требует --fsf-path, переходим в self-play")

        if fsf_enabled and cfg.curriculum_mode:
            # ── Curriculum: adaptive FSF as main teacher ──────────────────────
            sp_count  = int(cfg.games_per_iter * cfg.curriculum_self_play_ratio)
            fsf_count = cfg.games_per_iter - sp_count
            opp_lbl = "Random" if cfg.fsf_nodes_current == 0 else f"FSF nodes={cfg.fsf_nodes_current}"
            print(f"[Iter {iteration}] 📚 Curriculum | {opp_lbl}"
                  f"  self={sp_count}  fsf={fsf_count}")
            samples = []
            if sp_count > 0:
                # Split self-play into pure (vs current) + lagged (vs old checkpoint)
                has_lagged = cfg.lag_opponent_ratio > 0 and len(lagged_pool) > 0
                lag_count  = int(sp_count * cfg.lag_opponent_ratio) if has_lagged else 0
                pure_sp    = sp_count - lag_count

                if pure_sp > 0:
                    orig_games = cfg.games_per_iter
                    cfg.games_per_iter = pure_sp
                    try:
                        net.eval()
                        with torch.inference_mode():
                            samples = generate_games(net, cfg, device, iteration)
                    finally:
                        cfg.games_per_iter = orig_games

                if lag_count > 0:
                    lag_snap = lagged_pool.sample()
                    if lag_snap is not None:
                        lag_iter, lag_sd = lag_snap
                        print(f"  ⚔️  Lagged {lag_count} игр vs iter-{lag_iter} чекпоинт "
                              f"({cfg.lag_opponent_sims} sims)...")
                        net.eval()
                        with torch.inference_mode():
                            lag_samps, lw, ld, ll = generate_lagged_games(
                                net, lag_sd, cfg, device, lag_count, cfg.lag_opponent_sims)
                        samples = samples + lag_samps
            net.eval()
            with torch.inference_mode():
                fsf_samples, fsf_w, fsf_d, fsf_l = generate_fsf_games(
                    net, device, cfg,
                    num_games=fsf_count, fsf_path=fsf_path,
                    fsf_nodes=cfg.fsf_nodes_current, mcts_sims=cfg.fsf_mcts_sims,
                )
            samples = samples + fsf_samples
            total_fsf = fsf_w + fsf_d + fsf_l
            if total_fsf > 0:
                wr = (fsf_w + 0.5 * fsf_d) / total_fsf
                curriculum_winrate_history.append(wr)
                window = curriculum_winrate_history[-cfg.curriculum_window:]
                print(f"  📊 FSF winrate: {wr:.1%}  "
                      f"окно [{', '.join(f'{w:.0%}' for w in window)}]")
                if len(curriculum_winrate_history) >= cfg.curriculum_window:
                    avg_wr = float(np.mean(window))
                    def _nodes_label(n): return "Random" if n == 0 else f"FSF-{n}"
                    if avg_wr > cfg.curriculum_promote_threshold:
                        old = cfg.fsf_nodes_current
                        # 0 (random) → 1 (FSF-1), then 1→2→4→8...
                        if cfg.fsf_nodes_current == 0:
                            cfg.fsf_nodes_current = 1
                        else:
                            cfg.fsf_nodes_current = min(
                                int(cfg.fsf_nodes_current * 2), cfg.curriculum_nodes_max)
                        curriculum_winrate_history.clear()
                        print(f"  📈 Повышение: {_nodes_label(old)} → {_nodes_label(cfg.fsf_nodes_current)} "
                              f"(avg={avg_wr:.1%})")
                    elif avg_wr < cfg.curriculum_demote_threshold:
                        old = cfg.fsf_nodes_current
                        cfg.fsf_nodes_current = max(
                            cfg.fsf_nodes_current // 2, cfg.curriculum_nodes_min)
                        curriculum_winrate_history.clear()
                        print(f"  📉 Снижение: {_nodes_label(old)} → {_nodes_label(cfg.fsf_nodes_current)} "
                              f"(avg={avg_wr:.1%})")

        elif fsf_enabled:
            # ── Original FSF schedule ─────────────────────────────────────────
            print(f"[Iter {iteration}] ⚙️  Self-play: {cfg.games_per_iter} игр...")
            self_games_n, fsf_games_n, _ = get_fsf_schedule(iteration, cfg.games_per_iter)
            phase = ("🔴 FSF-heavy" if iteration < 30
                     else "🟡 FSF-fade" if iteration < 50 else "🟢 self-only")
            print(f"  {phase} | self={self_games_n}"
                  + (f" + fsf={fsf_games_n}" if fsf_games_n else ""))
            orig_games = cfg.games_per_iter
            cfg.games_per_iter = self_games_n
            try:
                net.eval()
                with torch.inference_mode():
                    samples = generate_games(net, cfg, device, iteration)
            finally:
                cfg.games_per_iter = orig_games
            if fsf_games_n > 0:
                print(f"  ⚔️  FSF: {fsf_games_n} игр vs Stockfish ({cfg.fsf_nodes} nodes)...")
                net.eval()
                with torch.inference_mode():
                    fsf_r = generate_fsf_games(
                        net, device, cfg,
                        num_games=fsf_games_n, fsf_path=fsf_path,
                        fsf_nodes=cfg.fsf_nodes, mcts_sims=cfg.fsf_mcts_sims,
                    )
                samples = samples + fsf_r[0]

        else:
            # ── Self-play only ────────────────────────────────────────────────
            print(f"[Iter {iteration}] ⚙️  Self-play: {cfg.games_per_iter} игр...")
            net.eval()
            with torch.inference_mode():
                samples = generate_games(net, cfg, device, iteration)

        # Restore training weights (only if EMA was applied)
        if use_ema_now:
            (net._orig_mod if hasattr(net, '_orig_mod') else net).load_state_dict(saved_state, strict=False)
        buffer.push(samples)

        sp_time = time.time() - sp_start
        print(f"  ✅ {len(samples):,} позиций за {sp_time:.1f}s "
              f"({cfg.games_per_iter / sp_time:.2f} игр/с, "
              f"{len(samples) / sp_time:.0f} поз/с)")
        print(f"  Буфер: {len(buffer):,} позиций")

        # Fresh data diagnostics
        stats = policy_diversity_stats(samples)
        print_diversity(stats)

        # Proactive temperature control by entropy (react BEFORE collapse, not after).
        # For 7000 outputs normal range: 2.5-4.0 bits.
        # Symmetric: raise fast on low entropy, then RELAX back toward the
        # configured baseline on healthy batches. The old code only decayed when
        # entropy>3.5 (rare) by 5%, so a few low-entropy batches ratcheted
        # temperature stuck near the 2.5 cap → permanently noisy self-play.
        entropy = stats.get('entropy_mean', 2.0)
        if entropy < 1.0:
            cfg.temperature = min(cfg.temperature * 1.25, 2.5)
            print(f"  ⚠️  entropy={entropy:.3f} < 1.0 → temperature→{cfg.temperature:.2f}")
        elif entropy < 1.6:
            cfg.temperature = min(cfg.temperature * 1.10, 2.0)
            print(f"  ⚡ entropy={entropy:.3f} < 1.6 → temperature→{cfg.temperature:.2f}")
        elif abs(cfg.temperature - base_temperature) > 1e-3:
            # healthy entropy → pull 25% back toward baseline each iteration
            cfg.temperature += (base_temperature - cfg.temperature) * 0.25
            if abs(cfg.temperature - base_temperature) < 1e-3:
                cfg.temperature = base_temperature
            print(f"  🌡️  entropy={entropy:.3f} ok → temperature→{cfg.temperature:.2f} "
                  f"(→ baseline {base_temperature:.2f})")
        print()

        # ── Training ─────────────────────────────────────────────────────────
        if len(buffer) < cfg.buffer_min_to_train:
            print(f"  ⏳ Мало данных ({len(buffer):,} < {cfg.buffer_min_to_train:,}), "
                  f"пропускаем тренировку\n")
            continue

        net.train()

        print(f"  🏋️  Тренировка (до {cfg.train_steps} шагов, ≤1 эпохи)...")
        train_start = time.time()
        # EMA is updated INSIDE train_epoch after each step (correct per-step semantics).
        metrics = train_epoch(net, optimizer, buffer, cfg, device, iteration, ema=ema)
        train_time = time.time() - train_start

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Collapse detector (--force-save bypasses the check)
        collapsed = metrics['policy_loss'] < cfg.collapse_threshold and not cfg.force_save
        collapse_warn = "  ⚠️  КОЛЛАПС ПОЛИТИКИ — чекпоинт не сохранён!" if collapsed else ""

        print(f"\n  ✅ Тренировка за {train_time:.1f}s ({metrics['steps']} шагов)")
        print(f"     policy_loss = {metrics['policy_loss']:.4f}{collapse_warn}")
        print(f"     value_loss  = {metrics['value_loss']:.4f}")
        print(f"     mlh_loss    = {metrics['mlh_loss']:.4f}")
        print(f"     future_loss = {metrics.get('future_loss', 0.0):.4f}")
        print(f"     total_loss  = {metrics['loss']:.4f}")
        print(f"     lr          = {current_lr:.2e}")

        if collapsed:
            print(f"\n  ⚠️  Рекомендация: перезапустить с --reset-buffer --reset-scheduler\n")

        iter_time = time.time() - iter_start
        print(f"\n  ⏱️  Итерация {iteration}: {iter_time:.1f}s total "
              f"(self-play {sp_time:.1f}s + train {train_time:.1f}s)\n")

        # ── Checkpoint ───────────────────────────────────────────────────────
        if not collapsed:
            # Save snapshot for lagged pool AFTER training (carries knowledge from this iteration)
            lagged_pool.maybe_save(net, iteration, cfg.lag_opponent_interval)

            model_to_save = net._orig_mod if hasattr(net, "_orig_mod") else net
            ckpt_data = {
                "iteration": iteration,
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": metrics,
                "curriculum_fsf_nodes": cfg.fsf_nodes_current,
                "curriculum_winrate_history": list(curriculum_winrate_history),
            }
            if ema is not None:
                ckpt_data["ema"] = ema.state_dict()
            # latest.pth — overwritten each iteration
            # On crash/stop there is always the latest state
            latest_path = os.path.join(cfg.checkpoint_dir, "latest.pth")
            torch.save(ckpt_data, latest_path)
            print(f"  💾 latest.pth (iter {iteration})")

            # Numbered archive checkpoint every save_every iterations
            if iteration % cfg.save_every == 0:
                path = os.path.join(cfg.checkpoint_dir, f"model_iter{iteration:05d}.pth")
                torch.save(ckpt_data, path)
                print(f"  💾 {os.path.basename(path)}")

            try:
                t_save = time.time()
                buffer.save_npz(buffer_path)
                # Drop the legacy pickle file if it's still around — saves disk
                # and removes ambiguity on the next load.
                if os.path.exists(buffer_path_pkl):
                    try: os.remove(buffer_path_pkl)
                    except OSError: pass
                print(f"  💾 Буфер сохранён ({len(buffer):,} позиций, "
                      f"{time.time() - t_save:.1f}s)\n")
            except Exception as e:
                print(f"  ⚠️  Не удалось сохранить буфер: {e}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capablanca Chess AlphaZero Training")
    parser.add_argument("--channels",           type=int,   default=64)
    parser.add_argument("--res-blocks",          type=int,   default=5)
    parser.add_argument("--transformer-blocks", type=int,   default=2,
                        help="Кол-во transformer encoder блоков с RPB ПОСЛЕ ResNet tower. "
                             "0 = чистый ResNet (для совместимости со старыми чекпоинтами).")
    parser.add_argument("--transformer-heads",  type=int,   default=8,
                        help="Кол-во attention heads. Должно делить --channels.")
    parser.add_argument("--no-mlh",             dest="enable_mlh", action="store_false",
                        default=True, help="Отключить Moves-Left-Head.")
    parser.add_argument("--no-future",          dest="enable_future", action="store_false",
                        default=True, help="Отключить Future Move Head.")
    parser.add_argument("--no-qkv-bias", dest="qkv_bias", action="store_false",
                        default=True,
                        help="BT5: убрать bias из QKV-проекций (~5%% быстрее inference, "
                             "без потерь качества). НЕ совместимо со старыми чекпоинтами — "
                             "включать только при дистилляции в новую архитектуру.")
    parser.add_argument("--rmsnorm", dest="use_rmsnorm", action="store_true",
                        default=False,
                        help="BT5: использовать RMSNorm вместо LayerNorm (без centering и bias). "
                             "Парная фича с --no-qkv-bias.")
    parser.add_argument("--qk-norm", action="store_true",
                        help="RMSNorm на Q и K перед скалярным произведением "
                             "(Gemma 2 / ViT-22B). Стабилизирует attention при большом lr.")
    parser.add_argument("--swiglu", action="store_true",
                        help="Gated FFN (SwiGLU) вместо Linear→Mish→Linear при том же "
                             "числе параметров.")
    parser.add_argument("--ffn-mult", type=int, default=2,
                        help="Множитель FFN в трансформерных блоках. По "
                             "умолчанию 2 (hidden = d*2*2/3 = 341 при 256d) — "
                             "мало; lc0 держит около 4.")
    parser.add_argument("--restricted-policy", action="store_true",
                        help="Policy-голова только на 2672 геометрически "
                             "достижимых индекса вместо 7000. Остальные 4328 "
                             "строк умеют лишь выдавать -inf: 11.1M мёртвых "
                             "параметров (model.reachable_policy_indices).")
    parser.add_argument("--abs-pos-embed", action="store_true",
                        help="Абсолютный позиционный эмбеддинг 80×C на токенах "
                             "перед трансформером (в attention только "
                             "ОТНОСИТЕЛЬНЫЙ bias, абсолютной позиции нет).")
    parser.add_argument("--wide-value", action="store_true",
                        help="Value-голова 32ch→512 вместо 8ch→256. 0.17M на "
                             "сети в 28M — узкое место, и именно value ломалась "
                             "при прошлых дистилляциях.")
    parser.add_argument("--attn-policy", action="store_true",
                        help="Билинейная policy-голова from→to вместо Linear(C*80, 7000). "
                             "17.9M → 0.18M параметров на голову (и столько же на future).")
    parser.add_argument("--registers", type=int, default=0,
                        help="Сколько register-токенов добавить к 80 клеточным (0 = выкл).")
    parser.add_argument("--value-residual", action="store_true",
                        help="ResFormer: подмешивать value первого блока во все "
                             "последующие. Лечит value-state drain в глубоких слоях.")
    parser.add_argument("--hyper-streams", type=int, default=0,
                        help="mHC (DeepSeek): расширить residual-поток до N параллельных "
                             "потоков, матрица смешивания проецируется на многогранник "
                             "Биркгофа через Синкхорна. 0=выкл, 4=как в статье. "
                             "При инициализации численно тождественно обычному residual.")
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                        help="muon: ортогонализация моментума для 2D-весов, AdamW для "
                             "остального. Меньше состояния оптимизатора и обычно быстрее "
                             "сходится на трансформерах.")
    parser.add_argument("--muon-lr", type=float, default=2e-2,
                        help="LR для Muon-групп (другая шкала, чем у AdamW; default 2e-2)")
    parser.add_argument("--piece-embed-dim", type=int, default=0,
                        help="BT3 trick: per-square Linear(16 → N) от plane piece-планов, "
                             "конкатенируется к input. 0 = выключено. 32 — рекомендуемое значение. "
                             "Только при тренировке новой архитектуры (ломает старые чекпоинты).")
    parser.add_argument("--simulations",         type=int,   default=100,
                        help="Полные симуляции (для PCR — больше fast_simulations, обучается на этих позициях)")
    parser.add_argument("--fast-simulations",     type=int,   default=80,
                        help="Симуляций на быстрых ходах playout cap (default: 80, меньше simulations)")
    parser.add_argument("--fast-sim-fraction",    type=float, default=0.75,
                        help="Доля ходов с быстрым поиском (0.75 = 75 пр. быстрых, 25 пр. полных)")
    parser.add_argument("--no-playout-cap",       dest="playout_cap_train_only_full",
                        action="store_false", default=True,
                        help="Отключить playout cap (учить на всех позициях)")
    parser.add_argument("--games",               type=int,   default=128)
    parser.add_argument("--mcts-batch",          type=int,   default=128)
    parser.add_argument("--temperature",          type=float, default=1.0,
                        help="Температура выборки хода (tau) в первые --temperature-moves ходов")
    parser.add_argument("--temperature-late",     type=float, default=0.0,
                        help="Температура после --temperature-moves (0.5=мягкий argmax, 0=жадный)")
    parser.add_argument("--temperature-moves",    type=int,   default=50,
                        help="Ходов с высокой температурой")
    parser.add_argument("--mcts-parallel-sims", type=int, default=8,
                        help="Листьев за шаг MCTS. ceil(--simulations/это) = число "
                             "ПОСЛЕДОВАТЕЛЬНЫХ раундов PUCT; держать >= 12, иначе "
                             "virtual loss размазывает визиты и цель обучения "
                             "вырождается в равномерную "
                             "(docs/experiments/policy_target_collapse.md). Было 32.")
    parser.add_argument("--compile-inference", type=str, default=None,
                        choices=[None, "default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode для inference (selfplay). None=off. "
                             "'default'=безопасный +15-25%%. 'reduce-overhead'=CUDA graphs +30-50%%. "
                             "'max-autotune'=макс. speedup но warmup 1-2 мин.")
    parser.add_argument("--kld-threshold", type=float, default=5e-3,
                        help="KLD-early-exit per-visit gain порог. Меньше = строже. "
                             "Реалистичный диапазон 1e-3..1e-2 для 100-400 sims. 0=disable.")
    parser.add_argument("--kld-check-every", type=int, default=2,
                        help="Проверять KL каждые N parallel-steps MCTS.")
    parser.add_argument("--kld-min-sims-frac", type=float, default=0.30,
                        help="Минимум доли sims перед early-exit (0.30 = 30%%).")
    parser.add_argument("--no-kld", dest="kld_enabled", action="store_false",
                        help="Отключить KLD-early-exit.")
    parser.set_defaults(kld_enabled=True)
    parser.add_argument("--batch-size",          type=int,   default=512)
    parser.add_argument("--train-steps",         type=int,   default=200)
    parser.add_argument("--min-train-steps",     type=int,   default=20)
    parser.add_argument("--buffer-min-to-train", type=int,   default=10_000)
    parser.add_argument("--lr",                  type=float, default=2e-4)
    parser.add_argument("--device",              type=str,   default="cuda")
    parser.add_argument("--checkpoint-dir",      type=str,   default="checkpoints")
    parser.add_argument("--save-every",          type=int,   default=5)
    parser.add_argument("--seed",                type=int,   default=-1,
                        help="Сид torch/numpy (-1 = не трогать). Нужен для абляций.")
    parser.add_argument("--max-iters",           type=int,   default=0,
                        help="Остановиться после N итераций (0 = бесконечно). "
                             "Нужно для контролируемых A/B-прогонов.")
    parser.add_argument("--log-every",           type=int,   default=50,
                        help="Печатать средние лоссы каждые N шагов тренировки")
    parser.add_argument("--value-loss-weight",   type=float, default=1.0)
    parser.add_argument("--value-q-weight",      type=float, default=0.0,
                        help="Blend MCTS root value into the value target "
                             "(KataGo/LC0): v=(1-w)*z + w*Q, full WDL with search D. "
                             "0=pure game outcome (legacy), ~0.5=KataGo-style.")
    parser.add_argument("--no-value-balance",    action="store_true",
                        help="Train on the natural outcome distribution instead of "
                             "rebalancing each batch to 33/33/33 W/D/L.")
    parser.add_argument("--policy-target",       choices=["visits", "gumbel"],
                        default="visits",
                        help="Policy training target: 'visits' (legacy visit-count "
                             "distribution) or 'gumbel' (completed-Q improved policy, "
                             "stronger at low sim counts). Move selection is unchanged.")
    parser.add_argument("--adjudicate",          action="store_true",
                        help="End clearly-decided games early (material ≥8, ≥20 ply "
                             "quiet, ≥15 moves) — saves timeout-tail compute.")
    parser.add_argument("--adjudicate-q-gate",   type=float, default=0.5,
                        help="Require the net's root value to confirm adjudication "
                             "(root_Q·verdict ≥ gate). 0 = material-only. Default 0.5.")
    parser.add_argument("--rep-search-perslot",  action="store_true",
                        help="Use per-slot repetition planes during search (align "
                             "with the training-side encoding).")
    parser.add_argument("--mlh-loss-weight",     type=float, default=0.1,
                        help="LC0 MLH loss weight (default: 0.1)")
    parser.add_argument("--future-loss-weight",  type=float, default=0.15,
                        help="Future move head loss weight (default: 0.15)")
    parser.add_argument("--pretrain-epochs",     type=int,   default=0,
                        help="Дистилляция: N эпох supervised на teacher-буфере перед self-play. 0=выкл.")
    parser.add_argument("--pretrain-only",       action="store_true",
                        help="Выйти сразу после дистилляции (не входить в self-play цикл)")
    parser.add_argument("--reset-scheduler",     action="store_true",
                        help="Пересоздать LR scheduler при загрузке чекпоинта")
    parser.add_argument("--reset-buffer",        action="store_true",
                        help="Очистить replay buffer при старте")
    parser.add_argument("--collapse-threshold",  type=float, default=0.01)
    parser.add_argument("--no-ema",              dest="use_ema", action="store_false",
                        default=True, help="Отключить EMA для self-play (по умолчанию включён)")
    parser.add_argument("--ema-decay",           type=float, default=0.9999,
                        help="EMA decay coefficient (default: 0.9999, per-step). "
                             "0.9999 = окно ~10K шагов ≈ 10 итераций (LC0 selfplay-style).")
    parser.add_argument("--ema-start-iter",      type=int,   default=10,
                        help="Не использовать EMA для self-play до этой итерации (default: 10)")
    parser.add_argument("--reset-ema",            action="store_true",
                        help="Сбросить EMA веса из чекпоинта (= скопировать из текущей сети). "
                             "Нужно если EMA испорчен (например, чекпоинт из старой версии где EMA "
                             "обновлялся раз в итерацию вместо per-step).")
    parser.add_argument("--resign-threshold",      type=float, default=-0.95,
                        help="Финальный порог сдачи (default: -0.95)")
    parser.add_argument("--resign-playthrough",    type=float, default=0.10,
                        help="Доля игр без resign для калибровки (LC0-style). "
                             "0.10 = 10%% играем до конца. (default: 0.10)")
    parser.add_argument("--resign-threshold-early", type=float, default=-0.99,
                        help="Порог сдачи на ранних итерациях (default: -0.99)")
    parser.add_argument("--resign-warmup-iters",  type=int,   default=30,
                        help="Итераций до перехода к финальному порогу (default: 30)")
    parser.add_argument("--resign-consec",        type=int,   default=3,
                        help="Ходов подряд для сдачи (default: 3)")
    parser.add_argument("--resign-min-move",      type=int,   default=20,
                        help="Минимальный ход для сдачи (default: 20)")
    parser.add_argument("--force-save",           action="store_true",
                        help="Сохранять чекпоинт даже если policy_loss < collapse_threshold")
    parser.add_argument("--contempt",             type=float, default=0.0,
                        help="Сдвиг Q в селекции PUCT на накопленный draw-prob. "
                             "0.0 = обычный режим. +N → избегать ничьих (полезно vs FSF), "
                             "-N → принимать ничьи. В чистом self-play оставлять 0.0 — "
                             "обе стороны применят его симметрично и исказят value-таргеты.")

    # FSF integration (optional)
    parser.add_argument("--fsf-path",             type=str, default=None,
                        help="Путь к Fairy-Stockfish бинарнику (включает FSF режим)")
    parser.add_argument("--fsf-nodes",            type=int, default=500,
                        help="Лимит nodes для FSF в обычном режиме (default: 500)")
    parser.add_argument("--fsf-mcts-sims",        type=int, default=100,
                        help="MCTS симуляций при игре против FSF (default: 100)")
    parser.add_argument("--fsf-value-alpha",      type=float, default=0.7,
                        help="Вес FSF eval в value target: alpha*eval + (1-alpha)*result (default: 0.7)")

    # Curriculum training
    parser.add_argument("--curriculum",           action="store_true",
                        help="Curriculum mode: FSF как адаптивный учитель (требует --fsf-path)")
    parser.add_argument("--fsf-nodes-start",      type=int, default=0,
                        help="Начальный уровень curriculum: 0=Random mover, 1+=FSF nodes (default: 0)")
    parser.add_argument("--fsf-nodes-max",        type=int, default=10000,
                        help="Максимальный уровень FSF nodes в curriculum (default: 10000)")
    parser.add_argument("--curriculum-sp-ratio",  type=float, default=0.0,
                        help="Доля self-play в curriculum (0.0=только FSF, 0.2=20%% self-play)")
    parser.add_argument("--curriculum-promote",   type=float, default=0.55,
                        help="Winrate для повышения сложности FSF (default: 0.55)")
    parser.add_argument("--curriculum-demote",    type=float, default=0.35,
                        help="Winrate для снижения сложности FSF (default: 0.35)")
    parser.add_argument("--curriculum-window",    type=int,   default=3,
                        help="Итераций для усреднения winrate (default: 3)")

    # Lagged opponent: play against an old checkpoint
    parser.add_argument("--lag-interval",     type=int,   default=0,
                        help="Сохранять снимок весов каждые N итераций (0=отключено, рек. 5)")
    parser.add_argument("--lag-ratio",        type=float, default=0.0,
                        help="Доля self-play игр против lagged чекпоинта (0.0=отключено, рек. 0.3)")
    parser.add_argument("--lag-pool-size",    type=int,   default=5,
                        help="Максимальное число хранимых снимков (default: 5)")
    parser.add_argument("--lag-sims",         type=int,   default=50,
                        help="MCTS симуляций для lagged оппонента (default: 50)")

    # Softened FSF
    parser.add_argument("--fsf-random-prob",  type=float, default=0.0,
                        help="Вероятность случайного хода FSF (0=детерминированный, 0.4=40%% рандом)")

    # Buffer window size and game length
    parser.add_argument("--buffer-max",       type=int,   default=1_000_000,
                        help="Максимальный размер replay буфера (default: 1000000, рек. 300000 при большом потоке данных)")
    parser.add_argument("--max-game-length",  type=int,   default=300,
                        help="Максимальная длина партии в полуходах (default: 300)")
    parser.add_argument("--timeout-as-draw",  action="store_true", default=False,
                        help="Таймаут = ничья (0.0) вместо оценки по материалу")

    args = parser.parse_args()

    # Reset buffer if requested — drop both legacy pickle and new npz form.
    if args.reset_buffer:
        for fname in ("buffer.npz", "buffer.pkl"):
            p = os.path.join(args.checkpoint_dir, fname)
            if os.path.exists(p):
                os.remove(p)
        print("🗑️  Буфер сброшен\n")

    cfg = Config(
        num_channels=args.channels,
        num_res_blocks=args.res_blocks,
        num_transformer_blocks=args.transformer_blocks,
        transformer_heads=args.transformer_heads,
        enable_mlh=args.enable_mlh,
        simulations=args.simulations,
        fast_simulations=args.fast_simulations,
        fast_sim_fraction=args.fast_sim_fraction,
        playout_cap_train_only_full=args.playout_cap_train_only_full,
        games_per_iter=args.games,
        mcts_batch=args.mcts_batch,
        temperature=args.temperature,
        temperature_late=args.temperature_late,
        temperature_moves=args.temperature_moves,
        mcts_parallel_sims=args.mcts_parallel_sims,
        compile_inference=args.compile_inference,
        kld_threshold=args.kld_threshold,
        kld_check_every=args.kld_check_every,
        kld_min_sims_frac=args.kld_min_sims_frac,
        kld_enabled=args.kld_enabled,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        min_train_steps=args.min_train_steps,
        buffer_min_to_train=args.buffer_min_to_train,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        max_iters=args.max_iters,
        seed=args.seed,
        value_loss_weight=args.value_loss_weight,
        value_q_weight=args.value_q_weight,
        value_balance=not args.no_value_balance,
        policy_target_mode=args.policy_target,
        adjudicate=args.adjudicate,
        adjudicate_q_gate=args.adjudicate_q_gate,
        rep_search_perslot=args.rep_search_perslot,
        mlh_loss_weight=args.mlh_loss_weight,
        future_loss_weight=args.future_loss_weight,
        enable_future=args.enable_future,
        qkv_bias=args.qkv_bias,
        use_rmsnorm=args.use_rmsnorm,
        piece_embed_dim=args.piece_embed_dim,
        qk_norm=args.qk_norm,
        swiglu=args.swiglu,
        attn_policy=args.attn_policy,
        ffn_mult=args.ffn_mult,
        restricted_policy=args.restricted_policy,
        abs_pos_embed=args.abs_pos_embed,
        wide_value=args.wide_value,
        num_registers=args.registers,
        value_residual=args.value_residual,
        hyper_streams=args.hyper_streams,
        optimizer=args.optimizer,
        muon_lr=args.muon_lr,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_only=args.pretrain_only,
        reset_scheduler=args.reset_scheduler,
        collapse_threshold=args.collapse_threshold,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        ema_start_iter=args.ema_start_iter,
        contempt=args.contempt,
        fsf_path=args.fsf_path,
        fsf_nodes=args.fsf_nodes,
        fsf_mcts_sims=args.fsf_mcts_sims,
        reset_ema=args.reset_ema,
        fsf_value_alpha=args.fsf_value_alpha,
        resign_threshold=args.resign_threshold,
        resign_threshold_early=args.resign_threshold_early,
        resign_warmup_iters=args.resign_warmup_iters,
        resign_consec=args.resign_consec,
        resign_min_move=args.resign_min_move,
        resign_playthrough=args.resign_playthrough,
        force_save=args.force_save,
        curriculum_mode=args.curriculum,
        fsf_nodes_current=args.fsf_nodes_start,
        curriculum_nodes_min=0,
        curriculum_nodes_max=args.fsf_nodes_max,
        curriculum_self_play_ratio=args.curriculum_sp_ratio,
        curriculum_promote_threshold=args.curriculum_promote,
        curriculum_demote_threshold=args.curriculum_demote,
        curriculum_window=args.curriculum_window,
        lag_opponent_interval=args.lag_interval,
        lag_opponent_ratio=args.lag_ratio,
        lag_opponent_pool_size=args.lag_pool_size,
        lag_opponent_sims=args.lag_sims,
        fsf_noise_prob=args.fsf_random_prob,
        buffer_max=args.buffer_max,
        max_game_length=args.max_game_length,
        timeout_as_draw=args.timeout_as_draw,
    )
    train(cfg)
