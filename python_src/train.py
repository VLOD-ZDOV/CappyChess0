# train.py — AlphaZero-style self-play training for Capablanca Chess
#
# Pipeline:
#   1. Self-play:  generate games with MCTS + current network → ReplayBuffer
#   2. Training:   sample minibatches → update network (policy + value + L2)
#   3. Repeat
#
# Run:  python train.py
# Requires: capablanca_engine (maturin develop --release)

import os
import random
import math
import time
import copy
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import CapablancaNet, POLICY_SIZE
from mcts import mcts_policy_vector, engine_to_tensor

try:
    from capablanca_engine import CapablancaEngine
except ImportError:
    raise ImportError(
        "capablanca_engine Rust extension not found.\n"
        "Build it with:  maturin develop --release"
    )


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Network
    num_channels:   int   = 128
    num_res_blocks: int   = 10

    # MCTS
    simulations:    int   = 400    # per move during self-play
    c_puct:         float = 1.25
    temperature_moves: int = 30    # use temperature=1 for first N moves, then greedy
    dirichlet_alpha: float = 0.3
    dirichlet_eps:   float = 0.25

    # Self-play
    games_per_iter:     int = 100
    max_game_length:    int = 300   # resign/draw after this many moves

    # Training
    batch_size:      int   = 256
    replay_buffer:   int   = 500_000   # max samples to keep
    min_buffer_size: int   = 10_000    # start training after this many samples
    learning_rate:   float = 2e-4
    weight_decay:    float = 1e-4
    train_steps:     int   = 1000      # gradient steps per iteration
    gradient_clip:   float = 1.0

    # Checkpoint
    save_every:      int   = 5         # save every N iterations
    checkpoint_dir:  str   = "checkpoints"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ── Replay buffer ─────────────────────────────────────────────────────────────

# A sample is: (board_tensor [1600 floats], policy_target [6720 floats], value_target float)
Sample = Tuple[np.ndarray, np.ndarray, float]


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.buffer: deque[Sample] = deque(maxlen=max_size)

    def push(self, samples: List[Sample]):
        self.buffer.extend(samples)

    def sample(self, batch_size: int) -> List[Sample]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class ChessDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        board, policy, value = self.samples[idx]
        return (
            torch.tensor(board,  dtype=torch.float32).view(20, 8, 10),
            torch.tensor(policy, dtype=torch.float32),
            torch.tensor([value], dtype=torch.float32),
        )


# ── Self-play ─────────────────────────────────────────────────────────────────

def self_play_game(net: CapablancaNet, cfg: Config) -> List[Sample]:
    """
    Play a full game via MCTS self-play.
    Returns list of (board_tensor, mcts_policy, game_result_from_white_perspective).
    """
    engine = CapablancaEngine()
    device = torch.device(cfg.device)
    net.eval()

    history: List[Tuple[np.ndarray, np.ndarray, int]] = []
    # history entries: (board_tensor, mcts_policy, side_that_played)

    move_num = 0
    while not engine.is_game_over() and move_num < cfg.max_game_length:
        temperature = 1.0 if move_num < cfg.temperature_moves else 0.0

        # Get MCTS policy
        mcts_pi = mcts_policy_vector(
            engine, net,
            simulations=cfg.simulations,
            c_puct=cfg.c_puct,
            temperature=temperature,
            device=device,
        )

        # Store state
        board_tensor = np.array(engine.get_board_tensor(), dtype=np.float32)
        side = engine.side_to_move()
        history.append((board_tensor, mcts_pi, side))

        # Choose and play move
        legal = engine.get_legal_moves()
        if not legal:
            break

        if temperature == 0:
            # Greedy: pick highest-visited move
            best_idx = int(np.argmax(mcts_pi))
            # Map back to move: find legal move whose idx matches
            best_move = legal[0]
            best_p = -1.0
            for m in legal:
                idx = engine.move_to_policy_idx(m)
                if idx is not None and mcts_pi[idx] > best_p:
                    best_p = mcts_pi[idx]
                    best_move = m
        else:
            # Sample
            probs = np.array([mcts_pi[engine.move_to_policy_idx(m) or 0] for m in legal], dtype=np.float64)
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
            else:
                probs = np.ones(len(legal)) / len(legal)
            best_move = np.random.choice(legal, p=probs)

        engine.make_move(best_move)
        move_num += 1

    # Game over: get result (from white's perspective)
    result = engine.game_result()  # +1 white, -1 black, 0 draw

    # Convert to samples: value target is from the perspective of the side that played
    samples: List[Sample] = []
    for board, policy, side in history:
        value_target = result if side == 0 else -result
        samples.append((board, policy, value_target))

    return samples


# ── Training step ─────────────────────────────────────────────────────────────

def train_step(
    net: CapablancaNet,
    optimizer: torch.optim.Optimizer,
    batch: tuple,
    device: torch.device,
    cfg: Config,
) -> Tuple[float, float, float]:
    """
    One gradient step. Returns (policy_loss, value_loss, total_loss).
    """
    boards, policies, values = batch
    boards   = boards.to(device)
    policies = policies.to(device)
    values   = values.to(device)

    net.train()
    policy_logits, pred_values = net(boards)

    # Policy loss: cross-entropy with MCTS visit distribution as soft target
    # = -sum(pi * log_softmax(logits))
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policies * log_probs).sum(dim=1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(pred_values, values)

    # L2 regularization is handled by weight_decay in optimizer
    total_loss = policy_loss + value_loss

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), cfg.gradient_clip)
    optimizer.step()

    return policy_loss.item(), value_loss.item(), total_loss.item()


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = torch.device(cfg.device)
    print(f"Training on device: {device}")

    # Network + optimizer
    net = CapablancaNet(
        num_channels=cfg.num_channels,
        num_res_blocks=cfg.num_res_blocks,
    ).to(device)

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    )

    buffer = ReplayBuffer(cfg.replay_buffer)

    # Load checkpoint if available
    start_iter = 0
    latest = _latest_checkpoint(cfg.checkpoint_dir)
    if latest:
        print(f"Resuming from {latest}")
        ckpt = torch.load(latest, map_location=device)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iteration", 0) + 1
        print(f"  → starting at iteration {start_iter}")

    print(f"\nNetwork parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(f"Policy size: {POLICY_SIZE}\n")

    for iteration in range(start_iter, 100_000):
        t0 = time.time()

        # ── 1. Self-play ────────────────────────────────────────────────────
        print(f"[Iter {iteration}] Self-play ({cfg.games_per_iter} games)...")
        new_samples = []
        for g in range(cfg.games_per_iter):
            samples = self_play_game(net, cfg)
            new_samples.extend(samples)
            if (g + 1) % 10 == 0:
                print(f"  Game {g+1}/{cfg.games_per_iter}, {len(samples)} positions")

        buffer.push(new_samples)
        sp_time = time.time() - t0
        print(f"  Buffer size: {len(buffer):,}  |  Self-play: {sp_time:.1f}s")

        # ── 2. Train ─────────────────────────────────────────────────────────
        if len(buffer) < cfg.min_buffer_size:
            print(f"  Waiting for buffer to fill ({len(buffer)}/{cfg.min_buffer_size})")
            continue

        print(f"[Iter {iteration}] Training ({cfg.train_steps} steps)...")
        t1 = time.time()
        total_pl = total_vl = total_tl = 0.0

        dataset = ChessDataset(buffer.sample(min(len(buffer), cfg.batch_size * cfg.train_steps)))
        loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                             num_workers=0, pin_memory=(cfg.device == "cuda"))

        steps = 0
        for batch in loader:
            if steps >= cfg.train_steps:
                break

            states, target_pis, target_vs = batch
            states = states.to(device)
            target_pis = target_pis.to(device)
            target_vs = target_vs.to(device)

            optimizer.zero_grad()

            # --- BFLOAT16 TRAINING ---
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                policy_logits, value = net(states)

                loss_p = F.cross_entropy(policy_logits, target_pis)
                loss_v = F.mse_loss(value, target_vs)
                total_loss = loss_p + loss_v

            total_loss.backward()
            optimizer.step()

            total_pl += loss_p.item()
            total_vl += loss_v.item()
            total_tl += total_loss.item()
            steps += 1

        scheduler.step()

        train_time = time.time() - t1
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"  policy_loss={total_pl/steps:.4f}  "
            f"value_loss={total_vl/steps:.4f}  "
            f"total={total_tl/steps:.4f}  "
            f"lr={lr:.6f}  "
            f"time={train_time:.1f}s"
        )

        # ── 3. Checkpoint ─────────────────────────────────────────────────────
        if iteration % cfg.save_every == 0:
            path = os.path.join(cfg.checkpoint_dir, f"model_iter{iteration:05d}.pth")
            torch.save({
                "iteration": iteration,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "buffer_size": len(buffer),
            }, path)
            print(f"  Saved checkpoint: {path}")

        print(f"  Total iteration time: {time.time()-t0:.1f}s\n")


def _latest_checkpoint(directory: str) -> str | None:
    if not os.path.isdir(directory):
        return None
    checkpoints = sorted(
        (f for f in os.listdir(directory) if f.endswith(".pth")),
        reverse=True,
    )
    return os.path.join(directory, checkpoints[0]) if checkpoints else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--channels",    type=int,   default=128)
    parser.add_argument("--res-blocks",  type=int,   default=10)
    parser.add_argument("--simulations", type=int,   default=400)
    parser.add_argument("--games",       type=int,   default=100)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--train-steps", type=int,   default=1000)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--device",      type=str,   default=None)
    args = parser.parse_args()

    cfg = Config(
        num_channels=args.channels,
        num_res_blocks=args.res_blocks,
        simulations=args.simulations,
        games_per_iter=args.games,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        learning_rate=args.lr,
    )
    if args.device:
        cfg.device = args.device

    train(cfg)
