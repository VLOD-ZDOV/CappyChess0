# train.py — Оптимизированный тренировочный цикл
#
# FIX v3:
#   - Детектор коллапса: policy_loss < threshold → предупреждение, чекпоинт не сохраняется
#   - Диагностика diversity после каждого self-play (entropy, top1, value_std)
#   - --reset-scheduler: пересоздаёт scheduler при загрузке чекпоинта (нужно при смене --lr)
#   - --reset-buffer: очищает replay buffer при старте
#   - Правильный подсчёт исходов (белые/чёрные/пат/таймаут)
#   - Фикс deprecation: torch.amp.GradScaler вместо torch.cuda.amp.GradScaler
#   - LR принудительно устанавливается в param_groups после загрузки оптимайзера
#   - train_steps ограничен 1 эпохой буфера
#   - value_loss_weight=1.0, train_steps default=200

import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Tuple

import queue

from model import CapablancaNet
from mcts import UltraFastMCTS, POLICY_SIZE

try:
    from capablanca_engine import CapablancaEngine
except ImportError:
    raise ImportError("capablanca_engine not found. Build with: maturin develop --release")

# NVIDIA оптимизации
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False

# ── Конфигурация ──────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Модель
    num_channels: int = 64
    num_res_blocks: int = 5

    # Self-play
    simulations: int = 80
    c_puct: float = 1.25
    temperature_moves: int = 30
    games_per_iter: int = 128
    max_game_length: int = 150
    mcts_batch: int = 128
    mcts_parallel_sims: int = 32  # листьев за шаг MCTS (больше = реже round-trip Python↔GPU)

    # Тренировка
    batch_size: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    train_steps: int = 200
    min_train_steps: int = 20
    value_loss_weight: float = 1.0

    # Буфер
    buffer_max: int = 500_000
    buffer_min_to_train: int = 10_000

    # Инфраструктура
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    log_every: int = 50

    # При смене --lr при рестарте передай --reset-scheduler чтобы
    # scheduler начал новый цикл, а не продолжил с середины старого
    reset_scheduler: bool = False

    # policy_loss ниже этого порога = коллапс — чекпоинт не сохраняется
    collapse_threshold: float = 0.01


CompactSample = Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], float]
Sample = CompactSample


def pack_sample(board: np.ndarray, policy: np.ndarray, value: float) -> CompactSample:
    board_f16 = board.astype(np.float16)
    nz = np.nonzero(policy)[0]
    pol_idx = nz.astype(np.int16)
    pol_val = policy[nz].astype(np.float16)
    return (board_f16, (pol_idx, pol_val), np.float32(value))


def unpack_policy(pol_sparse: Tuple[np.ndarray, np.ndarray],
                  size: int = 7000) -> np.ndarray:
    pol = np.zeros(size, dtype=np.float32)
    idx, val = pol_sparse
    pol[idx.astype(np.int32)] = val.astype(np.float32)
    return pol


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data: List[Sample] = []
        self._ptr = 0
        self._full = False

    def push(self, samples: List[Sample]):
        for s in samples:
            if not self._full:
                self.data.append(s)
                if len(self.data) == self.max_size:
                    self._full = True
            else:
                self.data[self._ptr] = s
            self._ptr = (self._ptr + 1) % self.max_size

    def sample(self, batch_size: int) -> List[Sample]:
        n = len(self.data)
        if n == 0:
            return []
        indices = np.random.choice(n, batch_size, replace=True)
        return [self.data[i] for i in indices]

    def __len__(self):
        return len(self.data)


# ── Диагностика разнообразия политик ─────────────────────────────────────────

def policy_diversity_stats(samples: List[Sample], n: int = 200) -> dict:
    """
    Считает метрики разнообразия по случайной выборке.
    entropy_mean  — средняя энтропия policy (норма ~1.5-4.0, коллапс < 0.3)
    top1_mean     — средняя вероятность лучшего хода (коллапс > 0.95)
    nonzero_mean  — среднее число ненулевых ходов
    value_std     — стандартное отклонение value (коллапс < 0.05)
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

def generate_games(net: nn.Module, cfg: Config, device: torch.device) -> List[Sample]:
    mcts = UltraFastMCTS(net, device, cfg.c_puct, batch_size=cfg.mcts_batch, parallel_sims=cfg.mcts_parallel_sims)
    all_samples: List[Sample] = []

    batch_sz = cfg.mcts_batch
    num_batches = (cfg.games_per_iter + batch_sz - 1) // batch_sz

    for b in range(num_batches):
        start = b * batch_sz
        n = min(batch_sz, cfg.games_per_iter - start)
        engines = [CapablancaEngine() for _ in range(n)]
        histories: List[List] = [[] for _ in range(n)]

        active = list(range(n))
        move_num = 0

        while active and move_num < cfg.max_game_length:
            cur_engines = [engines[i] for i in active]
            policies = mcts.search_games(cur_engines, cfg.simulations)

            new_active = []
            for j, game_idx in enumerate(active):
                eng = engines[game_idx]
                legal = eng.get_legal_moves_int()
                if not legal:
                    continue

                board_np = np.array(eng.get_board_tensor(), dtype=np.float32)
                side = eng.side_to_move()
                pol = policies[j]
                histories[game_idx].append((board_np, pol.copy(), side))

                if move_num < cfg.temperature_moves:
                    raw = np.array([
                        pol[eng.move_int_to_policy_idx(m) or 0] for m in legal
                    ], dtype=np.float64)
                    s = raw.sum()
                    probs = raw / s if s > 0 else np.ones(len(legal)) / len(legal)
                    move = int(np.random.choice(legal, p=probs))
                else:
                    move = max(legal,
                               key=lambda m: pol[eng.move_int_to_policy_idx(m) or 0])
                    move = int(move)

                eng.make_move_int(move)

                if not eng.is_game_over():
                    new_active.append(game_idx)

            active = new_active
            move_num += 1

        batch_positions = 0
        white_wins = black_wins = draws = timeouts = 0

        for i, eng in enumerate(engines):
            if eng.is_game_over():
                result = eng.game_result()
                if result == 1.0:
                    white_wins += 1
                elif result == -1.0:
                    black_wins += 1
                else:
                    draws += 1
            else:
                result = eng.material_result()
                timeouts += 1

            for board_np, pol, side in histories[i]:
                v = result if side == 0 else -result
                all_samples.append(pack_sample(board_np, pol, float(v)))
                batch_positions += 1

        print(f"  Batch {b+1}/{num_batches}: {n} games, "
              f"{batch_positions} positions, {move_num} ходов | "
              f"бел={white_wins} чёрн={black_wins} пат/50={draws} timeout={timeouts}")

    return all_samples


# ── Dataset ───────────────────────────────────────────────────────────────────

class SelfPlayDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[CompactSample]):
        self.boards = np.stack([s[0].astype(np.float32) for s in samples]).reshape(-1, 20, 8, 10)
        self.policies = np.stack([unpack_policy(s[1]) for s in samples])
        self.values = np.array([s[2] for s in samples], dtype=np.float32)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.boards[idx]),
            torch.from_numpy(self.policies[idx]),
            torch.tensor(self.values[idx]),
        )


# ── Тренировочный шаг ─────────────────────────────────────────────────────────

def train_epoch(net: nn.Module, optimizer: torch.optim.Optimizer,
                buffer: ReplayBuffer, cfg: Config, device: torch.device,
                scaler: torch.amp.GradScaler, iteration: int):
    net.train()

    max_steps_by_buffer = len(buffer) // cfg.batch_size
    effective_steps = max(cfg.min_train_steps,
                          min(cfg.train_steps, max_steps_by_buffer))

    if max_steps_by_buffer < cfg.train_steps:
        print(f"  ℹ️  Буфер {len(buffer):,} поз → {effective_steps} шагов "
              f"(ограничено 1 эпохой, потолок {cfg.train_steps})")

    samples = buffer.sample(effective_steps * cfg.batch_size)
    dataset = SelfPlayDataset(samples)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    steps = 0

    for boards, policies, values in loader:
        if steps >= effective_steps:
            break

        boards = boards.to(device, non_blocking=True,
                           memory_format=torch.channels_last)
        policies = policies.to(device, non_blocking=True)
        values = values.unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, preds = net(boards)
            logits = logits.float()
            preds = preds.float()

            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(policies * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(preds, values)
            loss = policy_loss + cfg.value_loss_weight * value_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        steps += 1

        if steps % cfg.log_every == 0:
            avg_p = total_policy_loss / steps
            avg_v = total_value_loss / steps
            avg_t = total_loss / steps
            print(f"    step {steps:4d}/{effective_steps} | "
                  f"policy_loss={avg_p:.4f}  value_loss={avg_v:.4f}  "
                  f"total={avg_t:.4f}")

    n = max(steps, 1)
    return {
        "loss": total_loss / n,
        "policy_loss": total_policy_loss / n,
        "value_loss": total_value_loss / n,
        "steps": steps,
    }


# ── Главный цикл ──────────────────────────────────────────────────────────────

def train(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
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

    net = CapablancaNet(cfg.num_channels, cfg.num_res_blocks).to(device)
    net = net.to(memory_format=torch.channels_last)

    if hasattr(torch, "compile"):
        try:
            net = torch.compile(net)  # FIX: убран mode="reduce-overhead" — несовместим с динамическим батчем MCTS (разный N листьев каждый шаг → CUDA-граф падает → NaN)
            print("✅ torch.compile() применён\n")
        except Exception as e:
            print(f"⚠️  torch.compile() недоступен: {e}\n")

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fused=True,
    )

    # FIX: новый API без deprecation warning
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    def make_scheduler(opt):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=50, T_mult=2, eta_min=cfg.learning_rate * 0.05
        )

    scheduler = make_scheduler(optimizer)
    buffer = ReplayBuffer(cfg.buffer_max)

    buffer_path = os.path.join(cfg.checkpoint_dir, "buffer.pkl")
    if os.path.exists(buffer_path):
        try:
            with open(buffer_path, "rb") as f:
                buffer.data, buffer._ptr, buffer._full = pickle.load(f)
            print(f"📦 Загружен буфер: {len(buffer):,} позиций\n")
        except Exception as e:
            print(f"⚠️  Не удалось загрузить буфер: {e}\n")

    start_iter = 0
    ckpts = sorted([f for f in os.listdir(cfg.checkpoint_dir) if f.endswith(".pth")])
    if ckpts:
        path = os.path.join(cfg.checkpoint_dir, ckpts[-1])
        ckpt = torch.load(path, map_location=device, weights_only=False)

        if hasattr(net, "_orig_mod"):
            net._orig_mod.load_state_dict(ckpt["model"])
        else:
            net.load_state_dict(ckpt["model"])

        optimizer.load_state_dict(ckpt["optimizer"])

        # FIX: принудительно устанавливаем LR из аргументов — загруженный
        # state_dict оптимайзера может содержать старый LR от предыдущего запуска
        for pg in optimizer.param_groups:
            pg['lr'] = cfg.learning_rate

        if cfg.reset_scheduler or "scheduler" not in ckpt:
            # Пересоздаём scheduler — старый цикл сброшен, LR стартует заново
            print("🔄 Scheduler сброшен (начинается новый косинусный цикл)\n")
        else:
            scheduler.load_state_dict(ckpt["scheduler"])

        start_iter = ckpt.get("iteration", 0) + 1
        print(f"📂 Загружен чекпоинт: {path} (итерация {start_iter})")

        # Диагностика буфера при старте — сразу видно если он скомпрометирован
        if len(buffer) > 0:
            stats = policy_diversity_stats(buffer.data)
            print_diversity(stats, prefix="   Буфер diversity")
            if stats.get('value_std', 1.0) < 0.05:
                print("   ⚠️  value_std < 0.05 — почти все value одинаковые!")
                print("   ⚠️  Рассмотри перезапуск с --reset-buffer\n")
            else:
                print()

    for iteration in range(start_iter, 100_000):
        iter_start = time.time()

        # ── Self-play ────────────────────────────────────────────────────────
        print(f"[Iter {iteration}] ⚙️  Self-play: {cfg.games_per_iter} игр...")
        sp_start = time.time()
        net.eval()
        torch.set_grad_enabled(False)

        samples = generate_games(net, cfg, device)
        buffer.push(samples)

        sp_time = time.time() - sp_start
        print(f"  ✅ {len(samples):,} позиций за {sp_time:.1f}s "
              f"({cfg.games_per_iter / sp_time:.2f} игр/с, "
              f"{len(samples) / sp_time:.0f} поз/с)")
        print(f"  Буфер: {len(buffer):,} позиций")

        # Диагностика свежих данных
        stats = policy_diversity_stats(samples)
        print_diversity(stats)
        print()

        # ── Тренировка ───────────────────────────────────────────────────────
        if len(buffer) < cfg.buffer_min_to_train:
            print(f"  ⏳ Мало данных ({len(buffer):,} < {cfg.buffer_min_to_train:,}), "
                  f"пропускаем тренировку\n")
            continue

        torch.set_grad_enabled(True)
        net.train()

        print(f"  🏋️  Тренировка (до {cfg.train_steps} шагов, ≤1 эпохи)...")
        train_start = time.time()
        metrics = train_epoch(net, optimizer, buffer, cfg, device, scaler, iteration)
        train_time = time.time() - train_start

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Детектор коллапса
        collapsed = metrics['policy_loss'] < cfg.collapse_threshold
        collapse_warn = "  ⚠️  КОЛЛАПС ПОЛИТИКИ — чекпоинт не сохранён!" if collapsed else ""

        print(f"\n  ✅ Тренировка за {train_time:.1f}s ({metrics['steps']} шагов)")
        print(f"     policy_loss = {metrics['policy_loss']:.4f}{collapse_warn}")
        print(f"     value_loss  = {metrics['value_loss']:.4f}")
        print(f"     total_loss  = {metrics['loss']:.4f}")
        print(f"     lr          = {current_lr:.2e}")

        if collapsed:
            print(f"\n  ⚠️  Рекомендация: перезапустить с --reset-buffer --reset-scheduler\n")

        iter_time = time.time() - iter_start
        print(f"\n  ⏱️  Итерация {iteration}: {iter_time:.1f}s total "
              f"(self-play {sp_time:.1f}s + train {train_time:.1f}s)\n")

        # ── Чекпоинт ─────────────────────────────────────────────────────────
        if iteration % cfg.save_every == 0 and not collapsed:
            model_to_save = net._orig_mod if hasattr(net, "_orig_mod") else net
            path = os.path.join(cfg.checkpoint_dir, f"model_iter{iteration:05d}.pth")
            torch.save({
                "iteration": iteration,
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": metrics,
            }, path)
            print(f"  💾 Сохранён: {path}")

            try:
                with open(buffer_path, "wb") as f:
                    pickle.dump((buffer.data, buffer._ptr, buffer._full), f)
                print(f"  💾 Буфер сохранён ({len(buffer):,} позиций)\n")
            except Exception as e:
                print(f"  ⚠️  Не удалось сохранить буфер: {e}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capablanca Chess AlphaZero Training")
    parser.add_argument("--channels",           type=int,   default=64)
    parser.add_argument("--res-blocks",          type=int,   default=5)
    parser.add_argument("--simulations",         type=int,   default=80)
    parser.add_argument("--games",               type=int,   default=128)
    parser.add_argument("--mcts-batch",          type=int,   default=128)
    parser.add_argument("--mcts-parallel-sims", type=int, default=32,
                        help="Листьев за шаг MCTS. Больше = меньше round-trips GPU.")
    parser.add_argument("--batch-size",          type=int,   default=512)
    parser.add_argument("--train-steps",         type=int,   default=200)
    parser.add_argument("--min-train-steps",     type=int,   default=20)
    parser.add_argument("--buffer-min-to-train", type=int,   default=10_000)
    parser.add_argument("--lr",                  type=float, default=2e-4)
    parser.add_argument("--device",              type=str,   default="cuda")
    parser.add_argument("--checkpoint-dir",      type=str,   default="checkpoints")
    parser.add_argument("--save-every",          type=int,   default=5)
    parser.add_argument("--value-loss-weight",   type=float, default=1.0)
    parser.add_argument("--reset-scheduler",     action="store_true",
                        help="Пересоздать LR scheduler при загрузке чекпоинта")
    parser.add_argument("--reset-buffer",        action="store_true",
                        help="Очистить replay buffer при старте")
    parser.add_argument("--collapse-threshold",  type=float, default=0.01)
    args = parser.parse_args()

    # Сброс буфера если запрошен
    if args.reset_buffer:
        buffer_path = os.path.join(args.checkpoint_dir, "buffer.pkl")
        if os.path.exists(buffer_path):
            os.remove(buffer_path)
            print("🗑️  Буфер сброшен\n")

    cfg = Config(
        num_channels=args.channels,
        num_res_blocks=args.res_blocks,
        simulations=args.simulations,
        games_per_iter=args.games,
        mcts_batch=args.mcts_batch,
        mcts_parallel_sims=args.mcts_parallel_sims,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        min_train_steps=args.min_train_steps,
        buffer_min_to_train=args.buffer_min_to_train,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        value_loss_weight=args.value_loss_weight,
        reset_scheduler=args.reset_scheduler,
        collapse_threshold=args.collapse_threshold,
    )
    train(cfg)
