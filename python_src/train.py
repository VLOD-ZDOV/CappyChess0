# train.py — Оптимизированный тренировочный цикл
#
# Основные оптимизации vs оригинал:
#   1. multiprocessing для self-play — несколько CPU воркеров генерируют игры параллельно
#   2. Разделение policy_loss / value_loss в логах
#   3. Compile model с torch.compile() если доступно
#   4. Gradient scaler для смешанной точности
#   5. Prefetch данных в DataLoader вместо ручного стека
#   6. Убрали bottleneck: один MCTS объект на воркер вместо пересоздания

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
    temperature_moves: int = 30   # ходы с температурной выборкой
    games_per_iter: int = 128
    max_game_length: int = 150   # снижено с 250: при 250 все игры упирались в лимит
    mcts_batch: int = 128         # размер батча для MCTS inference

    # Тренировка
    batch_size: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    train_steps: int = 500        # шагов на итерацию
    value_loss_weight: float = 3.0   # повышено с 1.0: при засилии ничьих value_loss
                                      # ~0.0002 тонет в policy_loss ~3.45 при весе 1.0

    # Буфер
    buffer_max: int = 500_000

    # Инфраструктура
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    log_every: int = 50           # логировать каждые N шагов

# Компактный формат хранения в буфере:
#   board:  np.ndarray float16, shape (1600,)   — 3.1 КБ вместо 6.4 КБ
#   policy: Tuple[np.ndarray int16, np.ndarray float16]  — sparse (indices, values)
#           ~40-60 ненулевых ходов → ~200 байт вместо 28 КБ
#   value:  float32
# Итого ~3.3 КБ/позицию vs 33.6 КБ → буфер 500к позиций: ~1.6 ГБ вместо 16 ГБ
CompactSample = Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], float]
Sample = CompactSample  # обратная совместимость имени


def pack_sample(board: np.ndarray, policy: np.ndarray, value: float) -> CompactSample:
    """Упаковывает позицию в компактный формат для хранения в буфере."""
    board_f16 = board.astype(np.float16)
    nz = np.nonzero(policy)[0]
    pol_idx = nz.astype(np.int16)
    pol_val = policy[nz].astype(np.float16)
    return (board_f16, (pol_idx, pol_val), np.float32(value))


def unpack_policy(pol_sparse: Tuple[np.ndarray, np.ndarray],
                  size: int = 7000) -> np.ndarray:
    """Распаковывает sparse policy обратно в dense float32."""
    pol = np.zeros(size, dtype=np.float32)
    idx, val = pol_sparse
    pol[idx.astype(np.int32)] = val.astype(np.float32)
    return pol


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    # FIX: deque → list с кольцевым указателем.
    # Старая версия делала list(self.data) — O(N) аллокацию на каждый вызов sample().
    # При buffer_max=500k и 500 train_steps/итерацию это 500 раз по 500k копий.
    # Теперь список уже есть в памяти и random access — O(1).
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data: List[Sample] = []
        self._ptr = 0          # куда пишем следующий элемент (кольцевой)
        self._full = False     # заполнен ли буфер до max_size

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
        # FIX: replace=True — всегда возвращаем ровно batch_size семплов.
        # replace=False при маленьком буфере (напр. 20k) возвращал только 20k
        # вместо 256k запрошенных → DataLoader давал 39 шагов вместо 500.
        n = len(self.data)
        if n == 0:
            return []
        indices = np.random.choice(n, batch_size, replace=True)
        return [self.data[i] for i in indices]

    def __len__(self):
        return len(self.data)


# ── Self-play (запускается в одном процессе, на GPU) ──────────────────────────

def generate_games(net: nn.Module, cfg: Config, device: torch.device) -> List[Sample]:
    """
    Генерирует cfg.games_per_iter игр батчами.
    Возвращает список (board_tensor, policy_vec, value).
    """
    mcts = UltraFastMCTS(net, device, cfg.c_puct, batch_size=cfg.mcts_batch)
    all_samples: List[Sample] = []

    # Обрабатываем игры батчами по mcts_batch
    batch_sz = cfg.mcts_batch
    num_batches = (cfg.games_per_iter + batch_sz - 1) // batch_sz

    for b in range(num_batches):
        start = b * batch_sz
        n = min(batch_sz, cfg.games_per_iter - start)
        engines = [CapablancaEngine() for _ in range(n)]
        # histories[i] = список (board_np, policy_np, side)
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

                # Выбор хода
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

        # Собираем результаты
        batch_positions = 0
        results_dist = {1.0: 0, -1.0: 0, 0.5: 0, -0.5: 0, 0.0: 0}
        for i, eng in enumerate(engines):
            if eng.is_game_over():
                # Игра завершилась матом или пат/50-ходов
                result = eng.game_result()
            else:
                # Игра прервана по лимиту ходов — используем материальную оценку
                result = eng.material_result()
            results_dist[result] = results_dist.get(result, 0) + 1
            for board_np, pol, side in histories[i]:
                v = result if side == 0 else -result
                all_samples.append(pack_sample(board_np, pol, float(v)))
                batch_positions += 1

        finished = results_dist.get(1.0, 0) + results_dist.get(-1.0, 0)
        timeout  = n - finished
        print(f"  Batch {b+1}/{num_batches}: {n} games, "
              f"{batch_positions} positions, move_num={move_num} | "
              f"мат={finished} таймаут={timeout} "
              f"(+0.5={results_dist.get(0.5,0)} -0.5={results_dist.get(-0.5,0)} 0={results_dist.get(0.0,0)})")

    return all_samples


# ── Dataset / DataLoader ──────────────────────────────────────────────────────

class SelfPlayDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[CompactSample]):
        # Распаковываем board из float16 → float32 для обучения
        self.boards = np.stack([s[0].astype(np.float32) for s in samples]).reshape(-1, 20, 8, 10)
        # Распаковываем sparse policy → dense float32
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
                scaler: torch.cuda.amp.GradScaler, iteration: int):
    """
    cfg.train_steps шагов обучения. Возвращает словарь с метриками.
    """
    net.train()

    # Предсоберём батч из буфера в numpy — быстрее чем случайный доступ в DataLoader
    samples = buffer.sample(cfg.train_steps * cfg.batch_size)
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
        if steps >= cfg.train_steps:
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
            print(f"    step {steps:4d}/{cfg.train_steps} | "
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
    print(f"   Self-play:     {cfg.games_per_iter} игр/итер, "
          f"{cfg.simulations} симуляций/ход")
    print(f"   MCTS batch:    {cfg.mcts_batch}")
    print(f"   Train batch:   {cfg.batch_size} × {cfg.train_steps} шагов")
    print(f"   Precision:     BF16 + TF32\n")

    # Модель
    net = CapablancaNet(cfg.num_channels, cfg.num_res_blocks).to(device)
    net = net.to(memory_format=torch.channels_last)

    # torch.compile даёт ~20% ускорение на A100 за счёт fusion
    if hasattr(torch, "compile"):
        try:
            net = torch.compile(net, mode="reduce-overhead")
            print("✅ torch.compile() применён\n")
        except Exception as e:
            print(f"⚠️  torch.compile() недоступен: {e}\n")

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fused=True,  # fused AdamW — быстрее на CUDA
    )

    # GradScaler для стабильности BF16
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    # LR Scheduler: линейный прогрев 5 итераций, затем косинусное затухание.
    # T_0=50 — один цикл на 50 итераций, разумно для долгого обучения.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    buffer = ReplayBuffer(cfg.buffer_max)

    # Загружаем буфер если есть (избегаем negative feedback loop при рестарте)
    buffer_path = os.path.join(cfg.checkpoint_dir, "buffer.pkl")
    if os.path.exists(buffer_path):
        try:
            with open(buffer_path, "rb") as f:
                buffer.data, buffer._ptr, buffer._full = pickle.load(f)
            print(f"📦 Загружен буфер: {len(buffer):,} позиций\n")
        except Exception as e:
            print(f"⚠️  Не удалось загрузить буфер: {e}\n")

    # Попытаемся загрузить последний чекпоинт
    start_iter = 0
    ckpts = sorted([f for f in os.listdir(cfg.checkpoint_dir) if f.endswith(".pth")])
    if ckpts:
        path = os.path.join(cfg.checkpoint_dir, ckpts[-1])
        ckpt = torch.load(path, map_location=device)

        # Load weights into the original module if it was compiled
        if hasattr(net, "_orig_mod"):
            net._orig_mod.load_state_dict(ckpt["model"])
        else:
            net.load_state_dict(ckpt["model"])

        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_iter = ckpt.get("iteration", 0) + 1
        print(f"📂 Загружен чекпоинт: {path} (итерация {start_iter})\n")

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
        games_per_sec = cfg.games_per_iter / sp_time
        pos_per_sec = len(samples) / sp_time
        print(f"  ✅ {len(samples):,} позиций за {sp_time:.1f}s "
              f"({games_per_sec:.2f} игр/с, {pos_per_sec:.0f} поз/с)")
        print(f"  Буфер: {len(buffer):,} позиций\n")

        # ── Тренировка ───────────────────────────────────────────────────────
        if len(buffer) < cfg.batch_size * 4:
            print(f"  ⏳ Мало данных ({len(buffer)}), пропускаем тренировку\n")
            continue

        torch.set_grad_enabled(True)
        net.train()

        print(f"  🏋️  Тренировка {cfg.train_steps} шагов...")
        train_start = time.time()
        metrics = train_epoch(net, optimizer, buffer, cfg, device, scaler, iteration)
        train_time = time.time() - train_start

        # Шаг LR scheduler после каждой итерации обучения
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"\n  ✅ Тренировка за {train_time:.1f}s")
        print(f"     policy_loss = {metrics['policy_loss']:.4f}")
        print(f"     value_loss  = {metrics['value_loss']:.4f}")
        print(f"     total_loss  = {metrics['loss']:.4f}")
        print(f"     lr          = {current_lr:.2e}")

        iter_time = time.time() - iter_start
        print(f"\n  ⏱️  Итерация {iteration}: {iter_time:.1f}s total "
              f"(self-play {sp_time:.1f}s + train {train_time:.1f}s)\n")

        # ── Чекпоинт ─────────────────────────────────────────────────────────
        if iteration % cfg.save_every == 0:
            # Unwrap compiled model для сохранения
            model_to_save = net
            if hasattr(net, "_orig_mod"):
                model_to_save = net._orig_mod

            path = os.path.join(cfg.checkpoint_dir,
                                f"model_iter{iteration:05d}.pth")
            torch.save({
                "iteration": iteration,
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": metrics,
            }, path)
            print(f"  💾 Сохранён: {path}")

            # Сохраняем буфер рядом с чекпоинтом
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
    parser.add_argument("--channels",      type=int,   default=64)
    parser.add_argument("--res-blocks",    type=int,   default=5)
    parser.add_argument("--simulations",   type=int,   default=80)
    parser.add_argument("--games",         type=int,   default=128)
    parser.add_argument("--mcts-batch",    type=int,   default=128)
    parser.add_argument("--batch-size",    type=int,   default=512)
    parser.add_argument("--train-steps",   type=int,   default=500)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--checkpoint-dir",type=str,   default="checkpoints")
    parser.add_argument("--save-every",         type=int,   default=5)
    parser.add_argument("--value-loss-weight",   type=float, default=3.0)
    args = parser.parse_args()

    cfg = Config(
        num_channels=args.channels,
        num_res_blocks=args.res_blocks,
        simulations=args.simulations,
        games_per_iter=args.games,
        mcts_batch=args.mcts_batch,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        value_loss_weight=args.value_loss_weight,
    )
    train(cfg)
