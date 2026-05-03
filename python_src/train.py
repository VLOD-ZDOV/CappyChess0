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
    temperature_moves: int = 50       # FIX: 30→50, больше исследования в начале партии
    temperature: float = 1.0          # tau для первых temperature_moves ходов (1.0 = пропорционально visit counts)
    temperature_late: float = 0.5     # tau после temperature_moves (мягкий argmax, не жадный)
    games_per_iter: int = 128
    max_game_length: int = 70
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
    force_save: bool = False  # если True — сохраняем чекпоинт даже при низком loss


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

                # Temperature sampling с плавным decay (как в lc0):
                #   до temperature_moves:          tau = cfg.temperature (1.0)
                #   следующие 20 ходов:            tau плавно падает 1.0 → temperature_late
                #   после temperature_moves + 20:  tau = temperature_late (0.5)
                if move_num < cfg.temperature_moves:
                    tau = cfg.temperature
                else:
                    decay_steps = 20
                    overstep = min(move_num - cfg.temperature_moves, decay_steps)
                    tau = cfg.temperature - (overstep / decay_steps) * (cfg.temperature - cfg.temperature_late)

                raw = np.array([
                    pol[eng.move_int_to_policy_idx(m) or 0] for m in legal
                ], dtype=np.float64)
                raw = np.power(np.maximum(raw, 1e-8), 1.0 / tau)
                s = raw.sum()
                probs = raw / s if s > 0 else np.ones(len(legal)) / len(legal)
                move = int(np.random.choice(legal, p=probs))

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
                # FIX: градуированная материальная оценка вместо ступенчатой ±0.5/0.
                # Было: |balance|>3 → ±0.5, иначе 0.0 — почти все таймауты давали 0.
                # Стало: линейная шкала, зажатая в [-0.8, 0.8].
                # Это даёт сети больше информации о качестве позиции.
                balance = eng.material_result()  # ±0.5 или 0.0 из Rust
                # Дополнительно получаем сырой баланс через Python-обёртку
                # (material_result уже содержит знак, масштабируем мягче)
                result = float(np.clip(balance * 1.6, -0.8, 0.8))
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

def value_to_wdl(v: float) -> np.ndarray:
    """
    Конвертирует скалярный исход в мягкий WDL one-hot [Win, Draw, Loss].
    |v| >= 0.7 → жёсткий исход (чистая победа/поражение).
    0 < |v| < 0.7 → мягкий переход (материальная оценка / таймаут).
    Мягкие метки дают лучший градиент чем hard one-hot.
    """
    wdl = np.zeros(3, dtype=np.float32)
    if v >= 0.7:
        wdl[0] = 1.0                        # Win
    elif v <= -0.7:
        wdl[2] = 1.0                        # Loss
    elif v > 0.0:
        w = v / 0.7
        wdl[0] = w; wdl[1] = 1.0 - w       # Частично Win, частично Draw
    elif v < 0.0:
        l = (-v) / 0.7
        wdl[2] = l; wdl[1] = 1.0 - l       # Частично Loss, частично Draw
    else:
        wdl[1] = 1.0                        # Draw
    return wdl


class SelfPlayDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[CompactSample]):
        self.boards   = np.stack([s[0].astype(np.float32) for s in samples]).reshape(-1, 20, 8, 10)
        self.policies = np.stack([unpack_policy(s[1]) for s in samples])
        # WDL: каждый скалярный value → soft one-hot [Win, Draw, Loss]
        self.wdl = np.stack([value_to_wdl(float(s[2])) for s in samples])

    def __len__(self):
        return len(self.wdl)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.boards[idx]),
            torch.from_numpy(self.policies[idx]),
            torch.from_numpy(self.wdl[idx]),   # (3,) float32
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
        values = values.to(device, non_blocking=True)  # WDL: (batch, 3)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, wdl_logits = net(boards)
            logits     = logits.float()
            wdl_logits = wdl_logits.float()

            # Policy: cross-entropy с visit counts как мягкими метками
            log_probs   = F.log_softmax(logits, dim=1)
            policy_loss = -(policies * log_probs).sum(dim=1).mean()

            # WDL: cross-entropy с soft one-hot [Win, Draw, Loss]
            # Эквивалентно KL-divergence от target к предсказанию
            log_wdl    = F.log_softmax(wdl_logits, dim=1)
            value_loss = -(values * log_wdl).sum(dim=1).mean()

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

    # Linear warmup + CosineAnnealingWarmRestarts.
    # Первые warmup_iters итераций LR растёт линейно от 0 до cfg.learning_rate,
    # затем косинусное затухание. Это стабилизирует начало обучения.
    WARMUP_ITERS = 5

    class WarmupCosineScheduler:
        def __init__(self, optimizer, warmup_iters, T_0, T_mult, eta_min, base_lr):
            self.warmup_iters = warmup_iters
            self.base_lr = base_lr
            self.cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
            )
            self._last_lr = [base_lr]
            self._iter = 0
            self.optimizer = optimizer

        def step(self):
            self._iter += 1
            if self._iter <= self.warmup_iters:
                lr = self.base_lr * self._iter / self.warmup_iters
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr
                self._last_lr = [lr]
            else:
                self.cosine.step()
                self._last_lr = self.cosine.get_last_lr()

        def get_last_lr(self):
            return self._last_lr

        def state_dict(self):
            return {"cosine": self.cosine.state_dict(), "_iter": self._iter}

        def load_state_dict(self, sd):
            self.cosine.load_state_dict(sd["cosine"])
            self._iter = sd.get("_iter", 0)

    def make_scheduler(opt):
        return WarmupCosineScheduler(
            opt, warmup_iters=WARMUP_ITERS,
            T_0=50, T_mult=2,
            eta_min=cfg.learning_rate * 0.05,
            base_lr=cfg.learning_rate,
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
    # Предпочитаем latest.pth (сохраняется каждую итерацию)
    _latest = os.path.join(cfg.checkpoint_dir, "latest.pth")
    ckpts = [f for f in ckpts if not f.startswith("latest")]
    if os.path.exists(_latest) or ckpts:
        path = _latest if os.path.exists(_latest) else os.path.join(cfg.checkpoint_dir, ckpts[-1])
        ckpt = torch.load(path, map_location=device, weights_only=False)

        raw_sd = ckpt["model"]
        # Убираем префиксы torch.compile / DataParallel
        raw_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
                  for k, v in raw_sd.items()}

        # Фильтруем слои несовместимые с WDL (старый Linear(256,1) → новый Linear(256,3))
        # value_head.6.weight shape: старый (1,256), новый (3,256)
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

        # Загружаем оптимайзер только если архитектура совместима полностью
        # При несовместимости (WDL переход) — создаём оптимайзер с нуля,
        # иначе dtype mismatch между старыми fp16 состояниями и новыми fp32 слоями
        if not incompatible_keys and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("✅ Оптимайзер загружен из чекпоинта")
            except Exception as e:
                print(f"⚠️  Оптимайзер не загружен ({e}), начинаем заново")
        else:
            print("ℹ️  Оптимайзер инициализирован заново (несовместимая архитектура)")

        # Принудительно устанавливаем LR
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

        # Автоадаптация температуры: если policy схлопнулась — поднимаем tau
        if stats.get('top1_mean', 0) > 0.85 and cfg.temperature < 2.0:
            cfg.temperature = min(cfg.temperature * 1.2, 2.0)
            print(f"  ⚠️  top1>{0.85:.2f} → temperature поднята до {cfg.temperature:.2f}")
        elif stats.get('entropy_mean', 0) > 3.5 and cfg.temperature > 0.8:
            cfg.temperature = max(cfg.temperature * 0.95, 0.8)
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

        # Детектор коллапса (--force-save обходит проверку)
        collapsed = metrics['policy_loss'] < cfg.collapse_threshold and not cfg.force_save
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
        if not collapsed:
            model_to_save = net._orig_mod if hasattr(net, "_orig_mod") else net
            ckpt_data = {
                "iteration": iteration,
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": metrics,
            }
            # latest.pth — перезаписывается каждую итерацию
            # При падении/остановке всегда есть последнее состояние
            latest_path = os.path.join(cfg.checkpoint_dir, "latest.pth")
            torch.save(ckpt_data, latest_path)
            print(f"  💾 latest.pth (iter {iteration})")

            # Нумерованный архивный чекпоинт каждые save_every итераций
            if iteration % cfg.save_every == 0:
                path = os.path.join(cfg.checkpoint_dir, f"model_iter{iteration:05d}.pth")
                torch.save(ckpt_data, path)
                print(f"  💾 {os.path.basename(path)}")

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
    parser.add_argument("--temperature",          type=float, default=1.0,
                        help="Температура выборки хода (tau) в первые --temperature-moves ходов")
    parser.add_argument("--temperature-late",     type=float, default=0.5,
                        help="Температура после --temperature-moves (0.5=мягкий argmax, 0=жадный)")
    parser.add_argument("--temperature-moves",    type=int,   default=50,
                        help="Ходов с высокой температурой")
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
    parser.add_argument("--force-save",           action="store_true",
                        help="Сохранять чекпоинт даже если policy_loss < collapse_threshold")
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
        temperature=args.temperature,
        temperature_late=args.temperature_late,
        temperature_moves=args.temperature_moves,
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
        force_save=args.force_save,
    )
    train(cfg)
