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
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Tuple


class ModelEMA:
    """Exponential Moving Average весов модели.

    AlphaZero использует EMA копию для self-play вместо текущих весов.
    Без этого свежие веса могут быть переобучены к последнему батчу,
    self-play выдаёт неконсистентные данные → нестабильное обучение.

    decay=0.999 → веса обновляются на 0.1% за каждый train step.
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


# ─────────────────────────────────────────────────────────────────────────────
# Fairy-Stockfish интеграция (опциональная, активируется через --fsf-path)
# ─────────────────────────────────────────────────────────────────────────────

def get_fsf_schedule(iteration: int, base_self_play: int):
    """Возвращает (self_games, fsf_games_this_iter, fsf_every) для данной итерации.

    Расписание:
      iter 0-29:  1152 self + 2688 FSF (каждые 3 итерации ~269 игр)
      iter 30-49: 2304 self + 1152 FSF (каждые 5 итераций ~288 игр)
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
    try:
        from capablanca_engine import CapablancaEngine
    except ImportError:
        return None
    for m in engine.get_legal_moves_int():
        if _int_to_uci(m) == uci:
            return m
    return None


class FairyStockfishWrapper:
    """UCI обёртка над Fairy-Stockfish для варианта Capablanca."""
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fairy-Stockfish не найден: {path}")
        self.proc = subprocess.Popen(
            [path], universal_newlines=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1,
        )
        self._send("uci");          self._wait("uciok")
        self._send("setoption name UCI_Variant value capablanca")
        self._send("isready");      self._wait("readyok")

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n"); self.proc.stdin.flush()

    def _wait(self, target: str) -> str:
        while True:
            line = self.proc.stdout.readline().strip()
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
    """Генерирует num_games партий против оппонента.

    fsf_nodes == 0 : random mover — первый уровень curriculum.
                     Позиции оппонента НЕ сохраняются. Value = game result.
    fsf_nodes >= 1 : Fairy-Stockfish. Позиции оппонента НЕ сохраняются.
                     Value = fsf_value_alpha * fsf_eval + (1-alpha) * game_result
                     где fsf_eval = -tanh(score_cp/400) после каждого хода NN.
                     Это даёт плотный, позиционно-специфичный value-сигнал
                     вместо одного задержанного game result на всю партию.
    """
    try:
        from capablanca_engine import CapablancaEngine
        from mcts import UltraFastMCTS
    except ImportError:
        print("  ❌ capablanca_engine или mcts не доступны")
        return [], 0, 0, 0

    fsf_value_alpha = getattr(cfg, 'fsf_value_alpha', 0.7)
    use_random = (fsf_nodes == 0)
    fsf = None
    if not use_random:
        try:
            fsf = FairyStockfishWrapper(fsf_path)
        except Exception as e:
            print(f"  ❌ FSF: {e}")
            return [], 0, 0, 0

    mcts = UltraFastMCTS(net, device, c_puct=1.745, batch_size=1,
                         add_dirichlet=False, parallel_sims=1)
    all_samples = []
    wins = draws = losses = errors = 0
    nn_wins = nn_draws = nn_losses = 0

    for game_idx in range(num_games):
        engine  = CapablancaEngine()
        nn_side = game_idx % 2
        uci_history = []
        # Только позиции NN: [board_np, pol, side, fsf_eval_or_None]
        # fsf_eval заполняется когда FSF ходит после NN (оценка позиции после хода NN)
        # score_cp > 0 → FSF (side-to-move) выигрывает → fsf_eval < 0 для NN
        nn_positions = []
        move_num, ok = 0, True
        adjudicated_result = None

        while not engine.is_game_over() and move_num < cfg.max_game_length and adjudicated_result is None:
            side = engine.side_to_move()
            legal = engine.get_legal_moves_int()
            if not legal: break
            board_np = np.array(engine.get_board_tensor(), dtype=np.float32)

            if side == nn_side:
                pol = mcts.search_games([engine], mcts_sims)[0]
                raw = np.array([pol[engine.move_int_to_policy_idx(m) or 0]
                               for m in legal], dtype=np.float64)
                raw = np.power(np.maximum(raw, 1e-8), 1.0 / 0.8)
                probs = raw / raw.sum()
                move = int(np.random.choice(legal, p=probs))
                # Позиция ПЕРЕД ходом NN, fsf_eval заполнится на следующем ходу FSF
                nn_positions.append([board_np, pol.copy(), side, None])

            elif use_random:
                # Random mover: не сохраняем, нет полезного policy-сигнала
                move = int(np.random.choice(legal))

            else:  # FSF's turn
                uci, score_cp = fsf.best_move(uci_history, nodes=fsf_nodes)
                if uci == "(none)": break
                move = _uci_to_int(uci, engine)
                if move is None:
                    errors += 1; ok = False; break
                # FSF оценивает позицию ПОСЛЕ последнего хода NN (сейчас очередь FSF)
                # score_cp > 0 → FSF (side to move) выигрывает → для NN плохо
                if nn_positions and nn_positions[-1][3] is None:
                    nn_positions[-1][3] = -float(np.tanh(score_cp / 400.0))

            engine.make_move_int(move)
            uci_history.append(_int_to_uci(move))
            move_num += 1
            adj = engine.adjudication_result()
            if adj is not None:
                adjudicated_result = adj

        if not ok: continue
        if adjudicated_result is not None: result = adjudicated_result
        elif engine.is_game_over(): result = engine.game_result()
        else: result = engine.material_result()

        if result > 0.5:    wins   += 1
        elif result < -0.5: losses += 1
        else:               draws  += 1
        nn_result = result if nn_side == 0 else -result
        if nn_result > 0.5:    nn_wins   += 1
        elif nn_result < -0.5: nn_losses += 1
        else:                  nn_draws  += 1

        for board_np, pol, side, fsf_eval in nn_positions:
            v_game = result if side == 0 else -result
            if not use_random and fsf_eval is not None:
                # Смешиваем: FSF eval (плотный, позиционный) + game result (долгосрочный)
                v = fsf_value_alpha * fsf_eval + (1.0 - fsf_value_alpha) * v_game
            else:
                v = v_game
            all_samples.append(pack_sample(board_np, pol, float(v)))

    if fsf is not None:
        fsf.close()
    opp_label = "Random" if use_random else f"FSF-{fsf_nodes}"
    print(f"  {opp_label}: {num_games-errors} партий | бел={wins} чёрн={losses} ничьи={draws} "
          f"ошибки={errors} | NN: +{nn_wins}/={nn_draws}/-{nn_losses} "
          f"| {len(all_samples)} позиций")
    return all_samples, nn_wins, nn_draws, nn_losses


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

    # Playout Cap Randomization (AlphaZero, lc0):
    # На fast_sim_fraction ходов делаем fast_simulations вместо simulations.
    # Только полные поиски попадают в обучающий буфер при playout_cap_train_only_full=True.
    fast_simulations: int = 100
    fast_sim_fraction: float = 0.75
    playout_cap_train_only_full: bool = True
    c_puct: float = 1.25
    temperature_moves: int = 30       # FIX: 30→50, больше исследования в начале партии
    temperature: float = 1.0          # tau для первых temperature_moves ходов (1.0 = пропорционально visit counts)
    temperature_late: float = 0.25     # tau после temperature_moves: 0.0 = жёсткий argmax (лучше матует)
    games_per_iter: int = 128
    max_game_length: int = 110
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
    buffer_max: int = 1_000_000
    buffer_min_to_train: int = 10_000

    # Сдача (Resignation): если V < resign_threshold на протяжении
    # resign_consec ходов подряд после resign_min_move — игра заканчивается поражением.
    # Убивает 30-40% таймаутов: сеть не играет 80 ходов голым королём.
    # На ранних итерациях (< resign_warmup_iters) порог жёсткий (-0.99),
    # потом переходит к resign_threshold (-0.95).
    # Это защищает от ошибок слабой сети в оценке позиции.
    resign_threshold: float = -0.95   # порог оценки (финальный)
    resign_threshold_early: float = -0.99  # порог на ранних итерациях
    resign_warmup_iters: int = 30     # итераций до перехода к resign_threshold
    resign_consec: int = 3            # ходов подряд ниже порога
    resign_min_move: int = 20         # не сдаёмся раньше этого хода

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

    # EMA весов модели (AlphaZero стабилизация self-play)
    use_ema: bool = True
    ema_decay: float = 0.999
    # EMA не применяется к self-play до этой итерации: ранние веса EMA = усреднение
    # случайных весов → worse чем live NN. Обновляется всегда, используется только с ema_start_iter.
    ema_start_iter: int = 10
    force_save: bool = False  # если True — сохраняем чекпоинт даже при низком loss

    # FSF eval как value target: смешиваем FSF-оценку позиции с game result.
    # fsf_eval = -tanh(score_cp/400) — оценка позиции после хода NN с точки зрения NN.
    # 0.7 = 70% FSF eval (плотный сигнал) + 30% game result (долгосрочный).
    fsf_value_alpha: float = 0.7

    # Curriculum обучение: FSF как адаптивный учитель
    # --curriculum --fsf-path ./binary --fsf-nodes-start 1 --fsf-nodes-max 10000
    curriculum_mode: bool = False
    fsf_nodes_current: int = 1        # текущий уровень FSF (авто-адаптируется)
    curriculum_nodes_min: int = 0  # 0 = Random mover (нижний предел)
    curriculum_nodes_max: int = 10000
    curriculum_self_play_ratio: float = 0.0   # 0.0 = только FSF, 0.2 = 20% self-play
    curriculum_promote_threshold: float = 0.55  # avg winrate выше → повышаем nodes
    curriculum_demote_threshold: float = 0.35   # avg winrate ниже → снижаем nodes
    curriculum_window: int = 3        # итераций для усреднения winrate


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
        # Параллельный массив float32 для быстрой стратификации по value.
        # Обновляется инкрементально в push() — O(1) на элемент, не O(N) при sampling.
        self._val_arr = np.zeros(max_size, dtype=np.float32)
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
            self._val_arr[self._ptr] = float(s[2])
            self._ptr = (self._ptr + 1) % self.max_size

    def rebuild_val_arr(self):
        """Пересобрать _val_arr из data после загрузки из pickle."""
        for i, s in enumerate(self.data):
            self._val_arr[i] = float(s[2])

    def sample(self, batch_size: int) -> List[Sample]:
        n = len(self.data)
        if n == 0:
            return []
        indices = np.random.choice(n, batch_size, replace=True)
        return [self.data[i] for i in indices]

    def sample_balanced(self, batch_size: int) -> List[Sample]:
        """Sample с балансировкой win/draw/loss (приближённо 33/33/33).

        Использует _val_arr для O(N) векторного поиска без Python-цикла по data.
        Если один класс отсутствует — балансирует по доступным.
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

        per_bin = batch_size // len(bins)
        result = []
        for b in bins:
            local_idx = np.random.randint(0, len(b), per_bin)
            result.extend([self.data[int(b[i])] for i in local_idx])
        while len(result) < batch_size:
            b = bins[np.random.randint(len(bins))]
            result.append(self.data[int(b[np.random.randint(len(b))])])
        np.random.shuffle(result)
        return result[:batch_size]

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

def generate_games(net: nn.Module, cfg: Config, device: torch.device, iteration: int = 0) -> List[Sample]:
    mcts = UltraFastMCTS(net, device, cfg.c_puct, batch_size=cfg.mcts_batch, parallel_sims=cfg.mcts_parallel_sims)
    all_samples: List[Sample] = []

    batch_sz = cfg.mcts_batch
    num_batches = (cfg.games_per_iter + batch_sz - 1) // batch_sz

    for b in range(num_batches):
        start = b * batch_sz
        n = min(batch_sz, cfg.games_per_iter - start)
        engines = [CapablancaEngine() for _ in range(n)]
        histories: List[List] = [[] for _ in range(n)]
        resign_counts = [0] * n
        resigned = [False] * n

        active = list(range(n))
        move_num = 0
        adjudicated = [None] * n

        # Tree reuse: один RustMCTS на весь батч игр.
        # После каждого хода вызываем make_move(game_idx, move) —
        # корень переносится на выбранного ребёнка, статистика сохраняется.
        # Это 2-3x лучшее качество при тех же затратах на inference.
        from capablanca_engine import RustMCTS as _RustMCTS
        _parallel = cfg.mcts_parallel_sims
        rust_mcts_reuse = _RustMCTS(engines, _parallel)

        while active and move_num < cfg.max_game_length:
            # Playout Cap Randomization: на fast_sim_fraction ходов используем fast_simulations
            # Решение применяется ко всему батчу одновременно (общий MCTS объект)
            use_full_search = np.random.random() >= cfg.fast_sim_fraction
            current_sims = cfg.simulations if use_full_search else cfg.fast_simulations

            # Tree reuse inference loop (без создания нового RustMCTS каждый ход)
            cur_engines = [engines[i] for i in active]
            steps = max(1, (current_sims + _parallel - 1) // _parallel)
            for _step in range(steps):
                _lm = rust_mcts_reuse.collect_leaves()
                if _lm.shape[0] == 0:
                    break
                _rp, _rv = mcts._infer(_lm)
                rust_mcts_reuse.apply_inference_buffered(
                    np.ascontiguousarray(_rp, dtype=np.float32),
                    np.ascontiguousarray(_rv, dtype=np.float32),
                    rust_mcts_reuse.get_current_batch_counts(),
                )
            raw_pols  = rust_mcts_reuse.get_policies()
            raw_vals  = rust_mcts_reuse.get_values()
            policies  = [np.array(p, dtype=np.float32) for p in raw_pols]
            values_np = np.array(raw_vals, dtype=np.float32)

            new_active = []
            for j, game_idx in enumerate(active):
                eng = engines[game_idx]
                legal = eng.get_legal_moves_int()
                if not legal:
                    continue

                board_np = np.array(eng.get_board_tensor(), dtype=np.float32)
                side = eng.side_to_move()
                pol = policies[j]
                # Сохраняем root_v и метку full_search для resign/timeout/playout-cap
                root_v_raw = float(values_np[j]) if j < len(values_np) else 0.0
                histories[game_idx].append((board_np, pol.copy(), side, root_v_raw, use_full_search))

                # Temperature decay с защитой от деления на ноль
                if move_num < cfg.temperature_moves:
                    tau = cfg.temperature
                elif move_num < cfg.temperature_moves + 20:
                    progress = (move_num - cfg.temperature_moves) / 20.0
                    tau = cfg.temperature * (1 - progress) + cfg.temperature_late * progress
                    tau = max(tau, 0.01)
                else:
                    tau = cfg.temperature_late

                raw = np.array([
                    pol[eng.move_int_to_policy_idx(m) or 0] for m in legal
                ], dtype=np.float64)

                if tau < 0.01:
                    # tau≈0 = жёсткий argmax (избегаем 1/0 = inf → NaN)
                    move = int(legal[int(np.argmax(raw))])
                else:
                    raw = np.power(np.maximum(raw, 1e-8), 1.0 / tau)
                    s = raw.sum()
                    probs = raw / s if s > 0 else np.ones(len(legal)) / len(legal)
                    move = int(np.random.choice(legal, p=probs))

                eng.make_move_int(move)
                rust_mcts_reuse.make_move(game_idx, move)  # tree reuse

                if eng.is_game_over():
                    continue

                # Resign: проверяем оценку позиции после хода
                # Используем root value из MCTS (уже вычислен выше)
                # sign: value с точки зрения стороны которая только что сделала ход
                if move_num >= cfg.resign_min_move:
                    # values_np[j] = оценка ПОСЛЕ хода с т.з. нового side_to_move.
                    # Инвертируем чтобы получить оценку с т.з. того кто только что ходил.
                    # Если v_for_mover < -0.95 — он проигрывает → сдаётся.
                    # j — уже правильный индекс в active (не нужен .index() = O(n))
                    v_before_move = -float(values_np[j]) if j < len(values_np) else 0.0
                    _resign_thr = (cfg.resign_threshold_early
                                   if iteration < cfg.resign_warmup_iters
                                   else cfg.resign_threshold)
                    if v_before_move < _resign_thr:
                        resign_counts[game_idx] += 1
                    else:
                        resign_counts[game_idx] = 0

                    if resign_counts[game_idx] >= cfg.resign_consec:
                        resigned[game_idx] = True
                        continue

                # Досрочное присуждение (Adjudication):
                # Если после хода есть решающий материальный перевес
                # (≥8 очков, ≥10 ходов без взятий, ≥15 полных ходов) —
                # заканчиваем игру не дожидаясь мата или таймаута.
                adj = eng.adjudication_result()
                if adj is not None:
                    adjudicated[game_idx] = adj
                else:
                    new_active.append(game_idx)

            active = new_active
            move_num += 1

        batch_positions = 0
        white_wins = black_wins = draws = timeouts = adjudications = 0

        for i, eng in enumerate(engines):
            if resigned[i]:
                # Сдача: сторона которая последней ходила — проиграла
                # Определяем кто сдался по side_to_move (ходит противник → предыдущий сдался)
                last_side = histories[i][-1][2] if histories[i] else 0
                result = -1.0 if last_side == 0 else 1.0
                timeouts += 1  # считаем как незавершённую для статистики
                if result > 0: white_wins += 1
                else:          black_wins += 1
            elif adjudicated[i] is not None:
                # Досрочное присуждение — решающий материальный перевес
                result = adjudicated[i]
                adjudications += 1
                if result > 0: white_wins += 1
                else:          black_wins += 1
            elif eng.is_game_over():
                result = eng.game_result()
                if result == 1.0:   white_wins += 1
                elif result == -1.0: black_wins += 1
                else:               draws += 1
            else:
                # Настоящий таймаут — мягкая материальная оценка
                result = eng.material_result()
                timeouts += 1

            for entry in histories[i]:
                board_np, pol, side = entry[0], entry[1], entry[2]
                # Playout cap: пропускаем fast-search позиции при обучении
                if cfg.playout_cap_train_only_full and len(entry) > 4 and not entry[4]:
                    continue
                v = result if side == 0 else -result
                all_samples.append(pack_sample(board_np, pol, float(v)))
                batch_positions += 1

        print(f"  Batch {b+1}/{num_batches}: {n} games, "
              f"{batch_positions} positions, {move_num} ходов | "
              f"бел={white_wins} чёрн={black_wins} пат={draws} "
              f"adj={adjudications} timeout={timeouts}")

    return all_samples


# ── Dataset ───────────────────────────────────────────────────────────────────

def value_to_wdl(v: float) -> np.ndarray:
    """
    Мягкая конвертация v∈[-1,1] → [P(Win), P(Draw), P(Loss)] без жёстких порогов.
    sqrt-нелинейность даёт более чёткий сигнал при материальном перевесе:
      v=0.5 → [0.71, 0.29, 0.0]   (раньше через порог: то же [0.71, 0.29, 0])
      v=0.9 → [0.95, 0.05, 0.0]   (раньше: жёсткий [1, 0, 0] — потеря информации)
      v=1.0 → [1.00, 0.00, 0.0]
    Преимущество: плавный градиент даже при v близком к ±1.
    """
    v = float(np.clip(v, -1.0, 1.0))
    p_win  = float(max(0.0, v) ** 0.5)
    p_loss = float(max(0.0, -v) ** 0.5)
    p_draw = max(0.0, 1.0 - p_win - p_loss)
    total = p_win + p_draw + p_loss
    if total > 0:
        p_win /= total; p_draw /= total; p_loss /= total
    return np.array([p_win, p_draw, p_loss], dtype=np.float32)


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

    samples = buffer.sample_balanced(effective_steps * cfg.batch_size)
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
        # Проверка NaN/Inf в градиентах — защита от взрывного градиента
        grad_ok = True
        for p in net.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                grad_ok = False
                break
        if not grad_ok:
            print("  ⚠️  NaN/Inf в градиентах — пропускаем шаг")
            optimizer.zero_grad(set_to_none=True)
            continue
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
            net = torch.compile(net, dynamic=True)  # dynamic=True — без рекомпиляций при разных размерах батча MCTS
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

    # EMA копия весов для self-play (AlphaZero-style стабилизация)
    ema = ModelEMA(net, decay=cfg.ema_decay) if cfg.use_ema else None

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

    curriculum_winrate_history: List[float] = []

    buffer_path = os.path.join(cfg.checkpoint_dir, "buffer.pkl")
    if os.path.exists(buffer_path):
        try:
            with open(buffer_path, "rb") as f:
                buffer.data, buffer._ptr, buffer._full = pickle.load(f)
            buffer.rebuild_val_arr()
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
            print("🔄 Scheduler сброшен (начинается новый косинусный цикл)\n")
        else:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"⚠️  Scheduler не загружен ({e}), используем свежий\n")

        # Загружаем EMA если есть
        if ema is not None and "ema" in ckpt:
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

        # ── Self-play / Curriculum ────────────────────────────────────────────
        sp_start = time.time()
        # EMA не используется для self-play в первые ema_start_iter итераций:
        # ранние EMA-веса = смесь случайных весов → хуже live NN для генерации данных.
        # EMA обновляется всегда (чтобы к iter=ema_start_iter уже отражало обученную сеть).
        use_ema_now = ema is not None and iteration >= cfg.ema_start_iter
        if use_ema_now:
            saved_state = {k: v.clone() for k, v in (net._orig_mod if hasattr(net, '_orig_mod') else net).state_dict().items()
                           if v.dtype.is_floating_point}
            ema.apply_to(net)
        elif ema is None:
            pass  # no EMA configured
        else:
            print(f"  ℹ️  EMA отложен до iter {cfg.ema_start_iter} (сейчас {iteration}), используется live NN")

        fsf_path = getattr(args, 'fsf_path', None)
        fsf_enabled = bool(fsf_path and os.path.exists(fsf_path))

        if cfg.curriculum_mode and not fsf_enabled:
            print(f"[Iter {iteration}] ⚠️  --curriculum требует --fsf-path, переходим в self-play")

        if fsf_enabled and cfg.curriculum_mode:
            # ── Curriculum: адаптивный FSF как основной учитель ──────────────
            sp_count  = int(cfg.games_per_iter * cfg.curriculum_self_play_ratio)
            fsf_count = cfg.games_per_iter - sp_count
            opp_lbl = "Random" if cfg.fsf_nodes_current == 0 else f"FSF nodes={cfg.fsf_nodes_current}"
            print(f"[Iter {iteration}] 📚 Curriculum | {opp_lbl}"
                  f"  self={sp_count}  fsf={fsf_count}")
            samples = []
            if sp_count > 0:
                orig_games = cfg.games_per_iter
                cfg.games_per_iter = sp_count
                net.eval()
                with torch.inference_mode():
                    samples = generate_games(net, cfg, device, iteration)
                cfg.games_per_iter = orig_games
            net.eval()
            with torch.inference_mode():
                fsf_samples, fsf_w, fsf_d, fsf_l = generate_fsf_games(
                    net, device, cfg,
                    num_games=fsf_count, fsf_path=fsf_path,
                    fsf_nodes=cfg.fsf_nodes_current, mcts_sims=args.fsf_mcts_sims,
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
                        # 0 (random) → 1 (FSF-1), затем 1→2→4→8...
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
            # ── Оригинальное расписание FSF ───────────────────────────────────
            print(f"[Iter {iteration}] ⚙️  Self-play: {cfg.games_per_iter} игр...")
            self_games_n, fsf_games_n, _ = get_fsf_schedule(iteration, cfg.games_per_iter)
            phase = ("🔴 FSF-heavy" if iteration < 30
                     else "🟡 FSF-fade" if iteration < 50 else "🟢 self-only")
            print(f"  {phase} | self={self_games_n}"
                  + (f" + fsf={fsf_games_n}" if fsf_games_n else ""))
            orig_games = cfg.games_per_iter
            cfg.games_per_iter = self_games_n
            net.eval()
            with torch.inference_mode():
                samples = generate_games(net, cfg, device, iteration)
            cfg.games_per_iter = orig_games
            if fsf_games_n > 0:
                print(f"  ⚔️  FSF: {fsf_games_n} игр vs Stockfish ({args.fsf_nodes} nodes)...")
                net.eval()
                with torch.inference_mode():
                    fsf_r = generate_fsf_games(
                        net, device, cfg,
                        num_games=fsf_games_n, fsf_path=fsf_path,
                        fsf_nodes=args.fsf_nodes, mcts_sims=args.fsf_mcts_sims,
                    )
                samples = samples + fsf_r[0]

        else:
            # ── Только self-play ──────────────────────────────────────────────
            print(f"[Iter {iteration}] ⚙️  Self-play: {cfg.games_per_iter} игр...")
            net.eval()
            with torch.inference_mode():
                samples = generate_games(net, cfg, device, iteration)

        # Восстанавливаем тренировочные веса (только если применяли EMA)
        if use_ema_now:
            (net._orig_mod if hasattr(net, '_orig_mod') else net).load_state_dict(saved_state, strict=False)
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

        net.train()

        print(f"  🏋️  Тренировка (до {cfg.train_steps} шагов, ≤1 эпохи)...")
        train_start = time.time()
        metrics = train_epoch(net, optimizer, buffer, cfg, device, scaler, iteration)
        # Обновляем EMA после каждой итерации обучения
        if ema is not None:
            ema.update(net)
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
                "curriculum_fsf_nodes": cfg.fsf_nodes_current,
                "curriculum_winrate_history": list(curriculum_winrate_history),
            }
            if ema is not None:
                ckpt_data["ema"] = ema.state_dict()
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
    parser.add_argument("--fast-simulations",     type=int,   default=100,
                        help="Симуляций на быстрых ходах playout cap (default: 100)")
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
    parser.add_argument("--use-ema",             action="store_true", default=True,
                        help="Использовать EMA веса для self-play (default: True)")
    parser.add_argument("--no-ema",              dest="use_ema", action="store_false",
                        help="Отключить EMA")
    parser.add_argument("--ema-decay",           type=float, default=0.999,
                        help="EMA decay coefficient (default: 0.999)")
    parser.add_argument("--ema-start-iter",      type=int,   default=10,
                        help="Не использовать EMA для self-play до этой итерации (default: 10)")
    parser.add_argument("--resign-threshold",      type=float, default=-0.95,
                        help="Финальный порог сдачи (default: -0.95)")
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

    # FSF интеграция (опциональная)
    parser.add_argument("--fsf-path",             type=str, default=None,
                        help="Путь к Fairy-Stockfish бинарнику (включает FSF режим)")
    parser.add_argument("--fsf-nodes",            type=int, default=500,
                        help="Лимит nodes для FSF в обычном режиме (default: 500)")
    parser.add_argument("--fsf-mcts-sims",        type=int, default=100,
                        help="MCTS симуляций при игре против FSF (default: 100)")
    parser.add_argument("--fsf-value-alpha",      type=float, default=0.7,
                        help="Вес FSF eval в value target: alpha*eval + (1-alpha)*result (default: 0.7)")

    # Curriculum обучение
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
        fast_simulations=args.fast_simulations,
        fast_sim_fraction=args.fast_sim_fraction,
        playout_cap_train_only_full=args.playout_cap_train_only_full,
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
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        ema_start_iter=args.ema_start_iter,
        fsf_value_alpha=args.fsf_value_alpha,
        resign_threshold=args.resign_threshold,
        resign_threshold_early=args.resign_threshold_early,
        resign_warmup_iters=args.resign_warmup_iters,
        resign_consec=args.resign_consec,
        resign_min_move=args.resign_min_move,
        force_save=args.force_save,
        curriculum_mode=args.curriculum,
        fsf_nodes_current=args.fsf_nodes_start,
        curriculum_nodes_min=0,
        curriculum_nodes_max=args.fsf_nodes_max,
        curriculum_self_play_ratio=args.curriculum_sp_ratio,
        curriculum_promote_threshold=args.curriculum_promote,
        curriculum_demote_threshold=args.curriculum_demote,
        curriculum_window=args.curriculum_window,
    )
    train(cfg)
