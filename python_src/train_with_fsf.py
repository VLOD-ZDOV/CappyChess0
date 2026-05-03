#!/usr/bin/env python3
# train_with_fsf.py — тренировка с адаптивной FSF интеграцией
#
# Логика пропорций self-play / FSF по итерациям:
#
#   iter 0-29:   1152 своих + 2688 FSF  (70% FSF — сеть слабая, учимся у Стокфиша)
#   iter 30-49:  2304 своих + 1152 FSF  (33% FSF — переходная фаза)
#   iter 50-69:  2304 своих + 0 FSF     (сеть уже превосходит FSF при малых nodes)
#   iter 70+:    2304 своих + 0 FSF
#
# FSF запускается только каждые --fsf-every итераций (default: 3 в фазе 0-29, 5 в 30-49)
# Количество FSF игр за запуск = fsf_games_total / (phase_length / fsf_every)
#
# Запуск:
#   python train_with_fsf.py --channels 128 --res-blocks 10 --simulations 400 \
#     --games 384 --mcts-batch 512 --batch-size 1024 --train-steps 400 \
#     --lr 1e-4 --fsf-path ./fairy-stockfish-largeboard_x86-64-bmi2

import os
import sys
import time
import pickle
import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import CapablancaNet
from mcts import UltraFastMCTS, POLICY_SIZE
from train import (
    Config, ReplayBuffer, CompactSample, Sample,
    pack_sample, unpack_policy,
    generate_games, train_epoch,
    policy_diversity_stats, print_diversity,
)

try:
    from capablanca_engine import CapablancaEngine
except ImportError:
    raise ImportError("capablanca_engine not found. Build with: maturin develop --release")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False


# ── Таблица фаз FSF ───────────────────────────────────────────────────────────

def get_fsf_schedule(iteration: int, base_self_play: int) -> Tuple[int, int, int]:
    """
    Возвращает (self_play_games, fsf_games_this_iter, fsf_every) для данной итерации.

    Пропорции:
      iter 0-29:  1152 своих + 2688 FSF  (запускаем FSF каждые 3 итерации → 269 игр за раз)
      iter 30-49: 2304 своих + 1152 FSF  (каждые 5 итераций → 230 игр за раз)
      iter 50+:   base_self_play своих + 0 FSF

    base_self_play — --games из аргументов (то что пользователь задал).
    """
    if iteration < 30:
        # Фаза 1: активное обучение у FSF
        # 2688 FSF игр за 30 итераций, каждые 3 итерации → 2688/10 ≈ 269 игр за запуск
        self_games = 1152
        fsf_every  = 3
        fsf_per_run = 269 if iteration % fsf_every == 0 else 0
    elif iteration < 50:
        # Фаза 2: переход, FSF всё меньше
        # 1152 FSF игр за 20 итераций, каждые 5 итераций → 1152/4 = 288 игр за запуск
        self_games  = 2304
        fsf_every   = 5
        fsf_per_run = 288 if iteration % fsf_every == 0 else 0
    else:
        # Фаза 3: только self-play
        self_games  = base_self_play
        fsf_every   = 0
        fsf_per_run = 0

    return self_games, fsf_per_run, fsf_every


# ── UCI helpers ───────────────────────────────────────────────────────────────

_PROMO_CHARS = {2: 'n', 3: 'b', 4: 'r', 5: 'q', 6: 'a', 7: 'c'}

def int_to_uci(m: int) -> str:
    p_val = m & 0b111
    t = (m >> 3) & 0x7F
    f = (m >> 10) & 0x7F
    uci = f"{chr(ord('a') + f%10)}{f//10+1}{chr(ord('a') + t%10)}{t//10+1}"
    if p_val in _PROMO_CHARS:
        uci += _PROMO_CHARS[p_val]
    return uci

def uci_to_int(uci: str, engine: CapablancaEngine) -> Optional[int]:
    for m in engine.get_legal_moves_int():
        if int_to_uci(m) == uci:
            return m
    return None


# ── Fairy-Stockfish wrapper ───────────────────────────────────────────────────

class FairyStockfishWrapper:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fairy-Stockfish не найден: {path}")
        self.proc = subprocess.Popen(
            [path], universal_newlines=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1
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

    def best_move(self, uci_history: List[str], nodes: int) -> str:
        moves = " ".join(uci_history) if uci_history else ""
        self._send(f"position startpos" + (f" moves {moves}" if moves else ""))
        self._send(f"go nodes {nodes}")
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                return line.split()[1]

    def close(self):
        try: self._send("quit"); self.proc.wait(timeout=3)
        except: self.proc.kill()


# ── Генерация FSF партий ──────────────────────────────────────────────────────

def generate_fsf_games(
    net: nn.Module, device: torch.device, cfg: Config,
    num_games: int, fsf_path: str, fsf_nodes: int,
    mcts_sims: int = 100,
) -> List[Sample]:
    """
    Генерирует партии: нейросеть vs Fairy-Stockfish.
    Чётные игры: сеть = белые. Нечётные: сеть = чёрные.
    Возвращает список обучающих примеров.
    """
    try:
        fsf = FairyStockfishWrapper(fsf_path)
    except Exception as e:
        print(f"  ❌ Не удалось запустить FSF: {e}")
        return []

    mcts = UltraFastMCTS(net, device, c_puct=1.25, batch_size=1,
                         add_dirichlet=False, parallel_sims=1)
    all_samples: List[Sample] = []
    wins = draws = losses = errors = 0

    for game_idx in range(num_games):
        engine      = CapablancaEngine()
        nn_side     = game_idx % 2          # 0=белые, 1=чёрные
        uci_history: List[str] = []
        history_tensors = []
        move_num    = 0
        ok          = True

        while not engine.is_game_over() and move_num < cfg.max_game_length:
            side   = engine.side_to_move()
            legal  = engine.get_legal_moves_int()
            if not legal: break

            board_np = np.array(engine.get_board_tensor(), dtype=np.float32)

            if side == nn_side:
                # ── Ход нейросети ──
                pol = mcts.search_games([engine], mcts_sims)[0]

                # Temperature 0.8 для разнообразия
                raw = np.array([pol[engine.move_int_to_policy_idx(m) or 0]
                                 for m in legal], dtype=np.float64)
                raw = np.power(np.maximum(raw, 1e-8), 1.0 / 0.8)
                probs = raw / raw.sum()
                move = int(np.random.choice(legal, p=probs))
                history_tensors.append((board_np, pol.copy(), side))
            else:
                # ── Ход FSF ──
                uci = fsf.best_move(uci_history, nodes=fsf_nodes)
                if uci == "(none)": break
                move = uci_to_int(uci, engine)
                if move is None:
                    errors += 1; ok = False; break

                # Мягкая метка для FSF хода — не one-hot, а немного сглаженная
                # Это лучше чем жёсткий one-hot: сеть не штрафуется за другие хорошие ходы
                pol = np.zeros(7000, dtype=np.float32)
                pol_idx = engine.move_int_to_policy_idx(move)
                if pol_idx is not None:
                    pol[pol_idx] = 0.9  # 90% вместо 100% — мягкость
                    # Остальные легальные ходы делим оставшиеся 10%
                    others = [engine.move_int_to_policy_idx(m) for m in legal
                              if m != move and engine.move_int_to_policy_idx(m) is not None]
                    if others:
                        w = 0.1 / len(others)
                        for idx in others: pol[idx] = w
                history_tensors.append((board_np, pol, side))

            engine.make_move_int(move)
            uci_history.append(int_to_uci(move))
            move_num += 1

        if not ok: continue

        result = engine.game_result() if engine.is_game_over() else float(np.clip(engine.material_result() * 1.6, -0.8, 0.8))

        if result > 0.5:   wins   += 1
        elif result < -0.5: losses += 1
        else:               draws  += 1

        for board_np, pol, side_t in history_tensors:
            v = result if side_t == 0 else -result
            all_samples.append(pack_sample(board_np, pol, float(v)))

    fsf.close()
    nn_wins   = wins   if True else losses  # сеть могла быть и белыми и чёрными
    total_ok  = num_games - errors
    print(f"  FSF: {total_ok} партий | бел={wins} чёрн={losses} ничьи={draws} "
          f"ошибки={errors} | {len(all_samples)} позиций")
    return all_samples


# ── Главный цикл ──────────────────────────────────────────────────────────────

def train_with_fsf(cfg: Config, args):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("⚠️  CUDA не найдена")

    fsf_available = os.path.exists(args.fsf_path) if args.fsf_path else False

    print(f"🚀 Тренировка на {device}")
    print(f"   Модель:     {cfg.num_channels}ch × {cfg.num_res_blocks} blocks")
    print(f"   Self-play:  {cfg.games_per_iter} игр/итер (base), {cfg.simulations} симуляций")
    print(f"   FSF:        {'✅ ' + args.fsf_path if fsf_available else '❌ не найден — только self-play'}")
    print(f"\n   Расписание FSF:")
    print(f"   iter  0-29: 1152 self + 2688 FSF (каждые 3 итерации по ~269 игр)")
    print(f"   iter 30-49: 2304 self + 1152 FSF (каждые 5 итераций по ~288 игр)")
    print(f"   iter  50+:  {cfg.games_per_iter} self + 0 FSF\n")

    net = CapablancaNet(cfg.num_channels, cfg.num_res_blocks).to(device)
    net = net.to(memory_format=torch.channels_last)

    if hasattr(torch, "compile"):
        try:
            net = torch.compile(net)
            print("✅ torch.compile() применён\n")
        except Exception as e:
            print(f"⚠️  torch.compile(): {e}\n")

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay, fused=True,
    )
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=cfg.learning_rate * 0.05
    )
    buffer    = ReplayBuffer(cfg.buffer_max)

    buffer_path = os.path.join(cfg.checkpoint_dir, "buffer.pkl")
    if os.path.exists(buffer_path):
        try:
            with open(buffer_path, "rb") as f:
                buffer.data, buffer._ptr, buffer._full = pickle.load(f)
            print(f"📦 Загружен буфер: {len(buffer):,} позиций\n")
        except Exception as e:
            print(f"⚠️  Буфер не загружен: {e}\n")

    start_iter = 0
    latest_path = os.path.join(cfg.checkpoint_dir, "latest.pth")
    numbered = sorted([f for f in os.listdir(cfg.checkpoint_dir)
                       if f.endswith(".pth") and not f.startswith("latest")])
    # Предпочитаем latest.pth (сохраняется каждую итерацию)
    load_path = latest_path if os.path.exists(latest_path) else (
        os.path.join(cfg.checkpoint_dir, numbered[-1]) if numbered else None
    )
    if load_path:
        ckpt = torch.load(load_path, map_location=device, weights_only=False)
        raw_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
                  for k, v in ckpt["model"].items()}
        # Фильтруем слои с несовместимой формой (scalar↔WDL, policy head resize)
        target_sd = (net._orig_mod if hasattr(net, "_orig_mod") else net).state_dict()
        bad = [k for k in raw_sd if k in target_sd and raw_sd[k].shape != target_sd[k].shape]
        for k in bad: del raw_sd[k]
        missing, _ = (net._orig_mod if hasattr(net, "_orig_mod") else net).load_state_dict(raw_sd, strict=False)
        if bad:    print(f"✅ Веса загружены, пропущено (несовм. форма): {bad}")
        if missing: print(f"   Инициализированы заново: {missing}")
        # Оптимайзер только при полной совместимости архитектуры
        if not bad and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"⚠️  Оптимайзер не загружен: {e}")
        else:
            print("ℹ️  Оптимайзер инициализирован заново")
        for pg in optimizer.param_groups: pg['lr'] = cfg.learning_rate
        if cfg.reset_scheduler or "scheduler" not in ckpt:
            print("🔄 Scheduler сброшен\n")
        else:
            try: scheduler.load_state_dict(ckpt["scheduler"])
            except: print("⚠️  Scheduler сброшен\n")
        start_iter = ckpt.get("iteration", 0) + 1
        print(f"📂 Загружен: {os.path.basename(load_path)} (итерация {start_iter})")
        if len(buffer) > 0:
            print_diversity(policy_diversity_stats(buffer.data), prefix="   Буфер")
            print()

    for iteration in range(start_iter, 100_000):
        iter_start = time.time()

        # ── Определяем пропорции для этой итерации ───────────────────────────
        self_games, fsf_games_now, fsf_every = get_fsf_schedule(
            iteration, cfg.games_per_iter
        )

        phase = "🔴 FSF-heavy" if iteration < 30 else ("🟡 FSF-fade" if iteration < 50 else "🟢 self-only")
        print(f"[Iter {iteration}] {phase} | self={self_games}"
              + (f" + fsf={fsf_games_now}" if fsf_games_now > 0 else ""))

        net.eval()
        torch.set_grad_enabled(False)
        sp_start = time.time()

        # ── Self-play ─────────────────────────────────────────────────────────
        orig_games = cfg.games_per_iter
        cfg.games_per_iter = self_games
        self_samples = generate_games(net, cfg, device)
        cfg.games_per_iter = orig_games

        sp_time = time.time() - sp_start
        print(f"  Self: {len(self_samples):,} поз за {sp_time:.1f}s "
              f"({self_games/sp_time:.1f} игр/с)")

        # ── FSF партии ────────────────────────────────────────────────────────
        fsf_samples: List[Sample] = []
        if fsf_games_now > 0 and fsf_available:
            print(f"  ⚔️  FSF: {fsf_games_now} игр vs Stockfish "
                  f"({args.fsf_nodes} nodes)...")
            fsf_start = time.time()
            fsf_samples = generate_fsf_games(
                net, device, cfg,
                num_games=fsf_games_now,
                fsf_path=args.fsf_path,
                fsf_nodes=args.fsf_nodes,
                mcts_sims=args.fsf_mcts_sims,
            )
            print(f"  FSF время: {time.time()-fsf_start:.1f}s")
        elif fsf_games_now > 0 and not fsf_available:
            # FSF не доступен — компенсируем self-play
            extra = fsf_games_now // 3  # добавляем треть от плановых FSF игр
            if extra > 0:
                print(f"  ℹ️  FSF недоступен → +{extra} self-play игр вместо")
                cfg.games_per_iter = extra
                fsf_samples = generate_games(net, cfg, device)
                cfg.games_per_iter = orig_games

        # ── Все данные в буфер ────────────────────────────────────────────────
        all_new = self_samples + fsf_samples
        buffer.push(all_new)

        total_time = time.time() - sp_start
        print(f"  Итого: {len(all_new):,} новых поз "
              f"({len(self_samples):,} self + {len(fsf_samples):,} FSF) | "
              f"Буфер: {len(buffer):,}")

        stats = policy_diversity_stats(self_samples)
        print_diversity(stats)

        if stats.get('top1_mean', 0) > 0.85 and cfg.temperature < 2.0:
            cfg.temperature = min(cfg.temperature * 1.2, 2.0)
            print(f"  ⚠️  top1 высокий → temperature={cfg.temperature:.2f}")
        elif stats.get('entropy_mean', 0) > 3.5 and cfg.temperature > 0.8:
            cfg.temperature = max(cfg.temperature * 0.95, 0.8)
        print()

        # ── Тренировка ────────────────────────────────────────────────────────
        if len(buffer) < cfg.buffer_min_to_train:
            print(f"  ⏳ Буфер {len(buffer):,} < {cfg.buffer_min_to_train:,}, пропуск\n")
            continue

        torch.set_grad_enabled(True)
        net.train()
        print(f"  🏋️  Тренировка (до {cfg.train_steps} шагов)...")
        t0 = time.time()
        metrics = train_epoch(net, optimizer, buffer, cfg, device, scaler, iteration)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        collapsed = metrics['policy_loss'] < cfg.collapse_threshold
        print(f"\n  ✅ {time.time()-t0:.1f}s ({metrics['steps']} шагов)"
              + ("  ⚠️ КОЛЛАПС!" if collapsed else ""))
        print(f"     policy={metrics['policy_loss']:.4f}  "
              f"value={metrics['value_loss']:.4f}  "
              f"total={metrics['loss']:.4f}  lr={current_lr:.2e}")
        if collapsed:
            print("  ⚠️  Попробуй --reset-buffer --reset-scheduler")

        print(f"\n  ⏱️  {iteration}: {time.time()-iter_start:.1f}s total\n")

        # ── Чекпоинт ──────────────────────────────────────────────────────────
        if not collapsed:
            m2s = net._orig_mod if hasattr(net, "_orig_mod") else net
            ckpt_data = {
                "iteration": iteration,
                "model": m2s.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": metrics,
            }
            # latest.pth — перезаписывается каждую итерацию
            latest_save = os.path.join(cfg.checkpoint_dir, "latest.pth")
            torch.save(ckpt_data, latest_save)
            print(f"  💾 latest.pth (iter {iteration})")

            # Нумерованный файл каждые save_every итераций
            if iteration % cfg.save_every == 0:
                num_path = os.path.join(cfg.checkpoint_dir, f"model_iter{iteration:05d}.pth")
                torch.save(ckpt_data, num_path)
                print(f"  💾 {os.path.basename(num_path)}")

            try:
                with open(buffer_path, "wb") as f:
                    pickle.dump((buffer.data, buffer._ptr, buffer._full), f)
                print(f"  💾 Буфер ({len(buffer):,} поз)\n")
            except Exception as e:
                print(f"  ⚠️  Буфер не сохранён: {e}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Capablanca AlphaZero + Fairy-Stockfish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Полный запуск с FSF
  python train_with_fsf.py --channels 128 --res-blocks 10 --simulations 400 \\
    --games 384 --mcts-batch 512 --batch-size 1024 --train-steps 400 \\
    --lr 1e-4 --fsf-path ./fairy-stockfish-largeboard_x86-64-bmi2

  # Без FSF (или если файл не указан) — работает как обычный train.py
  python train_with_fsf.py --channels 128 --res-blocks 10 --simulations 400 --games 384
        """
    )

    # Базовые параметры (те же что в train.py)
    parser.add_argument("--channels",            type=int,   default=128)
    parser.add_argument("--res-blocks",           type=int,   default=10)
    parser.add_argument("--simulations",          type=int,   default=400)
    parser.add_argument("--games",                type=int,   default=384,
                        help="Базовое число self-play игр (используется в фазе 50+)")
    parser.add_argument("--mcts-batch",           type=int,   default=512)
    parser.add_argument("--mcts-parallel-sims",   type=int,   default=32)
    parser.add_argument("--batch-size",           type=int,   default=1024)
    parser.add_argument("--train-steps",          type=int,   default=400)
    parser.add_argument("--min-train-steps",      type=int,   default=20)
    parser.add_argument("--buffer-min-to-train",  type=int,   default=10_000)
    parser.add_argument("--lr",                   type=float, default=1e-4)
    parser.add_argument("--temperature",          type=float, default=1.0)
    parser.add_argument("--temperature-late",     type=float, default=0.5)
    parser.add_argument("--temperature-moves",    type=int,   default=50)
    parser.add_argument("--value-loss-weight",    type=float, default=1.0)
    parser.add_argument("--device",               type=str,   default="cuda")
    parser.add_argument("--checkpoint-dir",       type=str,   default="checkpoints")
    parser.add_argument("--save-every",           type=int,   default=5)
    parser.add_argument("--reset-scheduler",      action="store_true")
    parser.add_argument("--reset-buffer",         action="store_true")
    parser.add_argument("--collapse-threshold",   type=float, default=0.01)

    # FSF параметры
    parser.add_argument("--fsf-path",      type=str, default=None,
                        help="Путь к Fairy-Stockfish бинарнику")
    parser.add_argument("--fsf-nodes",     type=int, default=500,
                        help="Лимит узлов FSF (500=быстро/слабо, 2000=медленно/сильно)")
    parser.add_argument("--fsf-mcts-sims", type=int, default=100,
                        help="MCTS симуляций при игре против FSF (меньше = быстрее)")

    args = parser.parse_args()

    if args.reset_buffer:
        bp = os.path.join(args.checkpoint_dir, "buffer.pkl")
        if os.path.exists(bp):
            os.remove(bp)
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
        temperature=args.temperature,
        temperature_late=args.temperature_late,
        temperature_moves=args.temperature_moves,
        value_loss_weight=args.value_loss_weight,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        reset_scheduler=args.reset_scheduler,
        collapse_threshold=args.collapse_threshold,
    )

    train_with_fsf(cfg, args)


if __name__ == "__main__":
    main()
