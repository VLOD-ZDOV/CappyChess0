#!/usr/bin/env python3
# train_with_fsf.py — оркестратор тренировки с периодической FSF интеграцией

import os
import sys
import argparse

# Добавляем текущую папку в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train, Config
from fsf_integration import run_fsf_duels, should_run_fsf, print_fsf_schedule, get_buffer_stats


def main():
    parser = argparse.ArgumentParser(description="Capablanca Chess AlphaZero + FSF")
    
    # Все параметры train.py
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=80)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--mcts-batch", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-late", type=float, default=0.5)
    parser.add_argument("--temperature-moves", type=int, default=50)
    parser.add_argument("--mcts-parallel-sims", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--min-train-steps", type=int, default=20)
    parser.add_argument("--buffer-min-to-train", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--reset-scheduler", action="store_true")
    parser.add_argument("--reset-buffer", action="store_true")
    parser.add_argument("--collapse-threshold", type=float, default=0.01)
    
    # Новые параметры FSF
    parser.add_argument("--fsf-every", type=int, default=5,
                        help="Запускать FSF дуэли каждые N итераций (0 = отключить)")
    parser.add_argument("--fsf-games", type=int, default=200,
                        help="Количество партий против FSF")
    parser.add_argument("--fsf-nodes", type=int, default=500,
                        help="Лимит узлов FSF (сила игры)")
    parser.add_argument("--fsf-timeout", type=int, default=3600,
                        help="Таймаут FSF дуэлей в секундах")
    
    args = parser.parse_args()
    
    # Создаём конфиг
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
    )
    
    # Печатаем расписание FSF
    if args.fsf_every > 0:
        print_fsf_schedule(args.fsf_every)
    else:
        print("\n📅 FSF интеграция: ОТКЛЮЧЕНА\n")
    
    # Запускаем тренировку с коллбэком
    train_with_fsf_callback(cfg, args)


def train_with_fsf_callback(cfg: Config, args):
    """
    Запускает тренировку с интеграцией FSF.
    Это обёртка вокруг оригинального train(), добавляющая FSF коллбэки.
    """
    
    # Импортируем здесь, чтобы избежать циклических импортов
    from train import (
        generate_games, train_epoch, policy_diversity_stats, print_diversity,
        ReplayBuffer, SelfPlayDataset, pack_sample, unpack_policy,
        CompactSample, Sample, Config
    )
    from model import CapablancaNet
    from mcts import UltraFastMCTS, POLICY_SIZE
    
    try:
        from capablanca_engine import CapablancaEngine
    except ImportError:
        raise ImportError("capablanca_engine not found. Build with: maturin develop --release")
    
    import os
    import time
    import pickle
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    # NVIDIA оптимизации
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    
    # === ИНИЦИАЛИЗАЦИЯ (копия из train.py) ===
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Тренировка на {device}")
    print(f"   Модель:        {cfg.num_channels}ch × {cfg.num_res_blocks} blocks")
    print(f"   Self-play:     {cfg.games_per_iter} игр/итер, {cfg.simulations} симуляций/ход")
    print(f"   FSF интеграция: каждые {args.fsf_every} итераций")
    print(f"   FSF параметры: {args.fsf_games} игр, {args.fsf_nodes} nodes\n")
    
    net = CapablancaNet(cfg.num_channels, cfg.num_res_blocks).to(device)
    net = net.to(memory_format=torch.channels_last)
    
    if hasattr(torch, "compile"):
        try:
            net = torch.compile(net)
            print("✅ torch.compile() применён\n")
        except Exception as e:
            print(f"⚠️  torch.compile() недоступен: {e}\n")
    
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fused=True,
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    
    def make_scheduler(opt):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=50, T_mult=2, eta_min=cfg.learning_rate * 0.05
        )
    
    scheduler = make_scheduler(optimizer)
    buffer = ReplayBuffer(cfg.buffer_max)
    
    # Загрузка буфера
    buffer_path = os.path.join(cfg.checkpoint_dir, "buffer.pkl")
    if os.path.exists(buffer_path) and not args.reset_buffer:
        try:
            with open(buffer_path, "rb") as f:
                buffer.data, buffer._ptr, buffer._full = pickle.load(f)
            print(f"📦 Загружен буфер: {len(buffer):,} позиций\n")
        except Exception as e:
            print(f"⚠️  Не удалось загрузить буфер: {e}\n")
    elif args.reset_buffer:
        print("🗑️  Буфер сброшен\n")
    
    # Загрузка чекпоинта
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
        for pg in optimizer.param_groups:
            pg['lr'] = cfg.learning_rate
        
        if args.reset_scheduler or "scheduler" not in ckpt:
            print("🔄 Scheduler сброшен\n")
        else:
            scheduler.load_state_dict(ckpt["scheduler"])
        
        start_iter = ckpt.get("iteration", 0) + 1
        print(f"📂 Загружен чекпоинт: {path} (итерация {start_iter})")
        
        if len(buffer) > 0:
            stats = policy_diversity_stats(buffer.data)
            print_diversity(stats, prefix="   Буфер diversity")
            print()
    
    # === ГЛАВНЫЙ ЦИКЛ ===
    for iteration in range(start_iter, 100_000):
        iter_start = time.time()
        
        # --- Self-play ---
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
        
        stats = policy_diversity_stats(samples)
        print_diversity(stats)
        
        # Автоадаптация температуры
        if stats.get('top1_mean', 0) > 0.85 and cfg.temperature < 2.0:
            cfg.temperature = min(cfg.temperature * 1.2, 2.0)
            print(f"  ⚠️  top1>{0.85:.2f} → temperature поднята до {cfg.temperature:.2f}")
        elif stats.get('entropy_mean', 0) > 3.5 and cfg.temperature > 0.8:
            cfg.temperature = max(cfg.temperature * 0.95, 0.8)
        print()
        
        # --- Тренировка ---
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
        
        collapsed = metrics['policy_loss'] < cfg.collapse_threshold
        collapse_warn = "  ⚠️  КОЛЛАПС ПОЛИТИКИ!" if collapsed else ""
        
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
        
        # --- Сохранение чекпоинта ---
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
                print(f"  💾 Буфер сохранён ({len(buffer):,} позиций)")
            except Exception as e:
                print(f"  ⚠️  Не удалось сохранить буфер: {e}")
            
            # === FSF ИНТЕГРАЦИЯ ===
            if args.fsf_every > 0 and should_run_fsf(iteration, args.fsf_every):
                print(f"\n{'='*60}")
                print(f"🔄 FSF ИНТЕГРАЦИЯ: Итерация {iteration} кратна {args.fsf_every}")
                print(f"{'='*60}")
                
                # Статистика до FSF
                stats_before = get_buffer_stats()
                if stats_before:
                    print(f"📊 Буфер до FSF: {stats_before['total']:,} позиций")
                    print(f"   Выигрыши: {stats_before['positive']}, "
                          f"Проигрыши: {stats_before['negative']}, "
                          f"Нейтральные: {stats_before['neutral']}")
                
                # Запуск FSF дуэлей
                success = run_fsf_duels(
                    model_path=path,
                    num_games=args.fsf_games,
                    fsf_nodes=args.fsf_nodes,
                    timeout_sec=args.fsf_timeout
                )
                
                if success:
                    # Перезагружаем буфер (play_fsf.py должен был обновить его)
                    if os.path.exists(buffer_path):
                        try:
                            with open(buffer_path, "rb") as f:
                                buffer.data, buffer._ptr, buffer._full = pickle.load(f)
                            
                            stats_after = get_buffer_stats()
                            if stats_after:
                                print(f"\n📊 Буфер после FSF: {stats_after['total']:,} позиций")
                                added = stats_after['total'] - stats_before['total']
                                print(f"   Добавлено: {added:,} позиций")
                                print(f"   Выигрыши: {stats_after['positive']}, "
                                      f"Проигрыши: {stats_after['negative']}")
                        except Exception as e:
                            print(f"⚠️  Не удалось перезагрузить буфер: {e}")
                
                print(f"\n{'='*60}")
                print(f"✅ FSF интеграция завершена. Продолжаем тренировку...")
                print(f"{'='*60}\n")
            
            print()  # Пустая строка для разделения


if __name__ == "__main__":
    main()
