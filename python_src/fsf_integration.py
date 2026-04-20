# fsf_integration.py — интеграция Fairy-Stockfish в тренировочный цикл

import os
import sys
import subprocess
import time
import pickle
from pathlib import Path
from typing import Optional

# Путь к папке с чекпоинтами
CHECKPOINT_DIR = Path("checkpoints")
BUFFER_PATH = CHECKPOINT_DIR / "buffer.pkl"


def run_fsf_duels(
    model_path: str,
    num_games: int = 200,
    fsf_nodes: int = 500,
    timeout_sec: int = 3600
) -> bool:
    """
    Запускает play_fsf.py для генерации партий против Fairy-Stockfish.
    
    Args:
        model_path: путь к .pth файлу модели
        num_games: количество партий (рекомендую 200-500)
        fsf_nodes: лимит узлов для FSF (500 = быстро, 1000 = сильнее)
        timeout_sec: максимальное время выполнения
    
    Returns:
        True если успешно, False если ошибка
    """
    if not os.path.exists("play_fsf.py"):
        print("⚠️  play_fsf.py не найден, пропускаем FSF интеграцию")
        return False
    
    cmd = [
        sys.executable, "play_fsf.py",
        "--model", model_path,
        "--games", str(num_games),
        "--fsf-nodes", str(fsf_nodes),
    ]
    
    print(f"\n{'='*60}")
    print(f"🤖 Запуск FSF дуэлей: {num_games} игр, {fsf_nodes} nodes")
    print(f"   Модель: {model_path}")
    print(f"{'='*60}\n")
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout_sec,
            capture_output=False,  # Показываем вывод в реальном времени
            text=True,
            check=True
        )
        elapsed = time.time() - start
        print(f"\n✅ FSF дуэли завершены за {elapsed:.1f}s")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"\n⏱️  FSF дуэли превысили лимит времени ({timeout_sec}s)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка FSF дуэлей: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        return False


def get_buffer_stats() -> Optional[dict]:
    """Возвращает статистику буфера, если он существует."""
    if not BUFFER_PATH.exists():
        return None
    
    try:
        with open(BUFFER_PATH, "rb") as f:
            data, ptr, full = pickle.load(f)
        
        # Подсчёт value распределения
        values = [sample[2] for sample in data]
        return {
            "total": len(data),
            "positive": sum(1 for v in values if v > 0.5),
            "negative": sum(1 for v in values if v < -0.5),
            "neutral": sum(1 for v in values if -0.5 <= v <= 0.5),
            "has_fsf": any(abs(v) > 0.9 for v in values)  # Признак FSF данных
        }
    except Exception as e:
        print(f"⚠️  Не удалось прочитать буфер: {e}")
        return None


def should_run_fsf(iteration: int, fsf_every: int = 5) -> bool:
    """
    Определяет, нужно ли запускать FSF дуэли.
    Запускается каждые N итераций, но НЕ на итерации 0.
    """
    return iteration > 0 and iteration % fsf_every == 0


def print_fsf_schedule(fsf_every: int = 5):
    """Печатает расписание FSF интеграции."""
    print(f"\n📅 FSF интеграция: каждые {fsf_every} итераций")
    print(f"   Будет запущена на итерациях: {fsf_every}, {fsf_every*2}, {fsf_every*3}...")
    print()
