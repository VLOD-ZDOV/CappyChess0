import os
import time
import pickle
import subprocess
import numpy as np
import torch
from typing import List
import sys
import argparse
from capablanca_engine import CapablancaEngine
from mcts import UltraFastMCTS
from model import CapablancaNet
from train import pack_sample, Sample, Config
def parse_args():
    parser = argparse.ArgumentParser(description="Play against Fairy-Stockfish")
    parser.add_argument("--model", type=str, default=None,
                        help="Путь к модели (если None — ищет последний чекпоинт)")
    parser.add_argument("--games", type=int, default=48,
                        help="Количество партий")
    parser.add_argument("--fsf-nodes", type=int, default=500,
                        help="Лимит узлов FSF")
    parser.add_argument("--output-buffer", type=str, default="checkpoints/buffer.pkl",
                        help="Путь к буферу для обновления")
    return parser.parse_args()
# Настройки
GAMES_TO_PLAY = 150
FSF_NODES_LIMIT = 300
MCTS_SIMULATIONS = 100
MAX_MOVES = 200
FSF_PATH = "./fairy-stockfish-largeboard_x86-64-bmi2" # Укажи свой путь

# Индексы из Rust: 0=PAWN, 1=KNIGHT, 2=BISHOP, 3=ROOK, 4=QUEEN, 5=ARCH, 6=CHANC
# Rust передает p_val = promo_idx + 1
PROMO_CHARS = {
    2: 'n', # KNIGHT (1) + 1
    3: 'b', # BISHOP (2) + 1
    4: 'r', # ROOK (3) + 1
    5: 'q', # QUEEN (4) + 1
    6: 'a', # ARCH (5) + 1
    7: 'c'  # CHANC (6) + 1
}

def int_to_uci(m: int) -> str:
    """Преобразует интовый ход Rust в строку UCI (например, e2e4 или f7f8q)"""
    p_val = m & 0b111 # Тип превращения
    t = (m >> 3) & 0x7F
    f = (m >> 10) & 0x7F

    file_f, rank_f = f % 10, f // 10
    file_t, rank_t = t % 10, t // 10

    uci = f"{chr(ord('a') + file_f)}{rank_f + 1}{chr(ord('a') + file_t)}{rank_t + 1}"

    # Если это превращение, добавляем букву из словаря
    if p_val in PROMO_CHARS:
        uci += PROMO_CHARS[p_val]

    return uci

def uci_to_int(uci: str, engine: CapablancaEngine) -> int:
    legal_moves = engine.get_legal_moves_int()

    # 1. Пробуем найти прямое совпадение
    for m in legal_moves:
        if int_to_uci(m) == uci:
            return m

    # 2. Если не нашли, выводим дебаг-информацию (только для первой ошибки)
    print(f"\n❓ Ход Стокфиша '{uci}' не найден в Rust-движке!")
    print(f"Доступные ходы движка: {[int_to_uci(m) for m in legal_moves[:10]]}...")

    # Спец-проверка для рокировки (если Стокфиш шлет e1g1, а у нас f1i1)
    # Здесь можно добавить ручной маппинг, если форматы разные

    return None

import datetime

def save_pgn(history: List[str], result: str, iteration: int, game_id: int, nn_side: int):
    now = datetime.datetime.now().strftime("%Y.%m.%d")
    white = "NeuralNet" if nn_side == 0 else "Fairy-Stockfish"
    black = "Fairy-Stockfish" if nn_side == 0 else "NeuralNet"

    with open("games.pgn", "a", encoding="utf-8") as f:
        f.write(f'[Event "Self-Play vs Fairy-Stockfish"]\n')
        f.write(f'[Site "https://www.pychess.org/analysis/capablanca"]\n')
        f.write(f'[Date "{now}"]\n')
        f.write(f'[White "{white}"]\n')
        f.write(f'[Black "{black}"]\n')
        f.write(f'[Result "{result}"]\n')
        f.write(f'[Variant "capablanca"]\n')
        # ВАЖНО: Используем FEN, который понимает PyChess (10 колонок)
        f.write(f'[FEN "rnabqkcbnr/pppppppppp/10/10/10/10/PPPPPPPPPP/RNABQKCBNR w KQkq - 0 1"]\n')
        f.write(f'[SetUp "1"]\n\n')

        # Записываем UCI ходы. Чтобы PyChess их съел,
        # лучше вставлять их в режиме "UCI" или использовать Analysis/Import.
        pgn_text = ""
        for i in range(0, len(history), 2):
            move_num = i // 2 + 1
            pgn_text += f"{move_num}. {history[i]} "
            if i + 1 < len(history):
                pgn_text += f"{history[i+1]} "

        f.write(pgn_text + f" {result}\n\n")

class FairyStockfishWrapper:
    def __init__(self, path: str):
        self.proc = subprocess.Popen([path], universal_newlines=True,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1)
        self.send("uci")
        self._wait_for("uciok")
        self.send("setoption name UCI_Variant value capablanca")
        self.send("isready")
        self._wait_for("readyok")

    def send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, target: str):
        while True:
            line = self.proc.stdout.readline().strip()
            if target in line: return line

    def get_best_move(self, move_history: List[str], nodes: int) -> str:
        self.send(f"position startpos moves {' '.join(move_history)}")
        self.send(f"go nodes {nodes}")
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                return line.split()[1]
    def close(self):
        self.send("quit")
        self.proc.wait()

# ==========================================
# Главный цикл
# ==========================================
def generate_fsf_games():
    cfg = Config(num_channels=128, num_res_blocks=10) # Твои настройки сети
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Загрузка сети
    net = CapablancaNet(cfg.num_channels, cfg.num_res_blocks).to(device)
    net = net.to(memory_format=torch.channels_last)

    ckpts = sorted([f for f in os.listdir(cfg.checkpoint_dir) if f.endswith(".pth")])
    if not ckpts:
        print("❌ Чекпоинты не найдены!")
        return

    latest_ckpt = os.path.join(cfg.checkpoint_dir, ckpts[-1])
    ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)

    if hasattr(net, "_orig_mod"):
        net._orig_mod.load_state_dict(ckpt["model"])
    else:
        net.load_state_dict(ckpt["model"])
    net.eval()

    print(f"🤖 Сеть загружена: {latest_ckpt}")

    # 2. Инициализация MCTS и Stockfish
    mcts = UltraFastMCTS(net, device, c_puct=1.25, batch_size=1, parallel_sims=1)
    try:
        fsf = FairyStockfishWrapper(FSF_PATH)
    except Exception as e:
        print(f"❌ Ошибка запуска Fairy-Stockfish: {e}")
        return

    all_samples: List[Sample] = []
    white_wins = black_wins = draws = 0

    print(f"⚔️ Запуск {GAMES_TO_PLAY} партий против Fairy-Stockfish (Лимит: {FSF_NODES_LIMIT} nodes)...")
    start_time = time.time()

    for game_idx in range(GAMES_TO_PLAY):
        engine = CapablancaEngine()
        history_tensors = []
        uci_history = []

        nn_side = 0 if game_idx % 2 == 0 else 1
        side_str = "Белые" if nn_side == 0 else "Черные"

        move_num = 0
        error_occurred = False

        while not engine.is_game_over() and move_num < MAX_MOVES:
            side = engine.side_to_move()
            legal_moves = engine.get_legal_moves_int()
            board_np = np.array(engine.get_board_tensor(), dtype=np.float32)

            if side == nn_side:
                # --- ХОД НЕЙРОСЕТИ ---
                pol = mcts.search_games([engine], MCTS_SIMULATIONS)[0]
                history_tensors.append((board_np, pol.copy(), side))

                # Выбор хода (температура 0.8 для разнообразия)
                move_indices = [engine.move_int_to_policy_idx(m) for m in legal_moves]
                raw = np.array([pol[idx] if idx is not None else 0.0 for idx in move_indices])
                raw = np.power(np.maximum(raw, 1e-8), 1.0 / 0.8)
                probs = raw / raw.sum()
                best_move = int(np.random.choice(legal_moves, p=probs))
                best_uci = int_to_uci(best_move)
            else:
                # --- ХОД FAIRY-STOCKFISH ---
                best_uci = fsf.get_best_move(uci_history, nodes=FSF_NODES_LIMIT)
                if best_uci == "(none)":
                    break
                best_move = uci_to_int(best_uci, engine)


                if best_move is None:
                    print(f"\n❌ РАССИНХРОНИЗАЦИЯ!")
                    print(f"Стокфиш предложил ход: {best_uci}")
                    print(f"История игры (UCI): {' '.join(uci_history)}")

                    # Печатаем доску только при ошибке!
                    if hasattr(engine, 'print_board'):
                        engine.print_board()
                    elif hasattr(engine, 'render'):
                        print(engine.render())

                    error_occurred = True
                    break  # Прерываем партию только если ход не найден

                # Создаем One-Hot вектор (учится на ходах Стокфиша)
                pol = np.zeros(7000, dtype=np.float32)
                pol_idx = engine.move_int_to_policy_idx(best_move)
                if pol_idx is not None:
                    pol[pol_idx] = 1.0
                history_tensors.append((board_np, pol, side))

            engine.make_move_int(best_move)
            uci_history.append(best_uci)
            move_num += 1

        if error_occurred:
            continue

        # Оценка результата и сохранение PGN
        result = engine.game_result()
        res_str = "1-0" if result == 1.0 else "0-1" if result == -1.0 else "1/2-1/2"

        if result == 1.0: white_wins += 1
        elif result == -1.0: black_wins += 1
        else: draws += 1

        # Сохраняем партию в файл
        save_pgn(uci_history, res_str, ckpt.get('iteration', 0), game_idx, nn_side)

        # Упаковка позиций для обучения
        for board, pol, side_t in history_tensors:
            v = result if side_t == 0 else -result
            all_samples.append(pack_sample(board, pol, float(v)))

        print(f"  [{game_idx+1}/{GAMES_TO_PLAY}] Сеть ({side_str}) vs FSF. Ходов: {move_num}. Итог: {res_str}")

    fsf.send("quit") # Корректно закрываем процесс Стокфиша
    elapsed = time.time() - start_time
    print(f"\n✅ Завершено за {elapsed:.1f} сек. Сгенерировано позиций: {len(all_samples)}")
    print(f"Счет: Белые={white_wins}, Черные={black_wins}, Ничьи={draws}")

    # 3. Обновление буфера (buffer.pkl)
    if len(all_samples) > 0:
        buffer_path = os.path.join(cfg.checkpoint_dir,"buffer.pkl")
        from train import ReplayBuffer
        buffer = ReplayBuffer(cfg.buffer_max)

        if os.path.exists(buffer_path):
            try:
                with open(buffer_path, "rb") as f:
                    buffer.data, buffer._ptr, buffer._full = pickle.load(f)
            except:
                print("⚠️ Не удалось загрузить старый буфер, создаем новый.")

        old_size = len(buffer)
        buffer.push(all_samples)
        new_size = len(buffer)

        with open(buffer_path, "wb") as f:
            pickle.dump((buffer.data, buffer._ptr, buffer._full), f)

        print(f"💾 Буфер обновлен: {old_size:,} -> {new_size:,} позиций")
def update_buffer(new_samples, buffer_path="checkpoints/buffer.pkl"):
    """Добавляет новые позиции в существующий буфер."""
    import pickle

    if os.path.exists(buffer_path):
        with open(buffer_path, "rb") as f:
            data, ptr, full = pickle.load(f)
    else:
        data, ptr, full = [], 0, False

    # Добавляем новые позиции (с value от FSF — обычно ±1.0)
    for sample in new_samples:
        if not full:
            data.append(sample)
            if len(data) >= 500_000:  # max_size
                full = True
        else:
            data[ptr] = sample
            ptr = (ptr + 1) % len(data)

    with open(buffer_path, "wb") as f:
        pickle.dump((data, ptr, full), f)

    print(f"💾 Буфер обновлен: {len(data):,} позиций (+{len(new_samples)})")

if __name__ == "__main__":
    args = parse_args()
    generate_fsf_games()
