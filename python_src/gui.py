import sys
import torch
import traceback
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QListWidget, QSpinBox, QButtonGroup, QRadioButton,
                             QGroupBox)
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QPolygonF
from PyQt5.QtCore import Qt, QRect, QPointF, QThread, pyqtSignal

try:
    from capablanca_engine import CapablancaEngine
    from model import CapablancaNet
    from mcts import UltraFastMCTS, MCTSNode, VIRTUAL_LOSS
except ImportError as e:
    print(f"Ошибка импорта! {e}")
    traceback.print_exc()
    sys.exit(1)

PIECE_CHARS = {
    (0, 0): '♙', (0, 1): '♘', (0, 2): '♗', (0, 3): '♖', (0, 4): '♕',
    (0, 5): 'A', (0, 6): 'C', (0, 7): '♔',
    (1, 0): '♟', (1, 1): '♞', (1, 2): '♝', (1, 3): '♜', (1, 4): '♛',
    (1, 5): 'a', (1, 6): 'c', (1, 7): '♚'
}


def decode_move(m_int):
    p_val = m_int & 0b111
    to_sq = (m_int >> 3) & 0x7F
    from_sq = (m_int >> 10) & 0x7F
    promo = ['q', 'r', 'b', 'n', 'a', 'c'][p_val - 1] if p_val > 0 else None
    return from_sq, to_sq, promo


def move_to_uci(m_int):
    f, t, p = decode_move(m_int)
    def sq_u(s):
        return f"{chr(ord('a') + (s % 10))}{(s // 10) + 1}"
    return sq_u(f) + sq_u(t) + (p if p else "")


class MCTSThread(QThread):
    """
    Поток GPU-MCTS. НЕ делает ходы — только накапливает дерево и шлёт статистику.
    analyze_for: 0 = белые, 1 = чёрные, -1 = всегда (оба цвета)
    """
    update_signal = pyqtSignal(list, float, int)   # [(move, prob)], value, sims

    def __init__(self, move_history, net, device, batch_size=64,
                 max_sims=100_000, analyze_for=-1):
        super().__init__()
        self.move_history = move_history.copy()
        self.net = net
        self.device = device
        self.batch_size = batch_size
        self.running = True
        self.max_sims = max_sims
        self.analyze_for = analyze_for  # -1=все, 0=белые, 1=чёрные

    def run(self):
        try:
            # --- Восстанавливаем позицию в локальном движке ---
            engine = CapablancaEngine()
            for m in self.move_history:
                engine.make_move_int(m)

            # --- Проверяем чей ход — нужно ли анализировать ---
            current_side = engine.side_to_move()  # 0=белые, 1=чёрные
            if self.analyze_for != -1 and self.analyze_for != current_side:
                # Не наш ход — ждём пока не остановят
                while self.running:
                    self.msleep(100)
                return

            # --- MCTS ---
            mcts = UltraFastMCTS(self.net, self.device,
                                 c_puct=1.25, batch_size=self.batch_size)
            root = MCTSNode(None, -1, 1.0)

            # Начальный expansion корня (1 инференс)
            tensor = np.array(engine.get_board_tensor(), dtype=np.float32)
            policy, _ = mcts._infer([tensor])
            mcts._expand_node(root, engine, policy[0])

            total_sims = 0
            report_every = 200   # чаще обновлять GUI

            while self.running and total_sims < self.max_sims:
                tensors = []   # батч тензоров для GPU
                metas = []     # (node, move_stack)

                # Собираем батч листьев
                attempts = 0
                while (len(tensors) < self.batch_size and
                       attempts < self.batch_size * 3 and self.running):

                    node, move_stack = mcts._select(root)

                    if node.is_terminal:
                        # Терминал — сразу backup, GPU не нужен
                        sim_eng = engine.copy()
                        for m in move_stack:
                            sim_eng.make_move_int(m)
                        result = sim_eng.game_result()
                        depth_parity = len(move_stack) % 2
                        value = result if depth_parity == 0 else -result
                        mcts._backup(node, move_stack, value)
                        attempts += 1
                        continue

                    # Воспроизводим позицию листа
                    sim_eng = engine.copy()
                    for m in move_stack:
                        sim_eng.make_move_int(m)

                    tensors.append(np.array(sim_eng.get_board_tensor(), dtype=np.float32))
                    metas.append((node, move_stack))
                    mcts._apply_virtual_loss(node, VIRTUAL_LOSS)
                    attempts += 1

                # --- Батчевый инференс на GPU ---
                if tensors:
                    policies, values = mcts._infer(tensors)

                    for i, (node, move_stack) in enumerate(metas):
                        sim_eng = engine.copy()
                        for m in move_stack:
                            sim_eng.make_move_int(m)

                        if not node.is_expanded:
                            mcts._expand_node(node, sim_eng, policies[i])

                        # Снимаем виртуальный лосс и пишем настоящий результат
                        mcts._apply_virtual_loss(node, -VIRTUAL_LOSS)
                        mcts._backup(node, move_stack, float(values[i]))

                    total_sims += len(tensors)

                # --- Шлём статистику в GUI ---
                if total_sims % report_every < self.batch_size or not self.running:
                    self._emit_stats(root, total_sims)

                self.msleep(2)   # даём GUI подышать

            self._emit_stats(root, total_sims)

        except Exception as e:
            print(f"[MCTSThread] Критическая ошибка: {e}")
            traceback.print_exc()

    def _emit_stats(self, root, total_sims):
        if not root.children:
            return
        total_visits = sum(c.visits for c in root.children.values())
        if total_visits == 0:
            return

        stats = []
        for m_int, node in root.children.items():
            if node.visits > 0:
                stats.append((m_int, node.visits / total_visits))
        stats.sort(key=lambda x: x[1], reverse=True)

        val = root.value_sum / root.visits if root.visits > 0 else 0.0
        self.update_signal.emit(stats[:5], float(val), total_sims)


class BoardWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setMinimumSize(800, 600)
        self.engine = CapablancaEngine()
        self.selected_sq = None
        self.flipped = False
        self.legal_moves = self.engine.get_legal_moves_int()
        self.top_moves_data = []   # [(m_int, prob, QColor)]

    # ---------- Геометрия доски 10x8 ----------
    def get_sq_rect(self, sq):
        r, f = divmod(sq, 10)
        v_rank = r if self.flipped else (7 - r)
        v_file = (9 - f) if self.flipped else f
        w, h = self.width() / 10, self.height() / 8
        return QRect(int(v_file * w), int(v_rank * h), int(w), int(h))

    def get_sq_at(self, pos):
        w, h = self.width() / 10, self.height() / 8
        vf, vr = int(pos.x() // w), int(pos.y() // h)
        if 0 <= vf < 10 and 0 <= vr < 8:
            r = vr if self.flipped else (7 - vr)
            f = (9 - vf) if self.flipped else vf
            return r * 10 + f
        return None

    # ---------- Отрисовка ----------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        tensor = self.engine.get_board_tensor()

        # Клетки
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        for r in range(8):
            for f in range(10):
                sq = r * 10 + f
                rect = self.get_sq_rect(sq)
                dark = (r + f) % 2 != 0
                bg = QColor("#ebecd0") if dark else QColor("#779556")
                tc = QColor("#779556") if dark else QColor("#ebecd0")
                painter.fillRect(rect, bg)

                if self.selected_sq == sq:
                    painter.fillRect(rect, QColor(255, 255, 0, 120))

                painter.setPen(tc)
                if (not self.flipped and r == 0) or (self.flipped and r == 7):
                    painter.drawText(rect.adjusted(0, 0, -4, -4),
                                     Qt.AlignBottom | Qt.AlignRight, chr(ord('a') + f))
                if (not self.flipped and f == 0) or (self.flipped and f == 9):
                    painter.drawText(rect.adjusted(4, 4, 0, 0),
                                     Qt.AlignTop | Qt.AlignLeft, str(r + 1))

        # Фигуры
        painter.setFont(QFont("Arial", int(self.height() / 8 * 0.65)))
        for sq in range(80):
            piece = None
            for i in range(8):
                if tensor[i * 80 + sq] > 0.5:
                    piece = (0, i)
                    break
                if tensor[(8 + i) * 80 + sq] > 0.5:
                    piece = (1, i)
                    break
            if piece:
                painter.setPen(Qt.black)
                painter.drawText(self.get_sq_rect(sq), Qt.AlignCenter,
                                 PIECE_CHARS.get(piece, '?'))

        # Подсветка легальных ходов
        if self.selected_sq is not None:
            best_dest = None
            if self.top_moves_data:
                bm = self.top_moves_data[0][0]
                fs, ts, _ = decode_move(bm)
                if fs == self.selected_sq:
                    best_dest = ts

            for m in self.legal_moves:
                fs, ts, _ = decode_move(m)
                if fs == self.selected_sq:
                    c = self.get_sq_rect(ts).center()
                    if ts == best_dest:
                        painter.setBrush(QColor(0, 255, 0, 180))
                        painter.drawEllipse(c, 12, 12)
                    else:
                        painter.setBrush(QColor(0, 0, 0, 50))
                        painter.drawEllipse(c, 7, 7)

        # Стрелки топ-ходов от MCTS
        for m_int, prob, color in self.top_moves_data:
            f, t, _ = decode_move(m_int)
            self.draw_arrow(painter, f, t, color, prob)

    def draw_arrow(self, painter, f, t, color, p):
        s = QPointF(self.get_sq_rect(f).center())
        e = QPointF(self.get_sq_rect(t).center())
        c = QColor(color)
        c.setAlpha(int(max(80, p * 255)))
        painter.setPen(QPen(c, 5, Qt.SolidLine, Qt.RoundCap))
        painter.setBrush(c)
        painter.drawLine(s, e)
        ang = np.arctan2(e.y() - s.y(), e.x() - s.x())
        p1 = e - QPointF(np.cos(ang - 0.5) * 16, np.sin(ang - 0.5) * 16)
        p2 = e - QPointF(np.cos(ang + 0.5) * 16, np.sin(ang + 0.5) * 16)
        painter.drawPolygon(QPolygonF([e, p1, p2]))

    # ---------- Ввод ----------
    def mousePressEvent(self, event):
        sq = self.get_sq_at(event.pos())
        if sq is None:
            return

        if self.selected_sq is not None:
            for m in self.legal_moves:
                f, t, _ = decode_move(m)
                if f == self.selected_sq and t == sq:
                    # Ход человека — останавливаем расчёт, применяем ход
                    self.main_window.stop_thinking()
                    self.engine.make_move_int(m)
                    self.legal_moves = self.engine.get_legal_moves_int()
                    self.selected_sq = None
                    self.top_moves_data = []          # сбрасываем стрелки
                    self.main_window.add_move(m)
                    self.update()
                    return

        self.selected_sq = sq
        self.update()


class CapablancaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Capablanca AI — GPU MCTS Analyzer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.think_thread = None
        self.move_history = []

        # ------- Layout -------
        main = QWidget()
        layout = QHBoxLayout(main)

        self.board_widget = BoardWidget(self)
        layout.addWidget(self.board_widget, 5)

        sidebar = QVBoxLayout()

        # ------- Загрузка модели -------
        btn_load = QPushButton("Загрузить веса")
        btn_load.clicked.connect(self.load_weights)
        sidebar.addWidget(btn_load)

        # ------- Анализировать за -------
        grp_side = QGroupBox("Анализировать за")
        side_layout = QVBoxLayout(grp_side)
        self.radio_all   = QRadioButton("Всех (всегда)")
        self.radio_white = QRadioButton("Белых")
        self.radio_black = QRadioButton("Чёрных")
        self.radio_all.setChecked(True)
        self._side_group = QButtonGroup()
        for i, r in enumerate([self.radio_all, self.radio_white, self.radio_black]):
            self._side_group.addButton(r, i)
            side_layout.addWidget(r)
        sidebar.addWidget(grp_side)

        # ------- Лимит симуляций -------
        grp_lim = QGroupBox("Лимит симуляций")
        lim_layout = QHBoxLayout(grp_lim)
        self.spin_sims = QSpinBox()
        self.spin_sims.setMinimum(0)           # 0 = безлимитно
        self.spin_sims.setMaximum(10_000_000)
        self.spin_sims.setSingleStep(1000)
        self.spin_sims.setValue(100_000)
        self.spin_sims.setSpecialValueText("∞ безлимитно")
        lim_layout.addWidget(self.spin_sims)
        sidebar.addWidget(grp_lim)

        # ------- Кнопки управления -------
        self.btn_think = QPushButton("▶ Думать (GPU MCTS)")
        self.btn_think.clicked.connect(self.toggle_thinking)
        self.btn_think.setEnabled(False)
        self.btn_think.setStyleSheet("font-weight: bold; font-size: 13px;")
        sidebar.addWidget(self.btn_think)

        btn_flip = QPushButton("Перевернуть доску")
        btn_flip.clicked.connect(self.flip_board)
        sidebar.addWidget(btn_flip)

        self.status = QLabel("Загрузите модель, чтобы начать анализ")
        self.status.setWordWrap(True)
        sidebar.addWidget(self.status)

        sidebar.addWidget(QLabel("Топ-5 (MCTS):"))
        self.move_list = QListWidget()
        self.move_list.setMaximumHeight(180)
        sidebar.addWidget(self.move_list)

        sidebar.addStretch()
        layout.addLayout(sidebar, 2)
        self.setCentralWidget(main)

    def flip_board(self):
        self.board_widget.flipped = not self.board_widget.flipped
        self.board_widget.update()

    def add_move(self, m_int):
        """Человек сделал ход — просто запоминаем, НЕ запускаем анализ автоматически."""
        self.move_history.append(m_int)

    # ------- Модель -------
    def load_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить веса", "", "*.pth")
        if not path:
            return
        try:
            ckpt = torch.load(path, map_location=self.device)
            sd = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
            sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
                  for k, v in sd.items()}

            ch = sd.get("input_conv.net.0.weight",
                        sd.get("input_block.0.weight")).shape[0]
            bl = len([k for k in sd.keys()
                      if "res_blocks" in k and "conv1.weight" in k])

            self.net = CapablancaNet(num_channels=ch, num_res_blocks=bl)
            self.net.load_state_dict(sd)
            self.net.to(self.device).eval()

            self.status.setText(f"Готово: {ch}ch×{bl}bl | {self.device}")
            self.btn_think.setEnabled(True)
        except Exception as e:
            self.status.setText(f"Ошибка: {e}")
            traceback.print_exc()

    # ------- MCTS управление -------
    def toggle_thinking(self):
        if self.think_thread and self.think_thread.isRunning():
            self.stop_thinking()
        else:
            self.start_thinking()

    def start_thinking(self):
        if not self.net:
            return
        self.stop_thinking()

        # Читаем настройки из UI
        side_id = self._side_group.checkedId()
        analyze_for = [-1, 0, 1][side_id]   # 0=все → -1, 1=белые → 0, 2=чёрные → 1
        max_sims = self.spin_sims.value() or 10_000_000  # 0 в спинбоксе = безлимитно

        side_label = ["всех", "белых", "чёрных"][side_id]
        lim_label = "∞" if self.spin_sims.value() == 0 else f"{max_sims:,}"
        self.btn_think.setText("⏹ Остановить")
        self.move_list.clear()
        self.status.setText(f"MCTS: за {side_label}, лимит {lim_label}…")

        self.think_thread = MCTSThread(
            self.move_history, self.net, self.device,
            max_sims=max_sims, analyze_for=analyze_for,
        )
        self.think_thread.update_signal.connect(self.on_mcts_update)
        self.think_thread.start()

    def stop_thinking(self):
        if self.think_thread and self.think_thread.isRunning():
            self.think_thread.running = False
            self.think_thread.wait(1500)

        self.btn_think.setText("▶ Думать (GPU MCTS)")
        self.board_widget.top_moves_data = []
        self.board_widget.update()

    def on_mcts_update(self, moves, val, sims):
        self.move_list.clear()
        self.board_widget.top_moves_data = []

        colors = [
            QColor(0, 200, 0),     # 1 зелёный
            QColor(0, 160, 255),   # 2 синий
            QColor(255, 140, 0),   # 3 оранжевый
            QColor(180, 100, 220), # 4 фиолетовый
            QColor(120, 120, 120)  # 5 серый
        ]

        for i, (m_int, prob) in enumerate(moves):
            uci = move_to_uci(m_int)
            self.move_list.addItem(f"{i + 1}. {uci}  ({prob * 100:.1f}%)")
            self.board_widget.top_moves_data.append((m_int, prob, colors[i]))

        winrate = (val + 1.0) / 2.0
        side = self.board_widget.engine.side_to_move()
        side_str = "Белые" if side == 0 else "Чёрные"
        lim = self.spin_sims.value()
        done = lim > 0 and sims >= lim
        self.status.setText(
            f"Ход: {side_str} | Сим: {sims:,} {'✓' if done else ''} | "
            f"Оценка: {winrate:.1%}"
        )
        self.board_widget.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CapablancaGUI()
    gui.show()
    sys.exit(app.exec_())
