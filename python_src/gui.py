import sys
import torch
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QListWidget)
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QRect

# Импорт движка и модели
try:
    from capablanca_engine import CapablancaEngine
    from model import CapablancaNet
except ImportError as e:
    print(f"Ошибка импорта! Убедитесь, что движок собран и model.py рядом.\n{e}")
    sys.exit(1)

# Цвета доски
COLOR_LIGHT = QColor("#D3D3D3")
COLOR_DARK = QColor("#808080")
COLOR_HIGHLIGHT = QColor(255, 255, 0, 100)
COLOR_HINT = QColor(0, 255, 0, 150)

# Фигуры
PIECE_CHARS = {
    # (Цвет: 0-белые, 1-черные), (Тип: 0-P, 1-N, 2-B, 3-R, 4-Q, 5-A, 6-C, 7-K)
    (0, 0): '♙', (0, 1): '♘', (0, 2): '♗', (0, 3): '♖', (0, 4): '♕',
    (0, 5): 'A', (0, 6): 'C', (0, 7): '♔',

    (1, 0): '♟', (1, 1): '♞', (1, 2): '♝', (1, 3): '♜', (1, 4): '♛',
    (1, 5): 'a', (1, 6): 'c', (1, 7): '♚',
}

def uci_to_sq(uci_str):
    f = ord(uci_str[0]) - ord('a')
    r = int(uci_str[1]) - 1
    return f, r

def sq_to_uci(f, r):
    return f"{chr(ord('a') + f)}{r + 1}"

class BoardWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = CapablancaEngine()
        self.flipped = False
        self.selected_sq = None
        self.legal_moves = []
        self.main_window = parent
        self.setMinimumSize(600, 480)

    def reset_game(self):
        self.engine = CapablancaEngine()
        self.selected_sq = None
        self._update_state()

    def _update_state(self):
        self.legal_moves = self.engine.get_legal_moves()
        self.main_window.analyze_position()
        self.update()

    def get_board_state(self):
        tensor = self.engine.get_board_tensor()
        board = {}
        # В Capablanca Chess 10x8: 20 плоскостей (8 фигур * 2 + доп. инфо)
        # Предполагаем порядок плоскостей: P, N, B, R, Q, A, C, K для белых, затем для черных
        for c in range(2):
            for p in range(8):
                plane_offset = (c * 8 + p) * 80
                for sq in range(80):
                    if tensor[plane_offset + sq] > 0.5:
                        file = sq % 10
                        rank = sq // 10
                        board[(file, rank)] = (c, p)
        return board

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        sq_size = min(self.width() // 10, self.height() // 8)
        offset_x = (self.width() - sq_size * 10) // 2
        offset_y = (self.height() - sq_size * 8) // 2

        board_state = self.get_board_state()

        for rank in range(8):
            for file in range(10):
                draw_f = 9 - file if self.flipped else file
                draw_r = rank if self.flipped else 7 - rank
                x, y = offset_x + draw_f * sq_size, offset_y + draw_r * sq_size

                painter.fillRect(x, y, sq_size, sq_size, COLOR_LIGHT if (file + rank) % 2 != 0 else COLOR_DARK)

                if self.selected_sq == (file, rank):
                    painter.fillRect(x, y, sq_size, sq_size, COLOR_HIGHLIGHT)

                if (file, rank) in board_state:
                    c, p = board_state[(file, rank)]
                    painter.setPen(Qt.white if c == 0 else Qt.black)
                    painter.setFont(QFont("Arial", sq_size // 2))
                    painter.drawText(QRect(x, y, sq_size, sq_size), Qt.AlignCenter, PIECE_CHARS[(c, p)])

                if self.selected_sq:
                    prefix = sq_to_uci(*self.selected_sq)
                    for m in self.legal_moves:
                        if m.startswith(prefix) and uci_to_sq(m[2:4]) == (file, rank):
                            painter.setBrush(COLOR_HINT)
                            painter.setPen(Qt.NoPen)
                            r_dot = sq_size // 8
                            painter.drawEllipse(x + sq_size//2 - r_dot, y + sq_size//2 - r_dot, r_dot*2, r_dot*2)

    def mousePressEvent(self, event):
        sq_size = min(self.width() // 10, self.height() // 8)
        offset_x = (self.width() - sq_size * 10) // 2
        offset_y = (self.height() - sq_size * 8) // 2
        f = (event.x() - offset_x) // sq_size
        r = (event.y() - offset_y) // sq_size

        if 0 <= f < 10 and 0 <= r < 8:
            file = 9 - f if self.flipped else f
            rank = r if self.flipped else 7 - r

            if self.selected_sq:
                move = sq_to_uci(*self.selected_sq) + sq_to_uci(file, rank)
                possible = [m for m in self.legal_moves if m.startswith(move)]
                if possible:
                    self.engine.make_move(possible[0])
                    self.selected_sq = None
                    self._update_state()
                    return

            board_state = self.get_board_state()
            if (file, rank) in board_state and board_state[(file, rank)][0] == self.engine.side_to_move():
                self.selected_sq = (file, rank)
            else:
                self.selected_sq = None
            self.update()

class CapablancaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Capablanca AI GUI")
        self.setStyleSheet("QMainWindow, QWidget { background-color: #1a1a1a; color: white; } "
                           "QPushButton { background-color: #333; border: 1px solid #555; padding: 10px; min-height: 20px; } "
                           "QListWidget { background-color: #000; border: 1px solid #333; }")

        self.net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        central = QWidget()
        layout = QHBoxLayout(central)

        self.board_widget = BoardWidget(self)
        layout.addWidget(self.board_widget, stretch=3)

        self.side_panel = QWidget()
        self.side_panel.setFixedWidth(300)
        side_layout = QVBoxLayout(self.side_panel)

        self.status_label = QLabel("Загрузите веса (64ch, 5blocks)")
        self.status_label.setWordWrap(True)
        self.status_label.setFont(QFont("Arial", 10, QFont.Bold))
        side_layout.addWidget(self.status_label)

        btn_load = QPushButton("Загрузить веса (.pth)")
        btn_load.clicked.connect(self.load_weights)
        side_layout.addWidget(btn_load)

        btn_flip = QPushButton("Повернуть доску")
        btn_flip.clicked.connect(self.flip_board)
        side_layout.addWidget(btn_flip)

        btn_reset = QPushButton("Сбросить партию")
        btn_reset.clicked.connect(self.reset_game)
        side_layout.addWidget(btn_reset)

        side_layout.addSpacing(20)
        side_layout.addWidget(QLabel("Анализ позиции:"))
        self.analysis_list = QListWidget()
        side_layout.addWidget(self.analysis_list)

        layout.addWidget(self.side_panel)
        self.setCentralWidget(central)
        self.board_widget._update_state()

    def load_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть веса", "", "Weights (*.pth)")
        if path:
            try:
                # ИСПРАВЛЕНО: Устанавливаем 64 канала и 5 блоков согласно вашей ошибке
                self.net = CapablancaNet(num_channels=64, num_res_blocks=5)

                checkpoint = torch.load(path, map_location=self.device, weights_only=False)

                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint

                self.net.load_state_dict(state_dict)
                self.net.to(self.device).eval()

                self.status_label.setText(f"Веса загружены успешно!\nDevice: {self.device}")
                self.analyze_position()
            except Exception as e:
                print(f"ОШИБКА:\n{traceback.format_exc()}")
                self.status_label.setText(f"Ошибка загрузки. Вероятно, параметры сети в файле отличаются от 64ch/5blocks.")

    def flip_board(self):
        self.board_widget.flipped = not self.board_widget.flipped
        self.board_widget.update()

    def reset_game(self):
        self.board_widget.reset_game()
        self.status_label.setText("Партия сброшена")

    def analyze_position(self):
        self.analysis_list.clear()
        if not self.net: return

        if self.board_widget.engine.is_game_over():
            self.status_label.setText("Игра завершена")
            return

        tensor_data = self.board_widget.engine.get_board_tensor()
        t = torch.tensor(tensor_data, dtype=torch.float32, device=self.device).view(1, 20, 8, 10)

        with torch.no_grad():
            policy_logits, value = self.net(t)
            probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        moves_with_probs = []
        for m in self.board_widget.legal_moves:
            idx = self.board_widget.engine.move_to_policy_idx(m)
            if idx is not None:
                moves_with_probs.append((m, probs[idx]))

        moves_with_probs.sort(key=lambda x: x[1], reverse=True)

        self.analysis_list.addItem(f"Value (оценка): {value.item():.3f}")
        self.analysis_list.addItem("-" * 15)
        for m, p in moves_with_probs[:10]:
            self.analysis_list.addItem(f"{m}: {p*100:.1f}%")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CapablancaGUI()
    window.show()
    sys.exit(app.exec_())
