"""Capablanca AI — Nibbler-style real-time analysis GUI.

A dark-themed analysis frontend for the Capablanca Chess network, inspired by
Nibbler (rooklift/nibbler). Talks to the network directly through onnx_engine.py
and renders the native 10x8 board with Archbishop and Chancellor.

Features:
  * Live MCTS analysis with a ranked move infobox (N / P / Q / WDL).
  * Opening Book / Explorer with WDL percentages from self-play archive.
  * Interactive Archive Game Browser with filters and game replay.
  * Transposition-aware search and tree reuse across moves.
  * Eval bar, per-game winrate graph and table of analyzed positions.
"""

import math
import re
import os
import subprocess
import sys
import json
import time
import traceback

import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QPushButton, QLabel,
                             QFileDialog, QSpinBox, QGroupBox, QDialog, QComboBox, QMessageBox,
                             QCheckBox, QSizePolicy, QShortcut, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QFrame, QTabWidget, QScrollArea)
from PyQt5.QtGui import (QPainter, QColor, QFont, QPen, QPolygonF, QKeySequence)
from PyQt5.QtCore import Qt, QRect, QRectF, QPointF, QThread, pyqtSignal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import archive as A
except ImportError:
    A = None

try:
    from capablanca_engine import CapablancaEngine
    from onnx_engine import OnnxEngine, VIRTUAL_LOSS
except ImportError as e:
    _venv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "venv")
    print(f"Ошибка импорта: {e}\n")
    if "capablanca_engine" in str(e):
        _py = os.path.join(_venv, "Scripts" if os.name == "nt" else "bin",
                           "python.exe" if os.name == "nt" else "python")
        if os.path.exists(_py):
            print("Похоже, GUI запущен системным Python. Запускай питоном из окружения проекта:\n")
            print(f"    {_py} {os.path.abspath(__file__)}\n")
        else:
            print("Движок не собран. Из корня проекта:\n")
            print("    python -m venv venv")
            print("    venv/bin/pip install -r requirements.txt")
            print("    venv/bin/pip install ./rust_engine\n")
    else:
        traceback.print_exc()
    sys.exit(1)


# ───────────────────────────── Constants ──────────────────────────────────

PIECE_GLYPHS = {0: '♟', 1: '♞', 2: '♝', 3: '♜', 4: '♛', 5: 'A', 6: 'C', 7: '♚'}
_PROMO_FROM_VAL = [None, None, 'n', 'b', 'r', 'q', 'a', 'c']
_PROMO_LABELS = [('q', 'Ферзь ♛'), ('r', 'Ладья ♜'), ('b', 'Слон ♝'),
                 ('n', 'Конь ♞'), ('a', 'Архиепископ A'), ('c', 'Канцлер C')]

RANK_COLORS = [QColor("#3aa655"), QColor("#3d7fd6"), QColor("#d98a1f"),
               QColor("#9b59b6"), QColor("#1aa3a3")]
GREY = QColor("#6f6f6f")

FPU_REDUCTION = 0.330
MAX_SELECT_DEPTH = 160
TT_MAX_NODES = 500_000
PV_DEPTH = 10

BG = "#262626"
BG2 = "#2f2f2f"
BG3 = "#383838"
FG = "#d8d8d8"
ACCENT = "#5b9bd5"
LIGHT_SQ = QColor("#e9edcc")
DARK_SQ = QColor("#7a9b5b")

W_COL = QColor("#3aa655")   # Win (Green)
D_COL = QColor("#7f8c8d")   # Draw (Grey)
L_COL = QColor("#cf4f3a")   # Loss (Red)


# ───────────────────────────── Move helpers ───────────────────────────────

def decode_move(m_int):
    p_val = m_int & 0b111
    to_sq = (m_int >> 3) & 0x7F
    from_sq = (m_int >> 10) & 0x7F
    promo = _PROMO_FROM_VAL[p_val] if 0 < p_val < len(_PROMO_FROM_VAL) else None
    return from_sq, to_sq, promo


def sq_to_str(s):
    return f"{chr(ord('a') + (s % 10))}{(s // 10) + 1}"


def move_to_uci(m_int):
    f, t, p = decode_move(m_int)
    return sq_to_str(f) + sq_to_str(t) + (p if p else "")


def value_to_wdl(v):
    v = max(-1.0, min(1.0, float(v)))
    p_win = max(0.0, v)
    p_loss = max(0.0, -v)
    p_draw = max(0.0, 1.0 - abs(v))
    return p_win, p_draw, p_loss


_HASH_WARNED = [False]


def position_key(engine):
    fn = getattr(engine, "position_hash", None)
    if fn is not None:
        return fn()
    if not _HASH_WARNED[0]:
        _HASH_WARNED[0] = True
        print("⚠️  engine.position_hash отсутствует — используется запасной ключ.")
    return hash((tuple(sorted(engine.get_pieces())), engine.side_to_move()))


# ─────────────────────── Фоновая загрузка архива ──────────────────────────

class ArchiveLoaderThread(QThread):
    """Индексирует партии и дебютную книгу без блокировки интерфейса."""
    progress = pyqtSignal(str)
    loaded = pyqtSignal(object, object, object)
    error = pyqtSignal(str)

    def __init__(self, directory, max_book_ply=32):
        super().__init__()
        self.directory = directory
        self.max_book_ply = max_book_ply

    def run(self):
        try:
            if A is None:
                raise ImportError("Модуль archive.py не найден рядом с gui.py")
            self.progress.emit("Чтение метаданных архива...")
            ar = A.Archive(self.directory)
            n_games = len(ar.g["result"])

            p_games = ar.p["game"]
            p_moves = ar.p["move_raw"]
            p_idxs = ar.p["move"]
            p_sides = ar.p["side"]

            self.progress.emit("Сборка цепочек ходов...")
            games_moves = [[] for _ in range(n_games)]
            for g_id, m_raw, m_idx, s in zip(p_games, p_moves, p_idxs, p_sides):
                if g_id < n_games:
                    if m_raw >= 0:
                        mv = int(m_raw)
                    else:
                        mv = A.idx_to_move(m_idx, s)
                    if mv is not None:
                        games_moves[g_id].append(mv)

            self.progress.emit(f"Построение книги дебютов из {n_games:,} партий...")
            book = {}
            for g_id, moves in enumerate(games_moves):
                res = int(ar.g["result"][g_id])
                eng = CapablancaEngine()
                for ply_num, mv in enumerate(moves[:self.max_book_ply]):
                    k = position_key(eng)
                    stm = eng.side_to_move()
                    if res == 0:
                        kind = "d"
                    elif (res > 0 and stm == 0) or (res < 0 and stm == 1):
                        kind = "w"
                    else:
                        kind = "l"

                    entry = book.setdefault(k, {"total": 0, "side": stm, "moves": {}})
                    entry["total"] += 1
                    m_stats = entry["moves"].setdefault(mv, {"w": 0, "d": 0, "l": 0, "n": 0})
                    m_stats["n"] += 1
                    m_stats[kind] += 1

                    try:
                        eng.make_move_int(mv)
                    except Exception:
                        break

            self.loaded.emit(ar, games_moves, book)
        except Exception as e:
            self.error.emit(str(e))


# ───────────────────── Transposition-aware search tree ────────────────────

class _TNode:
    __slots__ = ("visits", "value_sum", "draw_sum", "vloss", "is_expanded",
                 "is_terminal", "children")

    def __init__(self):
        self.visits = 0
        self.value_sum = 0.0
        self.draw_sum = 0.0
        self.vloss = 0
        self.is_expanded = False
        self.is_terminal = False
        self.children = {}

    def q(self):
        d = self.visits + self.vloss
        return (self.value_sum + self.vloss) / d if d > 0 else 0.0


class _TEdge:
    __slots__ = ("prior", "move", "child", "child_hash")

    def __init__(self, prior, move, child, child_hash):
        self.prior = prior
        self.move = move
        self.child = child
        self.child_hash = child_hash


# ───────────────────────────── Search thread ──────────────────────────────

class SearchThread(QThread):
    update = pyqtSignal(object)

    def __init__(self, move_history, mcts, max_sims, ttable, contempt=0.0):
        super().__init__()
        self.move_history = list(move_history)
        self.mcts = mcts
        self.max_sims = max_sims if max_sims > 0 else 50_000_000
        self.c_puct = mcts.c_puct
        self.contempt = float(contempt)
        self.running = True
        self.ttable = ttable
        self.tt_hits = 0
        self.reused = 0
        self.root_engine = None
        self.root_hash = 0

    def _replay(self, moves):
        eng = self.root_engine.copy()
        for m in moves:
            eng.make_move_int(m)
        return eng

    def _best_edge(self, node):
        parent_q = node.q()
        sqrt_n = math.sqrt(max(node.visits + node.vloss, 1))
        visited_pol = sum(e.prior for e in node.children.values()
                          if e.child.visits > 0 or e.child.vloss > 0)
        fpu = max(parent_q - FPU_REDUCTION * math.sqrt(max(visited_pol, 0.0)), -1.0)
        cont = self.contempt
        best, best_s = None, -1e18
        for e in node.children.values():
            c = e.child
            started = c.visits + c.vloss
            if started > 0:
                q_child = (c.value_sum + c.vloss + cont * c.draw_sum) / started
                q_in_parent = -q_child
            else:
                q_in_parent = fpu
            s = q_in_parent + self.c_puct * e.prior * sqrt_n / (1 + started)
            if s > best_s:
                best_s, best = s, e
        return best

    def _select(self, root):
        node = root
        path_nodes = [root]
        path_moves = []
        path_hashes = {self.root_hash}
        while (node.is_expanded and not node.is_terminal and node.children
               and len(path_moves) < MAX_SELECT_DEPTH):
            edge = self._best_edge(node)
            if edge.child_hash in path_hashes:
                return path_nodes, path_moves, 'rep'
            node = edge.child
            path_nodes.append(node)
            path_moves.append(edge.move)
            path_hashes.add(edge.child_hash)
        return path_nodes, path_moves, 'leaf'

    def _expand(self, leaf, sim, policy_vec):
        legal = sim.get_legal_moves_int()
        if not legal:
            leaf.is_terminal = True
            leaf.is_expanded = True
            return
        idxs = [sim.move_int_to_policy_idx(m) for m in legal]
        priors = np.array(
            [float(policy_vec[i]) if (i is not None and 0 <= i < len(policy_vec))
             else 1e-8 for i in idxs], dtype=np.float64)
        s = priors.sum()
        priors = priors / s if s > 1e-12 else np.ones(len(legal)) / len(legal)
        for m, pr in zip(legal, priors):
            child_eng = sim.copy()
            child_eng.make_move_int(m)
            h = position_key(child_eng)
            node = self.ttable.get(h)
            if node is None:
                node = _TNode()
                self.ttable[h] = node
            else:
                self.tt_hits += 1
            leaf.children[m] = _TEdge(float(pr), m, node, h)
        leaf.is_expanded = True

    @staticmethod
    def _vloss(path, delta):
        for n in path:
            n.vloss = max(0, n.vloss + delta)

    @staticmethod
    def _backup(path, leaf_value, leaf_draw=0.0):
        sign = 1.0
        for n in reversed(path):
            n.visits += 1
            n.value_sum += leaf_value * sign
            n.draw_sum  += leaf_draw
            sign = -sign

    @staticmethod
    def _terminal_value(sim):
        r = sim.game_result()
        return r if sim.side_to_move() == 0 else -r

    @staticmethod
    def _pv(node, max_depth):
        line = []
        seen = set()
        for _ in range(max_depth):
            if node.is_terminal or not node.is_expanded or not node.children:
                break
            best = max(node.children.values(), key=lambda e: e.child.visits)
            if best.child.visits == 0:
                break
            line.append(best.move)
            if best.child_hash in seen:
                break
            seen.add(best.child_hash)
            node = best.child
        return line

    def run(self):
        try:
            engine = CapablancaEngine()
            for m in self.move_history:
                engine.make_move_int(m)
            stm = engine.side_to_move()

            if engine.is_game_over():
                self.update.emit({"game_over": True, "result": engine.game_result(),
                                  "stm": stm, "moves": [], "root_q": 0.0,
                                  "sims": 0, "merges": 0, "finished": True})
                return

            self.root_engine = engine
            self.root_hash = position_key(engine)
            mcts = self.mcts

            root = self.ttable.get(self.root_hash)
            if root is None:
                root = _TNode()
                self.ttable[self.root_hash] = root
            if not root.is_expanded:
                tensor = np.asarray(engine.get_board_tensor(), dtype=np.float32)
                policy, _, _, _ = mcts._infer([tensor])
                self._expand(root, engine, policy[0])
            self.reused = root.visits

            child_nn = {}
            total = root.visits
            last_emit = 0.0
            bs = mcts.batch_size

            while self.running and total < self.max_sims:
                tensors, pending = [], []
                in_flight = set()
                attempts = 0
                while (len(tensors) < bs and attempts < bs * 4 and self.running):
                    attempts += 1
                    path_nodes, path_moves, kind = self._select(root)
                    leaf = path_nodes[-1]
                    if kind == 'leaf' and id(leaf) in in_flight:
                        continue

                    if kind == 'rep':
                        self._backup(path_nodes, 0.0, 1.0)
                        total += 1
                        continue
                    if leaf.is_terminal:
                        sim = self._replay(path_moves)
                        tv = self._terminal_value(sim)
                        self._backup(path_nodes, tv, 1.0 if tv == 0.0 else 0.0)
                        total += 1
                        continue

                    sim = self._replay(path_moves)
                    if sim.is_game_over():
                        leaf.is_terminal = True
                        leaf.is_expanded = True
                        tv = self._terminal_value(sim)
                        self._backup(path_nodes, tv, 1.0 if tv == 0.0 else 0.0)
                        total += 1
                        continue

                    tensors.append(np.asarray(sim.get_board_tensor(), dtype=np.float32))
                    pending.append((leaf, path_nodes, path_moves, sim))
                    in_flight.add(id(leaf))
                    self._vloss(path_nodes, VIRTUAL_LOSS)

                if pending:
                    pols, vals, draws, mlhs = mcts._infer(tensors)
                    for i, (leaf, path_nodes, path_moves, sim) in enumerate(pending):
                        if not leaf.is_expanded:
                            self._expand(leaf, sim, pols[i])
                        self._vloss(path_nodes, -VIRTUAL_LOSS)
                        v = float(vals[i])
                        d = float(draws[i])
                        m_ply = float(mlhs[i]) * 200.0
                        if len(path_moves) == 1:
                            child_nn[path_moves[0]] = (v, d, m_ply)
                        self._backup(path_nodes, v, d)
                    total += len(pending)

                now = time.time()
                if (now - last_emit > 0.13 or total >= self.max_sims or not self.running):
                    last_emit = now
                    self._emit(root, stm, total, child_nn, finished=(total >= self.max_sims))
                self.msleep(1)

            self._emit(root, stm, total, child_nn, finished=True)

        except Exception as e:
            print(f"[SearchThread] Ошибка: {e}")
            traceback.print_exc()

    def _emit(self, root, stm, total, child_nn, finished):
        payload = payload_from_node(root, stm, child_nn)
        payload["ply"] = len(self.move_history)
        payload["root_hash"] = self.root_hash
        payload["sims"] = total
        payload["merges"] = self.tt_hits
        payload["reused"] = self.reused
        payload["finished"] = finished
        self.update.emit(payload)


def payload_from_node(root, stm, child_nn=None):
    tv = sum(e.child.visits for e in root.children.values())
    moves = []
    for m, e in root.children.items():
        c = e.child
        if c.visits == 0:
            continue
        moves.append({
            "move": m,
            "visits": c.visits,
            "prior": e.prior,
            "q": -(c.value_sum / c.visits),
            "frac": (c.visits / tv) if tv > 0 else 0.0,
            "nn": child_nn.get(m) if child_nn else None,
            "pv": [m] + SearchThread._pv(c, PV_DEPTH),
        })
    moves.sort(key=lambda d: d["visits"], reverse=True)
    return {"game_over": False, "moves": moves, "root_q": root.q(),
            "stm": stm, "sims": root.visits, "merges": 0,
            "reused": root.visits, "finished": True}


# ──────────────────────────── Game logger ─────────────────────────────────

class GameLogger:
    DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
    RESULTS = {"draw": "ничья", "white": "победа белых", "black": "победа чёрных"}

    def __init__(self, mode, network):
        os.makedirs(self.DIR, exist_ok=True)
        stem = os.path.join(self.DIR, time.strftime("game_%Y%m%d_%H%M%S"))
        self.txt_path, self.path = stem + ".txt", stem + ".json"
        self.head = {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "network": os.path.basename(network) if network else None,
            "result": None,
        }
        self.plies = []
        self._t = time.time()

    def add(self, ply, by, m_int, snapshot, n_legal=None):
        now = time.time()
        rec = {"ply": ply, "by": by[0], "uci": move_to_uci(m_int),
               "sec": round(now - self._t, 1)}
        self._t = now
        if n_legal is not None:
            rec["legal"] = int(n_legal)
        if snapshot:
            tops = snapshot.get("moves") or []
            rec["q"] = round(float(snapshot.get("root_q", 0.0)), 3)
            rec["n"] = int(snapshot.get("sims", 0))
            rec["top"] = [[move_to_uci(d["move"]), round(d.get("frac", 0.0), 3)]
                          for d in tops[:3]]
            for i, d in enumerate(tops):
                if d["move"] == m_int:
                    rec["rank"] = i + 1
                    break
        self.plies.append(rec)
        self.flush()

    def finish(self, r):
        self.head["result"] = ("draw" if abs(r) < 1e-6
                               else ("white" if r > 0 else "black"))
        self.flush()

    @staticmethod
    def _atomic(path, text):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)

    def _as_text(self):
        side = {"play_white": "белые", "play_black": "чёрные"}.get(self.head["mode"], "—")
        out = [f"Capablanca — {self.head['started']}",
               f"сеть: {self.head['network']}   ты: {side}   "
               f"результат: {self.RESULTS.get(self.head['result'], 'не окончена')}",
               ""]
        line = []
        for i, rec in enumerate(self.plies):
            if i % 2 == 0:
                line = [f"{i // 2 + 1:>3}. {rec['uci']:<6}"]
            else:
                line.append(f"{rec['uci']:<6}")
                out.append("".join(line))
                line = []
        if line:
            out.append("".join(line))
        return "\n".join(out) + "\n"

    def _as_json(self):
        head = json.dumps(self.head, ensure_ascii=False)[1:-1]
        rows = ",\n  ".join(json.dumps(r, ensure_ascii=False,
                                       separators=(",", ":")) for r in self.plies)
        return "{" + head + ",\n \"plies\": [\n  " + rows + "\n ]\n}\n"

    def flush(self):
        self._atomic(self.txt_path, self._as_text())
        self._atomic(self.path, self._as_json())


# ───────────────────────────── Eval bar ───────────────────────────────────

class EvalBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(36)
        self.white_wr = 0.5
        self.mate_in = None

    def set_winrate(self, wr_white):
        self.white_wr = max(0.0, min(1.0, wr_white))
        self.mate_in = None
        self.update()

    def set_eval(self, wr_white, mate_in=None):
        self.white_wr = max(0.0, min(1.0, wr_white))
        self.mate_in = mate_in
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        h, w = self.height(), self.width()
        p.fillRect(self.rect(), QColor("#1c1c1c"))
        wh = int(h * self.white_wr)
        p.fillRect(0, h - wh, w, wh, QColor("#ededed"))
        p.setPen(QPen(QColor(ACCENT), 1))
        p.drawLine(0, h // 2, w, h // 2)

        if self.mate_in is not None:
            label = f"M{abs(self.mate_in)}" if self.mate_in > 0 else f"-M{abs(self.mate_in)}"
            p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        else:
            label = f"{self.white_wr * 100:.0f}"
            p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        if self.white_wr >= 0.5:
            p.setPen(QColor("#1c1c1c"))
            p.drawText(QRect(0, h - 20, w, 18), Qt.AlignCenter, label)
        else:
            p.setPen(QColor("#ededed"))
            p.drawText(QRect(0, 2, w, 18), Qt.AlignCenter, label)


# ───────────────────────────── Winrate graph ──────────────────────────────

class WinrateGraph(QWidget):
    seek = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(110)
        self.evals = {}
        self.cursor = 0
        self.total_plies = 0

    def set_data(self, evals, cursor, total_plies):
        self.evals = evals
        self.cursor = cursor
        self.total_plies = max(1, total_plies)
        self.update()

    def _x(self, ply):
        return 4 + ply / max(1, self.total_plies) * (self.width() - 8)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        h, w = self.height(), self.width()
        p.fillRect(self.rect(), QColor(BG2))

        mid = h / 2
        p.setPen(QPen(QColor("#4a4a4a"), 1, Qt.DashLine))
        p.drawLine(0, int(mid), w, int(mid))

        if self.evals:
            pts = sorted(self.evals.items())
            poly = [QPointF(self._x(ply), 4 + (1 - wr) * (h - 8))
                    for ply, wr in pts]
            area = QPolygonF([QPointF(poly[0].x(), mid)] + poly +
                             [QPointF(poly[-1].x(), mid)])
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(91, 155, 213, 70))
            p.drawPolygon(area)
            p.setPen(QPen(QColor(ACCENT), 2))
            p.setBrush(Qt.NoBrush)
            p.drawPolyline(QPolygonF(poly))
            for pt in poly:
                p.setBrush(QColor("#eaeaea"))
                p.setPen(Qt.NoPen)
                p.drawEllipse(pt, 2.6, 2.6)

        cx = self._x(self.cursor)
        p.setPen(QPen(QColor("#e8c84a"), 2))
        p.drawLine(int(cx), 0, int(cx), h)

    def mousePressEvent(self, ev):
        if self.total_plies <= 0:
            return
        frac = (ev.x() - 4) / max(1, self.width() - 8)
        ply = int(round(frac * self.total_plies))
        self.seek.emit(max(0, min(self.total_plies, ply)))


# ───────────────────────────── Infobox ────────────────────────────────────

class InfoBox(QWidget):
    play_move = pyqtSignal(int)
    ROW_H = 64

    def __init__(self):
        super().__init__()
        self.rows = []
        self.hover = -1
        self.setMouseTracking(True)
        self.setMinimumWidth(330)

    def set_moves(self, moves):
        self.rows = moves[:9]
        self.setMinimumHeight(max(1, len(self.rows)) * self.ROW_H + 4)
        self.update()

    def _row_at(self, y):
        i = (y - 2) // self.ROW_H
        return i if 0 <= i < len(self.rows) else -1

    def mouseMoveEvent(self, ev):
        h = self._row_at(ev.y())
        if h != self.hover:
            self.hover = h
            self.update()

    def leaveEvent(self, _):
        self.hover = -1
        self.update()

    def mousePressEvent(self, ev):
        i = self._row_at(ev.y())
        if i >= 0:
            self.play_move.emit(self.rows[i]["move"])

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BG2))
        w = self.width()

        if not self.rows:
            p.setPen(QColor("#7a7a7a"))
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(self.rect(), Qt.AlignCenter, "нет анализа")
            return

        for i, d in enumerate(self.rows):
            y = 2 + i * self.ROW_H
            row = QRect(0, y, w, self.ROW_H)
            if i == self.hover:
                p.fillRect(row, QColor(BG3))
            elif i % 2:
                p.fillRect(row, QColor("#2a2a2a"))

            color = RANK_COLORS[i] if i < len(RANK_COLORS) else GREY
            p.fillRect(QRect(0, y, 5, self.ROW_H), color)

            wr = (d["q"] + 1.0) / 2.0
            p.setPen(QColor(FG))
            p.setFont(QFont("Consolas", 12, QFont.Bold))
            p.drawText(QRect(14, y + 4, 96, 22), Qt.AlignVCenter | Qt.AlignLeft,
                       move_to_uci(d["move"]))
            p.setPen(color.lighter(125))
            p.setFont(QFont("Segoe UI", 13, QFont.Bold))
            p.drawText(QRect(w - 86, y + 3, 78, 24),
                       Qt.AlignVCenter | Qt.AlignRight, f"{wr * 100:.1f}%")
            p.setPen(QColor("#9a9a9a"))
            p.setFont(QFont("Segoe UI", 8))
            stats = (f"N {d['visits']}  ·  {d['frac'] * 100:.1f}%   "
                     f"P {d['prior'] * 100:.1f}%   Q {d['q']:+.3f}")
            p.drawText(QRect(14, y + 23, w - 100, 15),
                       Qt.AlignVCenter | Qt.AlignLeft, stats)

            pv = d.get("pv")
            if pv:
                p.setPen(QColor("#8fa6bd"))
                p.setFont(QFont("Consolas", 8))
                pv_str = "▸ " + " ".join(move_to_uci(x) for x in pv)
                pv_rect = QRect(14, y + 38, w - 22, 13)
                pv_str = p.fontMetrics().elidedText(
                    pv_str, Qt.ElideRight, pv_rect.width())
                p.drawText(pv_rect, Qt.AlignVCenter | Qt.AlignLeft, pv_str)

            if d["nn"] is not None:
                cq, cd = d["nn"][0], d["nn"][1]
                win = max(0.0, (1.0 - cd - cq) / 2.0)
                loss = max(0.0, (1.0 - cd + cq) / 2.0)
                draw = max(0.0, 1.0 - win - loss)
            else:
                win, draw, loss = value_to_wdl(d["q"])
            bx, bw, by = 14, w - 100, y + self.ROW_H - 9
            ww = int(bw * win)
            dw = int(bw * draw)
            p.setPen(Qt.NoPen)
            p.setBrush(W_COL)
            p.drawRect(bx, by, ww, 5)
            p.setBrush(D_COL)
            p.drawRect(bx + ww, by, dw, 5)
            p.setBrush(L_COL)
            p.drawRect(bx + ww + dw, by, bw - ww - dw, 5)


# ───────────────────── Book Explorer (WDL по ходам) ──────────────────────

class BookBox(QWidget):
    """Дебютная книга: список ходов, сыгранных из текущей позиции, с WDL-процентами."""
    play_move = pyqtSignal(int)
    ROW_H = 48

    def __init__(self):
        super().__init__()
        self.moves_data = []
        self.total_games = 0
        self.side_to_move = 0
        self.hover = -1
        self.has_archive = False
        self.setMouseTracking(True)
        self.setMinimumWidth(330)

    def set_book_data(self, entry, has_archive=True):
        self.has_archive = has_archive
        if entry is None or not entry.get("moves"):
            self.moves_data = []
            self.total_games = 0
            self.side_to_move = 0
        else:
            self.total_games = entry["total"]
            self.side_to_move = entry["side"]
            m_items = []
            for mv, st in entry["moves"].items():
                n = st["n"]
                w = st["w"]
                d = st["d"]
                l = st["l"]
                score = (w + 0.5 * d) / n if n > 0 else 0.5
                freq = n / self.total_games if self.total_games > 0 else 0.0
                m_items.append({
                    "move": mv, "n": n, "w": w, "d": d, "l": l,
                    "w_pct": w / n if n > 0 else 0.0,
                    "d_pct": d / n if n > 0 else 0.0,
                    "l_pct": l / n if n > 0 else 0.0,
                    "score": score, "freq": freq
                })
            m_items.sort(key=lambda x: x["n"], reverse=True)
            self.moves_data = m_items

        needed_h = max(1, len(self.moves_data)) * self.ROW_H + 28
        self.setMinimumHeight(needed_h)
        self.update()

    def _row_at(self, y):
        if y < 24:
            return -1
        i = (y - 24) // self.ROW_H
        return i if 0 <= i < len(self.moves_data) else -1

    def mouseMoveEvent(self, ev):
        h = self._row_at(ev.y())
        if h != self.hover:
            self.hover = h
            self.setCursor(Qt.PointingHandCursor if h >= 0 else Qt.ArrowCursor)
            self.update()

    def leaveEvent(self, _):
        self.hover = -1
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def mousePressEvent(self, ev):
        i = self._row_at(ev.y())
        if i >= 0:
            self.play_move.emit(self.moves_data[i]["move"])

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BG2))
        w = self.width()

        if not self.has_archive:
            p.setPen(QColor("#888"))
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Архив не загружен.\nНажмите «📦 Архив партий».")
            return

        if not self.moves_data:
            p.setPen(QColor("#777"))
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "В архиве нет партий с этой позицией\n(или глубже дебюта).")
            return

        # Заголовок
        who = "белые" if self.side_to_move == 0 else "чёрные"
        p.setPen(QColor("#aaa"))
        p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(QRect(8, 4, w - 16, 16), Qt.AlignVCenter | Qt.AlignLeft,
                   f"Всего партий: {self.total_games:,}  ·  ход: {who}")
        p.setPen(QPen(QColor("#3d3d3d"), 1))
        p.drawLine(8, 22, w - 8, 22)

        for i, d in enumerate(self.moves_data):
            y = 24 + i * self.ROW_H
            row = QRect(0, y, w, self.ROW_H)
            if i == self.hover:
                p.fillRect(row, QColor(BG3))
            elif i % 2:
                p.fillRect(row, QColor("#2a2a2a"))

            # UCI ход
            p.setPen(QColor(FG))
            p.setFont(QFont("Consolas", 11, QFont.Bold))
            p.drawText(QRect(12, y + 2, 80, 18), Qt.AlignVCenter | Qt.AlignLeft,
                       move_to_uci(d["move"]))

            # Игр + частота
            p.setPen(QColor("#9c9c9c"))
            p.setFont(QFont("Segoe UI", 8))
            n_txt = f"{d['n']:,} ({d['freq']*100:.0f}%)"
            p.drawText(QRect(96, y + 2, 130, 18), Qt.AlignVCenter | Qt.AlignLeft, n_txt)

            # Очки (%)
            p.setPen(QColor(ACCENT).lighter(120))
            p.setFont(QFont("Segoe UI", 9, QFont.Bold))
            p.drawText(QRect(w - 74, y + 2, 64, 18), Qt.AlignVCenter | Qt.AlignRight,
                       f"{d['score']*100:.1f}%")

            # WDL Полоса
            bx = 12
            bw = w - 24
            by = y + 22
            bh = 17

            ww = int(round(bw * d["w_pct"]))
            dw = int(round(bw * d["d_pct"]))
            lw = bw - ww - dw

            p.setPen(Qt.NoPen)
            # Победа
            if ww > 0:
                p.setBrush(W_COL)
                p.drawRoundedRect(QRectF(bx, by, ww, bh), 2, 2)
            # Ничья
            if dw > 0:
                p.setBrush(D_COL)
                p.drawRoundedRect(QRectF(bx + ww, by, dw, bh), 2, 2)
            # Поражение
            if lw > 0:
                p.setBrush(L_COL)
                p.drawRoundedRect(QRectF(bx + ww + dw, by, lw, bh), 2, 2)

            # Проценты внутри сегментов
            p.setFont(QFont("Segoe UI", 8, QFont.Bold))
            p.setPen(QColor("#ffffff"))

            if ww >= 28:
                p.drawText(QRect(bx, by, ww, bh), Qt.AlignCenter, f"{d['w_pct']*100:.0f}%")
            if dw >= 28:
                p.drawText(QRect(bx + ww, by, dw, bh), Qt.AlignCenter, f"{d['d_pct']*100:.0f}%")
            if lw >= 28:
                p.drawText(QRect(bx + ww + dw, by, lw, bh), Qt.AlignCenter, f"{d['l_pct']*100:.0f}%")


# ─────────────────────── Диалог выбора партий архива ───────────────────────

class ArchiveDialog(QDialog):
    """Окно просмотра и фильтрации партий из архива."""
    load_game = pyqtSignal(int)

    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.main_win = main_win
        self.setWindowTitle("Архив партий")
        self.resize(920, 620)
        self.filtered_ids = []

        lay = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.lbl_path = QLabel("Архив не выбран")
        self.lbl_path.setStyleSheet("color: #aaa; font-style: italic;")
        btn_browse = QPushButton("📂 Выбрать папку архива...")
        btn_browse.clicked.connect(self.choose_directory)
        top_row.addWidget(btn_browse)
        top_row.addWidget(self.lbl_path, 1)
        lay.addLayout(top_row)

        # Фильтры
        filter_box = QGroupBox("Фильтры отбора партий")
        fl = QHBoxLayout(filter_box)

        fl.addWidget(QLabel("Исход:"))
        self.cb_result = QComboBox()
        self.cb_result.addItems(["Все", "1-0 (Белые)", "½-½ (Ничья)", "0-1 (Чёрные)"])
        self.cb_result.currentIndexChanged.connect(self.apply_filter)
        fl.addWidget(self.cb_result)

        fl.addWidget(QLabel("Окончание:"))
        self.cb_term = QComboBox()
        self.cb_term.addItem("Все")
        if A:
            for t_id in sorted(A.TERM_NAMES):
                self.cb_term.addItem(A.TERM_NAMES[t_id], t_id)
        self.cb_term.currentIndexChanged.connect(self.apply_filter)
        fl.addWidget(self.cb_term)

        fl.addWidget(QLabel("Полуходов:"))
        self.spin_min_plies = QSpinBox()
        self.spin_min_plies.setRange(0, 500)
        self.spin_min_plies.setValue(0)
        self.spin_min_plies.valueChanged.connect(self.apply_filter)
        fl.addWidget(self.spin_min_plies)
        fl.addWidget(QLabel("–"))
        self.spin_max_plies = QSpinBox()
        self.spin_max_plies.setRange(0, 500)
        self.spin_max_plies.setValue(500)
        self.spin_max_plies.valueChanged.connect(self.apply_filter)
        fl.addWidget(self.spin_max_plies)

        lay.addWidget(filter_box)

        # Таблица партий
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["#", "Итерация", "Исход", "Ходов", "Окончание", "Источник"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.Stretch)
        hh.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.cellDoubleClicked.connect(self._row_double_clicked)
        lay.addWidget(self.table, 1)

        # Низ
        bottom = QHBoxLayout()
        self.lbl_count = QLabel("Партий: 0")
        bottom.addWidget(self.lbl_count)
        bottom.addStretch()

        btn_load = QPushButton("▶ Загрузить партию на доску")
        btn_load.clicked.connect(self._load_selected)
        bottom.addWidget(btn_load)

        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.accept)
        bottom.addWidget(btn_close)
        lay.addLayout(bottom)

    def choose_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Выбрать каталог архива")
        if d:
            self.main_win.load_archive_dir(d)

    def update_archive_view(self, ar):
        if ar is None:
            self.lbl_path.setText("Архив не найден")
            return
        self.lbl_path.setText(os.path.abspath(ar.dir))
        self.apply_filter()

    def apply_filter(self):
        ar = self.main_win.archive
        if ar is None:
            self.table.setRowCount(0)
            self.lbl_count.setText("Партий: 0")
            return

        res_idx = self.cb_result.currentIndex()
        res_filter = {1: 1, 2: 0, 3: -1}.get(res_idx, None)

        term_data = self.cb_term.currentData()
        min_p = self.spin_min_plies.value()
        max_p = self.spin_max_plies.value()

        mask = np.ones(ar.g["result"].shape[0], dtype=bool)
        if res_filter is not None:
            mask &= (ar.g["result"] == res_filter)
        if term_data is not None:
            mask &= (ar.g["term"] == term_data)
        if min_p > 0:
            mask &= (ar.g["plies"] >= min_p)
        if max_p < 500:
            mask &= (ar.g["plies"] <= max_p)

        self.filtered_ids = np.flatnonzero(mask)
        n = len(self.filtered_ids)
        self.lbl_count.setText(f"Отобрано: {n:,} из {len(mask):,} партий")

        # Показываем первые 2000 во избежание лагов таблицы
        show_n = min(n, 2000)
        self.table.blockSignals(True)
        self.table.setRowCount(show_n)

        for row in range(show_n):
            gid = int(self.filtered_ids[row])
            it = int(ar.g["iter"][gid])
            r = int(ar.g["result"][gid])
            pl = int(ar.g["plies"][gid])
            t = int(ar.g["term"][gid])
            src = int(ar.g["source"][gid])

            r_str = "1-0" if r > 0 else ("0-1" if r < 0 else "½-½")
            t_str = A.TERM_NAMES.get(t, str(t)) if A else str(t)
            s_str = A.SOURCE_NAMES.get(src, str(src)) if A else str(src)

            item_id = QTableWidgetItem(str(gid))
            item_it = QTableWidgetItem(str(it))
            item_res = QTableWidgetItem(r_str)
            item_pl = QTableWidgetItem(f"{pl} (ход {pl//2 + 1})")
            item_t = QTableWidgetItem(t_str)
            item_src = QTableWidgetItem(s_str)

            if r > 0:
                item_res.setForeground(QColor("#4ea863"))
            elif r < 0:
                item_res.setForeground(QColor("#e0614a"))
            else:
                item_res.setForeground(QColor("#9e9e9e"))

            for c, itm in enumerate([item_id, item_it, item_res, item_pl, item_t, item_src]):
                if c in (0, 1, 2):
                    itm.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, c, itm)

        self.table.blockSignals(False)

    def _row_double_clicked(self, row, _):
        if 0 <= row < len(self.filtered_ids):
            gid = int(self.filtered_ids[row])
            self.load_game.emit(gid)
            self.accept()

    def _load_selected(self):
        sel = self.table.selectedRanges()
        if not sel:
            return
        row = sel[0].topRow()
        if 0 <= row < len(self.filtered_ids):
            gid = int(self.filtered_ids[row])
            self.load_game.emit(gid)
            self.accept()


# ───────────────────────────── Promotion dialog ───────────────────────────

class PromotionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Превращение пешки")
        self.chosen = 'q'
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Выберите фигуру:"))
        row = QHBoxLayout()
        for ch, name in _PROMO_LABELS:
            b = QPushButton(name)
            b.setMinimumWidth(112)
            b.clicked.connect(lambda _, c=ch: self._pick(c))
            row.addWidget(b)
        lay.addLayout(row)

    def _pick(self, ch):
        self.chosen = ch
        self.accept()


# ───────────────────────────── Board widget ───────────────────────────────

class BoardWidget(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setMinimumSize(720, 576)
        self.engine = CapablancaEngine()
        self.legal_moves = self.engine.get_legal_moves_int()
        self.selected = None
        self.flipped = False
        self.last_move = None
        self.analysis = []

    def set_position(self, history, last_move):
        eng = CapablancaEngine()
        for m in history:
            eng.make_move_int(m)
        self.engine = eng
        self.legal_moves = eng.get_legal_moves_int()
        self.last_move = last_move
        self.selected = None
        self.update()

    def _metrics(self):
        cell = min(self.width() / 10.0, self.height() / 8.0)
        ox = (self.width() - cell * 10) / 2.0
        oy = (self.height() - cell * 8) / 2.0
        return cell, ox, oy

    def sq_rect(self, sq):
        cell, ox, oy = self._metrics()
        r, f = divmod(sq, 10)
        vr = r if self.flipped else (7 - r)
        vf = (9 - f) if self.flipped else f
        return QRectF(ox + vf * cell, oy + vr * cell, cell, cell)

    def sq_at(self, pos):
        cell, ox, oy = self._metrics()
        vf = int((pos.x() - ox) // cell)
        vr = int((pos.y() - oy) // cell)
        if 0 <= vf < 10 and 0 <= vr < 8:
            r = vr if self.flipped else (7 - vr)
            f = (9 - vf) if self.flipped else vf
            return r * 10 + f
        return None

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BG))
        cell, _, _ = self._metrics()

        p.setFont(QFont("Segoe UI", max(7, int(cell * 0.16)), QFont.Bold))
        for r in range(8):
            for f in range(10):
                sq = r * 10 + f
                rect = self.sq_rect(sq)
                light = (r + f) % 2 == 1
                base = LIGHT_SQ if light else DARK_SQ
                p.fillRect(rect, base)
                if self.last_move and sq in self.last_move:
                    p.fillRect(rect, QColor(232, 200, 74, 90))
                if self.selected == sq:
                    p.fillRect(rect, QColor(255, 241, 120, 120))
                p.setPen(base.darker(160))
                edge_rank = (r == 0) if not self.flipped else (r == 7)
                edge_file = (f == 0) if not self.flipped else (f == 9)
                if edge_rank:
                    p.drawText(rect.adjusted(0, 0, -3, -2),
                               Qt.AlignBottom | Qt.AlignRight, chr(ord('a') + f))
                if edge_file:
                    p.drawText(rect.adjusted(3, 2, 0, 0),
                               Qt.AlignTop | Qt.AlignLeft, str(r + 1))

        for color, ptype, sq in self.engine.get_pieces():
            self._draw_piece(p, self.sq_rect(int(sq)), color, ptype, cell)

        if self.selected is not None:
            shown_with_rank = [(g, d) for g, d in enumerate(self.analysis)
                               if decode_move(d["move"])[0] == self.selected]
        else:
            shown_with_rank = list(enumerate(self.analysis[:5]))

        if self.selected is not None:
            best_dest = (decode_move(shown_with_rank[0][1]["move"])[1]
                         if shown_with_rank else None)
            for m in self.legal_moves:
                fs, ts, _ = decode_move(m)
                if fs != self.selected:
                    continue
                c = self.sq_rect(ts).center()
                p.setPen(Qt.NoPen)
                if ts == best_dest:
                    p.setBrush(QColor(58, 166, 85, 200))
                    p.drawEllipse(c, cell * 0.16, cell * 0.16)
                else:
                    p.setBrush(QColor(0, 0, 0, 55))
                    p.drawEllipse(c, cell * 0.10, cell * 0.10)

        label_targets = set()
        for g, d in reversed(shown_with_rank):
            color = RANK_COLORS[g] if g < len(RANK_COLORS) else GREY
            self._draw_arrow(p, d, color, cell, rank=g, n_shown=len(shown_with_rank))
        for g, d in shown_with_rank:
            color = RANK_COLORS[g] if g < len(RANK_COLORS) else GREY
            _, t, _ = decode_move(d["move"])
            if t in label_targets:
                continue
            label_targets.add(t)
            self._draw_arrow_label(p, d, color, cell, rank=g)

    def _draw_piece(self, p, rect, color, ptype, cell):
        glyph = PIECE_GLYPHS.get(ptype, '?')
        is_white = color == 0
        fill = QColor("#f5f5f5") if is_white else QColor("#1b1b1b")
        edge = QColor("#1b1b1b") if is_white else QColor("#e6e6e6")
        letter = ptype in (5, 6)
        font = QFont("DejaVu Sans", int(cell * (0.5 if letter else 0.66)))
        font.setBold(True)
        p.setFont(font)
        p.setPen(edge)
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1),
                       (0, -2), (0, 2), (-2, 0), (2, 0)):
            p.drawText(rect.translated(dx, dy), Qt.AlignCenter, glyph)
        p.setPen(fill)
        p.drawText(rect, Qt.AlignCenter, glyph)

    def _draw_arrow(self, p, d, color, cell, rank, n_shown):
        f, t, _ = decode_move(d["move"])
        s = self.sq_rect(f).center()
        e = self.sq_rect(t).center()
        ang = np.arctan2(e.y() - s.y(), e.x() - s.x())
        alpha = max(0.35, 1.0 - rank * 0.15)
        wf    = max(0.30, 1.0 - rank * 0.18)
        c = QColor(color)
        c.setAlpha(int(255 * alpha))
        width = max(3.0, cell * 0.18 * wf)
        head  = cell * 0.34 * wf
        tip  = QPointF(e.x() - np.cos(ang) * head * 0.20,
                       e.y() - np.sin(ang) * head * 0.20)
        base = QPointF(tip.x() - np.cos(ang) * head,
                       tip.y() - np.sin(ang) * head)
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        src_r = max(3.0, cell * 0.07 * wf)
        p.drawEllipse(s, src_r, src_r)
        p.setPen(QPen(c, width, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(s, base)
        p.setBrush(c)
        p.setPen(Qt.NoPen)
        p1 = base + QPointF(-np.sin(ang) * head * 0.65, np.cos(ang) * head * 0.65)
        p2 = base - QPointF(-np.sin(ang) * head * 0.65, np.cos(ang) * head * 0.65)
        p.drawPolygon(QPolygonF([tip, p1, p2]))

    def _draw_arrow_label(self, p, d, color, cell, rank):
        if rank > 2:
            return
        _, t, _ = decode_move(d["move"])
        center = self.sq_rect(t).center()
        wr = (d["q"] + 1.0) / 2.0
        label = f"{wr * 100:.0f}%"
        font_size = max(9, int(cell * (0.20 if rank == 0 else 0.17)))
        p.setFont(QFont("Segoe UI", font_size, QFont.Bold))
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(label) + 10
        th = fm.height() + 2
        box = QRectF(center.x() - tw / 2, center.y() - th / 2, tw, th)
        dark = QColor(color.darker(180))
        dark.setAlpha(240)
        p.setBrush(dark)
        p.setPen(QPen(color.lighter(140), 1))
        p.drawRoundedRect(box, 4, 4)
        p.setPen(QColor("#ffffff"))
        p.drawText(box, Qt.AlignCenter, label)

    def mousePressEvent(self, ev):
        sq = self.sq_at(ev.pos())
        if sq is None:
            return
        if self.selected is not None:
            matching = [m for m in self.legal_moves
                        if decode_move(m)[0] == self.selected
                        and decode_move(m)[1] == sq]
            if matching:
                move = matching[0]
                if len(matching) > 1:
                    dlg = PromotionDialog(self)
                    if dlg.exec_() != QDialog.Accepted:
                        self.selected = None
                        self.update()
                        return
                    move = next((m for m in matching
                                 if decode_move(m)[2] == dlg.chosen), matching[0])
                self.selected = None
                self.main.try_human_move(move)
                return
        if sq == self.selected:
            self.selected = None
        elif any(decode_move(m)[0] == sq for m in self.legal_moves):
            self.selected = sq
        else:
            self.selected = None
        self.update()


# ───────────────────────────── Main window ────────────────────────────────

MODES = ["analyze", "play_white", "play_black", "selfplay"]
MODE_LABELS = ["Анализ", "Играть белыми", "Играть чёрными", "Самоигра"]


class NibblerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Capablanca AI — анализатор")
        self.resize(1320, 920)

        self.mcts = None
        self.search = None
        self.search_started = 0.0

        # Архив и дебютная книга
        self.archive = None
        self.games_moves = None
        self.book = None
        self.archive_loader = None
        self.archive_dialog = None

        self.history = []
        self.cursor = 0
        self.evals = {}
        self.snapshots = {}
        self.ttable = {}
        self.mode = "analyze"
        self.analysis_on = False
        self.hints_on = True
        self.contempt = 0.0

        self._build_ui()
        self._build_shortcuts()
        self._autoload()
        self._auto_find_archive()
        self.refresh()

    # ---- UI construction ----
    def _build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        outer.addWidget(self._build_toolbar())

        mid = QHBoxLayout()
        mid.setSpacing(8)
        self.eval_bar = EvalBar()
        mid.addWidget(self.eval_bar)
        self.board = BoardWidget(self)
        self.board.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mid.addWidget(self.board, 5)
        mid.addWidget(self._build_right_panel(), 0)
        outer.addLayout(mid, 5)

        graph_box = QGroupBox("График винрейта партии (за белых)")
        gl = QVBoxLayout(graph_box)
        self.graph = WinrateGraph()
        self.graph.seek.connect(self.seek_ply)
        gl.addWidget(self.graph)
        outer.addWidget(graph_box, 0)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Загрузите веса модели для анализа.")

    def _build_toolbar(self):
        bar = QFrame()
        bar.setStyleSheet(f"QFrame {{ background: {BG2}; border: 1px solid #444; "
                          f"border-radius: 5px; }}")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(8)

        btn_load = QPushButton("📂 Загрузить веса")
        btn_load.clicked.connect(self.load_weights)
        lay.addWidget(btn_load)

        btn_game = QPushButton("♟ Открыть партию")
        btn_game.clicked.connect(self.load_game)
        lay.addWidget(btn_game)

        btn_arch = QPushButton("📦 Архив партий")
        btn_arch.setStyleSheet("font-weight: bold; color: #5b9bd5;")
        btn_arch.clicked.connect(self.open_archive_dialog)
        lay.addWidget(btn_arch)
        lay.addWidget(self._vline())

        lay.addWidget(QLabel("Режим:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODE_LABELS)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        lay.addWidget(self.mode_combo)
        lay.addWidget(self._vline())

        lay.addWidget(QLabel("Лимит анализа:"))
        self.spin_analyze = QSpinBox()
        self.spin_analyze.setRange(0, 50_000_000)
        self.spin_analyze.setSingleStep(2000)
        self.spin_analyze.setValue(0)
        self.spin_analyze.setSpecialValueText("∞")
        lay.addWidget(self.spin_analyze)

        lay.addWidget(QLabel("Узлов на ход:"))
        self.spin_play = QSpinBox()
        self.spin_play.setRange(50, 10_000_000)
        self.spin_play.setSingleStep(200)
        self.spin_play.setValue(1200)
        lay.addWidget(self.spin_play)
        lay.addWidget(self._vline())

        lay.addWidget(QLabel("c_puct:"))
        self.spin_cpuct = QSpinBox()
        self.spin_cpuct.setRange(500, 5000)
        self.spin_cpuct.setSingleStep(100)
        self.spin_cpuct.setValue(1745)
        self.spin_cpuct.setSuffix("‰")
        self.spin_cpuct.valueChanged.connect(self._cpuct_changed)
        lay.addWidget(self.spin_cpuct)
        lay.addWidget(self._vline())

        lay.addWidget(QLabel("Contempt:"))
        self.spin_contempt = QSpinBox()
        self.spin_contempt.setRange(-100, 100)
        self.spin_contempt.setSingleStep(5)
        self.spin_contempt.setValue(0)
        self.spin_contempt.setSuffix("%")
        self.spin_contempt.valueChanged.connect(self._contempt_changed)
        lay.addWidget(self.spin_contempt)
        lay.addWidget(self._vline())

        self.btn_go = QPushButton("⏸ Остановить анализ" if self.analysis_on
                                  else "▶ Запустить анализ")
        self.btn_go.clicked.connect(self._toggle_analysis)
        lay.addWidget(self.btn_go)

        self.chk_hints = QCheckBox("Подсказки на моём ходу")
        self.chk_hints.setChecked(self.hints_on)
        self.chk_hints.toggled.connect(self._toggle_hints)
        lay.addWidget(self.chk_hints)

        btn_new = QPushButton("🆕 Новая")
        btn_new.clicked.connect(self.new_game)
        lay.addWidget(btn_new)
        btn_flip = QPushButton("⇅ Перевернуть")
        btn_flip.clicked.connect(self.flip_board)
        lay.addWidget(btn_flip)

        lay.addStretch()
        return bar

    def _vline(self):
        ln = QFrame()
        ln.setFrameShape(QFrame.VLine)
        ln.setStyleSheet("color: #4a4a4a;")
        return ln

    def _build_right_panel(self):
        panel = QWidget()
        panel.setFixedWidth(372)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # Вкладки: Анализ MCTS vs Дебютная книга
        self.right_tabs = QTabWidget()

        # Tab 1: MCTS InfoBox
        tab_mcts = QWidget()
        tl_mcts = QVBoxLayout(tab_mcts)
        tl_mcts.setContentsMargins(0, 4, 0, 0)
        self.infobox = InfoBox()
        self.infobox.play_move.connect(self.try_human_move)
        tl_mcts.addWidget(self.infobox)
        tl_mcts.addStretch()
        self.right_tabs.addTab(tab_mcts, "🧠 MCTS Анализ")

        # Tab 2: Book Explorer
        tab_book = QWidget()
        tl_book = QVBoxLayout(tab_book)
        tl_book.setContentsMargins(0, 4, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 0; background: transparent; }")
        self.book_box = BookBox()
        self.book_box.play_move.connect(self.try_human_move)
        scroll.setWidget(self.book_box)
        tl_book.addWidget(scroll)
        self.right_tabs.addTab(tab_book, "📖 Книга WDL")

        lay.addWidget(self.right_tabs, 3)

        lay.addWidget(QLabel("Проанализированные позиции:"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["#", "Сыграно", "Оценка", "Лучший", "Узлы"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.cellClicked.connect(self._table_clicked)
        self._table_plies = []
        lay.addWidget(self.table, 2)

        nav = QHBoxLayout()
        for txt, fn in [("⏮", lambda: self.seek_ply(0)),
                        ("◀", lambda: self.seek_ply(self.cursor - 1)),
                        ("▶", lambda: self.seek_ply(self.cursor + 1)),
                        ("⏭", lambda: self.seek_ply(len(self.history)))]:
            b = QPushButton(txt)
            b.clicked.connect(fn)
            nav.addWidget(b)
        lay.addLayout(nav)
        return panel

    def _build_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self,
                  lambda: self.seek_ply(self.cursor - 1))
        QShortcut(QKeySequence(Qt.Key_Right), self,
                  lambda: self.seek_ply(self.cursor + 1))
        QShortcut(QKeySequence(Qt.Key_Home), self, lambda: self.seek_ply(0))
        QShortcut(QKeySequence(Qt.Key_End), self,
                  lambda: self.seek_ply(len(self.history)))
        QShortcut(QKeySequence(Qt.Key_F), self, self.flip_board)
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_analysis)

    # ---- archive handling ----
    def _auto_find_archive(self):
        base = os.path.dirname(os.path.abspath(__file__))
        cands = [
            os.path.join(base, "archive"),
            os.path.join(base, "checkpoints_v11", "archive"),
            os.path.join(base, "..", "archive"),
        ]
        for c in cands:
            if os.path.isdir(c):
                self.load_archive_dir(c)
                break

    def open_archive_dialog(self):
        if self.archive_dialog is None:
            self.archive_dialog = ArchiveDialog(self, self)
            self.archive_dialog.load_game.connect(self.load_archive_game)
        self.archive_dialog.update_archive_view(self.archive)
        self.archive_dialog.show()

    def load_archive_dir(self, directory):
        # Тот же каталог используется и для записи сыгранных здесь партий.
        self.archive_dir = directory
        self.statusBar().showMessage(f"Загрузка архива: {directory}...")
        self.archive_loader = ArchiveLoaderThread(directory)
        self.archive_loader.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.archive_loader.loaded.connect(self._on_archive_loaded)
        self.archive_loader.error.connect(
            lambda err: QMessageBox.warning(self, "Ошибка архива", err))
        self.archive_loader.start()

    def _on_archive_loaded(self, ar, games_moves, book):
        self.archive = ar
        self.games_moves = games_moves
        self.book = book
        if self.archive_dialog is not None and self.archive_dialog.isVisible():
            self.archive_dialog.update_archive_view(ar)
        self.statusBar().showMessage(
            f"Архив загружен: {len(ar.g['result']):,} партий, "
            f"{len(book):,} позиций в книге дебютов.")
        self.refresh()

    def load_archive_game(self, game_id):
        if self.games_moves is None or game_id >= len(self.games_moves):
            return
        moves = self.games_moves[game_id]
        if not moves:
            QMessageBox.information(self, "Пустая партия", "В этой партии нет ходов.")
            return
        self.stop_search()
        self.history = list(moves)
        self.cursor = 0
        self.evals.clear()
        self.snapshots.clear()
        self.refresh()
        r = int(self.archive.g["result"][game_id])
        r_str = "1-0 (Белые)" if r > 0 else ("0-1 (Чёрные)" if r < 0 else "½-½ (Ничья)")
        self.statusBar().showMessage(
            f"Загружена партия #{game_id} из архива ({r_str}, {len(moves)} полуходов). "
            f"Стрелки ◀ ▶ для просмотра.")

    # ---- game loading ----
    def load_game(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть партию", "",
            "Партия (*.pgn *.txt *.uci);;Все файлы (*)")
        if not path:
            return
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError as e:
            QMessageBox.warning(self, "Не открылось", str(e))
            return

        tokens = re.findall(r"\b[a-j](?:10|[1-9])[a-j](?:10|[1-9])[qrbnacQRBNAC]?\b", text)
        if not tokens:
            QMessageBox.warning(self, "Пусто", "В файле не нашлось ходов вида e2e4.")
            return

        eng = CapablancaEngine()
        moves = []
        for i, uci in enumerate(tokens):
            found = None
            for m in eng.get_legal_moves_int():
                if move_to_uci(m) == uci:
                    found = m
                    break
            if found is None:
                QMessageBox.warning(
                    self, "Ход не по правилам",
                    f"Полуход {i + 1} ({uci}) не находится среди возможных.\n"
                    f"Загружено {len(moves)} полуходов до него.")
                break
            eng.make_move_int(found)
            moves.append(found)
            if eng.is_game_over():
                break

        if not moves:
            return
        self.stop_search()
        self.history = moves
        self.cursor = 0
        self.refresh()
        self.statusBar().showMessage(
            f"Загружена партия: {len(moves)} полуходов. Стрелки — вперёд и назад.")

    # ---- model loading ----
    def load_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить сеть", "",
            "Сеть (*.onnx *.pth);;ONNX-модель (*.onnx);;Чекпоинт PyTorch (*.pth)")
        if not path:
            return
        if path.lower().endswith(".pth"):
            path = self._onnx_from_checkpoint(path)
            if not path:
                return
        self._load_onnx(path)

    def _onnx_from_checkpoint(self, pth):
        dst = os.path.splitext(pth)[0] + ".onnx"
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(pth):
            return dst
        try:
            from export_onnx import export
        except ImportError as e:
            QMessageBox.warning(
                self, "Нужен PyTorch",
                f"Чтобы открыть .pth, нужен PyTorch в этом окружении:\n\n{e}\n\n"
                "Либо конвертируй заранее:\n    python export_onnx.py чекпоинт.pth сеть.onnx")
            return None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            export(pth, dst)
        except Exception as e:
            QMessageBox.critical(self, "Не удалось конвертировать",
                                 f"{type(e).__name__}: {e}")
            return None
        finally:
            QApplication.restoreOverrideCursor()
        return dst

    def _autoload(self):
        if getattr(sys, "frozen", False):
            base = os.path.dirname(os.path.abspath(sys.executable))
        else:
            base = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(base, "capablanca.onnx")
        if os.path.exists(cand):
            self._load_onnx(cand)

    def _load_onnx(self, path):
        try:
            self.stop_search()
            self.mcts = None
            self.mcts = OnnxEngine(path, c_puct=1.745, batch_size=96,
                                   nn_cache=True, nn_cache_max=64_000)
            self.net_path = path
            self.ttable.clear()
            dev = "GPU · CUDA" if self.mcts.gpu else "CPU"
            self.statusBar().showMessage(
                f"Сеть загружена: {os.path.basename(path)}  ·  {dev}")
            self.refresh()
        except Exception as e:
            self.statusBar().showMessage(f"Ошибка загрузки: {e}")
            traceback.print_exc()

    # ---- controls ----
    def _mode_changed(self, idx):
        self.mode = MODES[idx]
        self.refresh()

    def _toggle_hints(self, on):
        self.hints_on = bool(on)
        self.refresh()

    def hints_hidden(self):
        return (not self.hints_on
                and self.mode in ("play_white", "play_black")
                and self.cursor == len(self.history)
                and not self.engine_should_move())

    def _toggle_analysis(self):
        self.analysis_on = not self.analysis_on
        self.btn_go.setText("⏸ Остановить анализ" if self.analysis_on
                            else "▶ Запустить анализ")
        self.refresh()

    def _contempt_changed(self, val):
        self.contempt = val / 100.0

    def _cpuct_changed(self, val):
        if self.mcts is not None:
            self.mcts.c_puct = val / 1000.0

    def _prune_ttable(self, move_history):
        try:
            engine = CapablancaEngine()
            for m in move_history:
                engine.make_move_int(m)
            root_hash = position_key(engine)
        except Exception:
            self.ttable.clear()
            return
        root = self.ttable.get(root_hash)
        if root is None:
            self.ttable.clear()
            return
        keep = {root_hash: root}
        stack = [root]
        while stack:
            n = stack.pop()
            if not n.children:
                continue
            for e in n.children.values():
                if e.child_hash in keep:
                    continue
                keep[e.child_hash] = e.child
                stack.append(e.child)
        dropped = len(self.ttable) - len(keep)
        self.ttable.clear()
        self.ttable.update(keep)
        self.statusBar().showMessage(
            f"ttable prune: оставили {len(keep):,} узлов, выкинули {dropped:,}")

    def new_game(self):
        self.logger = GameLogger(self.mode, getattr(self, "net_path", None))
        self.history = []
        self.cursor = 0
        self.evals = {}
        self.snapshots = {}
        self.ttable.clear()
        if self.mcts is not None:
            self.mcts.clear_nn_cache()
        self.refresh()

    def flip_board(self):
        self.board.flipped = not self.board.flipped
        self.board.update()

    # ---- navigation ----
    def seek_ply(self, ply):
        ply = max(0, min(len(self.history), ply))
        if ply == self.cursor:
            return
        self.cursor = ply
        self.refresh()

    def _table_clicked(self, row, _col):
        if 0 <= row < len(self._table_plies):
            self.seek_ply(self._table_plies[row])

    # ---- moves ----
    def engine_should_move(self):
        if self.mode == "selfplay":
            return True
        stm = self.board.engine.side_to_move()
        if self.mode == "play_white":
            return stm == 1
        if self.mode == "play_black":
            return stm == 0
        return False

    def try_human_move(self, m):
        if self.mode == "selfplay":
            return
        if self.cursor == len(self.history) and self.engine_should_move():
            return
        if m not in self.board.engine.get_legal_moves_int():
            return
        self.push_move(m)

    def push_move(self, m, by="human"):
        snapshot = self.snapshots.get(self.cursor)
        if getattr(self, "logger", None) is None:
            self.logger = GameLogger(self.mode, getattr(self, "net_path", None))
        try:
            n_legal = len(self.board.engine.get_legal_moves_int())
            self.logger.add(self.cursor, by, m, snapshot, n_legal)
        except Exception as e:
            self.statusBar().showMessage(f"запись партии отключена: {e}")
            self.logger = None
        del self.history[self.cursor:]
        self.history.append(m)
        self.cursor += 1
        for ply in list(self.evals):
            if ply >= self.cursor:
                del self.evals[ply]
        for ply in list(self.snapshots):
            if ply >= self.cursor:
                del self.snapshots[ply]
        self.refresh()

    # ---- search lifecycle ----
    def stop_search(self):
        if self.search is not None:
            try:
                self.search.update.disconnect()
            except TypeError:
                pass
            self.search.running = False
            self.search.wait(10000)
            self.search = None

    def should_search(self):
        if self.mcts is None or self.board.engine.is_game_over():
            return False
        if self.analysis_on:
            return True
        return (self.cursor == len(self.history) and self.engine_should_move()
                and self.mode in ("play_white", "play_black"))

    def refresh(self):
        self.stop_search()
        view = self.history[:self.cursor]
        last = (decode_move(view[-1])[:2] if view else None)
        self.board.set_position(view, last)
        self.board.analysis = []

        # Обновление книги дебютов для текущей позиции
        if self.book is not None:
            k = position_key(self.board.engine)
            self.book_box.set_book_data(self.book.get(k), has_archive=True)
        else:
            self.book_box.set_book_data(None, has_archive=False)

        over = self.board.engine.is_game_over()
        snap = self.snapshots.get(self.cursor)

        if over:
            self.infobox.set_moves([])
            self.eval_bar.set_winrate(self.evals.get(self.cursor, 0.5))
            r = self.board.engine.game_result()
            msg = ("Ничья" if abs(r) < 1e-6
                   else ("Белые выиграли" if r > 0 else "Чёрные выиграли"))
            lg = getattr(self, "logger", None)
            try:
                if lg is not None and lg.head["result"] is None:
                    lg.finish(r)
                    msg += f"  ·  записано: games/{os.path.basename(lg.txt_path)}"
                    if A is not None and getattr(self, "archive_dir", ""):
                        n = A.archive_moves(
                            self.archive_dir, [x["uci"] for x in lg.plies],
                            int(round(r)) if abs(r) > 0.5 else 0,
                            A.SOURCE_HUMAN, white=lg.head.get("network", "?"),
                            black=lg.head.get("network", "?"))
                        if n:
                            msg += f"  ·  в архив: {n} позиций"
            except Exception as e:
                msg += f"  ·  запись не сохранена: {e}"
            self.statusBar().showMessage(f"Партия окончена — {msg}")
        elif snap is not None:
            self.apply_payload(snap, live=False)
        else:
            node = (self.ttable.get(position_key(self.board.engine))
                    if self.mcts is not None else None)
            if node is not None and node.visits > 0:
                self.apply_payload(
                    payload_from_node(node, self.board.engine.side_to_move()),
                    live=False)
            else:
                self.infobox.set_moves([])
                self.eval_bar.set_winrate(self.evals.get(self.cursor, 0.5))
                if (not over and self.mcts is not None and not self.should_search()):
                    self.statusBar().showMessage(
                        f"Позиция {self.cursor} · анализ остановлен — ▶ чтобы запустить")

        if not over and self.should_search():
            engine_turn = (self.cursor == len(self.history)
                           and self.engine_should_move())
            budget = (self.spin_play.value() if engine_turn
                      else self.spin_analyze.value())
            if len(self.ttable) > TT_MAX_NODES:
                self._prune_ttable(view)
            self.search = SearchThread(view, self.mcts, budget, self.ttable,
                                       contempt=self.contempt)
            self.search.update.connect(self.on_update)
            self.search_started = time.time()
            self.search.start()

        self.graph.set_data(self.evals, self.cursor, len(self.history))
        self._rebuild_table()

    _vram_cache = (0.0, "")

    def vram_note(self):
        now = time.time()
        if now - self._vram_cache[0] < 5.0:
            return self._vram_cache[1]
        note = ""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2).stdout
            for line in out.strip().splitlines():
                pid, mem = [x.strip() for x in line.split(",")]
                if int(pid) == os.getpid():
                    note = f"  ·  карта: {int(mem) / 1024:.1f} ГБ"
                    break
        except Exception:
            pass
        NibblerGUI._vram_cache = (now, note)
        return note

    def apply_payload(self, payload, live):
        moves = payload["moves"]
        stm = payload["stm"]
        hide = self.hints_hidden()
        self.infobox.set_moves([] if hide else moves)
        self.board.analysis = [] if hide else moves
        self.board.update()

        wr_stm = (payload["root_q"] + 1.0) / 2.0
        wr_white = wr_stm if stm == 0 else 1.0 - wr_stm

        mate_in = None
        if moves:
            top = moves[0]
            top_q = top["q"]
            nn = top.get("nn")
            top_d = nn[1] if nn else 0.5
            top_m_plies = nn[2] if (nn and len(nn) >= 3) else None
            if abs(top_q) > 0.92 and top_d < 0.15:
                if top_m_plies is not None and top_m_plies > 0:
                    n = max(1, int(round(top_m_plies / 2)))
                else:
                    n = max(1, int(round((1.0 - abs(top_q)) * 40)))
                n = min(n, 30)
                if top_q > 0:
                    mate_in = +n if stm == 0 else -n
                else:
                    mate_in = -n if stm == 0 else +n

        if hide:
            self.eval_bar.set_eval(0.5, None)
        else:
            self.eval_bar.set_eval(wr_white, mate_in)
        self.evals[self.cursor] = wr_white

        side = "белые" if stm == 0 else "чёрные"
        if live:
            elapsed = max(1e-3, time.time() - self.search_started)
            reused = payload.get("reused", 0)
            fresh = max(0, payload["sims"] - reused)
            self.statusBar().showMessage(
                f"Ход: {side}{self.vram_note()}  ·  "
                f"узлов: {payload['sims']:,} (♻ {reused:,})  ·  "
                f"{fresh / elapsed:.0f}/с  ·  слияний: {payload['merges']:,}" +
                ("" if hide else f"  ·  оценка (белые): {wr_white * 100:.1f}%"))
        else:
            self.statusBar().showMessage(
                f"Позиция {self.cursor} · слепок {payload['sims']:,} узлов" +
                ("" if hide else f" · оценка (белые): {wr_white * 100:.1f}%"))

    def on_update(self, payload):
        if payload.get("game_over"):
            return
        if payload.get("ply") is not None and payload["ply"] != self.cursor:
            return
        new_ply = self.cursor not in self.snapshots
        self.snapshots[self.cursor] = payload
        self.apply_payload(payload, live=True)

        shown = self.evals
        if self.hints_hidden():
            shown = {k: v for k, v in self.evals.items() if k != self.cursor}
        self.graph.set_data(shown, self.cursor, len(self.history))

        if new_ply:
            self._rebuild_table()
        elif self.cursor in self._table_plies:
            self.table.blockSignals(True)
            self._fill_table_row(self._table_plies.index(self.cursor), self.cursor)
            self.table.blockSignals(False)

        if (payload["finished"] and payload["moves"]
                and self.cursor == len(self.history)
                and self.engine_should_move()):
            self.push_move(payload["moves"][0]["move"], by="engine")

    # ---- analyzed-positions table ----
    def _fill_table_row(self, row, ply):
        snap = self.snapshots[ply]
        stm = snap["stm"]
        wr_stm = (snap["root_q"] + 1.0) / 2.0
        wr_white = wr_stm if stm == 0 else 1.0 - wr_stm
        played = move_to_uci(self.history[ply]) if ply < len(self.history) else "—"
        best = move_to_uci(snap["moves"][0]["move"]) if snap["moves"] else "—"
        cells = [str(ply), played, f"{wr_white * 100:.1f}%", best,
                 f"{snap['sims']:,}"]
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if col in (2, 4):
                item.setTextAlignment(Qt.AlignCenter)
            if ply == self.cursor:
                item.setBackground(QColor(ACCENT))
                item.setForeground(QColor("#ffffff"))
            elif played != "—" and best != "—" and played != best:
                item.setForeground(QColor("#d98a1f"))
            self.table.setItem(row, col, item)

    def _rebuild_table(self):
        plies = sorted(self.snapshots)
        self._table_plies = plies
        self.table.blockSignals(True)
        self.table.setRowCount(len(plies))
        for row, ply in enumerate(plies):
            self._fill_table_row(row, ply)
        if self.cursor in self._table_plies:
            self.table.selectRow(self._table_plies.index(self.cursor))
        self.table.blockSignals(False)

    def closeEvent(self, ev):
        self.stop_search()
        ev.accept()


# ───────────────────────────── Stylesheet ─────────────────────────────────

QSS = f"""
QMainWindow, QWidget {{ background: {BG}; color: {FG};
                        font-family: "Segoe UI", sans-serif; font-size: 12px; }}
QGroupBox {{ border: 1px solid #444; border-radius: 5px; margin-top: 9px;
             padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px;
                    color: {ACCENT}; font-weight: bold; }}
QPushButton {{ background: {BG3}; border: 1px solid #4d4d4d; border-radius: 4px;
               padding: 6px 9px; }}
QPushButton:hover {{ background: #454545; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #555; }}
QSpinBox, QComboBox {{ background: {BG2}; border: 1px solid #444;
                       border-radius: 4px; padding: 3px; min-width: 64px; }}
QComboBox QAbstractItemView {{ background: {BG2}; selection-background-color: {ACCENT}; }}
QTableWidget {{ background: {BG2}; border: 1px solid #444; border-radius: 4px;
                gridline-color: #3c3c3c; }}
QTableWidget::item {{ padding: 3px; }}
QTableWidget::item:selected {{ background: {ACCENT}; color: #fff; }}
QHeaderView::section {{ background: {BG3}; color: #b9b9b9; border: 0;
                        border-right: 1px solid #444; padding: 4px; }}
QTabWidget::pane {{ border: 1px solid #444; background: {BG2}; border-radius: 4px; }}
QTabBar::tab {{ background: {BG3}; color: #aaa; padding: 6px 12px; border-top-left-radius: 4px;
                border-top-right-radius: 4px; margin-right: 2px; }}
QTabBar::tab:selected {{ background: {BG2}; color: {FG}; font-weight: bold; border-top: 2px solid {ACCENT}; }}
QStatusBar {{ background: {BG2}; color: #b9b9b9; }}
"""


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    gui = NibblerGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
