"""Capablanca AI — Nibbler-style real-time analysis GUI.

A dark-themed analysis frontend for the Capablanca Chess network, inspired by
Nibbler (rooklift/nibbler). Talks to the network directly through mcts.py — no
UCI bridge — and renders the native 10x8 board with Archbishop and Chancellor.

Features:
  * Live MCTS analysis with a ranked move infobox (N / P / Q / WDL).
  * Transposition-aware search: positions reached by different move orders are
    merged into a single search node (see TranspositionSearch below).
  * Per-position analysis snapshots — reviewing history shows the cached
    analysis instead of launching a fresh search.
  * Eval bar, per-game winrate graph and a table of analyzed positions.
  * Play against the network (either colour) or watch engine self-play.
  * Coloured move arrows with the move winrate printed inside them.

UI text is in Russian to match the rest of the project; code comments are in
English.
"""

import math
import sys
import time
import traceback

import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QPushButton, QLabel,
                             QFileDialog, QSpinBox, QGroupBox, QDialog, QComboBox,
                             QCheckBox, QSizePolicy, QShortcut, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QFrame)
from PyQt5.QtGui import (QPainter, QColor, QFont, QPen, QPolygonF, QKeySequence)
from PyQt5.QtCore import Qt, QRect, QRectF, QPointF, QThread, pyqtSignal

try:
    from capablanca_engine import CapablancaEngine
    from model import CapablancaNet
    from mcts import UltraFastMCTS, VIRTUAL_LOSS
except ImportError as e:
    print(f"Ошибка импорта! {e}")
    traceback.print_exc()
    sys.exit(1)


# ───────────────────────────── Constants ──────────────────────────────────

# piece_type: PAWN=0 KNIGHT=1 BISHOP=2 ROOK=3 QUEEN=4 ARCH=5 CHANC=6 KING=7
PIECE_GLYPHS = {0: '♟', 1: '♞', 2: '♝', 3: '♜', 4: '♛', 5: 'A', 6: 'C', 7: '♚'}

# Promotion: m_int stores p_val = piece_const + 1 (see lib.rs encode_move).
_PROMO_FROM_VAL = [None, None, 'n', 'b', 'r', 'q', 'a', 'c']
_PROMO_LABELS = [('q', 'Ферзь ♛'), ('r', 'Ладья ♜'), ('b', 'Слон ♝'),
                 ('n', 'Конь ♞'), ('a', 'Архиепископ A'), ('c', 'Канцлер C')]

# Arrow / rank colours, brightest first (Nibbler-style ranked highlighting).
RANK_COLORS = [QColor("#3aa655"), QColor("#3d7fd6"), QColor("#d98a1f"),
               QColor("#9b59b6"), QColor("#1aa3a3")]
GREY = QColor("#6f6f6f")

FPU_REDUCTION = 0.330      # relative first-play-urgency, from lc0
MAX_SELECT_DEPTH = 160     # hard safety cap on selection descent
TT_MAX_NODES = 2_000_000   # transposition table soft cap — cleared when exceeded

# Theme
BG = "#262626"
BG2 = "#2f2f2f"
BG3 = "#383838"
FG = "#d8d8d8"
ACCENT = "#5b9bd5"
LIGHT_SQ = QColor("#e9edcc")
DARK_SQ = QColor("#7a9b5b")

W_COL = QColor("#3aa655")   # WDL bar: win
D_COL = QColor("#8a8a8a")   # draw
L_COL = QColor("#cf4f3a")   # loss


# ───────────────────────────── Move helpers ───────────────────────────────

def decode_move(m_int):
    """m_int -> (from_sq, to_sq, promo_letter|None). Squares are rank*10+file."""
    p_val = m_int & 0b111
    to_sq = (m_int >> 3) & 0x7F
    from_sq = (m_int >> 10) & 0x7F
    promo = _PROMO_FROM_VAL[p_val] if 0 < p_val < len(_PROMO_FROM_VAL) else None
    return from_sq, to_sq, promo


def sq_to_str(s):
    """Square index -> coordinate string, e.g. 0 -> 'a1', 79 -> 'j8'."""
    return f"{chr(ord('a') + (s % 10))}{(s // 10) + 1}"


def move_to_uci(m_int):
    f, t, p = decode_move(m_int)
    return sq_to_str(f) + sq_to_str(t) + (p if p else "")


def value_to_wdl(v):
    """Map a scalar value in [-1, 1] to a (win, draw, loss) triple."""
    v = max(-1.0, min(1.0, float(v)))
    p_win = max(0.0, v) ** 0.5
    p_loss = max(0.0, -v) ** 0.5
    p_draw = max(0.0, 1.0 - p_win - p_loss)
    s = p_win + p_draw + p_loss
    if s <= 1e-9:
        return 0.0, 1.0, 0.0
    return p_win / s, p_draw / s, p_loss / s


_HASH_WARNED = [False]


def position_key(engine):
    """Transposition key for a position.

    Prefers the engine's native ``position_hash`` (pieces + side + castling +
    en-passant). Falls back to a piece-list hash if the engine binary predates
    that method — that fallback ignores castling/ep rights, so rebuild the Rust
    engine (``maturin develop --release``) for fully correct transpositions.
    """
    fn = getattr(engine, "position_hash", None)
    if fn is not None:
        return fn()
    if not _HASH_WARNED[0]:
        _HASH_WARNED[0] = True
        print("⚠️  engine.position_hash отсутствует — пересоберите Rust-движок "
              "для точных транспозиций. Использую запасной ключ.")
    return hash((tuple(sorted(engine.get_pieces())), engine.side_to_move()))


# ───────────────────── Transposition-aware search tree ────────────────────

class _TNode:
    """A search node, uniquely identified by a board position.

    Statistics (visits / value) live on the node and are therefore shared by
    every move order that reaches this position — that is the merge. Move
    priors live on the *edges* instead, since a prior is parent-specific.
    """
    __slots__ = ("visits", "value_sum", "vloss", "is_expanded",
                 "is_terminal", "children")

    def __init__(self):
        self.visits = 0
        self.value_sum = 0.0
        self.vloss = 0
        self.is_expanded = False
        self.is_terminal = False
        self.children = {}          # move_int -> _TEdge

    def q(self):
        d = self.visits + self.vloss
        return self.value_sum / d if d > 0 else 0.0


class _TEdge:
    __slots__ = ("prior", "move", "child", "child_hash")

    def __init__(self, prior, move, child, child_hash):
        self.prior = prior
        self.move = move
        self.child = child
        self.child_hash = child_hash


# ───────────────────────────── Search thread ──────────────────────────────

class SearchThread(QThread):
    """Background transposition-aware MCTS worker.

    Builds a search DAG for one fixed position and streams ranked statistics
    back to the GUI. Two move orders that reach the same position share one
    _TNode, so their visits and values accumulate together. The DAG (``ttable``)
    is owned by the GUI and persists across moves, so a search picks up the
    sub-tree an earlier search already built for this position (tree reuse).
    A path-hash set guards against repetition cycles.
    """
    update = pyqtSignal(object)   # dict payload

    def __init__(self, move_history, mcts, max_sims, ttable):
        super().__init__()
        self.move_history = list(move_history)
        self.mcts = mcts
        self.max_sims = max_sims if max_sims > 0 else 50_000_000
        self.c_puct = mcts.c_puct
        self.running = True
        self.ttable = ttable        # shared, persists across moves -> tree reuse
        self.tt_hits = 0            # number of merged (transposed) edges
        self.reused = 0             # visits inherited from earlier searches
        self.root_engine = None
        self.root_hash = 0

    # ---- tree primitives ----
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
        best, best_s = None, -1e18
        for e in node.children.values():
            c = e.child
            started = c.visits + c.vloss
            q_in_parent = -(c.value_sum / started) if started > 0 else fpu
            s = q_in_parent + self.c_puct * e.prior * sqrt_n / (1 + started)
            if s > best_s:
                best_s, best = s, e
        return best

    def _select(self, root):
        """Descend from the root. Returns (path_nodes, path_moves, kind).

        kind is 'leaf' (needs evaluation / expansion) or 'rep' (the chosen edge
        re-enters a position already on this path — treated as a draw).
        """
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
                self.tt_hits += 1                 # transposition merged
            leaf.children[m] = _TEdge(float(pr), m, node, h)
        leaf.is_expanded = True

    @staticmethod
    def _vloss(path, delta):
        for n in path:
            n.vloss = max(0, n.vloss + delta)

    @staticmethod
    def _backup(path, leaf_value):
        # leaf_value is from the leaf's side-to-move; sign flips up the path.
        sign = 1.0
        for n in reversed(path):
            n.visits += 1
            n.value_sum += leaf_value * sign
            sign = -sign

    @staticmethod
    def _terminal_value(sim):
        r = sim.game_result()                      # white's perspective
        return r if sim.side_to_move() == 0 else -r

    # ---- main loop ----
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

            # Tree reuse: if this position was already explored (as the root of
            # an earlier search or as a sub-node of one) its _TNode is still in
            # the shared table — pick it up and keep accumulating visits.
            root = self.ttable.get(self.root_hash)
            if root is None:
                root = _TNode()
                self.ttable[self.root_hash] = root
            if not root.is_expanded:
                tensor = np.asarray(engine.get_board_tensor(), dtype=np.float32)
                policy, _, _, _ = mcts._infer([tensor])
                self._expand(root, engine, policy[0])
            self.reused = root.visits

            child_nn = {}        # root move -> (child_q, child_d) network eval
            total = root.visits  # continue the count, don't restart it
            last_emit = 0.0
            bs = mcts.batch_size

            while self.running and total < self.max_sims:
                tensors, pending = [], []
                attempts = 0
                while (len(tensors) < bs and attempts < bs * 4 and self.running):
                    attempts += 1
                    path_nodes, path_moves, kind = self._select(root)
                    leaf = path_nodes[-1]

                    if kind == 'rep':                    # repetition -> draw
                        self._backup(path_nodes, 0.0)
                        total += 1
                        continue
                    if leaf.is_terminal:
                        sim = self._replay(path_moves)
                        self._backup(path_nodes, self._terminal_value(sim))
                        total += 1
                        continue

                    sim = self._replay(path_moves)
                    if sim.is_game_over():
                        leaf.is_terminal = True
                        leaf.is_expanded = True
                        self._backup(path_nodes, self._terminal_value(sim))
                        total += 1
                        continue

                    tensors.append(np.asarray(sim.get_board_tensor(),
                                               dtype=np.float32))
                    pending.append((leaf, path_nodes, path_moves, sim))
                    self._vloss(path_nodes, VIRTUAL_LOSS)

                if pending:
                    pols, vals, draws, _ = mcts._infer(tensors)
                    for i, (leaf, path_nodes, path_moves, sim) in enumerate(pending):
                        if not leaf.is_expanded:
                            self._expand(leaf, sim, pols[i])
                        self._vloss(path_nodes, -VIRTUAL_LOSS)
                        v = float(vals[i])
                        if len(path_moves) == 1:         # direct child of root
                            child_nn[path_moves[0]] = (v, float(draws[i]))
                        self._backup(path_nodes, v)
                    total += len(pending)

                now = time.time()
                if (now - last_emit > 0.13 or total >= self.max_sims
                        or not self.running):
                    last_emit = now
                    self._emit(root, stm, total, child_nn,
                               finished=(total >= self.max_sims))
                self.msleep(1)

            self._emit(root, stm, total, child_nn, finished=True)

        except Exception as e:
            print(f"[SearchThread] Ошибка: {e}")
            traceback.print_exc()

    def _emit(self, root, stm, total, child_nn, finished):
        tv = sum(e.child.visits for e in root.children.values())
        moves = []
        for m, e in root.children.items():
            c = e.child
            if c.visits == 0:
                continue
            mq = -(c.value_sum / c.visits)       # value from the mover's side
            moves.append({
                "move": m,
                "visits": c.visits,
                "prior": e.prior,
                "q": mq,
                "frac": (c.visits / tv) if tv > 0 else 0.0,
                "nn": child_nn.get(m),
            })
        moves.sort(key=lambda d: d["visits"], reverse=True)
        self.update.emit({"game_over": False, "moves": moves,
                          "root_q": root.q(), "stm": stm, "sims": total,
                          "merges": self.tt_hits, "reused": self.reused,
                          "finished": finished})


# ───────────────────────────── Eval bar ───────────────────────────────────

class EvalBar(QWidget):
    """Vertical evaluation bar — white fill grows from the bottom."""

    def __init__(self):
        super().__init__()
        self.setFixedWidth(30)
        self.white_wr = 0.5

    def set_winrate(self, wr_white):
        self.white_wr = max(0.0, min(1.0, wr_white))
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
        pct = f"{self.white_wr * 100:.0f}"
        p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        if self.white_wr >= 0.5:
            p.setPen(QColor("#1c1c1c"))
            p.drawText(QRect(0, h - 18, w, 16), Qt.AlignCenter, pct)
        else:
            p.setPen(QColor("#ededed"))
            p.drawText(QRect(0, 2, w, 16), Qt.AlignCenter, pct)


# ───────────────────────────── Winrate graph ──────────────────────────────

class WinrateGraph(QWidget):
    """Per-game winrate graph (white POV). Click a point to jump to that ply."""
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
    """Ranked move list with N / P / Q / WDL — the Nibbler-style infobox."""
    play_move = pyqtSignal(int)

    ROW_H = 50

    def __init__(self):
        super().__init__()
        self.rows = []
        self.hover = -1
        self.setMouseTracking(True)
        self.setMinimumWidth(330)

    def set_moves(self, moves):
        self.rows = moves[:14]
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
            p.drawText(QRect(14, y + 24, w - 100, 16),
                       Qt.AlignVCenter | Qt.AlignLeft, stats)

            if d["nn"] is not None:
                cq, cd = d["nn"]
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
    """10x8 board: square-aspect rendering, piece outlines, arrows, dots."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setMinimumSize(720, 576)
        self.engine = CapablancaEngine()
        self.legal_moves = self.engine.get_legal_moves_int()
        self.selected = None
        self.flipped = False
        self.last_move = None
        self.analysis = []          # full ranked move list from the search

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

        # Which analysis arrows to show: when a piece is focused (selected and
        # it has analysed moves) — only that piece's moves; otherwise the
        # global top moves. Clicking the piece again clears the focus.
        piece_moves = []
        if self.selected is not None:
            piece_moves = [d for d in self.analysis
                           if decode_move(d["move"])[0] == self.selected]
        shown = piece_moves if piece_moves else self.analysis[:5]

        if self.selected is not None:
            best_dest = decode_move(piece_moves[0]["move"])[1] if piece_moves else None
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

        for i, d in enumerate(shown):
            color = RANK_COLORS[i] if i < len(RANK_COLORS) else GREY
            self._draw_arrow(p, d, color, cell)

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

    def _draw_arrow(self, p, d, color, cell):
        f, t, _ = decode_move(d["move"])
        s = self.sq_rect(f).center()
        e = self.sq_rect(t).center()
        c = QColor(color)
        c.setAlpha(220)
        width = max(5.0, cell * 0.14)
        ang = np.arctan2(e.y() - s.y(), e.x() - s.x())
        head = cell * 0.30
        tip = QPointF(e.x() - np.cos(ang) * head * 0.2,
                      e.y() - np.sin(ang) * head * 0.2)
        base = QPointF(tip.x() - np.cos(ang) * head,
                       tip.y() - np.sin(ang) * head)
        p.setPen(QPen(c, width, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(s, base)
        p.setBrush(c)
        p.setPen(Qt.NoPen)
        p1 = base + QPointF(-np.sin(ang) * head * 0.6, np.cos(ang) * head * 0.6)
        p2 = base - QPointF(-np.sin(ang) * head * 0.6, np.cos(ang) * head * 0.6)
        p.drawPolygon(QPolygonF([tip, p1, p2]))

        wr = (d["q"] + 1.0) / 2.0
        mid = QPointF(s.x() + (e.x() - s.x()) * 0.55,
                      s.y() + (e.y() - s.y()) * 0.55)
        label = f"{wr * 100:.0f}%"
        p.setFont(QFont("Segoe UI", max(8, int(cell * 0.18)), QFont.Bold))
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(label) + 10
        th = fm.height() + 2
        box = QRectF(mid.x() - tw / 2, mid.y() - th / 2, tw, th)
        dark = QColor(color.darker(180))
        dark.setAlpha(235)
        p.setBrush(dark)
        p.setPen(QPen(c.lighter(140), 1))
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
            self.selected = None             # second click — back to normal
        elif any(decode_move(m)[0] == sq for m in self.legal_moves):
            self.selected = sq               # focus this piece's moves
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = None
        self.mcts = None
        self.search = None
        self.search_started = 0.0

        self.history = []            # full move list (linear)
        self.cursor = 0              # plies shown (0..len(history))
        self.evals = {}              # ply -> white winrate
        self.snapshots = {}          # ply -> last analysis payload for that ply
        self.ttable = {}             # persistent search DAG — reused across moves
        self.mode = "analyze"
        self.analysis_on = True

        self._build_ui()
        self._build_shortcuts()
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

        self.btn_go = QPushButton("⏸ Остановить анализ")
        self.btn_go.clicked.connect(self._toggle_analysis)
        lay.addWidget(self.btn_go)

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

        lay.addWidget(QLabel("Анализ текущей позиции:"))
        self.infobox = InfoBox()
        self.infobox.play_move.connect(self.try_human_move)
        lay.addWidget(self.infobox)

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
        lay.addWidget(self.table, 1)

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

    # ---- model loading ----
    def load_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить веса", "", "*.pth")
        if not path:
            return
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            raw = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
            sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
                  for k, v in raw.items()}

            stem = sd.get("input_conv.net.0.weight", sd.get("input_block.0.weight"))
            ch = stem.shape[0]
            bl = sum(1 for k in sd if "res_blocks" in k and k.endswith("conv1.weight"))

            net = CapablancaNet(num_channels=ch, num_res_blocks=bl)
            tgt = net.state_dict()
            sd = {k: v for k, v in sd.items()
                  if not (k in tgt and v.shape != tgt[k].shape)}
            net.load_state_dict(sd, strict=False)
            net.to(self.device).eval()

            self.net = net
            self.mcts = UltraFastMCTS(net, self.device, c_puct=1.745,
                                      batch_size=96, add_dirichlet=False,
                                      nn_cache=True)
            self.ttable.clear()      # old stats came from the previous weights
            self.statusBar().showMessage(
                f"Модель загружена: {ch}ch × {bl}bl · {self.device}")
            self.refresh()
        except Exception as e:
            self.statusBar().showMessage(f"Ошибка загрузки: {e}")
            traceback.print_exc()

    # ---- controls ----
    def _mode_changed(self, idx):
        self.mode = MODES[idx]
        self.refresh()

    def _toggle_analysis(self):
        self.analysis_on = not self.analysis_on
        self.btn_go.setText("⏸ Остановить анализ" if self.analysis_on
                            else "▶ Запустить анализ")
        self.refresh()

    def new_game(self):
        self.history = []
        self.cursor = 0
        self.evals = {}
        self.snapshots = {}
        self.ttable.clear()
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
            return                       # not the human's turn
        if m not in self.board.engine.get_legal_moves_int():
            return
        self.push_move(m)

    def push_move(self, m):
        del self.history[self.cursor:]
        self.history.append(m)
        self.cursor += 1
        # snapshot/eval of the position we just left (== cursor-1) is kept;
        # anything strictly ahead belonged to a now-discarded line.
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
            # wait() returns as soon as the thread ends; the long cap only
            # matters for a slow first GPU inference (cuDNN autotune).
            self.search.wait(10000)
            self.search = None

    def should_search(self):
        """Whether a search should run for the currently viewed position."""
        if self.mcts is None or self.board.engine.is_game_over():
            return False
        if self.analysis_on:
            return True
        # Analysis display is off, but in a real game versus the engine it
        # must still answer the human's move.
        return (self.cursor == len(self.history) and self.engine_should_move()
                and self.mode in ("play_white", "play_black"))

    def refresh(self):
        """Rebuild the viewed position, show its snapshot, and search if needed.

        The cached snapshot (if any) is shown immediately as a placeholder so
        navigation feels instant; a live search then refines it. With analysis
        turned off, snapshots are shown but no new search is started — except
        that in Play mode the engine still replies to the human's move.
        """
        self.stop_search()
        view = self.history[:self.cursor]
        last = (decode_move(view[-1])[:2] if view else None)
        self.board.set_position(view, last)
        self.board.analysis = []

        over = self.board.engine.is_game_over()
        snap = self.snapshots.get(self.cursor)

        if over:
            self.infobox.set_moves([])
            self.eval_bar.set_winrate(self.evals.get(self.cursor, 0.5))
            r = self.board.engine.game_result()
            msg = ("Ничья" if abs(r) < 1e-6
                   else ("Белые выиграли" if r > 0 else "Чёрные выиграли"))
            self.statusBar().showMessage(f"Партия окончена — {msg}")
        elif snap is not None:
            self.apply_payload(snap, live=False)        # instant placeholder
        else:
            self.infobox.set_moves([])
            self.eval_bar.set_winrate(self.evals.get(self.cursor, 0.5))

        if not over and self.should_search():
            engine_turn = (self.cursor == len(self.history)
                           and self.engine_should_move())
            budget = (self.spin_play.value() if engine_turn
                      else self.spin_analyze.value())
            if len(self.ttable) > TT_MAX_NODES:
                self.ttable.clear()          # soft memory cap
            self.search = SearchThread(view, self.mcts, budget, self.ttable)
            self.search.update.connect(self.on_update)
            self.search_started = time.time()
            self.search.start()
        elif not over and self.mcts is not None and snap is None:
            self.statusBar().showMessage(
                f"Позиция {self.cursor} · анализ остановлен — ▶ чтобы запустить")

        self.graph.set_data(self.evals, self.cursor, len(self.history))
        self._rebuild_table()

    def apply_payload(self, payload, live):
        """Render an analysis payload (live result or cached snapshot)."""
        moves = payload["moves"]
        stm = payload["stm"]
        self.infobox.set_moves(moves)
        self.board.analysis = moves
        self.board.update()

        wr_stm = (payload["root_q"] + 1.0) / 2.0
        wr_white = wr_stm if stm == 0 else 1.0 - wr_stm
        self.eval_bar.set_winrate(wr_white)
        self.evals[self.cursor] = wr_white

        side = "белые" if stm == 0 else "чёрные"
        if live:
            elapsed = max(1e-3, time.time() - self.search_started)
            reused = payload.get("reused", 0)
            fresh = max(0, payload["sims"] - reused)
            reuse_txt = f" (♻ {reused:,})" if reused else ""
            self.statusBar().showMessage(
                f"Ход: {side}  ·  узлов: {payload['sims']:,}{reuse_txt}  ·  "
                f"{fresh / elapsed:.0f}/с  ·  слияний транспозиций: "
                f"{payload['merges']:,}  ·  оценка (белые): {wr_white * 100:.1f}%")
        else:
            self.statusBar().showMessage(
                f"Позиция {self.cursor} · слепок {payload['sims']:,} узлов · "
                f"оценка (белые): {wr_white * 100:.1f}%")

    def on_update(self, payload):
        if payload.get("game_over"):
            return
        new_ply = self.cursor not in self.snapshots
        self.snapshots[self.cursor] = payload      # cache before any move
        self.apply_payload(payload, live=True)
        self.graph.set_data(self.evals, self.cursor, len(self.history))

        # During continuous analysis only the current row changes — a full
        # rebuild every tick would flicker, so refresh just that row.
        if new_ply:
            self._rebuild_table()
        elif self.cursor in self._table_plies:
            self.table.blockSignals(True)
            self._fill_table_row(self._table_plies.index(self.cursor), self.cursor)
            self.table.blockSignals(False)

        if (payload["finished"] and payload["moves"]
                and self.cursor == len(self.history)
                and self.engine_should_move()):
            self.push_move(payload["moves"][0]["move"])

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
                item.setForeground(QColor("#d98a1f"))     # deviated from top
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
