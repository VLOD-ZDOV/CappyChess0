"""Capablanca AI — Nibbler-style real-time analysis GUI.

A dark-themed analysis frontend for the Capablanca Chess network, inspired by
Nibbler (rooklift/nibbler). Talks to the network directly through onnx_engine.py
and renders the native 10x8 board with Archbishop and Chancellor.

Features:
  * Live MCTS analysis with a ranked move infobox (N / P / Q / WDL);
    hovering a line previews its principal variation on the board.
  * Opening Book / Explorer with WDL percentages from the self-play archive.
  * Archive Game Browser (all games, sortable, filters) and game replay.
  * Transposition-aware search and tree reuse across moves.
  * Eval bar, per-game winrate graph and a move list with engine evaluations.

Usage:
    python gui.py [сеть.onnx|чекпоинт.pth] [партия.pgn] [--archive КАТАЛОГ]
"""

import argparse
import itertools
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
                             QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QSpinBox, QGroupBox, QDialog, QComboBox, QMessageBox,
                             QCheckBox, QSizePolicy, QShortcut, QTableWidget, QTableView,
                             QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QFrame, QTabWidget, QScrollArea, QSplitter)
from PyQt5.QtGui import (QPainter, QColor, QFont, QPen, QPolygonF, QKeySequence,
                         QPainterPath, QFontDatabase, QFontMetrics, QBrush)
from PyQt5.QtCore import (Qt, QRect, QRectF, QPointF, QThread, pyqtSignal,
                          QAbstractTableModel, QModelIndex)

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

_PROMO_FROM_VAL = [None, None, 'n', 'b', 'r', 'q', 'a', 'c']
_PROMO_LABELS = [('q', 'Ферзь'), ('c', 'Канцлер'), ('a', 'Архиепископ'),
                 ('r', 'Ладья'), ('b', 'Слон'), ('n', 'Конь')]
_PROMO_TYPE = {'n': 1, 'b': 2, 'r': 3, 'q': 4, 'a': 5, 'c': 6}

RANK_COLORS = [QColor("#3fb35f"), QColor("#4a8fe0"), QColor("#e0932b"),
               QColor("#a66bd0"), QColor("#26b0b0")]
GREY = QColor("#7a7a7a")

FPU_REDUCTION = 0.330
MAX_SELECT_DEPTH = 160
TT_MAX_NODES = 500_000
PV_DEPTH = 10
DEVICES = ("auto", "cuda", "cpu")
INFOBOX_ROWS = 12
BOOK_PLY = 32
# На процессоре один прогон пачки 96 длится ~0.9 с, и столько же ждёт любая
# реакция на смену позиции; пачка 8 даёт почти ту же скорость (замер: 91 против
# 108 поз/с) при задержке ~0.1 с.
BATCH_GPU = 96
BATCH_CPU = 8

BG = "#232323"
BG2 = "#2b2b2b"
BG3 = "#353535"
BORDER = "#3f3f3f"
FG = "#dcdcdc"
FG_DIM = "#9a9a9a"
ACCENT = "#5b9bd5"
LIGHT_SQ = QColor("#eeeed2")
DARK_SQ = QColor("#769656")
HL_LAST = QColor(246, 222, 90, 110)
HL_SEL = QColor(255, 236, 110, 150)

W_COL = QColor("#3aa655")   # победа
D_COL = QColor("#7f8c8d")   # ничья
L_COL = QColor("#cf4f3a")   # поражение

UI_FAMILIES = ["Segoe UI", "Inter", "Noto Sans", "DejaVu Sans", "Arial"]
MONO_FAMILIES = ["JetBrains Mono", "Consolas", "DejaVu Sans Mono", "Noto Sans Mono",
                 "Courier New"]


def ui_font(pt, bold=False, mono=False):
    f = QFont()
    f.setFamilies(MONO_FAMILIES if mono else UI_FAMILIES)
    if mono:
        f.setStyleHint(QFont.Monospace)
    f.setPointSizeF(pt)
    f.setBold(bold)
    return f


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


def wr_text(wr):
    return f"{wr * 100:.1f}%"


_HASH_WARNED = [False]


def position_key(engine):
    fn = getattr(engine, "position_hash", None)
    if fn is not None:
        return fn()
    if not _HASH_WARNED[0]:
        _HASH_WARNED[0] = True
        print("⚠️  engine.position_hash отсутствует — используется запасной ключ.")
    return hash((tuple(sorted(engine.get_pieces())), engine.side_to_move()))


def replay_legal(moves):
    """Проиграть ходы с проверкой: возвращает (движок, принятые ходы).
    Обрывается на первом ходе не по правилам, а не роняет интерфейс."""
    eng = CapablancaEngine()
    ok = []
    for m in moves:
        if eng.is_game_over() or m not in eng.get_legal_moves_int():
            break
        eng.make_move_int(m)
        ok.append(m)
    return eng, ok


# ─────────────────────── Индекс архива и дебютная книга ─────────────────────

class ArchiveIndex:
    """Ходы всех партий и книга дебютов в плоских массивах numpy.

    Прежняя версия держала миллион списков ходов и словарь на каждую позицию
    книги — это ~10 ГБ памяти на архиве в миллион партий. Здесь книга — это
    отсортированные ключи позиций, а статистика по ходам собирается при запросе.
    """

    def __init__(self, ar, moves, game_start, game_ok, book_keys, book_moves,
                 book_pov, book_side, max_book_ply):
        self.ar = ar
        self.moves = moves
        self.game_start = game_start
        self.game_ok = game_ok
        self.book_keys = book_keys
        self.book_moves = book_moves
        self.book_pov = book_pov
        self.book_side = book_side
        self.max_book_ply = max_book_ply
        self.n_games = len(ar.g["result"])
        self.n_book_positions = (int(np.count_nonzero(np.diff(book_keys)) + 1)
                                 if len(book_keys) else 0)

    def game_moves(self, gid):
        if not (0 <= gid < self.n_games) or not self.game_ok[gid]:
            return []
        mv = self.moves[self.game_start[gid]:self.game_start[gid + 1]]
        bad = np.flatnonzero(mv < 0)
        if len(bad):
            mv = mv[:bad[0]]
        return [int(x) for x in mv]

    def book_entry(self, key):
        k = np.uint64(key)
        lo = int(np.searchsorted(self.book_keys, k, "left"))
        hi = int(np.searchsorted(self.book_keys, k, "right"))
        if hi <= lo:
            return None
        mv = self.book_moves[lo:hi]
        pov = self.book_pov[lo:hi]
        uniq, inv = np.unique(mv, return_inverse=True)
        n = np.bincount(inv, minlength=len(uniq))
        w = np.bincount(inv, weights=(pov > 0), minlength=len(uniq))
        l = np.bincount(inv, weights=(pov < 0), minlength=len(uniq))
        moves = {}
        for i, u in enumerate(uniq):
            ni, wi, li = int(n[i]), int(w[i]), int(l[i])
            moves[int(u)] = {"n": ni, "w": wi, "l": li, "d": ni - wi - li}
        return {"total": hi - lo, "side": int(self.book_side[lo]), "moves": moves}


class ArchiveLoaderThread(QThread):
    """Индексирует партии и дебютную книгу без блокировки интерфейса."""
    progress = pyqtSignal(str)
    loaded = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, directory, max_book_ply=BOOK_PLY):
        super().__init__()
        self.directory = directory
        self.max_book_ply = max_book_ply
        self.cancelled = False

    def run(self):
        try:
            if A is None:
                raise ImportError("Модуль archive.py не найден рядом с gui.py")
            t0 = time.time()
            self.progress.emit("Архив: чтение метаданных…")
            ar = A.Archive(self.directory)
            # Archive держит в памяти политики всех позиций (~270 байт на
            # позицию, 5.5 ГБ на 20 млн); интерфейсу они не нужны.
            for _, _, meta in getattr(ar, "shards", []):
                meta.pop("pol_idx", None)
                meta.pop("pol_val", None)
            n_games = len(ar.g["result"])
            p_game = ar.p["game"]
            p_ply = ar.p["ply"]
            p_side = ar.p["side"]

            self.progress.emit("Архив: восстановление ходов…")
            raw = ar.p["move_raw"].astype(np.int32)
            idx = ar.p["move"].astype(np.int32)
            # Векторная копия archive.idx_to_move: ход по доске из индекса
            # политики для старых кусков без move_raw; превращения не
            # восстанавливаются (-1), и партия на них обрывается.
            f, t = idx // 80, idx % 80
            flip = p_side == 1
            f = np.where(flip, (7 - f // 10) * 10 + f % 10, f)
            t = np.where(flip, (7 - t // 10) * 10 + t % 10, t)
            fallback = np.where((idx >= 0) & (idx < 6400), (f << 10) | (t << 3), -1)
            moves = np.where(raw >= 0, raw, fallback).astype(np.int32)
            del raw, idx, f, t, fallback, flip

            if len(p_game) > 1 and np.any(p_game[1:] < p_game[:-1]):
                order = np.lexsort((p_ply, p_game))
                p_game, p_ply, p_side = p_game[order], p_ply[order], p_side[order]
                moves = moves[order]
            game_start = np.searchsorted(p_game, np.arange(n_games + 1))
            counts = np.diff(game_start)
            first_ply = np.full(n_games, -1, dtype=np.int64)
            has = counts > 0
            first_ply[has] = p_ply[game_start[:-1][has]]
            # Единицы партий в архиве начинаются не с нулевого полухода: такие
            # нельзя проиграть с начальной позиции.
            game_ok = has & (first_ply == 0)

            source = ar.g["source"]
            book_src = getattr(A, "SOURCE_BOOK", 4)
            # Записи перебора дебюта — не партии, исхода у них нет: в книге
            # они стали бы сотнями тысяч мнимых ничьих (как в archive.book_from).
            use = np.flatnonzero(game_ok & (source != book_src))
            lens = np.minimum(counts[use], self.max_book_ply)
            cap = int(lens.sum())
            keys = np.empty(cap, dtype=np.uint64)
            rows = np.empty(cap, dtype=np.int64)
            n = 0
            total = len(use)
            step = max(1, total // 50)
            self.progress.emit(f"Архив: книга дебютов по {total:,} партиям…")
            for j, gid in enumerate(use):
                if self.cancelled:
                    return
                if j % step == 0 and j:
                    self.progress.emit(
                        f"Архив: книга дебютов {100 * j // total}% ({j:,} из {total:,})…")
                s = int(game_start[gid])
                eng = CapablancaEngine()
                ph = getattr(eng, "position_hash", None)
                for r, mv in enumerate(moves[s:s + int(lens[j])].tolist(), start=s):
                    if mv < 0:
                        break
                    keys[n] = ph() if ph is not None else position_key(eng) & 0xFFFFFFFFFFFFFFFF
                    rows[n] = r
                    n += 1
                    try:
                        eng.make_move_int(mv)
                    except Exception:
                        break
            keys, rows = keys[:n], rows[:n]
            order = np.argsort(keys, kind="stable")
            keys, rows = keys[order], rows[order]
            side = p_side[rows].astype(np.int8)
            res = ar.g["result"][p_game[rows]].astype(np.int8)
            pov = np.where(side == 0, res, -res).astype(np.int8)
            index = ArchiveIndex(ar, moves, game_start, game_ok, keys,
                                 moves[rows].copy(), pov, side, self.max_book_ply)
            index.load_seconds = time.time() - t0
            self.loaded.emit(index)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")


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

_SEARCH_TOKENS = itertools.count(1)


class SearchThread(QThread):
    update = pyqtSignal(object)

    def __init__(self, move_history, mcts, max_sims, ttable, contempt=0.0, prev=None):
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
        self.token = next(_SEARCH_TOKENS)
        # Остановленный поиск доигрывает свою пачку в фоне; новый ждёт его
        # здесь, а не в потоке интерфейса, — дерево и сессия сети общие.
        self.prev = prev

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
        children = {}
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
            children[m] = _TEdge(float(pr), m, node, h)
        # Словарь подменяется целиком: читатель не должен увидеть его наполовину.
        leaf.children = children
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
            n.draw_sum += leaf_draw
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
            if self.prev is not None:
                self.prev.wait()
                self.prev = None
            if not self.running:
                return
            engine = CapablancaEngine()
            for m in self.move_history:
                engine.make_move_int(m)
            stm = engine.side_to_move()

            if engine.is_game_over():
                self.update.emit({"game_over": True, "result": engine.game_result(),
                                  "stm": stm, "moves": [], "root_q": 0.0,
                                  "sims": 0, "merges": 0, "finished": True,
                                  "token": self.token})
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
                    pols, vals, draws, _ = mcts._infer(tensors)
                    for i, (leaf, path_nodes, path_moves, sim) in enumerate(pending):
                        if not leaf.is_expanded:
                            self._expand(leaf, sim, pols[i])
                        self._vloss(path_nodes, -VIRTUAL_LOSS)
                        self._backup(path_nodes, float(vals[i]), float(draws[i]))
                    total += len(pending)

                now = time.time()
                if (now - last_emit > 0.13 or total >= self.max_sims or not self.running):
                    last_emit = now
                    self._emit(root, stm, total, finished=(total >= self.max_sims))
                self.msleep(1)

            self._emit(root, stm, total, finished=True)

        except Exception as e:
            print(f"[SearchThread] Ошибка: {e}")
            traceback.print_exc()

    def _emit(self, root, stm, total, finished):
        payload = payload_from_node(root, stm, self.root_engine)
        payload["ply"] = len(self.move_history)
        payload["root_hash"] = self.root_hash
        payload["sims"] = total
        payload["merges"] = self.tt_hits
        payload["reused"] = self.reused
        payload["finished"] = finished
        payload["token"] = self.token
        self.update.emit(payload)


def pv_marks(engine, pv):
    """Знак после каждого хода варианта: "#" мат, "+" шах, "" ничего.
    Модуль без is_check (старая сборка) даёт только маты."""
    if engine is None or not pv:
        return []
    e = engine.copy()
    has_check = hasattr(e, "is_check")
    marks = []
    for m in pv:
        if not e.make_move_int(m):
            break
        if e.is_game_over() and abs(e.game_result()) > 0.5:
            marks.append("#")
            break
        marks.append("+" if has_check and e.is_check() else "")
    return marks


def fmt_line(pv, marks):
    return " ".join(move_to_uci(m) + (marks[i] if i < len(marks) else "")
                    for i, m in enumerate(pv))


def payload_from_node(root, stm, engine=None):
    children = list(root.children.values())
    tv = sum(e.child.visits for e in children)
    moves = []
    for e in children:
        c = e.child
        n = c.visits
        if n == 0:
            continue
        q = -(c.value_sum / n)
        d = min(1.0, max(0.0, c.draw_sum / n))
        # Ход сразу в терминальную позицию, проигранную для соперника, — мат.
        mate = c.is_terminal and c.value_sum <= -n + 1e-6
        moves.append({
            "move": e.move,
            "visits": n,
            "prior": e.prior,
            "q": q,
            "d": d,
            "w": max(0.0, (1.0 - d + q) / 2.0),
            "l": max(0.0, (1.0 - d - q) / 2.0),
            "mate": bool(mate),
            "frac": (n / tv) if tv > 0 else 0.0,
            "pv": [e.move] + SearchThread._pv(c, PV_DEPTH),
        })
        moves[-1]["pv_marks"] = pv_marks(engine, moves[-1]["pv"])
    moves.sort(key=lambda d: (d["mate"], d["visits"]), reverse=True)
    rv = root.visits
    return {"game_over": False, "moves": moves,
            "root_q": root.q(),
            "root_d": (root.draw_sum / rv) if rv else 0.0,
            "stm": stm, "sims": rv, "merges": 0,
            "reused": rv, "finished": True}


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
        # Возврат назад и другой ход: журнал должен описывать сыгранную линию,
        # а не склейку всех попыток.
        self.plies = [r for r in self.plies if r["ply"] < ply]
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


# ───────────────────────────── Piece rendering ────────────────────────────

class PieceSet:
    """Фигуры как векторные контуры из шрифтовых глифов: чётко на любом размере
    и на HiDPI. Архиепископ и канцлер — составные значки (конь за слоном /
    ладьёй), как принято в досках для капабланки."""

    GLYPHS = {0: '♟', 1: '♞', 2: '♝', 3: '♜', 4: '♛', 7: '♚'}
    FONTS = ["DejaVu Sans", "Segoe UI Symbol", "Noto Sans Symbols2",
             "Noto Sans Symbols 2", "Arial Unicode MS", "FreeSerif"]
    KING_H = 0.80          # высота короля в долях клетки
    BASE = 0.40            # низ фигур ниже центра клетки

    def __init__(self):
        self.family = self._pick_family()
        self._cache = {}
        self._scale = None

    def _pick_family(self):
        fams = set(QFontDatabase().families())
        for fam in self.FONTS:
            if fam in fams:
                fm = QFontMetrics(QFont(fam, 40))
                if all(fm.inFont(ch) for ch in self.GLYPHS.values()):
                    return fam
        return QFont().defaultFamily()

    def _raw(self, ch):
        font = QFont(self.family)
        font.setPixelSize(400)
        path = QPainterPath()
        path.addText(0, 0, font, ch)
        return path

    def _unit_scale(self):
        if self._scale is None:
            h = self._raw('♚').boundingRect().height()
            self._scale = self.KING_H / h if h > 0 else 1.0 / 400
        return self._scale

    def _layer(self, ch, scale, cx, bottom):
        """Глиф → (силуэт, сам глиф) в координатах клетки."""
        path = self._raw(ch)
        br = path.boundingRect()
        s = self._unit_scale() * scale
        tr = [QPolygonF([QPointF((pt.x() - br.center().x()) * s + cx,
                                 (pt.y() - br.bottom()) * s + bottom) for pt in poly])
              for poly in path.toSubpathPolygons()]

        def area(q):
            return 0.5 * sum(q[i].x() * q[(i + 1) % len(q)].y()
                             - q[(i + 1) % len(q)].x() * q[i].y() for i in range(len(q)))
        # Детали рисунка — контуры с обратным направлением обхода; силуэт
        # собирается из контуров того же направления, что и самый большой.
        areas = [area(q) if len(q) > 2 else 0.0 for q in tr]
        big = max(range(len(tr)), key=lambda i: abs(areas[i]))
        sign = areas[big] > 0
        glyph, silhouette = QPainterPath(), QPainterPath()
        glyph.setFillRule(Qt.WindingFill)
        silhouette.setFillRule(Qt.WindingFill)
        for q, a in zip(tr, areas):
            if a == 0.0:
                continue
            glyph.addPolygon(q)
            glyph.closeSubpath()
            if (a > 0) == sign:
                silhouette.addPolygon(q)
                silhouette.closeSubpath()
        return silhouette.simplified(), glyph

    def layers(self, ptype):
        if ptype not in self._cache:
            if ptype == 5:
                self._cache[ptype] = [self._layer('♞', 0.80, -0.15, 0.31),
                                      self._layer('♝', 0.84, 0.14, self.BASE)]
            elif ptype == 6:
                self._cache[ptype] = [self._layer('♞', 0.80, -0.15, 0.31),
                                      self._layer('♜', 0.74, 0.14, self.BASE)]
            else:
                self._cache[ptype] = [self._layer(self.GLYPHS.get(ptype, '♟'),
                                                  1.0, 0.0, self.BASE)]
        return self._cache[ptype]

    def draw(self, p, center, cell, color, ptype, opacity=1.0):
        white = color == 0
        body = QColor("#fafafa") if white else QColor("#262626")
        edge = QColor("#151515") if white else QColor("#0a0a0a")
        detail = QColor("#151515") if white else QColor("#d8d8d8")
        p.save()
        p.setOpacity(opacity)
        p.translate(center)
        p.scale(cell, cell)
        for silhouette, glyph in self.layers(ptype):
            # Белая: белый силуэт и все контуры глифа тёмной линией. Чёрная:
            # светлый силуэт под тёмным глифом — дыры глифа дают светлые детали.
            p.setPen(QPen(edge, 0.034, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(body if white else detail)
            p.drawPath(silhouette)
            if white:
                p.setPen(QPen(detail, 0.022, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                p.setBrush(Qt.NoBrush)
            else:
                p.setPen(Qt.NoPen)
                p.setBrush(body)
            p.drawPath(glyph)
        p.restore()


# ───────────────────────────── Eval bar ───────────────────────────────────

class EvalBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(30)
        self.white_wr = 0.5
        self.mate_in = None
        self.flipped = False
        self.known = False

    def set_eval(self, wr_white, mate_in=None, known=True):
        self.white_wr = max(0.0, min(1.0, wr_white))
        self.mate_in = mate_in
        self.known = known
        w = self.white_wr
        self.setToolTip(f"Ожидаемый счёт — белые {w * 100:.1f}%, чёрные {(1 - w) * 100:.1f}%"
                        if known else "Нет оценки")
        self.update()

    def set_winrate(self, wr_white):
        self.set_eval(wr_white, None)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        h, w = self.height(), self.width()
        p.fillRect(self.rect(), QColor("#1a1a1a"))
        wh = int(round(h * self.white_wr))
        light = QColor("#ececec") if self.known else QColor("#5a5a5a")
        if self.flipped:
            p.fillRect(0, 0, w, wh, light)
        else:
            p.fillRect(0, h - wh, w, wh, light)
        p.setPen(QPen(QColor(ACCENT), 1, Qt.DashLine))
        p.drawLine(0, h // 2, w, h // 2)
        p.setPen(QPen(QColor(BORDER), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(0, 0, w - 1, h - 1)
        if not self.known:
            return

        white_leads = self.white_wr >= 0.5
        if self.mate_in is not None:
            label = f"#{abs(self.mate_in)}"
            white_leads = self.mate_in > 0
        else:
            lead = self.white_wr if white_leads else 1.0 - self.white_wr
            label = f"{lead * 100:.0f}"
        p.setFont(ui_font(8.5, bold=True))
        # Число — у стороны, которая впереди, и в её цвете.
        at_bottom = white_leads != self.flipped
        p.setPen(QColor("#1a1a1a") if white_leads else QColor("#ececec"))
        if at_bottom:
            p.drawText(QRect(0, h - 22, w, 18), Qt.AlignCenter, label)
        else:
            p.drawText(QRect(0, 4, w, 18), Qt.AlignCenter, label)


# ───────────────────────────── Winrate graph ──────────────────────────────

class WinrateGraph(QWidget):
    seek = pyqtSignal(int)
    PAD_L, PAD_R, PAD_T, PAD_B = 30, 8, 6, 16

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(96)
        self.evals = {}
        self.cursor = 0
        self.total_plies = 0
        self.setCursor(Qt.PointingHandCursor)

    def set_data(self, evals, cursor, total_plies):
        self.evals = dict(evals)
        self.cursor = cursor
        self.total_plies = max(1, total_plies)
        self.update()

    def _x(self, ply):
        return self.PAD_L + ply / max(1, self.total_plies) * (
            self.width() - self.PAD_L - self.PAD_R)

    def _y(self, wr):
        return self.PAD_T + (1 - wr) * (self.height() - self.PAD_T - self.PAD_B)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        h, w = self.height(), self.width()
        p.fillRect(self.rect(), QColor(BG2))
        x0, x1 = self.PAD_L, w - self.PAD_R
        p.setFont(ui_font(7))
        for wr, style in ((0.75, Qt.DotLine), (0.5, Qt.DashLine), (0.25, Qt.DotLine)):
            y = self._y(wr)
            p.setPen(QPen(QColor("#474747"), 1, style))
            p.drawLine(QPointF(x0, y), QPointF(x1, y))
            p.setPen(QColor("#7a7a7a"))
            p.drawText(QRectF(0, y - 7, x0 - 4, 14), Qt.AlignRight | Qt.AlignVCenter,
                       f"{int(wr * 100)}")
        # Номера ходов по оси X.
        n_moves = self.total_plies / 2
        step = max(1, int(math.ceil(n_moves / max(1, (x1 - x0) / 45) / 5.0)) * 5)
        p.setPen(QColor("#7a7a7a"))
        for mv in range(step, int(n_moves) + 1, step):
            x = self._x(mv * 2)
            p.drawText(QRectF(x - 15, h - self.PAD_B, 30, self.PAD_B),
                       Qt.AlignCenter, str(mv))

        if self.evals:
            pts = sorted(self.evals.items())
            poly = [QPointF(self._x(ply), self._y(wr)) for ply, wr in pts]
            mid = self._y(0.5)
            area = QPolygonF([QPointF(poly[0].x(), mid)] + poly +
                             [QPointF(poly[-1].x(), mid)])
            p.setPen(Qt.NoPen)
            p.save()
            p.setClipRect(QRectF(0, 0, w, mid))
            p.setBrush(QColor(236, 236, 236, 70))
            p.drawPolygon(area)
            p.restore()
            p.save()
            p.setClipRect(QRectF(0, mid, w, h - mid))
            p.setBrush(QColor(0, 0, 0, 110))
            p.drawPolygon(area)
            p.restore()
            p.setPen(QPen(QColor(ACCENT), 1.6))
            p.setBrush(Qt.NoBrush)
            if len(poly) > 1:
                p.drawPolyline(QPolygonF(poly))
            if len(poly) < 120:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor("#eaeaea"))
                for pt in poly:
                    p.drawEllipse(pt, 2.2, 2.2)
        else:
            p.setPen(QColor("#6f6f6f"))
            p.setFont(ui_font(8))
            p.drawText(QRectF(x0, 0, x1 - x0, h - self.PAD_B), Qt.AlignCenter,
                       "Оценки позиций появятся здесь по мере анализа")

        cx = self._x(self.cursor)
        p.setPen(QPen(QColor("#e8c84a"), 1.5))
        p.drawLine(QPointF(cx, 0), QPointF(cx, h - self.PAD_B))

    def _ply_at(self, x):
        frac = (x - self.PAD_L) / max(1, self.width() - self.PAD_L - self.PAD_R)
        return max(0, min(self.total_plies, int(round(frac * self.total_plies))))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.seek.emit(self._ply_at(ev.x()))

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.LeftButton:
            self.seek.emit(self._ply_at(ev.x()))


# ───────────────────────────── Infobox ────────────────────────────────────

class InfoBox(QWidget):
    play_move = pyqtSignal(int)
    hover_pv = pyqtSignal(object)
    ROW_H = 60

    def __init__(self):
        super().__init__()
        self.rows = []
        self.hover = -1
        self.message = "нет анализа"
        self.setMouseTracking(True)
        self.setMinimumWidth(300)

    def set_moves(self, moves, message=None):
        self.rows = moves[:INFOBOX_ROWS]
        if message is not None:
            self.message = message
        self.setMinimumHeight(max(1, len(self.rows)) * self.ROW_H + 4)
        if self.hover >= len(self.rows):
            self.hover = -1
            self.hover_pv.emit([])
        elif self.hover >= 0:
            self.hover_pv.emit(self.rows[self.hover].get("pv") or [])
        self.update()

    def _row_at(self, y):
        i = (y - 2) // self.ROW_H
        return i if 0 <= i < len(self.rows) else -1

    def mouseMoveEvent(self, ev):
        h = self._row_at(ev.y())
        if h != self.hover:
            self.hover = h
            self.setCursor(Qt.PointingHandCursor if h >= 0 else Qt.ArrowCursor)
            self.hover_pv.emit((self.rows[h].get("pv") or []) if h >= 0 else [])
            self.update()

    def leaveEvent(self, _):
        self.hover = -1
        self.hover_pv.emit([])
        self.update()

    def mousePressEvent(self, ev):
        i = self._row_at(ev.y())
        if i >= 0 and ev.button() == Qt.LeftButton:
            self.hover = -1
            self.hover_pv.emit([])
            self.play_move.emit(self.rows[i]["move"])

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BG2))
        w = self.width()

        if not self.rows:
            p.setPen(QColor("#7a7a7a"))
            p.setFont(ui_font(10))
            p.drawText(self.rect().adjusted(12, 12, -12, -12),
                       Qt.AlignCenter | Qt.TextWordWrap, self.message)
            return

        f_move = ui_font(11.5, bold=True, mono=True)
        f_score = ui_font(12.5, bold=True)
        f_stats = ui_font(8)
        f_pv = ui_font(8, mono=True)
        for i, d in enumerate(self.rows):
            y = 2 + i * self.ROW_H
            row = QRect(0, y, w, self.ROW_H)
            if i == self.hover:
                p.fillRect(row, QColor(BG3))
            elif i % 2:
                p.fillRect(row, QColor("#282828"))

            color = RANK_COLORS[i] if i < len(RANK_COLORS) else GREY
            p.fillRect(QRect(0, y + 2, 4, self.ROW_H - 4), color)

            p.setPen(QColor(FG))
            p.setFont(f_move)
            marks = d.get("pv_marks") or []
            p.drawText(QRect(14, y + 4, 110, 20), Qt.AlignVCenter | Qt.AlignLeft,
                       move_to_uci(d["move"]) + (marks[0] if marks else ""))
            p.setFont(f_score)
            p.setPen(color.lighter(125) if i < len(RANK_COLORS) else QColor(FG_DIM))
            score = "мат" if d.get("mate") else wr_text((d["q"] + 1.0) / 2.0)
            p.drawText(QRect(w - 110, y + 3, 100, 22),
                       Qt.AlignVCenter | Qt.AlignRight, score)

            p.setPen(QColor(FG_DIM))
            p.setFont(f_stats)
            stats = (f"N {d['visits']:,} ({d['frac'] * 100:.0f}%)   "
                     f"P {d['prior'] * 100:.1f}%   Q {d['q']:+.3f}   "
                     f"D {d.get('d', 0.0) * 100:.0f}%")
            p.drawText(QRect(14, y + 23, w - 24, 14),
                       Qt.AlignVCenter | Qt.AlignLeft, stats)

            pv = d.get("pv")
            if pv:
                p.setPen(QColor("#8fa9c2"))
                p.setFont(f_pv)
                pv_rect = QRect(14, y + 37, w - 24, 13)
                pv_str = p.fontMetrics().elidedText(
                    fmt_line(pv, marks), Qt.ElideRight, pv_rect.width())
                p.drawText(pv_rect, Qt.AlignVCenter | Qt.AlignLeft, pv_str)

            win, loss = d.get("w", 0.0), d.get("l", 0.0)
            draw = max(0.0, 1.0 - win - loss)
            bx, bw, by = 14, w - 24, y + self.ROW_H - 7
            ww = bw * win
            dw = bw * draw
            p.setPen(Qt.NoPen)
            p.setBrush(W_COL)
            p.drawRect(QRectF(bx, by, ww, 4))
            p.setBrush(D_COL)
            p.drawRect(QRectF(bx + ww, by, dw, 4))
            p.setBrush(L_COL)
            p.drawRect(QRectF(bx + ww + dw, by, max(0.0, bw - ww - dw), 4))


# ───────────────────── Book Explorer (WDL по ходам) ──────────────────────

class BookBox(QWidget):
    """Дебютная книга: ходы, сыгранные из текущей позиции, с WDL-процентами."""
    play_move = pyqtSignal(int)
    ROW_H = 46
    HEAD_H = 26

    def __init__(self):
        super().__init__()
        self.moves_data = []
        self.total_games = 0
        self.side_to_move = 0
        self.hover = -1
        self.state = "none"       # none | loading | ready
        self.status = ""
        self.ply = 0
        self.max_ply = BOOK_PLY
        self.setMouseTracking(True)
        self.setMinimumWidth(300)

    def set_state(self, state, status=""):
        self.state = state
        self.status = status
        self.update()

    def set_book_data(self, entry, has_archive=True, ply=0, max_ply=BOOK_PLY):
        if has_archive:
            self.state = "ready"
        elif self.state == "ready":
            self.state = "none"
        self.ply = ply
        self.max_ply = max_ply
        self.hover = -1
        if entry is None or not entry.get("moves"):
            self.moves_data = []
            self.total_games = 0
            self.side_to_move = 0
        else:
            self.total_games = entry["total"]
            self.side_to_move = entry["side"]
            m_items = []
            for mv, st in entry["moves"].items():
                n, w, d, l = st["n"], st["w"], st["d"], st["l"]
                m_items.append({
                    "move": mv, "n": n, "w": w, "d": d, "l": l,
                    "w_pct": w / n if n > 0 else 0.0,
                    "d_pct": d / n if n > 0 else 0.0,
                    "l_pct": l / n if n > 0 else 0.0,
                    "score": (w + 0.5 * d) / n if n > 0 else 0.5,
                    "freq": n / self.total_games if self.total_games > 0 else 0.0,
                })
            m_items.sort(key=lambda x: x["n"], reverse=True)
            self.moves_data = m_items

        self.setMinimumHeight(max(1, len(self.moves_data)) * self.ROW_H + self.HEAD_H + 4)
        self.update()

    def _row_at(self, y):
        if y < self.HEAD_H:
            return -1
        i = (y - self.HEAD_H) // self.ROW_H
        return i if 0 <= i < len(self.moves_data) else -1

    def mouseMoveEvent(self, ev):
        h = self._row_at(ev.y())
        if h != self.hover:
            self.hover = h
            self.setCursor(Qt.PointingHandCursor if h >= 0 else Qt.ArrowCursor)
            if h >= 0:
                d = self.moves_data[h]
                self.setToolTip(f"{move_to_uci(d['move'])}: {d['n']:,} партий — "
                                f"+{d['w']:,} ={d['d']:,} −{d['l']:,}")
            else:
                self.setToolTip("")
            self.update()

    def leaveEvent(self, _):
        self.hover = -1
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def mousePressEvent(self, ev):
        i = self._row_at(ev.y())
        if i >= 0 and ev.button() == Qt.LeftButton:
            self.play_move.emit(self.moves_data[i]["move"])

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(BG2))
        w = self.width()

        def centered(text):
            p.setPen(QColor("#8a8a8a"))
            p.setFont(ui_font(10))
            p.drawText(self.rect().adjusted(16, 16, -16, -16),
                       Qt.AlignCenter | Qt.TextWordWrap, text)

        if self.state == "loading":
            centered(self.status or "Архив загружается…")
            return
        if self.state != "ready":
            centered(self.status or "Архив не загружен.\nОткройте его кнопкой «Архив…».")
            return
        if not self.moves_data:
            if self.ply >= self.max_ply:
                centered(f"Книга строится по первым {self.max_ply} полуходам партий.")
            else:
                centered("В архиве нет партий с этой позицией.")
            return

        who = "белые" if self.side_to_move == 0 else "чёрные"
        p.setPen(QColor("#aaaaaa"))
        p.setFont(ui_font(8, bold=True))
        p.drawText(QRect(10, 4, w - 20, 18), Qt.AlignVCenter | Qt.AlignLeft,
                   f"Партий: {self.total_games:,}   ·   ходят {who}")
        p.drawText(QRect(10, 4, w - 20, 18), Qt.AlignVCenter | Qt.AlignRight, "очки")
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawLine(8, self.HEAD_H - 3, w - 8, self.HEAD_H - 3)

        f_move = ui_font(10.5, bold=True, mono=True)
        f_small = ui_font(8)
        f_score = ui_font(9, bold=True)
        f_bar = ui_font(7.5, bold=True)
        for i, d in enumerate(self.moves_data):
            y = self.HEAD_H + i * self.ROW_H
            row = QRect(0, y, w, self.ROW_H)
            if i == self.hover:
                p.fillRect(row, QColor(BG3))
            elif i % 2:
                p.fillRect(row, QColor("#282828"))

            p.setPen(QColor(FG))
            p.setFont(f_move)
            p.drawText(QRect(12, y + 3, 90, 18), Qt.AlignVCenter | Qt.AlignLeft,
                       move_to_uci(d["move"]))
            p.setPen(QColor(FG_DIM))
            p.setFont(f_small)
            p.drawText(QRect(100, y + 3, 150, 18), Qt.AlignVCenter | Qt.AlignLeft,
                       f"{d['n']:,} партий · {d['freq'] * 100:.0f}%")
            p.setPen(QColor(ACCENT).lighter(125))
            p.setFont(f_score)
            p.drawText(QRect(w - 84, y + 3, 72, 18), Qt.AlignVCenter | Qt.AlignRight,
                       f"{d['score'] * 100:.1f}%")

            bx, bw, by, bh = 12, w - 24, y + 23, 16
            ww = bw * d["w_pct"]
            dw = bw * d["d_pct"]
            lw = bw - ww - dw
            p.save()
            clip = QPainterPath()
            clip.addRoundedRect(QRectF(bx, by, bw, bh), 3, 3)
            p.setClipPath(clip)
            p.setPen(Qt.NoPen)
            for x, width, col in ((bx, ww, W_COL), (bx + ww, dw, D_COL),
                                  (bx + ww + dw, lw, L_COL)):
                if width > 0:
                    p.setBrush(col)
                    p.drawRect(QRectF(x, by, width, bh))
            p.restore()
            p.setFont(f_bar)
            p.setPen(QColor("#ffffff"))
            for x, width, pct in ((bx, ww, d["w_pct"]), (bx + ww, dw, d["d_pct"]),
                                  (bx + ww + dw, lw, d["l_pct"])):
                if width >= 30:
                    p.drawText(QRectF(x, by, width, bh), Qt.AlignCenter, f"{pct * 100:.0f}%")


# ─────────────────────── Диалог выбора партий архива ───────────────────────

class ArchiveGamesModel(QAbstractTableModel):
    """Все отобранные партии без ограничения числа строк: ячейки строятся
    только для видимых строк, сортировка — argsort по столбцу."""
    HEADERS = ["№", "Итерация", "Исход", "Полуходов", "Окончание", "Источник"]
    COLS = [None, "iter", "result", "plies", "term", "source"]

    def __init__(self):
        super().__init__()
        self.ar = None
        self.ids = np.zeros(0, dtype=np.int64)
        self.sort_col, self.sort_order = 0, Qt.DescendingOrder

    def set_rows(self, ar, ids):
        self.beginResetModel()
        self.ar = ar
        self.ids = np.asarray(ids, dtype=np.int64)
        self._apply_sort()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.ids)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self.ar is None:
            return None
        gid = int(self.ids[index.row()])
        col = index.column()
        g = self.ar.g
        if role == Qt.DisplayRole:
            if col == 0:
                return str(gid)
            if col == 1:
                it = int(g["iter"][gid])
                return str(it) if 0 <= it < 100_000 else "—"
            if col == 2:
                r = int(g["result"][gid])
                return "1-0" if r > 0 else ("0-1" if r < 0 else "½-½")
            if col == 3:
                pl = int(g["plies"][gid])
                return f"{pl}  (ход {(pl + 1) // 2})"
            if col == 4:
                t = int(g["term"][gid])
                return A.TERM_NAMES.get(t, str(t)) if A else str(t)
            if col == 5:
                s = int(g["source"][gid])
                return (A.SOURCE_NAMES.get(s, "неизвестно") if A else str(s))
        elif role == Qt.ForegroundRole and col == 2:
            r = int(g["result"][gid])
            return QBrush(QColor("#5cc275") if r > 0 else
                          (QColor("#e86a52") if r < 0 else QColor("#a8a8a8")))
        elif role == Qt.TextAlignmentRole and col in (0, 1, 2, 3):
            return int(Qt.AlignCenter)
        return None

    def sort(self, column, order=Qt.AscendingOrder):
        self.layoutAboutToBeChanged.emit()
        self.sort_col, self.sort_order = column, order
        self._apply_sort()
        self.layoutChanged.emit()

    def _apply_sort(self):
        if self.ar is None or not len(self.ids):
            return
        name = self.COLS[self.sort_col]
        key = self.ids if name is None else self.ar.g[name][self.ids]
        order = np.argsort(key, kind="stable")
        if self.sort_order == Qt.DescendingOrder:
            order = order[::-1]
        self.ids = self.ids[order]


class ArchiveDialog(QDialog):
    """Окно просмотра и фильтрации партий из архива."""
    load_game = pyqtSignal(int)

    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.main_win = main_win
        self.setWindowTitle("Архив партий")
        self.resize(940, 640)

        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        top_row = QHBoxLayout()
        btn_browse = QPushButton("Выбрать каталог архива…")
        btn_browse.clicked.connect(self.choose_directory)
        top_row.addWidget(btn_browse)
        self.lbl_path = QLabel("Архив не выбран")
        self.lbl_path.setStyleSheet(f"color: {FG_DIM};")
        self.lbl_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top_row.addWidget(self.lbl_path, 1)
        lay.addLayout(top_row)

        filter_box = QGroupBox("Фильтры")
        fl = QHBoxLayout(filter_box)
        fl.setSpacing(6)

        fl.addWidget(QLabel("Исход"))
        self.cb_result = QComboBox()
        self.cb_result.addItems(["все", "1-0", "½-½", "0-1"])
        fl.addWidget(self.cb_result)

        fl.addWidget(QLabel("Окончание"))
        self.cb_term = QComboBox()
        self.cb_term.addItem("все")
        if A:
            for t_id in sorted(A.TERM_NAMES):
                self.cb_term.addItem(A.TERM_NAMES[t_id], t_id)
        fl.addWidget(self.cb_term)

        fl.addWidget(QLabel("Источник"))
        self.cb_source = QComboBox()
        book_src = getattr(A, "SOURCE_BOOK", 4) if A else 4
        self.cb_source.addItem("все партии", ("not", book_src))
        self.cb_source.addItem("всё, с перебором дебюта", None)
        if A:
            for s_id in sorted(A.SOURCE_NAMES):
                self.cb_source.addItem(A.SOURCE_NAMES[s_id], ("eq", s_id))
        fl.addWidget(self.cb_source)

        fl.addWidget(QLabel("Полуходов"))
        self.spin_min_plies = QSpinBox()
        self.spin_min_plies.setRange(0, 2000)
        fl.addWidget(self.spin_min_plies)
        fl.addWidget(QLabel("–"))
        self.spin_max_plies = QSpinBox()
        self.spin_max_plies.setRange(0, 2000)
        self.spin_max_plies.setSpecialValueText("∞")
        fl.addWidget(self.spin_max_plies)
        fl.addStretch()
        for wdg in (self.cb_result, self.cb_term, self.cb_source):
            wdg.currentIndexChanged.connect(self.apply_filter)
        for wdg in (self.spin_min_plies, self.spin_max_plies):
            wdg.valueChanged.connect(self.apply_filter)
        lay.addWidget(filter_box)

        self.model = ArchiveGamesModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSortIndicator(0, Qt.DescendingOrder)
        hh = self.table.horizontalHeader()
        for c, wdt in enumerate([90, 80, 60, 120, 0, 150]):
            if wdt:
                hh.resizeSection(c, wdt)
        hh.setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.doubleClicked.connect(lambda idx: self._emit_row(idx.row()))
        lay.addWidget(self.table, 1)
        QShortcut(QKeySequence(Qt.Key_Return), self.table, self._load_selected)
        QShortcut(QKeySequence(Qt.Key_Enter), self.table, self._load_selected)

        bottom = QHBoxLayout()
        self.lbl_count = QLabel("Партий: 0")
        self.lbl_count.setStyleSheet(f"color: {FG_DIM};")
        bottom.addWidget(self.lbl_count)
        bottom.addStretch()
        btn_load = QPushButton("Открыть на доске")
        btn_load.setDefault(True)
        btn_load.clicked.connect(self._load_selected)
        bottom.addWidget(btn_load)
        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.reject)
        bottom.addWidget(btn_close)
        lay.addLayout(bottom)

    def choose_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Выбрать каталог архива")
        if d:
            self.main_win.load_archive_dir(d)

    def update_archive_view(self, index, status=None):
        if index is None:
            self.lbl_path.setText(status or "Архив не загружен")
            self.model.set_rows(None, [])
            self.lbl_count.setText("")
            return
        self.lbl_path.setText(os.path.abspath(index.ar.dir))
        self.apply_filter()

    def apply_filter(self):
        index = self.main_win.archive
        if index is None:
            self.model.set_rows(None, [])
            self.lbl_count.setText("")
            return
        g = index.ar.g
        mask = index.game_ok.copy()
        res_filter = {1: 1, 2: 0, 3: -1}.get(self.cb_result.currentIndex())
        if res_filter is not None:
            mask &= (g["result"] == res_filter)
        term_data = self.cb_term.currentData()
        if term_data is not None:
            mask &= (g["term"] == term_data)
        src = self.cb_source.currentData()
        if src is not None:
            kind, val = src
            mask &= (g["source"] != val) if kind == "not" else (g["source"] == val)
        min_p, max_p = self.spin_min_plies.value(), self.spin_max_plies.value()
        if min_p > 0:
            mask &= (g["plies"] >= min_p)
        if max_p > 0:
            mask &= (g["plies"] <= max_p)
        ids = np.flatnonzero(mask)
        self.model.set_rows(index.ar, ids)
        self.lbl_count.setText(f"Отобрано {len(ids):,} из {len(mask):,} записей · "
                               f"двойной щелчок или Enter — открыть")

    def _emit_row(self, row):
        if 0 <= row < len(self.model.ids):
            self.load_game.emit(int(self.model.ids[row]))
            self.accept()

    def _load_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self._emit_row(rows[0].row())


# ───────────────────────────── Promotion dialog ───────────────────────────

# ───────────────────────────── Board widget ───────────────────────────────

class BoardWidget(QWidget):
    wheel_step = pyqtSignal(int)

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setMinimumSize(420, 336)
        self.setFocusPolicy(Qt.ClickFocus)
        self.pieces = PieceSet()
        self.engine = CapablancaEngine()
        self.legal_moves = self.engine.get_legal_moves_int()
        self.selected = None
        self.flipped = False
        self.last_move = None
        self.analysis = []
        self.preview = []
        self.drag_from = None
        self.drag_pos = None
        self._press_pos = None
        # Выбор превращения прямо на доске, как в Nibbler: столбец фигур на
        # вертикали хода от края доски внутрь. [(поле, ход, тип фигуры), ...]
        self.promo = None
        self.promo_hover = None
        self.setMouseTracking(True)

    def set_position(self, history, last_move, engine=None):
        if engine is None:
            engine = CapablancaEngine()
            for m in history:
                engine.make_move_int(m)
        self.engine = engine
        self.legal_moves = engine.get_legal_moves_int()
        self.last_move = last_move
        self.selected = None
        self.drag_from = None
        self.preview = []
        self.promo = None
        self.update()

    def cancel_promotion(self):
        if self.promo is not None:
            self.promo = None
            self.promo_hover = None
            self.update()
            return True
        return False

    def set_preview(self, pv):
        self.preview = list(pv or [])
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
        cell, ox, oy = self._metrics()

        coord_font = ui_font(max(6.0, cell * 0.12), bold=True)
        coord_font.setPixelSize(max(9, int(cell * 0.15)))
        p.setFont(coord_font)
        for r in range(8):
            for f in range(10):
                sq = r * 10 + f
                rect = self.sq_rect(sq)
                light = (r + f) % 2 == 1
                base = LIGHT_SQ if light else DARK_SQ
                p.fillRect(rect, base)
                if self.last_move and sq in self.last_move:
                    p.fillRect(rect, HL_LAST)
                if self.selected == sq:
                    p.fillRect(rect, HL_SEL)
                p.setPen(DARK_SQ if light else LIGHT_SQ)
                edge_rank = (r == 0) if not self.flipped else (r == 7)
                edge_file = (f == 0) if not self.flipped else (f == 9)
                if edge_rank:
                    p.drawText(rect.adjusted(0, 0, -cell * 0.05, -cell * 0.02),
                               Qt.AlignBottom | Qt.AlignRight, chr(ord('a') + f))
                if edge_file:
                    p.drawText(rect.adjusted(cell * 0.05, cell * 0.02, 0, 0),
                               Qt.AlignTop | Qt.AlignLeft, str(r + 1))

        for color, ptype, sq in self.engine.get_pieces():
            sq = int(sq)
            if self.drag_from == sq and self.drag_pos is not None:
                self.pieces.draw(p, self.sq_rect(sq).center(), cell, color, ptype, 0.35)
                continue
            self.pieces.draw(p, self.sq_rect(sq).center(), cell, color, ptype)

        if self.selected is not None:
            self._draw_targets(p, cell)

        if self.preview:
            self._draw_preview(p, cell)
        else:
            if self.selected is not None:
                shown = [(g, d) for g, d in enumerate(self.analysis)
                         if decode_move(d["move"])[0] == self.selected]
            else:
                shown = list(enumerate(self.analysis[:5]))
            for g, d in reversed(shown):
                color = RANK_COLORS[g] if g < len(RANK_COLORS) else GREY
                f, t, _ = decode_move(d["move"])
                self._draw_arrow(p, f, t, color, cell, rank=g)
            label_targets = set()
            for g, d in shown:
                _, t, _ = decode_move(d["move"])
                if t in label_targets or g > 2:
                    continue
                label_targets.add(t)
                color = RANK_COLORS[g] if g < len(RANK_COLORS) else GREY
                label = "мат" if d.get("mate") else f"{(d['q'] + 1.0) * 50:.0f}%"
                self._draw_badge(p, self.sq_rect(t).center(), label, color, cell,
                                 big=(g == 0))

        if self.drag_from is not None and self.drag_pos is not None:
            for color, ptype, sq in self.engine.get_pieces():
                if int(sq) == self.drag_from:
                    self.pieces.draw(p, QPointF(self.drag_pos), cell * 1.05, color, ptype)

        if self.promo:
            p.fillRect(QRectF(ox, oy, cell * 10, cell * 8), QColor(0, 0, 0, 150))
            color = self.engine.side_to_move()
            for sq, _, ptype in self.promo:
                rect = self.sq_rect(sq)
                hot = sq == self.promo_hover
                p.setPen(Qt.NoPen)
                p.setBrush(QColor("#f0f0f0") if hot else QColor("#b8b8b8"))
                p.drawEllipse(rect.center(), cell * 0.47, cell * 0.47)
                self.pieces.draw(p, rect.center(), cell * (0.92 if hot else 0.82),
                                 color, ptype)

    def _draw_targets(self, p, cell):
        occupied = {int(sq) for _, _, sq in self.engine.get_pieces()}
        best = next((d for d in self.analysis
                     if decode_move(d["move"])[0] == self.selected), None)
        best_dest = decode_move(best["move"])[1] if best else None
        seen = set()
        for m in self.legal_moves:
            fs, ts, _ = decode_move(m)
            if fs != self.selected or ts in seen:
                continue
            seen.add(ts)
            rect = self.sq_rect(ts)
            c = rect.center()
            col = QColor(58, 166, 85, 190) if ts == best_dest else QColor(0, 0, 0, 60)
            if ts in occupied:
                p.setPen(QPen(col, cell * 0.07))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(c, cell * 0.44, cell * 0.44)
            else:
                p.setPen(Qt.NoPen)
                p.setBrush(col)
                r = cell * (0.16 if ts == best_dest else 0.13)
                p.drawEllipse(c, r, r)

    def _draw_arrow(self, p, f, t, color, cell, rank, alpha=None, width_f=None):
        s = self.sq_rect(f).center()
        e = self.sq_rect(t).center()
        ang = math.atan2(e.y() - s.y(), e.x() - s.x())
        a = alpha if alpha is not None else max(0.40, 0.92 - rank * 0.14)
        wf = width_f if width_f is not None else max(0.45, 1.0 - rank * 0.14)
        c = QColor(color)
        c.setAlphaF(a)
        width = max(3.0, cell * 0.15 * wf)
        head = max(9.0, cell * 0.36 * wf)
        tip = QPointF(e.x() - math.cos(ang) * cell * 0.06, e.y() - math.sin(ang) * cell * 0.06)
        base = QPointF(tip.x() - math.cos(ang) * head, tip.y() - math.sin(ang) * head)
        nx, ny = -math.sin(ang), math.cos(ang)
        # Стрелка одним контуром: у полупрозрачных линии и наконечника
        # иначе видно тёмное наложение.
        hw, hh = width / 2, head * 0.62
        start = QPointF(s.x() + math.cos(ang) * cell * 0.12,
                        s.y() + math.sin(ang) * cell * 0.12)
        poly = QPolygonF([
            QPointF(start.x() + nx * hw, start.y() + ny * hw),
            QPointF(base.x() + nx * hw, base.y() + ny * hw),
            QPointF(base.x() + nx * hh, base.y() + ny * hh),
            tip,
            QPointF(base.x() - nx * hh, base.y() - ny * hh),
            QPointF(base.x() - nx * hw, base.y() - ny * hw),
            QPointF(start.x() - nx * hw, start.y() - ny * hw),
        ])
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawPolygon(poly)

    def _draw_badge(self, p, center, label, color, cell, big=False):
        font = ui_font(8, bold=True)
        font.setPixelSize(max(10, int(cell * (0.19 if big else 0.16))))
        p.setFont(font)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(label) + cell * 0.12
        th = fm.height() + 2
        box = QRectF(center.x() - tw / 2, center.y() - th / 2, tw, th)
        fill = QColor(color.darker(165))
        fill.setAlpha(235)
        p.setBrush(fill)
        p.setPen(QPen(color.lighter(135), 1.2))
        p.drawRoundedRect(box, th / 2.6, th / 2.6)
        p.setPen(QColor("#ffffff"))
        p.drawText(box, Qt.AlignCenter, label)

    def _draw_preview(self, p, cell):
        """Главная линия выбранного хода: стрелки с номерами полуходов."""
        stm = self.engine.side_to_move()
        pv = self.preview[:8]
        for k in range(len(pv) - 1, -1, -1):
            f, t, _ = decode_move(pv[k])
            mover = stm if k % 2 == 0 else 1 - stm
            col = QColor("#f2f2f2") if mover == 0 else QColor("#1e1e1e")
            if k == 0:
                col = RANK_COLORS[0]
            self._draw_arrow(p, f, t, col, cell, rank=0,
                             alpha=max(0.45, 0.9 - k * 0.07),
                             width_f=max(0.5, 0.9 - k * 0.06))
        for k, m in enumerate(pv):
            f, t, _ = decode_move(m)
            s, e = self.sq_rect(f).center(), self.sq_rect(t).center()
            mid = QPointF(s.x() + (e.x() - s.x()) * 0.55, s.y() + (e.y() - s.y()) * 0.55)
            mover = stm if k % 2 == 0 else 1 - stm
            bg = QColor("#f2f2f2") if mover == 0 else QColor("#1e1e1e")
            fg = QColor("#1e1e1e") if mover == 0 else QColor("#f2f2f2")
            r = max(8.0, cell * 0.14)
            p.setPen(QPen(QColor("#777777"), 1))
            p.setBrush(bg)
            p.drawEllipse(mid, r, r)
            font = ui_font(8, bold=True)
            font.setPixelSize(max(9, int(r * 1.2)))
            p.setFont(font)
            p.setPen(fg)
            p.drawText(QRectF(mid.x() - r, mid.y() - r, 2 * r, 2 * r), Qt.AlignCenter,
                       str(k + 1))

    # ---- мышь: щелчок-щелчок и перетаскивание ----
    def _movable(self, sq):
        return any(decode_move(m)[0] == sq for m in self.legal_moves)

    def _try_move(self, frm, to):
        matching = [m for m in self.legal_moves
                    if decode_move(m)[0] == frm and decode_move(m)[1] == to]
        if not matching:
            return False
        move = matching[0]
        if len(matching) > 1:
            by_piece = {decode_move(m)[2]: m for m in matching}
            r, f = divmod(to, 10)
            step = -1 if r == 7 else 1
            self.promo = []
            for ch, _ in _PROMO_LABELS:
                if ch in by_piece and 0 <= r < 8:
                    self.promo.append((r * 10 + f, by_piece[ch], _PROMO_TYPE[ch]))
                    r += step
            self.selected = None
            self.update()
            return True
        self.selected = None
        self.main.try_human_move(move)
        return True

    def mousePressEvent(self, ev):
        if self.promo is not None:
            sq = self.sq_at(ev.pos())
            move = next((m for s_, m, _ in self.promo if s_ == sq), None)
            self.cancel_promotion()
            if move is not None and ev.button() == Qt.LeftButton:
                self.main.try_human_move(move)
            return
        if ev.button() == Qt.RightButton:
            self.selected = None
            self.update()
            return
        if ev.button() != Qt.LeftButton:
            return
        sq = self.sq_at(ev.pos())
        if sq is None:
            return
        if self.selected is not None and sq != self.selected:
            if self._try_move(self.selected, sq):
                return
        if self._movable(sq):
            self.selected = sq
            self.drag_from = sq
            self._press_pos = ev.pos()
        else:
            self.selected = None
        self.update()

    def mouseMoveEvent(self, ev):
        if self.promo is not None:
            sq = self.sq_at(ev.pos())
            hot = sq if any(s_ == sq for s_, _, _ in self.promo) else None
            if hot != self.promo_hover:
                self.promo_hover = hot
                self.update()
            return
        if self.drag_from is None or not (ev.buttons() & Qt.LeftButton):
            return
        if self.drag_pos is None and self._press_pos is not None and \
                (ev.pos() - self._press_pos).manhattanLength() < 5:
            return
        self.drag_pos = ev.pos()
        self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        frm, dragged = self.drag_from, self.drag_pos is not None
        self.drag_from = None
        self.drag_pos = None
        if frm is not None and dragged:
            sq = self.sq_at(ev.pos())
            if sq is not None and sq != frm and self._try_move(frm, sq):
                return
        self.update()

    def wheelEvent(self, ev):
        dy = ev.angleDelta().y()
        if dy:
            self.wheel_step.emit(-1 if dy > 0 else 1)


# ───────────────────────────── Main window ────────────────────────────────

MODES = ["analyze", "play_white", "play_black", "selfplay"]
MODE_LABELS = ["Анализ", "Играть белыми", "Играть чёрными", "Самоигра"]


class NibblerGUI(QMainWindow):
    def __init__(self, net=None, game=None, archive_dir=None, find_archive=True,
                 device="auto"):
        super().__init__()
        self.device = device
        self.setWindowTitle("Capablanca AI — анализатор")
        self.resize(1360, 900)

        self.mcts = None
        self.net_path = None
        self.search = None
        self.search_started = 0.0
        self._retired = []

        # Архив и дебютная книга
        self.archive = None
        self.archive_dir = ""
        self.archive_loader = None
        self.archive_dialog = None

        self.history = []
        self.history_loaded = False
        self.cursor = 0
        self.evals = {}
        self.snapshots = {}
        self.ttable = {}
        self.mode = "analyze"
        self.analysis_on = False
        self.hints_on = True
        self.contempt = 0.0
        self.logger = None
        self._moves_sig = None

        self._build_ui()
        self._build_shortcuts()
        if net:
            self.open_network(net)
        else:
            self._autoload()
        if archive_dir:
            self.load_archive_dir(archive_dir)
        elif find_archive:
            self._auto_find_archive()
        if game:
            self.open_game_file(game)
        self.refresh()
        # Иначе фокус получает первое поле ввода и забирает стрелки у навигации.
        self.board.setFocus()

    # ---- UI construction ----
    def _build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 4)
        outer.setSpacing(8)
        outer.addWidget(self._build_toolbar())

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)
        board_row = QHBoxLayout()
        board_row.setSpacing(6)
        self.eval_bar = EvalBar()
        board_row.addWidget(self.eval_bar)
        self.board = BoardWidget(self)
        self.board.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.board.wheel_step.connect(lambda d: self.seek_ply(self.cursor + d))
        board_row.addWidget(self.board, 1)
        ll.addLayout(board_row, 1)
        self.graph = WinrateGraph()
        self.graph.setToolTip("Ожидаемый счёт белых по ходам партии. Щелчок — перейти к позиции.")
        self.graph.seek.connect(self.seek_ply)
        self.graph.setFixedHeight(110)
        ll.addWidget(self.graph)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(left)
        split.addWidget(self._build_right_panel())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([960, 380])
        outer.addWidget(split, 1)

        self.setCentralWidget(central)
        self.lbl_net = QLabel("сеть не загружена")
        self.lbl_net.setStyleSheet(f"color: {FG_DIM}; padding: 0 6px;")
        self.statusBar().addPermanentWidget(self.lbl_net)
        self.statusBar().showMessage("Загрузите сеть (.onnx или .pth), чтобы включить анализ.")

    def _button(self, text, slot, tip=None):
        b = QPushButton(text)
        b.setFocusPolicy(Qt.NoFocus)
        b.clicked.connect(slot)
        if tip:
            b.setToolTip(tip)
        return b

    def _build_toolbar(self):
        bar = QFrame()
        bar.setObjectName("toolbar")
        col = QVBoxLayout(bar)
        col.setContentsMargins(8, 6, 8, 6)
        col.setSpacing(6)
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        col.addLayout(row1)
        col.addLayout(row2)

        row1.addWidget(self._button("Сеть…", self.load_weights,
                                    "Загрузить сеть: .onnx или чекпоинт .pth (Ctrl+L)"))
        row1.addWidget(self._button("Открыть партию…", self.load_game,
                                    "PGN или текст с ходами вида e2e4 (Ctrl+O)"))
        row1.addWidget(self._button("Архив…", self.open_archive_dialog,
                                    "Партии из архива самоигры (Ctrl+B)"))
        row1.addWidget(self._vline())

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODE_LABELS)
        self.mode_combo.setFocusPolicy(Qt.NoFocus)
        self.mode_combo.setToolTip("Анализ — свободный разбор; в режимах игры сеть "
                                   "отвечает сама, «Узлов на ход» задаёт её силу")
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        row1.addWidget(self.mode_combo)

        self.btn_go = self._button("▶ Анализ", self._toggle_analysis,
                                   "Запустить или остановить анализ (Пробел)")
        self.btn_go.setObjectName("primary")
        self.btn_go.setCheckable(True)
        self.btn_go.setMinimumWidth(110)
        row1.addWidget(self.btn_go)
        row1.addWidget(self._button("Новая", self.new_game, "Новая партия (Ctrl+N)"))
        row1.addWidget(self._button("⇅ Перевернуть", self.flip_board, "Перевернуть доску (F)"))
        row1.addStretch()
        self.lbl_title = QLabel("")
        self.lbl_title.setStyleSheet(f"color: {FG_DIM};")
        row1.addWidget(self.lbl_title)

        def spin(label, tip, lo, hi, step, val, suffix="", special=None):
            lab = QLabel(label)
            lab.setToolTip(tip)
            row2.addWidget(lab)
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setValue(val)
            s.setToolTip(tip)
            s.setGroupSeparatorShown(hi > 100_000)
            if suffix:
                s.setSuffix(suffix)
            if special:
                s.setSpecialValueText(special)
            s.setKeyboardTracking(False)
            s.editingFinished.connect(lambda: self.board.setFocus())
            row2.addWidget(s)
            return s

        self.spin_analyze = spin("Лимит анализа", "Узлов на позицию в режиме анализа; ∞ — "
                                 "пока не остановите", 0, 50_000_000, 2000, 0, special="∞")
        self.spin_play = spin("Узлов на ход", "Сколько узлов сеть думает над своим ходом "
                              "в режимах игры", 50, 10_000_000, 200, 1200)
        row2.addWidget(self._vline())
        self.spin_cpuct = spin("c_puct", "Коэффициент исследования поиска, в тысячных",
                               500, 5000, 100, 1745, suffix="‰")
        self.spin_cpuct.valueChanged.connect(self._cpuct_changed)
        self.spin_contempt = spin("Contempt", "Штраф за ничьи в поиске: плюс — "
                                  "ничья хуже для ходящего", -100, 100, 5, 0, suffix="%")
        self.spin_contempt.valueChanged.connect(self._contempt_changed)
        row2.addWidget(self._vline())
        row2.addWidget(QLabel("Считать на"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["авто", "видеокарте", "процессоре"])
        self.device_combo.setCurrentIndex(DEVICES.index(self.device))
        self.device_combo.setToolTip("Где считать сеть. Авто — видеокарта, если есть CUDA")
        self.device_combo.setFocusPolicy(Qt.NoFocus)
        self.device_combo.currentIndexChanged.connect(self._device_changed)
        row2.addWidget(self.device_combo)
        row2.addWidget(self._vline())
        self.chk_hints = QCheckBox("Подсказки на моём ходу")
        self.chk_hints.setFocusPolicy(Qt.NoFocus)
        self.chk_hints.setToolTip("В режиме игры показывать анализ, когда ходите вы")
        self.chk_hints.setChecked(self.hints_on)
        self.chk_hints.toggled.connect(self._toggle_hints)
        row2.addWidget(self.chk_hints)
        row2.addStretch()
        return bar

    def _vline(self):
        ln = QFrame()
        ln.setFrameShape(QFrame.VLine)
        ln.setObjectName("vline")
        ln.setFixedWidth(9)
        return ln

    def _build_right_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(330)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.right_tabs = QTabWidget()
        self.right_tabs.setFocusPolicy(Qt.NoFocus)

        def scrolled(widget):
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setFrameShape(QFrame.NoFrame)
            sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sc.setWidget(widget)
            return sc

        self.infobox = InfoBox()
        self.infobox.play_move.connect(self.try_human_move)
        self.infobox.hover_pv.connect(lambda pv: self.board.set_preview(pv))
        self.right_tabs.addTab(scrolled(self.infobox), "Анализ")
        self.book_box = BookBox()
        self.book_box.play_move.connect(self.try_human_move)
        self.right_tabs.addTab(scrolled(self.book_box), "Книга")

        vsplit = QSplitter(Qt.Vertical)
        vsplit.setChildrenCollapsible(False)
        vsplit.addWidget(self.right_tabs)

        moves_box = QWidget()
        ml = QVBoxLayout(moves_box)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(4)
        head = QLabel("Ходы партии")
        head.setObjectName("sectionTitle")
        ml.addWidget(head)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "Ход", "Белые", "Лучший", "Узлы"])
        self.table.horizontalHeaderItem(2).setToolTip("Ожидаемый счёт белых после хода")
        self.table.horizontalHeaderItem(3).setToolTip(
            "Лучший ход по анализу позиции перед ходом; оранжевым — сыграно другое")
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setShowGrid(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.cellClicked.connect(self._table_clicked)
        ml.addWidget(self.table, 1)

        nav = QHBoxLayout()
        nav.setSpacing(4)
        for txt, tip, fn in [("⏮", "В начало (Home)", lambda: self.seek_ply(0)),
                             ("◀", "Назад (←, колесо вверх)", lambda: self.seek_ply(self.cursor - 1)),
                             ("▶", "Вперёд (→, колесо вниз)", lambda: self.seek_ply(self.cursor + 1)),
                             ("⏭", "В конец (End)", lambda: self.seek_ply(len(self.history)))]:
            nav.addWidget(self._button(txt, fn, tip))
        ml.addLayout(nav)
        vsplit.addWidget(moves_box)
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 2)
        vsplit.setSizes([480, 320])
        lay.addWidget(vsplit, 1)
        return panel

    def _build_shortcuts(self):
        keys = [
            (Qt.Key_Left, lambda: self.seek_ply(self.cursor - 1)),
            (Qt.Key_Right, lambda: self.seek_ply(self.cursor + 1)),
            (Qt.Key_Up, lambda: self.seek_ply(self.cursor - 2)),
            (Qt.Key_Down, lambda: self.seek_ply(self.cursor + 2)),
            (Qt.Key_Home, lambda: self.seek_ply(0)),
            (Qt.Key_End, lambda: self.seek_ply(len(self.history))),
            (Qt.Key_F, self.flip_board),
            (Qt.Key_Space, self._toggle_analysis),
            (Qt.Key_Escape, self._escape),
            ("Ctrl+N", self.new_game),
            ("Ctrl+O", self.load_game),
            ("Ctrl+L", self.load_weights),
            ("Ctrl+B", self.open_archive_dialog),
        ]
        for key, fn in keys:
            QShortcut(QKeySequence(key), self, fn)

    def _escape(self):
        if self.board.cancel_promotion():
            return
        self.board.selected = None
        self.board.set_preview([])

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
        self.archive_dialog.update_archive_view(
            self.archive, "Архив загружается…" if self._archive_loading() else None)
        self.archive_dialog.show()
        self.archive_dialog.raise_()

    def _archive_loading(self):
        return self.archive_loader is not None and self.archive_loader.isRunning()

    def load_archive_dir(self, directory):
        if self._archive_loading():
            self.archive_loader.cancelled = True
            self.archive_loader.wait()
        # Тот же каталог используется и для записи сыгранных здесь партий.
        self.archive_dir = directory
        self.book_box.set_state("loading", "Архив загружается…")
        self.statusBar().showMessage(f"Загрузка архива: {directory}…")
        loader = ArchiveLoaderThread(directory)
        self.archive_loader = loader
        loader.progress.connect(self._archive_progress)
        loader.loaded.connect(self._on_archive_loaded)
        loader.error.connect(self._on_archive_error)
        loader.start()

    def _archive_progress(self, msg):
        self.book_box.set_state("loading", msg)
        self.statusBar().showMessage(msg)

    def _on_archive_error(self, err):
        self.book_box.set_state("none", f"Архив не загрузился:\n{err}")
        self.statusBar().showMessage(f"Архив не загрузился: {err}")
        if self.archive_dialog is not None and self.archive_dialog.isVisible():
            self.archive_dialog.update_archive_view(None, f"Ошибка: {err}")
            QMessageBox.warning(self, "Ошибка архива", err)

    def _on_archive_loaded(self, index):
        self.archive = index
        if self.archive_dialog is not None and self.archive_dialog.isVisible():
            self.archive_dialog.update_archive_view(index)
        n_real = int((index.game_ok & (index.ar.g["source"] != getattr(A, "SOURCE_BOOK", 4))).sum())
        self.statusBar().showMessage(
            f"Архив: {n_real:,} партий, "
            f"{index.n_book_positions:,} позиций в книге "
            f"(первые {index.max_book_ply} полуходов) · {index.load_seconds:.0f} с")
        self._update_book()

    def load_archive_game(self, game_id):
        if self.archive is None:
            return
        _, moves = replay_legal(self.archive.game_moves(game_id))
        if not moves:
            QMessageBox.information(self, "Пустая партия",
                                    "У этой записи нет ходов, которые можно проиграть.")
            return
        self.set_game(moves)
        g = self.archive.ar.g
        r = int(g["result"][game_id])
        r_str = "1-0" if r > 0 else ("0-1" if r < 0 else "½-½")
        term = A.TERM_NAMES.get(int(g["term"][game_id]), "?") if A else "?"
        full = int(g["plies"][game_id])
        cut = "" if len(moves) >= full else f" (восстановлено {len(moves)} из {full})"
        self.lbl_title.setText(f"Архив #{game_id} · {r_str} · {term}")
        self.statusBar().showMessage(
            f"Партия #{game_id} из архива: {r_str}, {term}, {len(moves)} полуходов{cut}. "
            f"← → или колесо мыши — листать.")

    # ---- game loading ----
    def set_game(self, moves, loaded=True):
        self.stop_search()
        self.history = list(moves)
        self.history_loaded = loaded
        self.cursor = 0
        self.evals.clear()
        self.snapshots.clear()
        self.logger = None
        self.refresh()

    def load_game(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть партию", "",
            "Партия (*.pgn *.txt *.uci);;Все файлы (*)")
        if path:
            self.open_game_file(path)

    def open_game_file(self, path):
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except OSError as e:
            QMessageBox.warning(self, "Не открылось", str(e))
            return
        # Теги и комментарии PGN могут содержать что угодно похожее на ход.
        body = re.sub(r"\[[^\]]*\]|\{[^}]*\}|;[^\n]*", " ", text)
        tokens = re.findall(r"\b([a-j][1-8][a-j][1-8][qrbnacQRBNAC]?)\b", body)
        if not tokens:
            QMessageBox.warning(self, "Пусто", "В файле не нашлось ходов вида e2e4.")
            return

        eng = CapablancaEngine()
        moves = []
        for i, uci in enumerate(tokens):
            uci = uci.lower()
            found = next((m for m in eng.get_legal_moves_int()
                          if move_to_uci(m) == uci), None)
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
        self.set_game(moves)
        self.lbl_title.setText(os.path.basename(path))
        self.statusBar().showMessage(
            f"Загружена партия: {len(moves)} полуходов. ← → или колесо мыши — листать.")

    # ---- model loading ----
    def load_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить сеть", "",
            "Сеть (*.onnx *.pth);;ONNX-модель (*.onnx);;Чекпоинт PyTorch (*.pth)")
        if path:
            self.open_network(path)

    def open_network(self, path):
        if not os.path.exists(path):
            QMessageBox.warning(self, "Нет файла", f"Файл сети не найден:\n{path}")
            return
        if path.lower().endswith(".pth"):
            path = self._onnx_from_checkpoint(path)
            if not path:
                return
        self._load_onnx(path)

    @staticmethod
    def _onnx_target(pth):
        dst = os.path.splitext(pth)[0] + ".onnx"
        if os.access(os.path.dirname(os.path.abspath(pth)), os.W_OK):
            return dst
        # Каталог чекпоинта только для чтения — кладём копию в кэш пользователя.
        cache = os.path.join(os.path.expanduser("~"), ".cache", "capablanca_gui")
        os.makedirs(cache, exist_ok=True)
        return os.path.join(cache, os.path.basename(dst))

    def _onnx_from_checkpoint(self, pth):
        dst = self._onnx_target(pth)
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
        self.statusBar().showMessage(f"Конвертация {os.path.basename(pth)} в ONNX…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
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
            self._load_onnx(cand, quiet=True)

    def _load_onnx(self, path, quiet=False):
        self.stop_search()
        self._join_retired()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Кэш оценок по полному входу почти не попадает: во входе история
            # из 8 полуходов, а повторы позиций и так общие через ttable. При
            # этом каждая запись весила ~72 КБ — до 4.6 ГБ на 64 тыс. записей.
            eng = OnnxEngine(path, c_puct=self.spin_cpuct.value() / 1000.0,
                             batch_size=BATCH_GPU, nn_cache=False,
                             device=self.device)
            eng.batch_size = BATCH_GPU if eng.gpu else BATCH_CPU
        except Exception as e:
            traceback.print_exc()
            msg = f"{type(e).__name__}: {e}"
            self.statusBar().showMessage(f"Сеть не загрузилась: {msg}")
            if not quiet:
                QMessageBox.critical(self, "Сеть не загрузилась", msg)
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.mcts = eng
        self.net_path = path
        self.ttable.clear()
        self.snapshots.clear()
        self.evals.clear()
        dev = "видеокарта (CUDA)" if eng.gpu else "процессор"
        self.lbl_net.setText(f"{os.path.basename(path)} · {dev}")
        self.setWindowTitle(f"Capablanca AI — {os.path.basename(path)}")
        self.statusBar().showMessage(
            f"Сеть загружена: {os.path.basename(path)} · {dev}. "
            f"Пробел — запустить анализ.")
        self.refresh()

    # ---- controls ----
    def _mode_changed(self, idx):
        self.mode = MODES[idx]
        self.refresh()

    def _device_changed(self, idx):
        self.device = DEVICES[idx]
        if self.net_path:
            self._load_onnx(self.net_path)

    def _toggle_hints(self, on):
        self.hints_on = bool(on)
        self.refresh()

    def hints_hidden(self):
        return (not self.hints_on
                and self.mode in ("play_white", "play_black")
                and self.cursor == len(self.history)
                and not self.engine_should_move())

    def _toggle_analysis(self):
        if self.mcts is None and not self.analysis_on:
            self.statusBar().showMessage("Сначала загрузите сеть: кнопка «Сеть…» (Ctrl+L).")
            self.btn_go.setChecked(False)
            return
        self.analysis_on = not self.analysis_on
        self.btn_go.setChecked(self.analysis_on)
        self.btn_go.setText("⏸ Стоп" if self.analysis_on else "▶ Анализ")
        self.refresh()

    def _contempt_changed(self, val):
        self.contempt = val / 100.0
        if self.search is not None:
            self.refresh()

    def _cpuct_changed(self, val):
        if self.mcts is not None:
            self.mcts.c_puct = val / 1000.0
            if self.search is not None:
                self.refresh()

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
        self.ttable.clear()
        self.ttable.update(keep)

    def new_game(self):
        self.stop_search()
        self._join_retired()
        self.history = []
        self.history_loaded = False
        self.cursor = 0
        self.evals = {}
        self.snapshots = {}
        self.ttable.clear()
        self.logger = None
        self.lbl_title.setText("")
        if self.mcts is not None:
            self.mcts.clear_nn_cache()
        self.refresh()

    def flip_board(self):
        self.board.flipped = not self.board.flipped
        self.eval_bar.flipped = self.board.flipped
        self.board.update()
        self.eval_bar.update()

    # ---- navigation ----
    def seek_ply(self, ply):
        ply = max(0, min(len(self.history), ply))
        if ply == self.cursor:
            return
        self.cursor = ply
        self.refresh()

    def _table_clicked(self, row, _col):
        self.seek_ply(row + 1)

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
            self.statusBar().showMessage("В режиме самоигры ходит сеть. Смените режим, "
                                         "чтобы ходить самому.")
            return
        if self.cursor == len(self.history) and self.engine_should_move():
            self.statusBar().showMessage("Сейчас ход сети.")
            return
        if m not in self.board.legal_moves:
            return
        if self.cursor < len(self.history) and self.history[self.cursor] == m:
            # Тот же ход, что в партии, — просто шаг вперёд без потери продолжения.
            self.seek_ply(self.cursor + 1)
            return
        self.push_move(m)

    def push_move(self, m, by="human"):
        snapshot = self.snapshots.get(self.cursor)
        if self.logger is None:
            try:
                self.logger = GameLogger(self.mode, self.net_path)
            except OSError as e:
                self.statusBar().showMessage(f"запись партии отключена: {e}")
        if self.logger is not None:
            try:
                self.logger.add(self.cursor, by, m, snapshot,
                                len(self.board.legal_moves))
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
        """Не ждёт поток: он доиграет текущую пачку в фоне, а следующий поиск
        дождётся его сам. Иначе каждое нажатие стрелки подвешивало окно на
        время прогона пачки (на процессоре — до секунды)."""
        s = self.search
        if s is None:
            return
        self.search = None
        try:
            s.update.disconnect()
        except TypeError:
            pass
        s.running = False
        self._retired = [t for t in self._retired if t.isRunning()]
        if s.isRunning():
            self._retired.append(s)

    def _join_retired(self):
        for t in self._retired:
            t.running = False
            t.wait()
        self._retired = []

    def should_search(self):
        if self.mcts is None or self.board.engine.is_game_over():
            return False
        if self.analysis_on:
            return True
        return (self.cursor == len(self.history) and self.engine_should_move()
                and self.mode in ("play_white", "play_black", "selfplay"))

    def _graph_evals(self):
        if self.hints_hidden():
            return {k: v for k, v in self.evals.items() if k != self.cursor}
        return self.evals

    def _update_book(self):
        if self.archive is not None:
            k = position_key(self.board.engine)
            self.book_box.set_book_data(self.archive.book_entry(k), has_archive=True,
                                        ply=self.cursor, max_ply=self.archive.max_book_ply)
        elif not self._archive_loading():
            self.book_box.set_book_data(None, has_archive=False)

    def refresh(self):
        self.stop_search()
        view = self.history[:self.cursor]
        last = (decode_move(view[-1])[:2] if view else None)
        self.board.set_position(view, last)
        self.board.analysis = []
        self._update_book()

        over = self.board.engine.is_game_over()
        snap = self.snapshots.get(self.cursor)

        if over:
            r = self.board.engine.game_result()
            msg = ("Ничья" if abs(r) < 1e-6
                   else ("Белые выиграли" if r > 0 else "Чёрные выиграли"))
            self.infobox.set_moves([], f"Партия окончена: {msg.lower()}")
            wr = 0.5 if abs(r) < 1e-6 else (1.0 if r > 0 else 0.0)
            self.eval_bar.set_eval(wr, None)
            self.evals[self.cursor] = wr
            msg += self._record_finished_game(r)
            self.statusBar().showMessage(f"Партия окончена — {msg}")
        elif snap is not None:
            self.apply_payload(snap, live=False)
        else:
            node = (self.ttable.get(position_key(self.board.engine))
                    if self.mcts is not None else None)
            if node is not None and node.visits > 0 and node.is_expanded:
                self.apply_payload(
                    payload_from_node(node, self.board.engine.side_to_move(),
                                      self.board.engine),
                    live=False)
            else:
                if self.mcts is None:
                    hint = "Сеть не загружена.\nКнопка «Сеть…» или Ctrl+L."
                elif self.should_search():
                    hint = "Анализ запускается…"
                else:
                    hint = "Анализ остановлен.\nПробел — запустить."
                self.infobox.set_moves([], hint)
                if self.cursor in self.evals:
                    self.eval_bar.set_eval(self.evals[self.cursor], None)
                else:
                    self.eval_bar.set_eval(0.5, None, known=False)
                if self.mcts is not None and not self.should_search():
                    self.statusBar().showMessage(
                        f"Полуход {self.cursor} · анализ остановлен — Пробел, чтобы запустить")

        if not over and self.should_search():
            engine_turn = (self.cursor == len(self.history)
                           and self.engine_should_move())
            budget = (self.spin_play.value() if engine_turn
                      else self.spin_analyze.value())
            prev = self._retired[-1] if self._retired else None
            if len(self.ttable) > TT_MAX_NODES and prev is None:
                self._prune_ttable(view)
            self.search = SearchThread(view, self.mcts, budget, self.ttable,
                                       contempt=self.contempt, prev=prev)
            self.search.update.connect(self.on_update)
            self.search_started = time.time()
            self.search.start()

        self.graph.set_data(self._graph_evals(), self.cursor, len(self.history))
        self._sync_move_list()

    def _record_finished_game(self, r):
        lg = self.logger
        if lg is None or lg.head["result"] is not None or self.cursor != len(self.history):
            return ""
        out = ""
        try:
            lg.finish(r)
            out += f"  ·  записано: games/{os.path.basename(lg.txt_path)}"
            # В архив — только партия, сыгранная здесь с начальной позиции,
            # и целиком, а не ходы с момента создания журнала.
            if A is not None and self.archive_dir and not self.history_loaded:
                n = A.archive_moves(
                    self.archive_dir, [move_to_uci(m) for m in self.history],
                    int(round(r)) if abs(r) > 0.5 else 0,
                    A.SOURCE_HUMAN, white=lg.head.get("network") or "?",
                    black=lg.head.get("network") or "?")
                if n:
                    out += f"  ·  в архив: {n} позиций"
        except Exception as e:
            out += f"  ·  запись не сохранена: {e}"
        return out

    _vram_cache = (0.0, "")

    def vram_note(self):
        if self.mcts is None or not self.mcts.gpu:
            return ""
        now = time.time()
        if now - self._vram_cache[0] < 10.0:
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
                    note = f"  ·  видеопамять {int(mem) / 1024:.1f} ГБ"
                    break
        except Exception:
            pass
        NibblerGUI._vram_cache = (now, note)
        return note

    def apply_payload(self, payload, live):
        moves = payload["moves"]
        stm = payload["stm"]
        hide = self.hints_hidden()
        self.infobox.set_moves([] if hide else moves,
                               "Подсказки скрыты — ваш ход." if hide else "нет анализа")
        self.board.analysis = [] if hide else moves
        self.board.update()

        wr_stm = (payload["root_q"] + 1.0) / 2.0
        mate_in = None
        if moves and moves[0].get("mate"):
            wr_stm = 1.0
            mate_in = 1 if stm == 0 else -1
        wr_white = wr_stm if stm == 0 else 1.0 - wr_stm

        if hide:
            self.eval_bar.set_eval(0.5, None, known=False)
        else:
            self.eval_bar.set_eval(wr_white, mate_in)
        self.evals[self.cursor] = wr_white

        side = "белые" if stm == 0 else "чёрные"
        ev = "" if hide else f"  ·  белые {wr_white * 100:.1f}%"
        if live:
            elapsed = max(1e-3, time.time() - self.search_started)
            reused = payload.get("reused", 0)
            fresh = max(0, payload["sims"] - reused)
            state = "готово" if payload.get("finished") else f"{fresh / elapsed:,.0f} узл/с"
            self.statusBar().showMessage(
                f"Ходят {side}  ·  узлов {payload['sims']:,}"
                + (f" (из прошлого поиска {reused:,})" if reused else "")
                + f"  ·  {state}{ev}{self.vram_note()}")
        else:
            self.statusBar().showMessage(
                f"Полуход {self.cursor}  ·  сохранённый анализ, {payload['sims']:,} узлов{ev}")

    def on_update(self, payload):
        s = self.search
        if s is None or payload.get("token") != s.token:
            return      # опоздавший сигнал уже остановленного поиска
        if payload.get("game_over"):
            return
        self.snapshots[self.cursor] = payload
        self.apply_payload(payload, live=True)
        self.graph.set_data(self._graph_evals(), self.cursor, len(self.history))
        self._update_move_rows({self.cursor - 1, self.cursor})

        if (payload["finished"] and payload["moves"]
                and self.cursor == len(self.history)
                and self.engine_should_move()):
            self.push_move(payload["moves"][0]["move"], by="engine")

    # ---- список ходов ----
    def _sync_move_list(self):
        sig = tuple(self.history)
        if sig != self._moves_sig:
            self._moves_sig = sig
            self.table.setRowCount(len(self.history))
            for i, m in enumerate(self.history):
                num = f"{i // 2 + 1}." if i % 2 == 0 else f"{i // 2 + 1}…"
                for col, text in ((0, num), (1, move_to_uci(m))):
                    item = QTableWidgetItem(text)
                    if col == 0:
                        item.setForeground(QColor("#808080"))
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    else:
                        item.setFont(ui_font(9.5, mono=True))
                    self.table.setItem(i, col, item)
        self._update_move_rows(range(len(self.history)))
        self.table.blockSignals(True)
        row = self.cursor - 1
        if 0 <= row < self.table.rowCount():
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, 0),
                                    QAbstractItemView.PositionAtCenter)
        else:
            self.table.clearSelection()
            if self.cursor == 0:
                self.table.scrollToTop()
        self.table.blockSignals(False)

    def _update_move_rows(self, rows):
        evals = self._graph_evals()
        hide_best = self.hints_hidden()
        for i in rows:
            if not (0 <= i < len(self.history)) or i >= self.table.rowCount():
                continue
            played = self.history[i]
            wr = evals.get(i + 1)
            snap = self.snapshots.get(i)
            best = snap["moves"][0]["move"] if (snap and snap["moves"]) else None
            if hide_best and i == self.cursor:
                best = None
            cells = {2: f"{wr * 100:.0f}%" if wr is not None else "",
                     3: move_to_uci(best) if best is not None else "",
                     4: f"{snap['sims']:,}" if snap else ""}
            for col, text in cells.items():
                item = self.table.item(i, col)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(i, col, item)
                item.setText(text)
                item.setTextAlignment(Qt.AlignCenter if col != 3 else
                                      Qt.AlignLeft | Qt.AlignVCenter)
                if col == 3:
                    item.setFont(ui_font(9.5, mono=True))
                    item.setForeground(QColor("#8a8a8a"))
                elif col == 4:
                    item.setForeground(QColor("#8a8a8a"))
            mv_item = self.table.item(i, 1)
            if mv_item is not None:
                mv_item.setForeground(QColor("#e0932b") if (best is not None and best != played)
                                      else QColor(FG))

    def closeEvent(self, ev):
        self.stop_search()
        self._join_retired()
        if self._archive_loading():
            self.archive_loader.cancelled = True
            self.archive_loader.wait()
        ev.accept()


# ───────────────────────────── Stylesheet ─────────────────────────────────

_UI_CSS_FAMILIES = ", ".join(f'"{f}"' for f in UI_FAMILIES)
QSS = f"""
QMainWindow, QWidget {{ background: {BG}; color: {FG};
                        font-family: {_UI_CSS_FAMILIES}, sans-serif; font-size: 12px; }}
QToolTip {{ background: #111; color: {FG}; border: 1px solid {BORDER}; padding: 4px; }}
QFrame#toolbar {{ background: {BG2}; border: 1px solid {BORDER}; border-radius: 6px; }}
QFrame#toolbar QLabel {{ background: transparent; color: #b8b8b8; }}
QFrame#vline {{ border: 0; border-left: 1px solid {BORDER}; margin: 3px 4px; background: transparent; }}
QLabel#sectionTitle {{ color: {ACCENT}; font-weight: bold; padding: 2px 2px 0 2px; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 5px; margin-top: 9px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px;
                    color: {ACCENT}; font-weight: bold; }}
QPushButton {{ background: {BG3}; border: 1px solid #484848; border-radius: 4px;
               padding: 5px 10px; }}
QPushButton:hover {{ background: #404040; border-color: {ACCENT}; }}
QPushButton:pressed {{ background: #4a4a4a; }}
QPushButton#primary {{ background: #2f4b66; border-color: #41698f; font-weight: bold; }}
QPushButton#primary:hover {{ background: #36587a; }}
QPushButton#primary:checked {{ background: #7a3b30; border-color: #a0503f; }}
QSpinBox, QComboBox {{ background: {BG}; border: 1px solid #474747;
                       border-radius: 4px; padding: 3px 4px; min-width: 60px; }}
QSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {BG2}; border: 1px solid {BORDER};
                               selection-background-color: {ACCENT}; }}
QCheckBox {{ background: transparent; spacing: 6px; }}
QTableWidget, QTableView {{ background: {BG2}; alternate-background-color: #2f2f2f;
                border: 1px solid {BORDER}; border-radius: 4px; gridline-color: #363636; }}
QTableWidget::item, QTableView::item {{ padding: 1px 4px; }}
QTableWidget::item:selected, QTableView::item:selected {{ background: #34587d; color: #fff; }}
QHeaderView::section {{ background: {BG3}; color: #b0b0b0; border: 0;
                        border-right: 1px solid {BORDER}; padding: 4px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; background: {BG2}; border-radius: 4px; top: -1px; }}
QTabBar::tab {{ background: {BG3}; color: #9a9a9a; padding: 6px 16px;
                border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }}
QTabBar::tab:selected {{ background: {BG2}; color: {FG}; font-weight: bold;
                         border-top: 2px solid {ACCENT}; }}
QScrollArea {{ border: 0; background: {BG2}; }}
QScrollBar:vertical {{ background: {BG2}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #4a4a4a; border-radius: 4px; min-height: 24px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: #5a5a5a; }}
QScrollBar:horizontal {{ background: {BG2}; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #4a4a4a; border-radius: 4px; min-width: 24px; margin: 2px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:hover {{ background: #3a3a3a; }}
QStatusBar {{ background: {BG2}; color: #b0b0b0; border-top: 1px solid {BORDER}; }}
QStatusBar QLabel {{ background: transparent; }}
QDialog {{ background: {BG}; }}
"""


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    ap = argparse.ArgumentParser(description="Анализатор Capablanca")
    ap.add_argument("files", nargs="*", help="сеть (.onnx/.pth) и/или партия (.pgn/.txt)")
    ap.add_argument("--archive", help="каталог архива самоигры")
    ap.add_argument("--cpu", action="store_true", help="считать сеть на процессоре")
    args, qt_args = ap.parse_known_args()
    net = next((f for f in args.files if f.lower().endswith((".onnx", ".pth"))), None)
    game = next((f for f in args.files if not f.lower().endswith((".onnx", ".pth"))), None)

    app = QApplication([sys.argv[0]] + qt_args)
    app.setStyleSheet(QSS)
    gui = NibblerGUI(net=net, game=game, archive_dir=args.archive,
                     device="cpu" if args.cpu else "auto")
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
