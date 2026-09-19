#!/usr/bin/env python
"""Полный перебор первых N полуходов с оценкой движка.

    python gen_book4.py book4.jsonl --plies 4 --nodes 1000000 --threads 12

Позиций после 4 полуходов ровно 812 341 (ветвление 28 → 28 → 32.2 → 32.2).
Пишется построчно в jsonl и возобновляется с места обрыва: шестнадцать часов
работы нельзя ставить на то, что ничего не случится.

Оценка — NNUE (+417 Elo к классической по замеру на 24 партиях). Без неё
движок в капабланке считает классикой, и прежняя книга на 2 полухода этим и
страдала: направление оценок было верным, масштаб враньём.
"""
import argparse, json, os, subprocess, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capablanca_engine import CapablancaEngine

HERE = os.path.dirname(os.path.abspath(__file__))
FSF = os.path.join(HERE, "fairy-stockfish-largeboard_x86-64-bmi2")
NET = os.path.join(HERE, "capablanca-bb644ef32758.nnue")


def uci(m):
    f, t, p = (m >> 10) & 0x7F, (m >> 3) & 0x7F, m & 0b111
    s = f"{chr(97+f%10)}{f//10+1}{chr(97+t%10)}{t//10+1}"
    return s + " nbrqac"[p] if p else s


def enumerate_lines(plies):
    lvl = [[]]
    for _ in range(plies):
        nxt = []
        for seq in lvl:
            e = CapablancaEngine()
            for m in seq:
                e.make_move_int(m)
            nxt.extend(seq + [m] for m in e.get_legal_moves_int())
        lvl = nxt
    return [[uci(m) for m in seq] for seq in lvl]


class Engine:
    def __init__(self, nodes):
        self.nodes = nodes
        self.p = subprocess.Popen([FSF], universal_newlines=True, bufsize=1,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._c("uci", "uciok")
        self.s("setoption name UCI_Variant value capablanca")
        self.s(f"setoption name EvalFile value {NET}")
        self._c("isready", "readyok")

    def s(self, c):
        self.p.stdin.write(c + "\n"); self.p.stdin.flush()

    def _c(self, c, w):
        self.s(c)
        while w not in self.p.stdout.readline():
            pass

    def eval(self, moves):
        self.s("position startpos moves " + " ".join(moves))
        self.s(f"go nodes {self.nodes}")
        cp = mate = None; depth = 0; best = None
        while True:
            l = self.p.stdout.readline()
            if not l:
                return None
            if l.startswith("info") and " score " in l:
                t = l.split()
                i = t.index("score")
                if t[i+1] == "cp": cp, mate = int(t[i+2]), None
                else: mate, cp = int(t[i+2]), None
                depth = int(t[t.index("depth")+1])
            if l.startswith("bestmove"):
                best = l.split()[1]
                break
        return {"moves": moves, "cp": cp, "mate": mate, "depth": depth, "best": best}


def worker(wid, lines, nodes, out_path, lock):
    eng = Engine(nodes)
    buf = []
    t0 = time.time()
    for i, mv in enumerate(lines):
        r = eng.eval(mv)
        if r is None:
            break
        buf.append(json.dumps(r, ensure_ascii=False))
        if len(buf) >= 200:
            with lock:
                with open(out_path, "a") as f:
                    f.write("\n".join(buf) + "\n")
            buf = []
            if wid == 0:
                done = i + 1
                rate = done / (time.time() - t0)
                print(f"[поток 0] {done}/{len(lines)}, {rate:.1f} поз/с, "
                      f"осталось {(len(lines)-done)/rate/3600:.1f} ч", flush=True)
    if buf:
        with lock:
            with open(out_path, "a") as f:
                f.write("\n".join(buf) + "\n")
    eng.s("quit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--plies", type=int, default=4)
    ap.add_argument("--nodes", type=int, default=1_000_000)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()

    print("строю список позиций...", flush=True)
    lines = enumerate_lines(a.plies)
    print(f"{len(lines):,} позиций после {a.plies} полуходов".replace(",", " "), flush=True)

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, errors="ignore"):
            try:
                done.add(" ".join(json.loads(l)["moves"]))
            except Exception:
                pass
        print(f"уже посчитано {len(done):,}, продолжаю".replace(",", " "), flush=True)
    todo = [m for m in lines if " ".join(m) not in done]
    if not todo:
        print("всё готово"); return

    lock = mp.Lock()
    chunks = [todo[i::a.threads] for i in range(a.threads)]
    ps = [mp.Process(target=worker, args=(i, chunks[i], a.nodes, a.out, lock))
          for i in range(a.threads)]
    for p in ps: p.start()
    try:
        for p in ps: p.join()
    except KeyboardInterrupt:
        for p in ps: p.terminate()
    print(f"готово: {a.out}")


if __name__ == "__main__":
    main()
