#!/usr/bin/env python3
"""Play a game against a checkpoint from a terminal — no GUI, no display.

    python play_cli.py <checkpoint.pth> [--sims 800] [--side white|black]

Interactive: type moves in UCI (e2e4, and a promotion suffix q/r/b/n/a/c).
Non-interactive, for driving over ssh: `--move e2e4` applies one move from a
saved game state, lets the engine reply, prints the position and exits. State
lives in `--state` (default play_state.json) so each call resumes the game.

Commands in interactive mode: `moves` lists legal moves, `undo` is not
supported (restart instead), `quit` exits.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from capablanca_engine import CapablancaEngine
from model import build_net_from_state_dict
from mcts import UltraFastMCTS

PIECES = "PNBRQACK"
GLYPH = {0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "A", 6: "C", 7: "K"}
_PROMO_CHARS = {2: "n", 3: "b", 4: "r", 5: "q", 6: "a", 7: "c"}


def move_to_uci(m: int) -> str:
    p = m & 0b111
    t = (m >> 3) & 0x7F
    f = (m >> 10) & 0x7F
    s = f"{chr(ord('a') + f % 10)}{f // 10 + 1}{chr(ord('a') + t % 10)}{t // 10 + 1}"
    return s + _PROMO_CHARS.get(p, "")


def render(engine: CapablancaEngine, last: str = "") -> str:
    """ASCII board, white at the bottom. Uppercase = white, lowercase = black."""
    grid = [["." for _ in range(10)] for _ in range(8)]
    for color, piece, sq in engine.get_pieces():
        ch = GLYPH[piece]
        grid[sq // 10][sq % 10] = ch if color == 0 else ch.lower()
    out = ["    a b c d e f g h i j"]
    for r in range(7, -1, -1):
        out.append(f" {r + 1}  " + " ".join(grid[r]) + f"  {r + 1}")
    out.append("    a b c d e f g h i j")
    stm = "белые" if engine.side_to_move() == 0 else "чёрные"
    out.append(f"\nход: {stm}" + (f"   последний: {last}" if last else ""))
    return "\n".join(out)


def load_engine_from(moves):
    e = CapablancaEngine()
    for m in moves:
        e.make_move_int(m)
    return e


def net_move(net, device, engine, sims, batch, parallel):
    mcts = UltraFastMCTS(net, device, c_puct=1.745, batch_size=1,
                         add_dirichlet=False, parallel_sims=parallel)
    tree = mcts.new_tree([engine])
    mcts.run_search(tree, sims)
    idxs, vals = tree.get_policies_sparse()[0]
    lookup = {int(i): float(v) for i, v in zip(idxs, vals)}
    legal = engine.get_legal_moves_int()
    scored = [(lookup.get(engine.move_int_to_policy_idx(m), 0.0), m) for m in legal]
    scored.sort(reverse=True)
    q = float(np.asarray(tree.get_values(), dtype=np.float32)[0])
    d = float(np.asarray(tree.get_draws(), dtype=np.float32)[0])
    return scored[0][1], q, d, scored[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--sims", type=int, default=800)
    ap.add_argument("--parallel", type=int, default=32)
    ap.add_argument("--side", choices=["white", "black"], default="white",
                    help="сторона ЧЕЛОВЕКА")
    ap.add_argument("--move", default=None, help="сделать один ход и выйти")
    ap.add_argument("--state", default="play_state.json")
    ap.add_argument("--new", action="store_true", help="начать партию заново")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resign", type=float, default=0.90,
                    help="Сдаться, если P(проигрыша) выше порога подряд "
                         "--resign-consec ходов. 0 = никогда. Тот же WDL-критерий, "
                         "что в self-play (train.py Config.resign_wdl_threshold).")
    ap.add_argument("--resign-consec", type=int, default=3)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    raw = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net, sd = build_net_from_state_dict(raw)
    net.load_state_dict(sd, strict=False)
    net.to(device).eval()

    state = {"moves": [], "human": args.side}
    if os.path.exists(args.state) and not args.new:
        with open(args.state) as f:
            state = json.load(f)
    human_white = state["human"] == "white"
    engine = load_engine_from(state["moves"])

    def save():
        with open(args.state, "w") as f:
            json.dump(state, f)

    def engine_reply():
        """Let the net move if it is its turn and the game is still on."""
        if engine.is_game_over():
            return None
        if (engine.side_to_move() == 0) == human_white:
            return None
        with torch.inference_mode():
            mv, q, d, top = net_move(net, device, engine, args.sims,
                                     1, args.parallel)
        engine.make_move_int(mv)
        state["moves"].append(mv)
        alts = "  ".join(f"{move_to_uci(m)} {p:.0%}" for p, m in top)
        # P(loss) from the WDL head: Q = W - L and D = draw, so L = (1 - Q - D)/2.
        # Same criterion self-play resignation uses; a Q threshold alone confuses
        # "dead draw" with "losing".
        p_loss = max(0.0, min(1.0, (1.0 - q - d) / 2.0))
        keep = max(0, args.resign_consec - 1)
        state["p_loss"] = (state.get("p_loss", [])[-keep:] if keep else []) + [p_loss]
        line = (f"сеть: {move_to_uci(mv)}   Q={q:+.3f} D={d:.2f} "
                f"P(проигрыша)={p_loss:.0%}\n  топ-5 по визитам: {alts}")
        if (args.resign > 0 and len(state["p_loss"]) >= args.resign_consec
                and all(x > args.resign for x in state["p_loss"])):
            state["resigned"] = True
            line += f"\n\n  🏳️  сеть сдаётся (P(проигрыша) > {args.resign:.0%} "
            line += f"{args.resign_consec} хода подряд)"
        return line

    def status():
        if state.get("resigned"):
            return "\n*** сеть сдалась ***"
        if engine.is_game_over():
            r = engine.game_result()
            return "\n*** " + ("1-0" if r > 0.5 else "0-1" if r < -0.5 else "½-½") + " ***"
        return ""

    # one-shot mode: apply the human move, let the net answer, print, exit
    if args.move is not None or args.new:
        note = ""
        if args.move:
            legal = engine.get_legal_moves_int()
            found = next((m for m in legal if move_to_uci(m) == args.move), None)
            if found is None:
                print(f"нелегальный ход: {args.move}")
                print("легальные:", " ".join(sorted(move_to_uci(m) for m in legal)))
                sys.exit(2)
            engine.make_move_int(found)
            state["moves"].append(found)
            note = f"ход человека: {args.move}"
        elif args.new:
            note = "новая партия"
        reply = engine_reply()
        save()
        print(render(engine, note))
        if reply:
            print("\n" + reply)
        print(status())
        return

    # interactive
    print(render(engine))
    r = engine_reply()
    if r:
        print("\n" + r + "\n"); print(render(engine))
    while not engine.is_game_over():
        try:
            cmd = input("\nваш ход (UCI) > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd in ("quit", "q"):
            break
        if cmd == "moves":
            print(" ".join(sorted(move_to_uci(m) for m in engine.get_legal_moves_int())))
            continue
        legal = engine.get_legal_moves_int()
        found = next((m for m in legal if move_to_uci(m) == cmd), None)
        if found is None:
            print("нелегальный ход; `moves` покажет список")
            continue
        engine.make_move_int(found)
        state["moves"].append(found)
        r = engine_reply()
        save()
        print("\n" + render(engine, f"ход человека: {cmd}"))
        if r:
            print("\n" + r)
    print(status())
    save()


if __name__ == "__main__":
    main()
