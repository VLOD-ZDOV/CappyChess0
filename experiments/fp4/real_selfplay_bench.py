#!/usr/bin/env python
"""Real self-play throughput: TRT-FP16 vs torch.compile, inside the actual
UltraFastMCTS loop (Rust engine built for cp312). Measures wall-clock and
positions/s including Rust round-trip, dynamic batches and D2H — the honest
number, not a clean-forward microbench. Run in venv312.
"""
import sys
import os
import time
import argparse
import numpy as np
import torch
import tensorrt as trt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python_src"))
import mcts as M  # noqa: E402
from mcts import UltraFastMCTS  # noqa: E402
from model import build_net_from_state_dict  # noqa: E402
from capablanca_engine import CapablancaEngine  # noqa: E402

L = trt.Logger(trt.Logger.ERROR)


class TRTNet:
    """Wraps a dynamic-batch TRT engine to look like model(x) → (policy, wdl)."""
    def __init__(self, onnx_path, max_batch):
        b = trt.Builder(L)
        net = b.create_network(0)
        p = trt.OnnxParser(net, L)
        assert p.parse(open(onnx_path, "rb").read()), "onnx parse failed"
        cfg = b.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        inp = net.get_input(0)
        c, h, w = inp.shape[1], inp.shape[2], inp.shape[3]
        prof = b.create_optimization_profile()
        prof.set_shape(inp.name, (1, c, h, w),
                       (max_batch // 2, c, h, w), (max_batch, c, h, w))
        cfg.add_optimization_profile(prof)
        plan = b.build_serialized_network(net, cfg)
        self.eng = trt.Runtime(L).deserialize_cuda_engine(bytes(plan))
        self.ctx = self.eng.create_execution_context()
        self.in_name = self.eng.get_tensor_name(0)
        self.out_names = [self.eng.get_tensor_name(i)
                          for i in range(self.eng.num_io_tensors)
                          if self.eng.get_tensor_mode(self.eng.get_tensor_name(i))
                          == trt.TensorIOMode.OUTPUT]
        self.max_batch = max_batch
        self.stream = torch.cuda.Stream()

    def eval(self):
        return self

    def __call__(self, x):
        n = x.shape[0]
        xi = x.to(torch.float16).contiguous()   # onnx is fp16 NCHW
        self.ctx.set_input_shape(self.in_name, tuple(xi.shape))
        self.ctx.set_tensor_address(self.in_name, xi.data_ptr())
        outs = []
        for name in self.out_names:
            shp = tuple(self.ctx.get_tensor_shape(name))
            o = torch.empty(shp, dtype=torch.float16, device="cuda")
            self.ctx.set_tensor_address(name, o.data_ptr())
            outs.append(o)
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        # order: onnx outputs were [policy, wdl]
        return tuple(outs)


def make_counter(mcts):
    """Wrap _infer to count total positions processed."""
    orig = mcts._infer
    stats = {"pos": 0, "calls": 0}

    def wrapped(tensors, hashes=None):
        stats["pos"] += tensors.shape[0]
        stats["calls"] += 1
        return orig(tensors, hashes=hashes)
    mcts._infer = wrapped
    return stats


def run(backend, net, device, games, sims, psims, max_moves):
    mcts = UltraFastMCTS(net, device, c_puct=1.25, batch_size=games,
                         add_dirichlet=True, parallel_sims=psims,
                         compile_mode=("default" if backend == "compile" else None),
                         bf16_weights=(backend == "compile"))
    stats = make_counter(mcts)
    engines = [CapablancaEngine() for _ in range(games)]
    # warmup (compile/TRT first-call cost) — one full move
    mcts.search_games_with_values(engines, sims)
    stats["pos"] = 0
    stats["calls"] = 0
    torch.cuda.synchronize()
    t0 = time.time()
    moves = 0
    while moves < max_moves:
        pols, vals = mcts.search_games_with_values(engines, sims)
        # play the argmax move in each live game to advance the trees
        alive = 0
        for i, eng in enumerate(engines):
            if eng.is_game_over():
                continue
            alive += 1
            mv = int(np.asarray(pols[i]).argmax())
            legal = eng.get_legal_moves_int()
            if legal:
                eng.make_move_int(mv if mv in legal else legal[0])
        moves += 1
        if alive == 0:
            break
    torch.cuda.synchronize()
    dt = time.time() - t0
    return dt, stats["pos"], stats["pos"] / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../../python_src/checkpoints_big/latest.pth")
    ap.add_argument("--fp16-onnx", default="model.fp16.onnx")
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--psims", type=int, default=32)
    ap.add_argument("--moves", type=int, default=12)
    args = ap.parse_args()

    device = torch.device("cuda")
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    net, sd = build_net_from_state_dict(sd)
    net.load_state_dict(sd, strict=False)
    net = net.eval().cuda()
    print(f"games={args.games} sims={args.sims} psims={args.psims} moves={args.moves}")
    max_batch = max(512, args.games * args.psims)

    print("\n[compile] building/warming...")
    dt_c, pos_c, tput_c = run("compile", net, device, args.games, args.sims,
                              args.psims, args.moves)
    print(f"  compile bf16:  {dt_c:6.1f}s  {pos_c:,} pos  →  {tput_c:,.0f} pos/s")

    print("\n[trt] building engine...")
    trtnet = TRTNet(args.fp16_onnx, max_batch)
    dt_t, pos_t, tput_t = run("trt", trtnet, device, args.games, args.sims,
                              args.psims, args.moves)
    print(f"  TRT fp16:      {dt_t:6.1f}s  {pos_t:,} pos  →  {tput_t:,.0f} pos/s")

    print(f"\nREAL self-play speedup TRT-fp16 vs compile-bf16: {tput_t/tput_c:.2f}x")


if __name__ == "__main__":
    main()
