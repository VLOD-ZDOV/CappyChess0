#!/usr/bin/env python
"""Build + benchmark TensorRT engines for CapablancaNet on Blackwell.
Compares a high-precision baseline engine against a low-precision (FP8/INT8)
quantized engine. Run in venv312.

TRT 11 uses strongly-typed / graph-driven precision: the quantized ONNX carries
Q/DQ nodes that TRT lowers onto FP8 tensor cores automatically — no builder
precision flag needed. The baseline builds from the plain fp32 ONNX (TF32 tensor
cores on by default).
"""
import argparse
import time
import numpy as np
import tensorrt as trt
import torch

L = trt.Logger(trt.Logger.ERROR)


def build(onnx_path, batch, fp16=False):
    b = trt.Builder(L)
    net = b.create_network(0)
    p = trt.OnnxParser(net, L)
    if not p.parse(open(onnx_path, "rb").read()):
        for i in range(p.num_errors):
            print("  parse err:", p.get_error(i))
        raise RuntimeError("parse failed")
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if fp16 and hasattr(trt.BuilderFlag, "FP16"):
        cfg.set_flag(trt.BuilderFlag.FP16)
    inp = net.get_input(0)
    c, h, w = inp.shape[1], inp.shape[2], inp.shape[3]
    prof = b.create_optimization_profile()
    prof.set_shape(inp.name, (1, c, h, w), (batch, c, h, w), (batch, c, h, w))
    cfg.add_optimization_profile(prof)
    t = time.time()
    plan = b.build_serialized_network(net, cfg)
    if plan is None:
        raise RuntimeError("engine build failed")
    return bytes(plan), time.time() - t


def bench(engine_bytes, x, iters=100, warmup=20):
    rt = trt.Runtime(L)
    eng = rt.deserialize_cuda_engine(engine_bytes)
    ctx = eng.create_execution_context()
    in_name = eng.get_tensor_name(0)
    ctx.set_input_shape(in_name, x.shape)
    d_in = torch.from_numpy(x).cuda().contiguous()
    ctx.set_tensor_address(in_name, d_in.data_ptr())
    outs = {}
    for i in range(eng.num_io_tensors):
        name = eng.get_tensor_name(i)
        if eng.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shp = tuple(ctx.get_tensor_shape(name))
            outs[name] = torch.empty(shp, dtype=torch.float32, device="cuda")
            ctx.set_tensor_address(name, outs[name].data_ptr())
    s = torch.cuda.Stream()
    for _ in range(warmup):
        ctx.execute_async_v3(s.cuda_stream)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        ctx.execute_async_v3(s.cuda_stream)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return dt, x.shape[0] / dt, {k: v.clone() for k, v in outs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="model.onnx")
    ap.add_argument("--quant", default="model.fp8.onnx")
    ap.add_argument("--qlabel", default="fp8")
    ap.add_argument("--buffer", default="../../python_src/checkpoints_big/buffer.npz")
    ap.add_argument("--batches", default="64,256,1024")
    args = ap.parse_args()

    print(f"TensorRT {trt.__version__}  GPU {torch.cuda.get_device_name(0)}")
    z = np.load(args.buffer, allow_pickle=True)
    boards = z["boards"]

    for bs in [int(x) for x in args.batches.split(",")]:
        idx = np.sort(np.random.default_rng(0).choice(len(boards), bs, replace=False))
        x = boards[idx].astype(np.float32).reshape(bs, 139, 8, 10)
        print(f"\n=== batch {bs} ===")
        eng_b, tb = build(args.baseline, bs)
        db, tputb, ob = bench(eng_b, x)
        print(f"  baseline (tf32/fp32)  build {tb:.1f}s  {db*1e3:6.2f} ms  {tputb:10,.0f} pos/s")
        try:
            eng_q, tq = build(args.quant, bs)
            dq, tputq, oq = bench(eng_q, x)
            print(f"  {args.qlabel:18s}    build {tq:.1f}s  {dq*1e3:6.2f} ms  {tputq:10,.0f} pos/s"
                  f"   →  {tputq/tputb:.2f}x")
            # accuracy vs baseline
            pk = [k for k in ob if "policy" in k.lower()] or list(ob)[:1]
            if pk:
                k = pk[0]
                match = (ob[k].argmax(1) == oq[k].argmax(1)).float().mean().item()
                print(f"  top-1 move match {args.qlabel} vs baseline: {match*100:.1f}%")
        except Exception as e:
            print(f"  {args.qlabel} FAILED: {e}")


if __name__ == "__main__":
    main()
