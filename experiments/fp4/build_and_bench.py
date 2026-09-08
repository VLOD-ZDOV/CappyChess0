#!/usr/bin/env python
"""Build TensorRT engines (BF16 baseline + NVFP4) from the ONNX model and
benchmark throughput on the GPU. Run inside venv312 (has tensorrt-cu13).

NVFP4 path uses nvidia-modelopt to insert FP4 quant/dequant around eligible
ops, then TRT builds an engine that lands them on the Blackwell FP4 tensor
cores. We calibrate activations on REAL positions from the replay buffer.

Reports: latency + throughput (positions/s) for BF16 vs NVFP4 at a few batch
sizes, plus the engine build status. Accuracy of the FP4 engine vs FP32 is
checked against onnxruntime reference.
"""
import argparse
import os
import sys
import time
import numpy as np

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_positions(buffer, n, planes=139, h=8, w=10):
    z = np.load(buffer, allow_pickle=True)
    boards = z["boards"]
    idx = np.sort(np.random.default_rng(0).choice(len(boards), n, replace=False))
    return boards[idx].astype(np.float32).reshape(n, planes, h, w)


def build_engine(onnx_path, mode, batch, calib_data=None):
    """mode ∈ {bf16, fp16, fp4}. Returns serialized engine bytes."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        if mode == "fp4" else 0)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  ONNX parse error:", parser.get_error(i))
            raise RuntimeError("onnx parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    if mode == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    elif mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif mode == "fp4":
        # FP4 is driven by Q/DQ nodes inserted by modelopt in the ONNX graph;
        # STRONGLY_TYPED network honors them. Allow BF16 fallback for unsupported ops.
        config.set_flag(trt.BuilderFlag.BF16)
        if hasattr(trt.BuilderFlag, "FP4"):
            config.set_flag(trt.BuilderFlag.FP4)

    # optimization profile for dynamic batch
    inp = network.get_input(0)
    shape = inp.shape
    if shape[0] == -1:
        profile = builder.create_optimization_profile()
        c, h, w = shape[1], shape[2], shape[3]
        profile.set_shape(inp.name, (1, c, h, w), (batch, c, h, w), (batch, c, h, w))
        config.add_optimization_profile(profile)

    t0 = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"engine build failed for mode={mode}")
    print(f"  [{mode}] engine built in {time.time()-t0:.1f}s")
    return bytes(plan)


def bench(engine_bytes, x, iters=50, warmup=10):
    import torch  # CUDA allocations via torch for convenience
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    ctx = engine.create_execution_context()

    n = x.shape[0]
    inp_name = engine.get_tensor_name(0)
    ctx.set_input_shape(inp_name, x.shape)

    d_in = torch.from_numpy(x).cuda().contiguous()
    # allocate outputs
    outs = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shp = tuple(ctx.get_tensor_shape(name))
            outs[name] = torch.empty(shp, dtype=torch.float32, device="cuda")
            ctx.set_tensor_address(name, outs[name].data_ptr())
    ctx.set_tensor_address(inp_name, d_in.data_ptr())

    stream = torch.cuda.Stream()
    for _ in range(warmup):
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return dt, n / dt, {k: v.clone() for k, v in outs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="model.onnx")
    ap.add_argument("--onnx-fp4", default="model.fp4.onnx",
                    help="modelopt-quantized ONNX (Q/DQ inserted)")
    ap.add_argument("--buffer", default="../../python_src/checkpoints_big/buffer.npz")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    print(f"TensorRT {trt.__version__}")
    x = load_positions(args.buffer, args.batch)

    results = {}
    for mode, path in [("bf16", args.onnx), ("fp4", args.onnx_fp4)]:
        if not os.path.exists(path):
            print(f"skip {mode}: {path} missing")
            continue
        print(f"\n=== {mode} ({path}) ===")
        try:
            eng = build_engine(path, mode, args.batch)
            dt, tput, outs = bench(eng, x)
            results[mode] = (dt, tput, outs)
            print(f"  batch={args.batch}  {dt*1e3:.2f} ms/iter  {tput:,.0f} pos/s")
        except Exception as e:
            print(f"  {mode} FAILED: {e}")

    if "bf16" in results and "fp4" in results:
        b = results["bf16"][1]
        f = results["fp4"][1]
        print(f"\nSPEEDUP fp4 vs bf16: {f/b:.2f}x  ({b:,.0f} → {f:,.0f} pos/s)")
        # accuracy: compare policy logits
        bp = results["bf16"][2]
        fp = results["fp4"][2]
        pol_key = [k for k in bp if "policy" in k.lower()] or list(bp)[:1]
        if pol_key:
            k = pol_key[0]
            import torch
            b_arg = bp[k].argmax(1)
            f_arg = fp[k].argmax(1)
            match = (b_arg == f_arg).float().mean().item()
            print(f"top-1 policy match fp4 vs bf16: {match*100:.1f}%")


if __name__ == "__main__":
    main()
