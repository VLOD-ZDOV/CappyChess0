#!/usr/bin/env python
"""Post-training NVFP4 quantization of the ONNX model via nvidia-modelopt.

Inserts Q/DQ nodes calibrated on REAL positions so TensorRT can place the
eligible matmuls/convs on Blackwell FP4 tensor cores. Output ONNX → build_and_bench.py.
"""
import argparse
import numpy as np
from modelopt.onnx.quantization import quantize


def load_calib(buffer, n, planes=139, h=8, w=10):
    z = np.load(buffer, allow_pickle=True)
    boards = z["boards"]
    idx = np.sort(np.random.default_rng(1).choice(len(boards), n, replace=False))
    return boards[idx].astype(np.float32).reshape(n, planes, h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="model.onnx")
    ap.add_argument("--out", default="model.fp4.onnx")
    ap.add_argument("--buffer", default="../../python_src/checkpoints_big/buffer.npz")
    ap.add_argument("--calib", type=int, default=512)
    ap.add_argument("--mode", default="nvfp4",
                    choices=["nvfp4", "int8", "int4_rtn", "fp8"])
    args = ap.parse_args()

    calib = load_calib(args.buffer, args.calib)
    print(f"calibration: {calib.shape} from {args.buffer}  mode={args.mode}")

    quantize(
        onnx_path=args.onnx,
        quantize_mode=args.mode,
        calibration_data={"board": calib},
        output_path=args.out,
        high_precision_dtype="fp16",
    )
    print(f"OK: wrote {args.out}")


if __name__ == "__main__":
    main()
