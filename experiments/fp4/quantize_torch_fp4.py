#!/usr/bin/env python
"""NVFP4 quantize CapablancaNet via modelopt.torch (mtq), calibrate on real
positions, export to ONNX with FP4 Q/DQ for TensorRT. Run in venv312.
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python_src"))
from model import build_net_from_state_dict, CapablancaNet  # noqa: E402

import modelopt.torch.quantization as mtq  # noqa: E402


class InferWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        policy, wdl, _, _ = self.net(x)
        return policy, wdl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="model.fp4.onnx")
    ap.add_argument("--buffer", default="../../python_src/checkpoints_big/buffer.npz")
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cfg", default="NVFP4_DEFAULT_CFG")
    args = ap.parse_args()

    device = "cuda"
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    net, sd = build_net_from_state_dict(sd)
    net.load_state_dict(sd, strict=False)
    wrap = InferWrapper(net).eval().to(device)
    print(f"model: {net.num_channels}ch × {net.num_res_blocks}bl  cfg={args.cfg}")

    z = np.load(args.buffer, allow_pickle=True)
    boards = z["boards"]
    idx = np.sort(np.random.default_rng(1).choice(len(boards), args.calib, replace=False))
    calib = torch.from_numpy(
        boards[idx].astype(np.float32)
    ).reshape(args.calib, CapablancaNet.INPUT_PLANES,
              CapablancaNet.BOARD_H, CapablancaNet.BOARD_W).to(device)

    def forward_loop(model):
        bs = args.batch
        with torch.no_grad():
            for i in range(0, calib.shape[0], bs):
                model(calib[i:i + bs])

    config = getattr(mtq, args.cfg)
    wrap = mtq.quantize(wrap, config, forward_loop)
    print("quantized. quant summary:")
    mtq.print_quant_summary(wrap)

    # export to ONNX with Q/DQ (legacy exporter handles SDPA path)
    dummy = calib[:args.batch]
    torch.onnx.export(
        wrap, dummy, args.out,
        input_names=["board"], output_names=["policy", "wdl"],
        dynamic_axes={"board": {0: "batch"}, "policy": {0: "batch"},
                      "wdl": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
