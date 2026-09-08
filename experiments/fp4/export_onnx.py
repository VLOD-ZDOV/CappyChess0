#!/usr/bin/env python
"""Export CapablancaNet to ONNX for TensorRT FP4 engine building.

We export ONLY the inference-relevant outputs (policy logits, wdl logits) — the
future head is training-only and the MLH head is tiny; keeping the graph lean
helps TRT. Dynamic batch axis so one engine serves the variable self-play batch.

The RPB attention path uses F.scaled_dot_product_attention with an additive
bias; opset 17+ lowers SDPA cleanly. We export at a fixed sample batch and mark
axis 0 dynamic.
"""
import argparse
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python_src"))
from model import build_net_from_state_dict, CapablancaNet  # noqa: E402


class InferWrapper(nn.Module):
    """Trim to (policy_logits, wdl_logits) — what MCTS consumes."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        policy, wdl, _, _ = self.net(x)
        return policy, wdl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("-o", "--out", default="model.onnx")
    ap.add_argument("--batch", type=int, default=256,
                    help="sample batch for tracing (axis is dynamic)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--static", action="store_true",
                    help="fixed batch (no dynamic axis) — sometimes faster FP4 engine")
    args = ap.parse_args()

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    net, sd = build_net_from_state_dict(sd)
    net.load_state_dict(sd, strict=False)
    net.eval()
    print(f"model: {net.num_channels}ch × {net.num_res_blocks}bl, "
          f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M params")

    wrap = InferWrapper(net).eval()
    dummy = torch.randn(args.batch, CapablancaNet.INPUT_PLANES,
                        CapablancaNet.BOARD_H, CapablancaNet.BOARD_W)

    dyn = None if args.static else {
        "board":  {0: "batch"},
        "policy": {0: "batch"},
        "wdl":    {0: "batch"},
    }
    # dynamo=False → legacy TorchScript exporter; the dynamo path mis-handles
    # the transpose().flatten() after SDPA under a dynamic batch axis.
    torch.onnx.export(
        wrap, dummy, args.out,
        input_names=["board"], output_names=["policy", "wdl"],
        dynamic_axes=dyn, opset_version=args.opset,
        do_constant_folding=True, dynamo=False,
    )
    print(f"wrote {args.out}  (batch={'static '+str(args.batch) if args.static else 'dynamic'})")

    # sanity: onnxruntime forward vs torch
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        xb = dummy.numpy()
        op, ow = sess.run(None, {"board": xb})
        with torch.no_grad():
            tp, tw = wrap(dummy)
        dp = np.abs(op - tp.numpy()).max()
        dw = np.abs(ow - tw.numpy()).max()
        print(f"onnxruntime vs torch  policy max|Δ|={dp:.2e}  wdl max|Δ|={dw:.2e}")
    except Exception as e:
        print(f"(skip ORT check: {e})")


if __name__ == "__main__":
    main()
