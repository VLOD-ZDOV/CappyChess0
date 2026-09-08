"""Export a trained CapablancaNet checkpoint (.pth) to a self-contained ONNX
graph for GPU inference via onnxruntime.

The exported graph already applies softmax / WDL reduction, so its four outputs
are exactly what the GUI consumes: policy, value Q, draw probability D, and the
moves-left estimate M. Architecture (channels / blocks / heads) is recovered
straight from the checkpoint, so no flags are needed.

Usage:
    python export_onnx.py <checkpoint.pth> [output.onnx]

This is a one-off, dev-side step and needs PyTorch. The GUI itself
(gui.py + onnx_engine.py) needs only onnxruntime — see the README.
"""

import os
import sys

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CapablancaNet


class _InferWrap(nn.Module):
    """Wraps CapablancaNet so the ONNX graph emits inference-ready tensors:
    softmaxed policy, scalar Q = P(win) - P(loss), draw prob D, moves-left M."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        policy, wdl, mlh, _ = self.net(x)        # future head is training-only
        wdl = F.softmax(wdl, dim=1)
        q = wdl[:, 0] - wdl[:, 2]
        d = wdl[:, 1]
        m = torch.sigmoid(mlh).reshape(-1) if mlh is not None \
            else torch.zeros_like(q)
        return F.softmax(policy, dim=1), q, d, m


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_onnx.py <checkpoint.pth> [output.onnx]")
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "capablanca.onnx"

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    raw = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Use the canonical helper — it recovers ALL architecture knobs including
    # the BT5 trim (qkv_bias / use_rmsnorm / piece_embed_dim). The inline
    # detection this replaced ignored those flags, so a BT5-trained checkpoint
    # silently exported with LayerNorm instead of RMSNorm and no piece_embed —
    # a wrong-architecture ONNX with no error.
    from model import build_net_from_state_dict, describe_arch
    net, sd = build_net_from_state_dict(raw)
    print(f"Архитектура: {describe_arch(net)}")
    result = net.load_state_dict(sd, strict=False)
    if result.missing_keys:
        print(f"⚠️  не заполнено ключей: {len(result.missing_keys)} "
              f"(напр. {result.missing_keys[0]})")
    net.eval()

    wrap = _InferWrap(net).eval()
    dummy = torch.zeros(2, CapablancaNet.INPUT_PLANES,
                        CapablancaNet.BOARD_H, CapablancaNet.BOARD_W)
    dyn = {0: "batch"}
    # Try the modern (dynamo) exporter first — produces cleaner graphs and is
    # the recommended path on PyTorch 2.5+. If it can't prove shape constraints
    # through some op (happens with SDPA + transpose + reshape on some PyTorch
    # builds), fall back to the legacy TorchScript exporter, which just traces
    # the graph without symbolic shape analysis.
    common = dict(
        input_names=["board"],
        output_names=["policy", "q", "d", "m"],
        dynamic_axes={"board": dyn, "policy": dyn, "q": dyn, "d": dyn, "m": dyn},
        opset_version=18, do_constant_folding=True,
    )
    try:
        torch.onnx.export(wrap, dummy, dst, **common)
    except Exception as e:
        print(f"⚠️  dynamo экспорт не справился ({type(e).__name__}). "
              f"Откат на legacy TorchScript exporter…")
        torch.onnx.export(wrap, dummy, dst, dynamo=False, **common)

    # The modern exporter writes weights to a sidecar <name>.data file.
    # Consolidate into a single self-contained .onnx for easy distribution.
    model = onnx.load(dst)                       # resolves external data
    onnx.save_model(model, dst, save_as_external_data=False)
    sidecar = dst + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)
    print(f"✅ Сохранено (один файл): {dst}")


if __name__ == "__main__":
    main()
