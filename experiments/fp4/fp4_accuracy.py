#!/usr/bin/env python
"""NVFP4 accuracy probe for CapablancaNet — answers "будет ли менее точным?".

No TensorRT, no extra packages: we *simulate* NVFP4 numerics in pure torch
(fake-quant) and measure how much the policy/value outputs drift away from the
FP32 reference on REAL positions sampled from the replay buffer.

NVFP4 recipe (NVIDIA Blackwell) modelled here:
  - element format E2M1: sign + 2 exp + 1 mantissa,
    representable magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}, amax = 6
  - per-block scale over 16 contiguous elements, stored in FP8 (E4M3, amax 448)
  - per-tensor FP32 global scale so block-scales fit the E4M3 range

We fake-quant the weight AND the input activation of every Conv2d / Linear in
the trunk + heads (exactly the ops TRT would lower onto FP4 tensor cores), and
leave norms / softmax / residual adds in high precision (TRT keeps those too).

Reference points reported per output:
  FP4 vs FP32   — the real question
  BF16 vs FP32  — the dtype self-play already runs in (sanity baseline)
"""
import argparse
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python_src"))
from model import build_net_from_state_dict, CapablancaNet  # noqa: E402

# ── E2M1 / NVFP4 fake-quant ────────────────────────────────────────────────
# Positive representable magnitudes of E2M1.
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round magnitudes to the nearest E2M1 level, keep sign."""
    levels = _E2M1.to(x.device, x.dtype)
    s = torch.sign(x)
    a = x.abs().clamp(max=_E2M1_MAX)
    # nearest level
    idx = torch.bucketize(a, (levels[1:] + levels[:-1]) / 2.0)
    return s * levels[idx]


def _round_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to FP8 E4M3 (used for the per-block scales)."""
    return x.clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn).to(x.dtype)


def nvfp4_quant(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Fake-quantise the last dim of x with NVFP4 two-level scaling."""
    orig_shape = x.shape
    n = orig_shape[-1]
    pad = (block - n % block) % block
    if pad:
        x = F.pad(x, (0, pad))
    xb = x.reshape(*x.shape[:-1], -1, block)              # (..., nblk, 16)

    amax = xb.abs().amax(dim=-1, keepdim=True)            # per-block amax
    tensor_amax = amax.amax()
    # global per-tensor scale so block scales land inside E4M3 range
    global_scale = (tensor_amax / (_E2M1_MAX * _E4M3_MAX)).clamp(min=1e-12)
    block_scale = (amax / _E2M1_MAX) / global_scale       # → ~E4M3 range
    block_scale = _round_e4m3(block_scale) * global_scale
    block_scale = block_scale.clamp(min=1e-12)

    q = _round_e2m1(xb / block_scale) * block_scale
    q = q.reshape(*x.shape)
    if pad:
        q = q[..., :n]
    return q.reshape(orig_shape)


# ── Layer wrappers that fake-quant weight + activation ─────────────────────
class FP4Linear(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.lin = lin
        self.register_buffer("wq", nvfp4_quant(lin.weight.data.float()).to(lin.weight.dtype))

    def forward(self, x):
        xq = nvfp4_quant(x.float()).to(x.dtype)
        return F.linear(xq, self.wq, self.lin.bias)


class FP4Conv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        w = conv.weight.data.float()                      # (O, I, kh, kw)
        wq = nvfp4_quant(w.reshape(w.shape[0], -1)).reshape(w.shape)
        self.register_buffer("wq", wq.to(conv.weight.dtype))

    def forward(self, x):
        # quantise activation along channel dim (move C to last, block over C)
        xperm = x.permute(0, 2, 3, 1).contiguous()
        xq = nvfp4_quant(xperm.float()).to(x.dtype)
        xq = xq.permute(0, 3, 1, 2).contiguous()
        return F.conv2d(xq, self.wq, self.conv.bias,
                        self.conv.stride, self.conv.padding,
                        self.conv.dilation, self.conv.groups)


def fp4_wrap(module: nn.Module, exclude_prefixes=(), _path=""):
    """Replace Conv2d/Linear with FP4 fake-quant versions in-place.
    Any submodule whose dotted path starts with one of exclude_prefixes is
    left in full precision (models a TRT 'keep this layer higher-precision'
    config — e.g. the value head or the input conv)."""
    for name, child in list(module.named_children()):
        path = f"{_path}{name}"
        if any(path.startswith(p) for p in exclude_prefixes):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, FP4Conv2d(child))
        elif isinstance(child, nn.Linear):
            setattr(module, name, FP4Linear(child))
        else:
            fp4_wrap(child, exclude_prefixes, path + ".")


# ── Metrics ────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_infer(net, x, device, dtype):
    net = net.to(device=device, dtype=dtype).eval()
    xb = x.to(device=device, dtype=dtype)
    logits, wdl_logits, mlh, _ = net(xb)
    p = F.softmax(logits.float(), dim=1)
    wdl = F.softmax(wdl_logits.float(), dim=1)
    q = wdl[:, 0] - wdl[:, 2]
    return p, q, wdl


def compare(ref_p, ref_q, p, q, tag):
    top1 = (p.argmax(1) == ref_p.argmax(1)).float().mean().item()
    ref_top3 = ref_p.topk(3, dim=1).indices
    p_top1 = p.argmax(1, keepdim=True)
    in_top3 = (p_top1 == ref_top3).any(1).float().mean().item()
    kl = (ref_p * (ref_p.clamp_min(1e-9).log()
                   - p.clamp_min(1e-9).log())).sum(1).mean().item()
    q_mae = (q - ref_q).abs().mean().item()
    q_max = (q - ref_q).abs().max().item()
    print(f"  {tag:12s} | top1-match {top1*100:5.1f}%  "
          f"top1∈ref-top3 {in_top3*100:5.1f}%  "
          f"policyKL {kl:.4f}  Q-MAE {q_mae:.4f}  Q-max {q_max:.4f}")
    return dict(top1=top1, kl=kl, q_mae=q_mae, q_max=q_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--buffer", default=None,
                    help="replay .npz to draw real positions from")
    ap.add_argument("-n", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    net0, sd = build_net_from_state_dict(sd)
    net0.load_state_dict(sd, strict=False)
    net0.eval()
    nparams = sum(p.numel() for p in net0.parameters())
    print(f"model: {net0.num_channels}ch × {net0.num_res_blocks}bl  "
          f"{nparams/1e6:.1f}M params  | n={args.n} positions")

    # real positions
    if args.buffer and os.path.exists(args.buffer):
        z = np.load(args.buffer, allow_pickle=True)
        boards = z["boards"]
        idx = np.random.default_rng(0).choice(len(boards), args.n, replace=False)
        x = torch.from_numpy(
            boards[np.sort(idx)].astype(np.float32)
        ).reshape(args.n, CapablancaNet.INPUT_PLANES,
                  CapablancaNet.BOARD_H, CapablancaNet.BOARD_W)
        print(f"positions: real, from {args.buffer}")
    else:
        x = torch.randn(args.n, CapablancaNet.INPUT_PLANES,
                        CapablancaNet.BOARD_H, CapablancaNet.BOARD_W)
        print("positions: RANDOM (no buffer given — accuracy numbers are upper-bound noise)")

    import copy
    # FP32 reference
    ref_p, ref_q, _ = run_infer(copy.deepcopy(net0), x, device, torch.float32)
    print("\nDrift vs FP32 reference:")

    # BF16 baseline (what self-play already runs)
    bf_p, bf_q, _ = run_infer(copy.deepcopy(net0), x, device, torch.bfloat16)
    compare(ref_p, ref_q, bf_p, bf_q, "BF16")

    # NVFP4 fake-quant (weights+acts), everything — worst case floor
    net_fp4 = copy.deepcopy(net0)
    fp4_wrap(net_fp4)
    fp4_p, fp4_q, _ = run_infer(net_fp4, x, device, torch.float32)
    compare(ref_p, ref_q, fp4_p, fp4_q, "NVFP4-all")

    # NVFP4 but keep sensitive layers high-precision (realistic TRT config):
    # value head (Q matters a lot), input conv (raw sparse planes), and the
    # policy head's final projection (low-rank → sensitive).
    net_mix = copy.deepcopy(net0)
    fp4_wrap(net_mix, exclude_prefixes=("value_head", "input_conv",
                                        "policy_head.4", "mlh_head"))
    mix_p, mix_q, _ = run_infer(net_mix, x, device, torch.float32)
    compare(ref_p, ref_q, mix_p, mix_q, "NVFP4-mixed")

    print("\nNote: NVFP4 numbers are a simulation of Blackwell FP4 tensor-core")
    print("math; real TensorRT may differ slightly (kernel rounding, which")
    print("layers it keeps in higher precision). This is the accuracy gate.")


if __name__ == "__main__":
    main()
