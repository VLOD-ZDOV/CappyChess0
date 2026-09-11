"""Тихие ошибки вычислений на текущей кривой частота/напряжение карты.

Одно и то же вычисление на фиксированных входах в детерминированном режиме.
На исправном железе каждый повтор совпадает с первым бит в бит; любое
расхождение — ошибка вычисления, а не особенность кода. Нужен после каждой
правки кривой или разгона: пограничная точка андервольта в играх работает, а
под длительной тензорной нагрузкой даёт не падение, а испорченные числа.

    python gpu_stability.py checkpoints_v9/latest.pth checkpoints_v9/buffer.npz
    python gpu_stability.py ... --gemm-seconds 1200 --net-seconds 600   # долгий

Два теста: большое матричное умножение в bf16 на тензорных ядрах и прямой
проход самой сети по позициям из буфера. Короткий прогон ловит заметную
нестабильность; редкие ошибки проявляются на прогретой карте за десятки минут.

Пример прогона вместе с обучением: 6335 повторов GEMM и 1395 прогонов сети —
0 расхождений.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch


def run(name, fn, seconds):
    ref = fn()
    reps = bad = 0
    worst = 0.0
    t0 = time.time()
    while time.time() - t0 < seconds:
        out = fn()
        reps += 1
        if not all(torch.equal(a, b) for a, b in zip(out, ref)):
            bad += 1
            worst = max(worst, max((a.float() - b.float()).abs().max().item()
                                   for a, b in zip(out, ref)))
    torch.cuda.synchronize()
    tail = f"   макс. отличие {worst:.3g}" if bad else "   — всё бит в бит"
    print(f"{name:<36} повторов {reps:>6}   расхождений {bad:>4}{tail}", flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="чекпоинт сети (.pth, можно с :ema)")
    ap.add_argument("buffer", help="буфер (.npz) — берутся первые позиции как вход")
    ap.add_argument("--gemm-seconds", type=float, default=120)
    ap.add_argument("--net-seconds", type=float, default=60)
    ap.add_argument("--gemm-size", type=int, default=8192)
    ap.add_argument("--positions", type=int, default=256)
    a = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    dev = "cuda"

    g = torch.Generator(device=dev).manual_seed(0)
    n = a.gemm_size
    A = torch.randn(n, n, device=dev, dtype=torch.bfloat16, generator=g)
    B = torch.randn(n, n, device=dev, dtype=torch.bfloat16, generator=g)
    total = run(f"GEMM {n}² bf16, тензорные ядра", lambda: (A @ B,), a.gemm_seconds)
    del A, B
    torch.cuda.empty_cache()

    from model import build_net_from_state_dict, pick_state_dict, split_weights, CapablancaNet
    path, which = split_weights(a.checkpoint)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    net, sd = build_net_from_state_dict(pick_state_dict(raw, which))
    res = net.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        # Сеть со случайными весами тоже детерминирована, но тест тогда
        # гоняет не ту нагрузку — лучше остановиться.
        sys.exit(f"{path}: не сошлись веса, пропущено {len(res.missing_keys)}, "
                 f"лишних {len(res.unexpected_keys)}")
    net = net.to(dev).eval()
    import buffer_io
    boards = buffer_io.sample_boards(a.buffer, a.positions)
    x = torch.from_numpy(boards.astype(np.float32)).reshape(
        -1, CapablancaNet.INPUT_PLANES, CapablancaNet.BOARD_H,
        CapablancaNet.BOARD_W).to(dev)

    @torch.no_grad()
    def fwd():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p, v, _, _ = net(x)
        return (p, v)

    label = f"сеть, {x.shape[0]} позиций, bf16"
    total += run(label, fwd, a.net_seconds)
    print("\nИТОГ:", "ошибок вычислений не найдено" if total == 0
          else f"{total} расхождений — железо считает с ошибками")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
