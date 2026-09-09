"""Заучивает сеть буфер или учится обобщать.

Позиции в буфере лежат в порядке появления: старые прошли через много шагов
градиента, свежие — почти ни через один. Одна и та же сеть на тех и других.
Большой разрыв = заучивание; тогда `value_loss` в логе обучения измеряет
память, а не силу, и доверять ему нельзя.

Замер 2026-09-09 на прогоне v6 (он потерял 198 Elo за 30 итераций):
    старые 0.163 / свежие 0.891 — разрыв 5.5x
    configA, не видевшая ни одной позиции: 0.708 / 0.836 — разрыва нет
"""
import argparse, sys
import numpy as np, torch, torch.nn.functional as F
from model import build_net_from_state_dict, CapablancaNet
from train import unpack_policy, value_draw_to_wdl


def slice_rows(d, rows):
    b = d["boards"][rows].astype(np.float32).reshape(
        -1, CapablancaNet.INPUT_PLANES, CapablancaNet.BOARD_H, CapablancaNet.BOARD_W)
    pol = np.stack([unpack_policy((d["pol_idx"][r], d["pol_val"][r])) for r in rows])
    wdl = np.stack([value_draw_to_wdl(float(d["values"][r]), float(d["draws"][r]))
                    for r in rows])
    return torch.from_numpy(b), torch.from_numpy(pol), torch.from_numpy(wdl)


def load(path, dev):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    w = raw.get("ema") or raw.get("model") or raw
    net, sd = build_net_from_state_dict(w)
    res = net.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        # Молчаливая потеря тензоров уже стоила проекту ложного вывода
        # «голова мертва»: сеть тогда осталась со случайными весами.
        sys.exit(f"{path}: не сошлись веса, пропущено {len(res.missing_keys)}, "
                 f"лишних {len(res.unexpected_keys)}")
    return net.to(dev).eval()


@torch.no_grad()
def losses(net, data, dev):
    b, pol, wdl = data
    ps, vs = [], []
    for i in range(0, b.shape[0], 256):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, wdl_logits, _, _ = net(b[i:i + 256].to(dev))
        ps.append(-(pol[i:i + 256].to(dev) * F.log_softmax(logits.float(), 1)).sum(1))
        vs.append(-(wdl[i:i + 256].to(dev) * F.log_softmax(wdl_logits.float(), 1)).sum(1))
    return torch.cat(ps).mean().item(), torch.cat(vs).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("buffer")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--n", type=int, default=3072)
    a = ap.parse_args()

    dev = "cuda"
    d = np.load(a.buffer)
    n = d["values"].shape[0]
    N = min(a.n, n // 3)
    old = slice_rows(d, np.arange(N))
    new = slice_rows(d, np.arange(n - N, n))

    print(f"{n:,} позиций в буфере, по {N} с каждого края\n")
    print(f"{'чекпоинт':<28} {'policy старые':>13} {'свежие':>8} "
          f"{'value старые':>13} {'свежие':>8} {'разрыв':>7}")
    for p in a.checkpoints:
        net = load(p, dev)
        po, vo = losses(net, old, dev)
        pn, vn = losses(net, new, dev)
        print(f"{p.split('/')[-1]:<28} {po:>13.3f} {pn:>8.3f} "
              f"{vo:>13.3f} {vn:>8.3f} {vn / max(vo, 1e-6):>6.1f}x")
        del net
        torch.cuda.empty_cache()
    print("\nразрыв ~1x = обобщает, 2x+ = заучивает буфер")


if __name__ == "__main__":
    main()
