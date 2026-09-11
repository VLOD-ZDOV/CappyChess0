#!/usr/bin/env python3
"""Distil a trained net into a different architecture.

The point is to change the shape of the network without paying for the strength
it already has. A from-scratch run would have to re-earn it; here the student
learns to reproduce the teacher's outputs on a fixed position set, which takes
minutes instead of days.

Recipe, from this project's own history (docs/notes/distil.txt): a SHORT cycle
with a LARGE learning rate. The counterexample is on record — 128->256 with
900 steps x 8 epochs at 4e-4 worked; 256->384 with 40K steps at a small lr
destroyed the value head, and the signature was a winrate that FELL as search
got deeper (39% at 200 sims -> 31% at 5000). Verify with --check after.

    python distill.py teacher.pth out.pth --buffer checkpoints/buffer.npz \
        --channels 256 --res-blocks 10 --transformer-blocks 10 --ffn-mult 4 \
        --restricted-policy --abs-pos-embed --wide-value --no-future
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import (CapablancaNet, build_net_from_state_dict, describe_arch,
                   reachable_policy_indices)

PLANES, H, W = 139, 8, 10


def load_teacher(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    raw = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    net, sd = build_net_from_state_dict(raw)
    net.load_state_dict(sd, strict=False)
    return net.to(device).eval(), raw


def warm_start(student, teacher_sd):
    """Copy every tensor whose name and shape still match, plus the policy rows.

    The policy head keeps only the reachable indices, so its final Linear is
    (2672, D) against the teacher's (7000, D) — the rows are sliced rather than
    dropped, which is the whole point of the restriction.
    """
    ssd = student.state_dict()
    copied = skipped = 0
    for k, v in ssd.items():
        t = teacher_sd.get(k)
        if t is not None and t.shape == v.shape:
            v.copy_(t); copied += 1
        else:
            skipped += 1
    for s_key, t_key in (("policy_head.fc.weight", "policy_head.4.weight"),
                         ("policy_head.fc.bias", "policy_head.4.bias")):
        if s_key in ssd and t_key in teacher_sd:
            src = teacher_sd[t_key]
            idx = torch.tensor(reachable_policy_indices(), dtype=torch.long,
                               device=src.device)
            if src.shape[0] == 7000 and ssd[s_key].shape[0] == len(idx):
                ssd[s_key].copy_(src.index_select(0, idx)); copied += 1; skipped -= 1
    # policy_head conv/norm live under body.* in the restricted head
    for i in (0, 1):
        for suf in ("weight", "bias"):
            s_key, t_key = f"policy_head.body.{i}.{suf}", f"policy_head.{i}.{suf}"
            if s_key in ssd and t_key in teacher_sd and ssd[s_key].shape == teacher_sd[t_key].shape:
                ssd[s_key].copy_(teacher_sd[t_key]); copied += 1; skipped -= 1
    student.load_state_dict(ssd)
    return copied, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher"); ap.add_argument("out")
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--positions", type=int, default=0, help="0 = весь буфер")
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--res-blocks", type=int, default=10)
    ap.add_argument("--transformer-blocks", type=int, default=10)
    ap.add_argument("--transformer-heads", type=int, default=8)
    ap.add_argument("--ffn-mult", type=int, default=4)
    ap.add_argument("--restricted-policy", action="store_true")
    ap.add_argument("--abs-pos-embed", action="store_true")
    ap.add_argument("--wide-value", action="store_true")
    ap.add_argument("--no-future", dest="enable_future", action="store_false")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps-per-epoch", type=int, default=900)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--teacher-batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true",
                    help="Считать в fp32. По умолчанию BF16, как в train.py — "
                         "в fp32 те же 8 эпох занимали ~12.9 ГБ видеопамяти "
                         "против примерно вдвое меньшего с BF16.")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    teacher, tsd = load_teacher(a.teacher, dev)
    print(f"учитель:  {describe_arch(teacher)}")

    student = CapablancaNet(
        num_channels=a.channels, num_res_blocks=a.res_blocks,
        num_transformer_blocks=a.transformer_blocks,
        transformer_heads=a.transformer_heads, ffn_mult=a.ffn_mult,
        swiglu=True, qk_norm=True, use_rmsnorm=True, qkv_bias=False,
        enable_future=a.enable_future,
        restricted_policy=a.restricted_policy, abs_pos_embed=a.abs_pos_embed,
        wide_value=a.wide_value).to(dev)
    print(f"студент:  {describe_arch(student)}")
    with torch.no_grad():
        c, s = warm_start(student, tsd)
    print(f"тёплый старт: перенесено {c} тензоров, заново {s}")

    # mmap_mode на .npz NumPy молча игнорирует — поле всегда читалось целиком.
    import buffer_io
    boards = buffer_io.read_field(a.buffer, "boards")
    n = len(boards) if a.positions <= 0 else min(a.positions, len(boards))
    print(f"позиций: {n:,} из {a.buffer}")

    idx = torch.tensor(reachable_policy_indices(), dtype=torch.long, device=dev)
    K = len(idx)
    # Цели учителя считаем один раз: только по достижимым индексам (2672 из
    # 7000) — это и память втрое, и ровно то, что студент способен выдать.
    t_pol = torch.empty((n, K), dtype=torch.float16)
    t_wdl = torch.empty((n, 3), dtype=torch.float16)
    t0 = time.time()
    amp = torch.autocast("cuda", dtype=torch.bfloat16,
                         enabled=(not a.fp32 and dev.type == "cuda"))
    with torch.inference_mode(), amp:
        for i in range(0, n, a.teacher_batch):
            j = min(i + a.teacher_batch, n)
            xb = torch.from_numpy(np.asarray(boards[i:j])).to(dev).float().view(-1, PLANES, H, W)
            pl, wl, _, _ = teacher(xb)
            t_pol[i:j] = F.softmax(pl.float().index_select(1, idx), dim=1).half().cpu()
            t_wdl[i:j] = F.softmax(wl.float(), dim=1).half().cpu()
            if i % (a.teacher_batch * 50) == 0:
                print(f"  разметка {j:,}/{n:,}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"цели учителя готовы за {time.time()-t0:.0f}s")
    del teacher; torch.cuda.empty_cache()

    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * a.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total,
                                                pct_start=0.1)
    rng = np.random.default_rng(0)
    step = 0
    student.train()
    for ep in range(a.epochs):
        pl_sum = vl_sum = 0.0
        for _ in range(a.steps_per_epoch):
            sel = np.sort(rng.choice(n, a.batch, replace=False))
            xb = torch.from_numpy(np.asarray(boards[sel])).to(dev).float().view(-1, PLANES, H, W)
            tp = t_pol[sel].to(dev).float()
            tw = t_wdl[sel].to(dev).float()
            with amp:
                pl, wl, _, _ = student(xb)
            # Лоссы — в fp32: log_softmax по 2672 классам в bf16 теряет точность
            # там, где она и нужна, на хвосте распределения.
            logp = F.log_softmax(pl.float().index_select(1, idx), dim=1)
            p_loss = -(tp * logp).sum(dim=1).mean()
            v_loss = -(tw * F.log_softmax(wl.float(), dim=1)).sum(dim=1).mean()
            loss = p_loss + a.value_weight * v_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step(); sched.step(); step += 1
            pl_sum += p_loss.item(); vl_sum += v_loss.item()
        print(f"  эпоха {ep+1}/{a.epochs}: policy_CE={pl_sum/a.steps_per_epoch:.4f} "
              f"value_CE={vl_sum/a.steps_per_epoch:.4f} lr={sched.get_last_lr()[0]:.2e}",
              flush=True)

    torch.save({"iteration": 0, "model": student.state_dict(),
                "ema": {k: v.clone() for k, v in student.state_dict().items()},
                "metrics": {}}, a.out)
    print(f"\nсохранено: {a.out}")


if __name__ == "__main__":
    main()
