"""
Изолированный стенд: измеряет вклад PCIe-трансфера в _infer и потенциал трёх
улучшений на реальной CapablancaNet и реальных размерах батча.

Варианты:
  V0 baseline    — как сейчас: softmax(fp32) -> .cpu().numpy(), вход bf16 H2D
  V1 bf16-policy — политика отдаётся на CPU в fp16 (вдвое меньше байт D2H)
  V2 sparse-out  — на GPU gather только нужных (~K легальных) индексов на лист,
                   трансфер K вместо 7000 (потолок sparse-вывода политики)
  V3 double-buf  — перекрытие H2D+forward+D2H следующего батча с CPU-постобработкой
                   текущего (два CUDA-стрима, два pinned-буфера вывода)

Запуск:  python bench_transfer.py
"""
import sys, time, numpy as np, torch
import torch._dynamo
sys.path.insert(0, "../../python_src")
from model import CapablancaNet

DEV = "cuda"
IP, H, W = CapablancaNet.INPUT_PLANES, 8, 10
POLICY = 7000
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def make_net(ch=256, rb=15, tb=4):
    net = CapablancaNet(num_channels=ch, num_res_blocks=rb,
                        num_transformer_blocks=tb, transformer_heads=8).to(DEV).eval()
    return net.to(memory_format=torch.channels_last).to(torch.bfloat16)


def synth_input(n):
    """Имитация выхода Rust collect_leaves: (n, IP*H*W) float32 на хосте (pinned)."""
    arr = torch.randn(n, IP, H, W, dtype=torch.float32).contiguous()
    return arr


@torch.no_grad()
def run(net, mode, n, iters=25, warm=10, K=48):
    # pinned bf16 буфер входа (как в проекте), channels_last
    pin_in = torch.empty(n, IP, H, W, pin_memory=True, dtype=torch.bfloat16,
                         memory_format=torch.channels_last)
    host = synth_input(n)
    # для sparse: случайные "легальные" индексы на лист (K штук)
    gather_idx = torch.randint(0, POLICY, (n, K), device=DEV)
    # pinned выходные буферы (для double-buffer)
    out_pol = torch.empty(n, POLICY, pin_memory=True, dtype=torch.float32)

    def one_call():
        pin_in.copy_(host)                                  # CPU fp32->bf16 cast
        x = pin_in.to(DEV, non_blocking=True)               # H2D ~bf16
        out = net(x)
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.softmax(logits.float(), dim=1)
        if mode == "V0":
            p = probs.cpu().numpy()                         # D2H fp32 7000
        elif mode == "V1":
            p = probs.to(torch.float16).cpu().numpy()       # D2H fp16 7000
        elif mode == "V2":
            g = torch.gather(probs, 1, gather_idx)          # (n,K) на GPU
            p = g.cpu().numpy()                             # D2H fp32 K
        return p

    for _ in range(warm):
        one_call()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        one_call()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / iters
    return dt


@torch.no_grad()
def run_doublebuf(net, n, iters=25, warm=10, cpu_ms=0.0):
    """V3: пока GPU считает батч i+1, CPU обрабатывает вывод батча i.
    cpu_ms — имитация CPU-работы (collect_leaves+apply+numpy) на батч."""
    pin_in = torch.empty(n, IP, H, W, pin_memory=True, dtype=torch.bfloat16,
                         memory_format=torch.channels_last)
    host = synth_input(n)
    s_compute = torch.cuda.Stream()
    bufs = [torch.empty(n, POLICY, pin_memory=True, dtype=torch.float32) for _ in range(2)]
    evts = [torch.cuda.Event() for _ in range(2)]

    def issue(slot):
        with torch.cuda.stream(s_compute):
            pin_in.copy_(host)
            x = pin_in.to(DEV, non_blocking=True)
            out = net(x)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits.float(), dim=1)
            bufs[slot].copy_(probs, non_blocking=True)
            evts[slot].record(s_compute)

    def cpu_work():
        if cpu_ms > 0:
            end = time.perf_counter() + cpu_ms / 1000.0
            x = 0.0
            while time.perf_counter() < end:
                x += 1.0  # busy-wait имитация CPU-стадии

    for _ in range(warm):
        issue(0); evts[0].synchronize()
    torch.cuda.synchronize()
    t = time.perf_counter()
    issue(0)
    for i in range(iters):
        slot = i % 2
        nxt = (i + 1) % 2
        issue(nxt)                 # GPU стартует следующий батч...
        cpu_work()                 # ...пока CPU обрабатывает текущий
        evts[slot].synchronize()
    evts[(iters) % 2].synchronize()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / iters
    return dt


def main():
    ITERS, WARM = 10, 5
    for (ch, tag) in [(256, "256ch×15+4T"), (384, "384ch×15+4T")]:
        net = make_net(ch=ch)
        params = sum(p.numel() for p in net.parameters()) / 1e6
        print(f"\n=== {tag}  ({params:.0f}M)  GPU={torch.cuda.get_device_name(0)} ===", flush=True)
        for n in (12288,):
            base = run(net, "V0", n, iters=ITERS, warm=WARM)
            v1 = run(net, "V1", n, iters=ITERS, warm=WARM)
            v2 = run(net, "V2", n, iters=ITERS, warm=WARM)
            print(f"  batch {n}:", flush=True)
            print(f"    V0 baseline (fp32 policy)  : {base*1000:7.1f} ms  {n/base:8.0f} leaves/s  (1.00x)", flush=True)
            print(f"    V1 bf16 policy D2H         : {v1*1000:7.1f} ms  {n/v1:8.0f} leaves/s  ({base/v1:.2f}x)", flush=True)
            print(f"    V2 sparse-out (gather K=48): {v2*1000:7.1f} ms  {n/v2:8.0f} leaves/s  ({base/v2:.2f}x)", flush=True)
            # double-buffer: имитируем CPU-стадию = ~13% от baseline (collect+apply из профиля)
            cpu_ms = base * 1000 * 0.13
            v3 = run_doublebuf(net, n, iters=ITERS, warm=WARM, cpu_ms=cpu_ms)
            # сравниваем с baseline+CPU (последовательно)
            seq = base + cpu_ms / 1000.0
            print(f"    V3 double-buf (hide CPU {cpu_ms:.0f}ms): {v3*1000:7.1f} ms  {n/v3:8.0f} leaves/s  ({seq/v3:.2f}x vs seq)", flush=True)
            # Компилированный полный пайплайн (решающее сравнение: compile end-to-end)
            torch._dynamo.reset()
            netc = torch.compile(net, mode="default", dynamic=False)
            vc = run(netc, "V0", n, iters=ITERS, warm=WARM + 6)
            print(f"    Vc compile(default) full   : {vc*1000:7.1f} ms  {n/vc:8.0f} leaves/s  ({base/vc:.2f}x vs eager)", flush=True)
            vc2 = run(netc, "V2", n, iters=ITERS, warm=WARM + 6)
            print(f"    Vc+sparse (compile+gather) : {vc2*1000:7.1f} ms  {n/vc2:8.0f} leaves/s  ({base/vc2:.2f}x vs eager)", flush=True)
        del net; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
