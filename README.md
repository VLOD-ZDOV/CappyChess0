# ♛ Capablanca Chess Zero

**An AlphaZero-style engine trained entirely from self-play** — no human games,
no opening books, no handcrafted evaluation. The network discovers everything on
its own board.

The project has three parts:

| Part | Stack | Role |
|------|-------|------|
| **Engine** | Rust + PyO3 | move generation, rules, MCTS primitives |
| **Training** | Python + PyTorch | self-play, replay buffer, network training |
| **Analysis** | PyQt5 | a Nibbler-style GUI for reviewing games |

---

## ✨ Highlights

- **A pure AlphaZero loop**: self-play → replay buffer → training → a stronger network.
- **Hybrid network**: a convolutional ResNet trunk with Squeeze-Excitation,
  topped by Transformer blocks with a Relative Position Bias.
- **Four output heads**: policy, value (WDL), moves-left, and a training-only
  "future move" head.
- **Transposition-aware MCTS**: positions reached by different move orders are
  merged into a single search node — something Lc0 deliberately avoids.
- **Curriculum vs. Fairy-Stockfish**: an adaptive teacher that tracks the
  network's current strength.

---

## 🔄 How it works

One training iteration is a closed loop:

```
        ┌─────────────────────────────────────────────┐
        │                                             │
   self-play games           train on sampled         │
   (batched MCTS +     ─────► mini-batches      ──────►│ new weights
    the current net)         (policy + value +        │
        ▲                     MLH + future loss)      │
        │                                             │
        └──────────── replay buffer (FIFO) ◄──────────┘
```

1. **Self-play.** The current network plays a large batch of games against
   itself. Every move is chosen by an MCTS search guided by the network's
   policy and value outputs. The search visit counts become the *policy target*;
   the game result (mixed with bootstrap values) becomes the *value target*.
2. **Replay buffer.** Positions from recent iterations are kept in a FIFO buffer
   and sampled with win/draw/loss balancing.
3. **Training.** The network is trained to predict the search policy, the game
   outcome, the moves-left estimate, and the future move — a multi-task loss.
4. **Repeat.** The improved network feeds the next round of self-play. Strength
   compounds iteration over iteration.

Optionally, a fraction of games are played against **Fairy-Stockfish** as an
external teacher (see *Curriculum* below).

---

## 🧠 Network architecture — `CapablancaNet`

### Input encoding — 139 planes

The board is fed to the network as a `(139, 8, 10)` tensor, **canonically
flipped** so the side to move always "plays up". This means the network only
ever has to learn one point of view.

```
139 planes = 8 history positions × 17  +  3 meta planes

per history slot (newest first):
  8  our pieces   (P N B R Q A C K)
  8  their pieces
  1  repetition flag
meta:
  1  castling rights (4 zones packed)
  1  halfmove clock / 100
  1  all-ones plane (a CNN edge-detection helper)
```

History planes give the network a sense of motion and let it detect repetitions.

### Trunk

```
input  (139, 8, 10)
   │
   ▼  Conv 3×3 → GroupNorm → Mish              ── input tower
   │
   ▼  N × ResBlock                             ── residual tower
   │     Conv→GN→Mish → Conv→GN
   │     Squeeze-Excitation gate (scale + bias)
   │     + residual → Mish
   │
   ▼  reshape to 80 tokens (one per square)
   │
   ▼  K × TransformerBlock                     ── attention tower
   │     Pre-LN → MHA(+RPB) → +residual
   │     Pre-LN → FFN(Mish) → +residual
   │
   ▼  reshape back to (C, 8, 10) → heads
```

A few deliberate choices:

- **GroupNorm, not BatchNorm.** GroupNorm has no running statistics, so it
  behaves identically at batch size 1 (MCTS leaf inference) and does not drift
  when the data distribution shifts during curriculum training.
- **Mish activation** (`x · tanh(softplus(x))`) — smooth and self-gated, more
  stable than ReLU in deep stacks.
- **Squeeze-Excitation** in every residual block. The SE branch outputs both a
  per-channel **scale and a bias**, letting the network re-weight feature maps
  based on global board context.
- **Transformer with a Relative Position Bias (RPB).** The 80 squares become
  tokens, and self-attention connects any two squares in a single step. The bias
  is keyed purely on the *offset* `(Δrank, Δfile)` between two squares: a knight
  jump is the same offset everywhere, so the network learns board geometry from
  the start. RPB costs ~2.3K parameters per block — three orders of magnitude
  cheaper than Smolgen-style alternatives.

### Output heads

| Head | Output | Purpose |
|------|--------|---------|
| **Policy** | 7000 logits | move probabilities — the MCTS prior |
| **Value (WDL)** | 3 logits → softmax | `P(Win)`, `P(Draw)`, `P(Loss)`; the scalar score is `Q = P(Win) − P(Loss)` |
| **Moves-Left** | 1 scalar (sigmoid) | estimated half-moves remaining — lets MCTS prefer faster wins / slower losses |
| **Future move** | 7000 logits | predicts *our* move two plies ahead; **training only** — an auxiliary task that pushes the trunk toward planning-oriented features |

The policy and future heads use a 32-channel `1×1` bottleneck before the final
linear layer. That width is deliberate: a narrower bottleneck makes the
7000-way output physically low-rank and blunts tactical sharpness.

A WDL value head (rather than a single `tanh` scalar) gives cleaner gradients —
the network explicitly separates "sharp, double-edged" from "dead drawn", and
trains under cross-entropy instead of MSE.

At init the head output layers are scaled down (`gain = 0.01`), so a fresh
network starts from near-uniform move priors and a flat `[⅓, ⅓, ⅓]` WDL — no
unfounded bias before it has learned anything.

A typical configuration: **256 channels × 15 residual blocks + 4 Transformer
blocks** (8 attention heads).

---

## 🔍 Inside the search (MCTS)

- **PUCT selection** with parameters carried over from Lc0 (`c_puct = 1.745`,
  logarithmic `c_puct` growth, `FPU = 0.330`, virtual loss).
- **Board-less nodes** — a node is ~60 bytes; the position is reconstructed by
  replaying moves, which keeps the tree cache-friendly.
- **Batched search** — every self-play game in an iteration is searched together
  so the GPU sees one large inference batch.
- **Double buffering** — leaf collection for the next batch overlaps with the
  inference of the current one.
- **Tree reuse** — after a move is played the relevant subtree becomes the new
  root; visits are not thrown away.
- **Transposition merging** (in the analysis GUI) — search statistics live on a
  *position* node, priors live on the *edges*, so different move orders that
  reach the same position share one node and pool their visits.

---

## 🧩 Ideas borrowed from Leela Chess Zero

[Leela Chess Zero](https://lczero.org/) is the open AlphaZero-style chess
project. Several of its community-tuned ideas are reused here:

- **PUCT parameters** straight from Lc0's `params.cc` — tuned over millions of
  games, not guessed.
- **WDL value head** — predicting a Win/Draw/Loss triple instead of one scalar.
- **Moves-Left Head** — for converting won positions instead of shuffling.
- **Future-move heads** — an auxiliary planning signal, from Lc0's BT4 networks.
- **Transformer trunk with relative position encoding** — in the spirit of
  Lc0's BT3+ attention nets (RPB used in place of Smolgen).
- **Mish activation** — adopted in Lc0 BT3+.
- **Playout-cap randomization** — most moves use a cheap shallow search; only
  full-search positions are written to the training buffer.
- **KLD early exit** — a search stops once the visit distribution settles.
- **EMA weights** — self-play runs on exponentially-averaged weights for a
  steadier feedback loop.

## 🎛 Ideas borrowed from Nibbler

[Nibbler](https://github.com/rooklift/nibbler) is a well-loved analysis GUI for
Lc0. It is hardwired to an 8×8 board and the UCI protocol, so this network
cannot be plugged into it — but its ideas shaped the project's own GUI
(`python_src/gui.py`):

- a **ranked move infobox** showing `N` (visits), `P` (network prior), `Q`
  (evaluation) and a WDL bar per move;
- **live background analysis** that refreshes continuously;
- an **evaluation bar** and a **per-game win-rate graph**;
- **coloured best-move arrows** drawn on the board with the win-% inside;
- **analysis snapshots** — scrubbing through history shows cached results
  instead of re-searching.

On top of the Nibbler ideas the GUI adds its own: transposition merging in the
search tree, and tree reuse across moves (visit accumulation).

---

## 📁 Project layout

```
rust_engine/
  src/lib.rs         — engine: board, move generation, rules, MCTS primitives
  Cargo.toml         — PyO3 module build
python_src/
  train.py           — training loop + Fairy-Stockfish integration
  mcts.py            — Python wrapper over the Rust MCTS (batching, double buffering)
  model.py           — CapablancaNet (ResNet + Transformer + 4 heads)
  eval.py            — round-robin tournament between checkpoints
  export_onnx.py     — convert a .pth checkpoint into a self-contained .onnx
  onnx_engine.py     — onnxruntime inference backend used by the GUI
  gui.py             — Nibbler-style analysis GUI (PyQt5, GPU via onnxruntime)
  fsf_integration.py — Fairy-Stockfish wrapper
  requirements.txt   — GUI / inference dependencies
```

---

## 🚀 Running from source

This is the raw, run-it-yourself path — install the dependencies and launch the
scripts directly. (For shipping a one-click binary instead, see *Packaging* below.)

There are two dependency sets: the **GUI** needs only a lightweight inference
stack, **training** additionally needs PyTorch.

### Prerequisites

- Python 3.10+
- A Rust toolchain (`rustup`) — to build the engine
- For GPU inference: an up-to-date NVIDIA driver (the CUDA runtime itself ships
  inside the `onnxruntime-gpu` wheel — no CUDA Toolkit install needed)

### 1. Build the Rust engine

The engine compiles into a Python module called `capablanca_engine`. Run inside
`rust_engine/`:

```bash
pip install maturin
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

### 2. Install Python dependencies

GUI / inference only:

```bash
pip install -r python_src/requirements.txt
```

That is `numpy`, `PyQt5` and `onnxruntime-gpu` — for a machine without an NVIDIA
GPU, replace `onnxruntime-gpu` with `onnxruntime`.

Training additionally needs PyTorch — install the build matching your CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Train a network *(optional — needs PyTorch)*

From scratch (self-play):

```bash
python train.py --channels 256 --res-blocks 15 \
                --transformer-blocks 4 --transformer-heads 8 \
                --simulations 1000 --games 2048
```

With a Fairy-Stockfish teacher (curriculum) — the opponent's strength rises when
the win-rate is high and drops when it falls, so the network always plays at the
edge of its ability:

```bash
python train.py --channels 256 --res-blocks 15 \
                --fsf-path ./fairy-stockfish-largeboard_x86-64-bmi2 \
                --curriculum --fsf-nodes-start 1 --curriculum-sp-ratio 0.5
```

### 4. Export the network to ONNX *(needs PyTorch)*

The GUI runs an `.onnx` graph, not a `.pth` checkpoint. Convert once:

```bash
python export_onnx.py checkpoints_big/latest.pth capablanca.onnx
```

This bakes architecture **and** weights into one binary file and pre-applies the
softmax / WDL reduction — so the GUI itself carries no PyTorch dependency.

### 5. Run the analysis GUI

```bash
python gui.py
```

If `capablanca.onnx` sits next to `gui.py` it loads automatically; otherwise
pick one through «Загрузить сеть». GPU acceleration is automatic when an NVIDIA
GPU is present (the status bar shows `GPU · CUDA`), with a silent CPU fallback.

### Tournament between two checkpoints *(needs PyTorch)*

```bash
python eval.py <weights_A.pth> <weights_B.pth> --max 100
```

---

## 📦 Packaging a standalone build

To distribute the GUI without making users install Python and dependencies:

1. Export the final network with `export_onnx.py` (single `capablanca.onnx`).
2. Build the Rust engine for the target OS (`.so` on Linux, `.pyd` on Windows).
3. Bundle with **PyInstaller**:

   ```bash
   pyinstaller --onedir --windowed --name capablanca-gui gui.py
   ```

4. Place `capablanca.onnx` next to the produced executable — the GUI auto-loads it.

`onnxruntime-gpu` carries the CUDA runtime, so the bundle runs GPU-accelerated on
any machine with an NVIDIA driver — no PyTorch, no CUDA Toolkit. Builds are
per-OS (no cross-compilation): build on Linux for Linux, on Windows for Windows.

---

## 📜 Credits

A learning / research project. Ideas and thanks:

- **[AlphaZero](https://www.science.org/doi/10.1126/science.aar6404)** (DeepMind) — the overall method.
- **[Leela Chess Zero](https://lczero.org/)** — MCTS parameters, the WDL / MLH / future heads, the Transformer trunk.
- **[Nibbler](https://github.com/rooklift/nibbler)** — the analysis-GUI concept.
- **[Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish)** — teacher and sparring partner.

---

## Кратко по-русски

AlphaZero-движок, обучаемый **только** на self-play — без человеческих партий и
дебютных книг.

**Цикл обучения:** сеть играет батч партий сама с собой, поиск MCTS даёт
policy-таргеты (визиты) и value-таргеты (результат); позиции копятся в
FIFO-буфере; сеть обучается на мульти-таргет лоссе (policy + value + moves-left
+ future) → новые веса → следующий раунд self-play.

**Сеть `CapablancaNet`:** вход 139 плоскостей (8 досок истории × 17 + 3 мета,
канонический флип под сторону хода) → input-conv → башня ResNet-блоков со
Squeeze-Excitation → блоки Transformer с относительным позиционным смещением
(RPB) → четыре головы:

- **policy** — 7000 логитов, prior для MCTS;
- **value (WDL)** — Win/Draw/Loss, оценка `Q = P(Win) − P(Loss)`;
- **moves-left** — сколько полуходов до конца (доводить выигрыш до мата);
- **future** — ход на 2 полухода вперёд, вспомогательная голова только для
  обучения (тянет ствол к «планирующим» признакам).

Особенности: GroupNorm вместо BatchNorm (корректен при batch=1 и сдвиге
распределения), активация Mish, RPB вместо Smolgen (≈2.3К параметров на блок).

**MCTS:** PUCT с параметрами из Lc0, безбордовые узлы (~60 байт), батчевый
поиск, двойная буферизация, переиспользование дерева, слияние транспозиций (в
GUI). Опционально — curriculum-обучение против Fairy-Stockfish с адаптивной
силой учителя.
