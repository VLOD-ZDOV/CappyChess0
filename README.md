# ♛ Capablanca Chess Zero

**An AlphaZero-style engine trained entirely from self-play** — no human games,
no opening books, no handcrafted evaluation. The network discovers everything
on its own board.

The project has three parts:

| Part | Stack | Role |
|------|-------|------|
| **Engine** | Rust + PyO3 | move generation, rules, MCTS primitives |
| **Training** | Python + PyTorch | self-play, replay buffer, network training |
| **Analysis** | PyQt5 + onnxruntime | a Nibbler-style GUI for reviewing games |

---

## ⚡ Quick start

Step by step, in order. Commands are given for **Linux/macOS** and **Windows**;
pick your column and keep to it. Everything runs from the repository root
unless a step says otherwise.

### 0. The game, in thirty seconds

The board is **10×8** — files `a`…`j`, ranks `1`…`8`. Two pieces do not exist in
normal chess:

| | moves as |
|---|---|
| **A** — archbishop | bishop **+** knight |
| **C** — chancellor | rook **+** knight |

Opening rank: `R N A B Q K B C N R`, so the king starts on **f1** (`f8` for
Black).

**Castling moves the king three squares, not two:**

| | king | rook | must be empty |
|---|---|---|---|
| kingside | `f1` → `i1` | `j1` → `h1` | `g1 h1 i1` |
| queenside | `f1` → `c1` | `a1` → `d1` | `b1 c1 d1 e1` |

Type castling as the king move: `f1i1` or `f1c1`. Moves are UCI (`e2e4`);
promotion adds a suffix — `q r b n` as usual, plus **`a`** archbishop and
**`c`** chancellor (`e7e8a`).

### 1. Prerequisites

- **Python 3.10+**
- **Rust** — [rustup.rs](https://rustup.rs). The search engine is a compiled
  Rust module; there is no pure-Python fallback.
- **NVIDIA GPU with CUDA** for training. Playing works on CPU, just slower.

### 2. Create the environment

<table><tr><th>Linux / macOS</th><th>Windows (PowerShell)</th></tr><tr><td>

```bash
python -m venv venv
source venv/bin/activate
# fish: source venv/bin/activate.fish
pip install maturin numpy torch
```

</td><td>

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install maturin numpy torch
```

</td></tr></table>

### 3. Build the Rust engine

It installs as a Python module named `capablanca_engine`. **Nothing else
imports until this succeeds.**

<table><tr><th>Linux / macOS</th><th>Windows</th></tr><tr><td>

```bash
cd rust_engine
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
  maturin develop --release
cd ..
```

</td><td>

```powershell
cd rust_engine
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
maturin develop --release
cd ..
```

</td></tr></table>

Check it:

```
python -c "import capablanca_engine; print('ok')"
```

> **The single most common error is `No module named 'capablanca_engine'`.**
> It almost always means the environment is not active and the *system* Python
> ran instead. Either activate it (step 2) or call the interpreter by full
> path: `venv/bin/python …` on Linux, `venv\Scripts\python.exe …` on Windows.

### 4. Get a network

Download a `.pth` from [Releases](../../releases), or train your own (step 6).
Put it anywhere; the commands below take the path as an argument.

### 5. Play against it

**In the GUI** — board, arrows ranked by the network's own preference,
evaluation bar:

```
pip install -r python_src/requirements.txt   # PyQt5 + onnxruntime-gpu
pip install onnx onnxscript
python python_src/gui.py
```

Open your `.pth` straight from *Load network* — the GUI converts it to ONNX on
first open and caches the result next to the checkpoint, so later opens are
instant. A `capablanca.onnx` sitting next to `gui.py` loads automatically at
startup.

Convert by hand instead, if you prefer, or if PyTorch is not installed
alongside the GUI:

```
python python_src/export_onnx.py weights.pth capablanca.onnx
```

Export from a *snapshot*, never from a `latest.pth` that a training run is
still overwriting.

**In the terminal** — no GUI, no display:

```
cd python_src
python play_cli.py ../weights.pth --sims 800 --side white
```

`moves` lists legal moves, `quit` exits. One move per invocation, with the game
kept in a state file (handy over ssh, or for scripting an opponent):

```
python play_cli.py ../weights.pth --new --side white --move e2e4
python play_cli.py ../weights.pth --move d2d4
```

After each move the engine prints its own evaluation: `Q` is **from the
network's own point of view** — `Q=+0.9` means the network thinks *it* is
winning.

### 6. Train

```
cd python_src
python -u train.py \
    --channels 256 --res-blocks 10 \
    --transformer-blocks 10 --transformer-heads 8 --ffn-mult 4 \
    --swiglu --qk-norm --rmsnorm --no-qkv-bias \
    --restricted-policy --abs-pos-embed --wide-value --no-future \
    --games 256 --mcts-batch 128 --mcts-parallel-sims 8 \
    --simulations 400 --fast-simulations 100 \
    --batch-size 512 --train-steps 40 \
    --buffer-max 200000 --lr 2e-4 \
    --save-every 5 --checkpoint-dir checkpoints
```

On Windows drop the backslashes and put it on one line, or use a backtick `` ` ``
for line continuation.

Three settings wreck training *quietly* — no crash, just a weaker network:

- **`--mcts-parallel-sims`** — how many leaves go to the GPU per call.
  `ceil(--simulations / --mcts-parallel-sims)` is the number of *sequential*
  PUCT rounds. Below about 12 rounds virtual loss spreads visits flat across
  the root and the stored policy target degenerates towards uniform.
  **Keep that ratio at 12 or above.**
- **`--train-steps` against `--games`** — the ratio
  `train_steps × batch_size / new_positions_per_iteration` is how many times
  the gradient walks over each position. Around **5 is healthy**; at 38 the
  network memorises the replay buffer instead of learning chess, and loses
  strength while its training loss falls towards zero. Leela keeps this
  between 1 and 4.
- **Architecture flags must match exactly when resuming.** The network is
  rebuilt from the flags and weights are loaded with `strict=False`, so a
  mismatch does not raise — it silently drops tensors.

Run long training detached, not inside a terminal multiplexer pane — if the
multiplexer server dies it takes the run with it:

```bash
setsid nohup python -u train.py … >> train.log 2>&1 < /dev/null &
tail -f train.log
```

### 7. Check that it is actually learning

The training loss will not tell you — a *falling* loss is compatible with a
network getting weaker. Three commands, cheapest first:

```
python overfit_check.py checkpoints/buffer.npz checkpoints/model_iter*.pth
```

Runs the network over the oldest and the newest positions in the replay buffer.
Old ones have had many gradient passes, fresh ones almost none, so a **gap of
2× or more means the network is memorising** the buffer rather than
generalising — and its reported `value_loss` is measuring memory. Takes seconds.

```
python policy_health.py checkpoints/model_iter*.pth
```

Reports `max(p)·n_legal` — how much sharper than uniform the best move is.
`1.0` is uniform, `1.04` a randomly initialised head. Catches a policy head
being flattened, which looks like a loss plateau. Note that *sharper is not
stronger*: in this project sharpness has gone up while strength went down.

```
python eval.py old.pth new.pth --games 200 --simulations 200 \
       --mcts-parallel-sims 8
```

Head-to-head is **the only trustworthy strength measure**. Score against a
random mover ranks networks in the *wrong order* — a stronger network draws
more against a random opponent because it fails to convert, not because it
plays worse.

For an absolute reading use the Fairy-Stockfish ladder with `UCI_Elo`
(`fsf_ladder.py`). Do not use `--fsf-nodes`: node limits barely weaken the
engine, so every network here scored 0% against it regardless of strength.

---

## ✨ Highlights

- **A pure AlphaZero loop**: self-play → replay buffer → training → a stronger
  network.
- **Hybrid network**: a convolutional ResNet trunk with Squeeze-Excitation,
  topped by Transformer blocks with a Relative Position Bias. Attention runs
  through PyTorch's fused `scaled_dot_product_attention` — picks FlashAttention
  or memory-efficient backend automatically.
- **Four output heads**: policy, value (WDL), moves-left, and a training-only
  "future move" head.
- **Transposition-aware MCTS** (in the GUI): positions reached by different
  move orders are merged into a single search node — something Lc0
  deliberately avoids.
- **Tree reuse across moves**: after a played move the relevant sub-tree
  becomes the new root; statistics carry over.
- **Curriculum vs. Fairy-Stockfish**: an adaptive teacher that tracks the
  network's current strength.
- **Self-contained GUI**: a single `.onnx` file + Rust engine binary. No
  PyTorch needed at inference time. Optional CUDA acceleration via
  `onnxruntime-gpu` (CUDA runtime ships inside the wheel).

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
   policy and value outputs. The search visit counts become the *policy
   target*; the game result (mixed with bootstrap values) becomes the
   *value target*.
2. **Replay buffer.** Positions from recent iterations are kept in a FIFO
   buffer and sampled with win/draw/loss balancing.
3. **Training.** The network is trained to predict the search policy, the game
   outcome, the moves-left estimate, and the future move — a multi-task loss.
4. **Repeat.** The improved network feeds the next round of self-play.
   Strength compounds iteration over iteration.

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

History planes give the network a sense of motion and let it detect
repetitions.

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
   │     Pre-LN → MHA(+RPB via SDPA) → +residual
   │     Pre-LN → FFN(Mish) → +residual
   │
   ▼  reshape back to (C, 8, 10) → heads
```

Deliberate choices:

- **GroupNorm, not BatchNorm.** GroupNorm has no running statistics, so it
  behaves identically at batch size 1 (MCTS leaf inference) and does not drift
  when the data distribution shifts during curriculum training.
- **Mish activation** (`x · tanh(softplus(x))`) — smooth and self-gated, more
  stable than ReLU in deep stacks.
- **Squeeze-Excitation** in every residual block. The SE branch outputs both a
  per-channel **scale and a bias**, letting the network re-weight feature maps
  based on global board context.
- **Transformer with a Relative Position Bias (RPB).** The 80 squares become
  tokens, and self-attention connects any two squares in a single step. The
  bias is keyed purely on the *offset* `(Δrank, Δfile)` between two squares:
  a knight jump is the same offset everywhere, so the network learns board
  geometry from the start. RPB costs ~2.3K parameters per block — three orders
  of magnitude cheaper than Smolgen-style alternatives.
- **`scaled_dot_product_attention`** instead of hand-rolled `matmul + softmax`.
  PyTorch picks the best backend (FlashAttention, memory-efficient, or math)
  based on shapes and hardware. RPB rides in as an additive `attn_mask`.

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

A WDL value head (rather than a single `tanh` scalar) gives cleaner gradients
— the network explicitly separates "sharp, double-edged" from "dead drawn",
and trains under cross-entropy instead of MSE.

At init the head output layers are scaled down (`gain = 0.01`), so a fresh
network starts from near-uniform move priors and a flat `[⅓, ⅓, ⅓]` WDL — no
unfounded bias before it has learned anything.

A typical configuration: **256 channels × 15 residual blocks + 4 Transformer
blocks** (8 attention heads).

---

## 🔍 Inside the search (MCTS)

- **PUCT selection** with parameters carried over from Lc0 (`c_puct = 1.745`,
  logarithmic `c_puct` growth, `FPU = 0.330`, virtual loss).
- **Virtual loss as a real divergence force.** Each in-flight leaf marks its
  ancestors with a `-1` penalty in the numerator of Q. Parallel selects
  therefore scatter across distinct moves instead of all collapsing onto the
  current best line. The vloss is removed exactly once, when the leaf's NN
  evaluation comes back — never inside `backup`.
- **Board-less nodes** — a node is ~60 bytes; the position is reconstructed by
  replaying moves, which keeps the tree cache-friendly.
- **Batched search** — every self-play game in an iteration is searched
  together so the GPU sees one large inference batch.
- **Bucket-padded inference** for `torch.compile(mode='reduce-overhead')`.
  Input batch is padded up to the next power-of-two so only ~10 distinct
  shapes ever reach the compiled graph — CUDA Graphs stick instead of
  recompiling on every step.
- **Tree reuse** — after a move is played the relevant subtree becomes the
  new root; visits are not thrown away.
- **Transposition merging** (in the analysis GUI) — search statistics live on
  a *position* node, priors live on the *edges*, so different move orders
  that reach the same position share one node and pool their visits.
- **KLD early-exit** — search stops once the visit distribution settles
  (Lc0-style).
- **Contempt** (optional) — Q is biased by `contempt * draw_prob`. Positive
  contempt makes the engine avoid draws (good vs weaker external opponents);
  negative makes it accept them.
- **Add-dirichlet toggle.** Self-play turns it on (exploration); eval / FSF /
  lagged play turn it off for a deterministic measurement of the network's
  actual choice.

---

## 🎛 GUI features

The analysis GUI (`python_src/gui.py`) loads a single `.onnx` graph and runs
through `onnxruntime`. No PyTorch dependency at inference time.

- **Two-pass arrow rendering** — Nibbler-style. All arrows drawn first,
  labels on top in a second pass. Width and alpha scale with rank, so the
  best move visually dominates. Labels are anchored to the destination
  square (not over the shaft) and never overlap on crossing arrows.
- **Mate display** in the side eval bar. When the top move's Q is near ±1
  and the network's draw probability is low, the bar shows `M<n>` /
  `-M<n>` in white POV instead of a percentage.
- **Tree reuse across moves.** The transposition table persists between
  searches; when a new search starts at a position already in the table, it
  picks up the accumulated statistics.
- **Smart prune**, not `clear`. When the ttable exceeds 500 000 nodes, only
  the sub-tree reachable from the current position is kept. The rest is
  dropped — no work the next search would have reused is lost.
- **Bounded NN cache** (~150 000 entries by default, ~4 GB RAM ceiling).
  Long sessions no longer balloon memory usage.
- **c_puct and contempt sliders** in the toolbar. For analysis, raise
  c_puct to 2.5-3.0 to widen the search; flip contempt positive to bias
  away from draws.
- **Live PV** per move (10 plies deep) inside the info box.

---

## 🧩 Ideas borrowed from Leela Chess Zero

[Leela Chess Zero](https://lczero.org/) is the open AlphaZero-style chess
project. Several of its community-tuned ideas are reused here:

- **PUCT parameters** straight from Lc0's `params.cc` — tuned over millions
  of games, not guessed.
- **WDL value head** — predicting a Win/Draw/Loss triple instead of one
  scalar.
- **Moves-Left Head** — for converting won positions instead of shuffling.
- **Future-move heads** — an auxiliary planning signal, from Lc0's BT4
  networks.
- **Transformer trunk with relative position encoding** — in the spirit of
  Lc0's BT3+ attention nets (RPB used in place of Smolgen).
- **Mish activation** — adopted in Lc0 BT3+.
- **Playout-cap randomization** — most moves use a cheap shallow search;
  only full-search positions are written to the training buffer.
- **KLD early exit** — a search stops once the visit distribution settles.
- **EMA weights** — self-play runs on exponentially-averaged weights for a
  steadier feedback loop.
- **Bounds propagation** (StickyEndgames) — proven terminal nodes bias
  selection through ±100 score shifts so the search commits to known
  outcomes.

## 🎛 Ideas borrowed from Nibbler

[Nibbler](https://github.com/rooklift/nibbler) is a well-loved analysis GUI
for Lc0. It is hardwired to an 8×8 board and the UCI protocol, so this
network cannot be plugged into it — but its ideas shaped the project's own
GUI:

- a **ranked move infobox** showing `N` (visits), `P` (network prior), `Q`
  (evaluation) and a WDL bar per move;
- **live background analysis** that refreshes continuously;
- an **evaluation bar** and a **per-game win-rate graph**;
- **coloured best-move arrows** drawn on the board with the win-% inside;
- **analysis snapshots** — scrubbing through history shows cached results
  instead of re-searching.

On top of the Nibbler ideas the GUI adds its own: transposition merging in
the search tree, tree reuse across moves (visit accumulation), and PV lines
for every ranked move.

---

## 📁 Project layout

```
rust_engine/
  src/lib.rs              — engine: board, move generation, rules, MCTS primitives
  Cargo.toml              — PyO3 module build
python_src/
  train.py                — training loop; FSF curriculum, lagged opponent and
                            distillation are flags on this one entry point
  mcts.py                 — Python driver for the Rust MCTS (batching, KLD)
  model.py                — CapablancaNet (ResNet + Transformer + 4 heads)
                            plus build_net_from_state_dict() helper
  eval.py                 — round-robin tournament between checkpoints
  game_stats.py           — diagnostics on a single checkpoint
  export_onnx.py          — convert a .pth checkpoint into a self-contained .onnx
  onnx_engine.py          — onnxruntime inference backend used by the GUI
  gui.py                  — Nibbler-style analysis GUI
  play_cli.py             — play against a checkpoint from a terminal (no GUI)
  policy_health.py        — is the policy head still ranking moves, or flat?
  distill.py              — move a trained net into a different architecture
  bootstrap_policy.py     — re-teach the policy head from a deep search
  check_buffer.py         — value-target distribution of a replay buffer
  slim_checkpoints.py     — shrink checkpoints/buffers into sibling *_slim/ dirs
  requirements.txt        — GUI / inference dependencies
experiments/              — scripts behind the measurements quoted here
```

---

## 🚀 Running from source

This is the raw, run-it-yourself path — install the dependencies and launch
the scripts directly. (For shipping a one-click binary instead, see
*Packaging* below.)

There are two dependency sets: the **GUI** needs only a lightweight inference
stack, **training** additionally needs PyTorch.

### Prerequisites

- Python 3.10+
- A Rust toolchain (`rustup`) — to build the engine
- For GPU inference: an up-to-date NVIDIA driver (the CUDA runtime itself
  ships inside the `onnxruntime-gpu` wheel — no CUDA Toolkit install needed)

### 1. Build the Rust engine

The engine compiles into a Python module called `capablanca_engine`. Run
inside `rust_engine/`:

```bash
pip install maturin
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

### 2. Install Python dependencies

GUI / inference only:

```bash
pip install -r python_src/requirements.txt
```

That is `numpy`, `PyQt5` and `onnxruntime-gpu` — for a machine without an
NVIDIA GPU, replace `onnxruntime-gpu` with `onnxruntime`.

Training additionally needs PyTorch — install the build matching your CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Train a network *(optional — needs PyTorch)*

From scratch (self-play):

```bash
python train.py --channels 256 --res-blocks 10 \
                --transformer-blocks 10 --transformer-heads 8 --ffn-mult 4 \
                --restricted-policy --abs-pos-embed --wide-value --no-future \
                --simulations 600 --games 128 \
                --mcts-batch 128 --mcts-parallel-sims 8 \
                --kld-threshold 8e-3 \
                --compile-inference default
```

With a Fairy-Stockfish teacher (curriculum) — the opponent's strength rises
when the win-rate is high and drops when it falls, so the network always
plays at the edge of its ability:

```bash
python train.py --channels 256 --res-blocks 15 \
                --fsf-path ./fairy-stockfish-largeboard_x86-64-bmi2 \
                --curriculum --fsf-nodes-start 1 --curriculum-sp-ratio 0.5 \
                --contempt 0.2
```

`--contempt 0.2` is useful **only when playing against an external opponent**
(FSF). In pure self-play both sides apply contempt symmetrically and the
training signal gets a structural shift — keep contempt at 0 for pure
self-play.

### 4. Export the network to ONNX *(needs PyTorch)*

The GUI runs an `.onnx` graph, not a `.pth` checkpoint. The exporter uses
PyTorch's ONNX path, which also needs `onnx` and `onnxscript`:

```bash
pip install onnx onnxscript
python export_onnx.py checkpoints_big/latest.pth capablanca.onnx
```

This bakes architecture **and** weights into one binary file and pre-applies
the softmax / WDL reduction — so the GUI itself carries no PyTorch
dependency. The exporter first tries the modern dynamo path and silently
falls back to the legacy TorchScript exporter if needed (see
`KNOWN_QUIRKS.md`).

### 5. Run the analysis GUI

```bash
python gui.py
```

If `capablanca.onnx` sits next to `gui.py` it loads automatically; otherwise
pick one through «Загрузить сеть» — that dialog takes a training checkpoint
(`.pth`) as well and converts it on the fly. GPU acceleration is automatic when an
NVIDIA GPU is present (the status bar shows `GPU · CUDA`), with a silent
CPU fallback.

### Tournament between two checkpoints *(needs PyTorch)*

```bash
python eval.py <weights_A.pth> <weights_B.pth> --games 200 \
       --simulations 200 --mcts-parallel-sims 8
```

`eval.py` recovers the **full** architecture from the checkpoint (channels,
res blocks, transformer blocks, heads, mlh/future toggles), so comparing two
checkpoints with different shapes works correctly.

---

## 📦 Packaging a standalone build

To distribute the GUI without making users install Python and dependencies:

1. Export the final network with `export_onnx.py` (single `capablanca.onnx`).
2. Build the Rust engine for the target OS (`.so` on Linux, `.pyd` on
   Windows).
3. Bundle with **PyInstaller**:

   ```bash
   pyinstaller --onedir --windowed --name capablanca-gui gui.py
   ```

4. Place `capablanca.onnx` next to the produced executable — the GUI
   auto-loads it.

`onnxruntime-gpu` carries the CUDA runtime, so the bundle runs
GPU-accelerated on any machine with an NVIDIA driver — no PyTorch, no
CUDA Toolkit. Builds are per-OS (no cross-compilation): build on Linux for
Linux, on Windows for Windows.

---

## 📜 Credits

A learning / research project. Ideas and thanks:

- **[AlphaZero](https://www.science.org/doi/10.1126/science.aar6404)**
  (DeepMind) — the overall method.
- **[Leela Chess Zero](https://lczero.org/)** — MCTS parameters, the WDL /
  MLH / future heads, the Transformer trunk.
- **[Nibbler](https://github.com/rooklift/nibbler)** — the analysis-GUI
  concept.
- **[Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish)** —
  teacher and sparring partner.

See `ROADMAP.md` for planned improvements and `KNOWN_QUIRKS.md` for
intentional design choices and PyTorch/ONNX ecosystem quirks.

---

## 🇷🇺 Кратко по-русски

AlphaZero-движок, обучаемый **только** на self-play — без человеческих
партий и дебютных книг.

**Цикл обучения:** сеть играет батч партий сама с собой, поиск MCTS даёт
policy-таргеты (визиты) и value-таргеты (результат); позиции копятся в
FIFO-буфере; сеть обучается на мульти-таргет лоссе (policy + value +
moves-left + future) → новые веса → следующий раунд self-play.

**Сеть `CapablancaNet`:** вход 139 плоскостей (8 досок истории × 17 + 3
мета, канонический флип под сторону хода) → input-conv → башня
ResNet-блоков со Squeeze-Excitation → блоки Transformer с относительным
позиционным смещением (RPB, через `scaled_dot_product_attention`) →
четыре головы:

- **policy** — 7000 логитов, prior для MCTS;
- **value (WDL)** — Win/Draw/Loss, оценка `Q = P(Win) − P(Loss)`;
- **moves-left** — сколько полуходов до конца (доводить выигрыш до мата);
- **future** — ход на 2 полухода вперёд, вспомогательная голова только
  для обучения.

Особенности: GroupNorm вместо BatchNorm (корректен при batch=1 и сдвиге
распределения), активация Mish, RPB вместо Smolgen (≈2.3К параметров на
блок).

**MCTS:** PUCT с параметрами из Lc0, безбордовые узлы (~60 байт), батчевый
поиск, переиспользование дерева, слияние транспозиций (в GUI), virtual
loss с правильной формулой `(W − vloss)/N` для разброса параллельных
селектов. Опционально — curriculum-обучение против Fairy-Stockfish с
адаптивной силой учителя и contempt-сдвигом.

**GUI:** один self-contained `.onnx` файл, инференс через onnxruntime, без
PyTorch. Tree reuse между ходами, smart prune вместо очистки таблицы,
ограниченный NN-кэш. Стрелки в стиле Nibbler с двумя проходами, mate
display в боковой шкале, c_puct/contempt слайдеры в тулбаре.

Подробности про будущие улучшения — в `ROADMAP.md`. Известные особенности
и баги-фичи — в `KNOWN_QUIRKS.md`.
