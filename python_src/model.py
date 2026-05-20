# model.py — Neural network for Capablanca Chess (10×8 board)
# Architecture: AlphaZero-style residual network
# Input:  (batch, 139, 8, 10)  — 139 feature planes (8 history × 17 + 3 meta)
# Output: policy (batch, 7000), wdl (batch, 3)  — Win/Draw/Loss логиты

import torch
import torch.nn as nn
import torch.nn.functional as F

# Policy vector layout (must match Rust engine):
#   0..6400        : from_sq * 80 + to_sq  (normal moves)
#   6400..6880     : promotions (6 types × 80 to-squares)
POLICY_SIZE = 7000  # FIX: было 6880 — макс. индекс промоушена = 6400+99*6+5 = 6999

# Кол-во каналов в bottleneck policy/future голов перед финальным Linear.
# Определяет ранг отображения в 7000-мерный policy. 8 → low-rank, 32 → достаточно.
POLICY_HEAD_CHANNELS = 32


def _gn_groups(channels: int) -> int:
    """GroupNorm: 8 каналов на группу (стандартный эвристик).
    GroupNorm не имеет running stats → работает корректно при batch=1 (MCTS inference)
    и не накапливает устаревшие статистики при смене распределения данных (curriculum).
    Веса (weight/bias) совместимы с BatchNorm checkpoint по форме — strict=False загружает корректно.
    """
    return max(1, channels // 8)


class ConvBnRelu(nn.Module):
    """Conv → GroupNorm → Mish.
    Mish (LC0 BT3+, Misra 2019): f(x) = x * tanh(softplus(x)).
    Гладкая и self-gated — стабильнее ReLU при глубоком стеке (~10-30 Elo бесплатно).
    """
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class RelativePositionBias(nn.Module):
    """2D обучаемый position bias по геометрии доски (Swin/T5-style).

    Альтернатива Smolgen из LC0 BT3+. Идея: ход коня — это ВСЕГДА дельта (±1,±2)
    или (±2,±1) независимо от того, где он стоит. Учим один параметр для каждого
    смещения (Δrank, Δfile), применяем ко всем парам клеток с таким же смещением.

    Для доски 8×10:
      Δrank ∈ [-7, +7] = 15 вариантов
      Δfile ∈ [-9, +9] = 19 вариантов
    Всего: heads × 15 × 19 = 2280 параметров (vs 1.7M в Smolgen).

    Скорость: один lookup в таблице вместо 4 GEMM'ов. wall-clock импакт ~2-3%.
    Геометрический baseline: с первого шага сеть понимает, что "ход коня"
    — это специфическое смещение, а не случайная пара клеток.
    """
    def __init__(self, heads: int, board_h: int = 8, board_w: int = 10):
        super().__init__()
        self.heads = heads
        self.board_h = board_h
        self.board_w = board_w
        n_dr = 2 * board_h - 1   # 15
        n_df = 2 * board_w - 1   # 19
        # Параметры. Инициализация нулём → на старте трансформер ведёт себя
        # как обычный MHA, потом постепенно учит геометрические биасы.
        self.bias_table = nn.Parameter(torch.zeros(heads, n_dr * n_df))

        # Предвычисленная карта индексов (n_sq, n_sq) для O(1) lookup при forward.
        n_sq = board_h * board_w
        indices = torch.zeros(n_sq, n_sq, dtype=torch.long)
        for i in range(n_sq):
            ri, fi = i // board_w, i % board_w
            for j in range(n_sq):
                rj, fj = j // board_w, j % board_w
                dr = rj - ri + (board_h - 1)  # → [0, n_dr-1]
                df = fj - fi + (board_w - 1)  # → [0, n_df-1]
                indices[i, j] = dr * n_df + df
        self.register_buffer("relative_indices", indices)

    def forward(self) -> torch.Tensor:
        # (heads, n_sq, n_sq) — broadcast в attention по batch'у.
        # Index lookup: bias_table[:, indices] dims (heads, n_sq, n_sq)
        return self.bias_table[:, self.relative_indices]


class MultiHeadAttentionRPB(nn.Module):
    """MHA с Relative Position Bias (без Smolgen). Pre-LN style."""
    def __init__(self, d_model: int, heads: int = 8,
                 board_h: int = 8, board_w: int = 10):
        super().__init__()
        assert d_model % heads == 0, f"d_model={d_model} должно делиться на heads={heads}"
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rpb = RelativePositionBias(heads, board_h, board_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        B, S, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # (B, S, h, hd) → (B, h, S, hd)
        q = q.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        # Attention: QK^T/√d + RPB bias
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, S, S)
        scores = scores + self.rpb().unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                                 # (B, h, S, hd)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN transformer encoder block: LN → MHA(RPB) → residual → LN → FFN → residual.
    Pre-LN стабильнее post-LN при обучении без warmup'а трансформера.
    """
    def __init__(self, d_model: int, heads: int = 8, ffn_mult: int = 2,
                 board_h: int = 8, board_w: int = 10):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionRPB(d_model, heads, board_h, board_w)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.Mish(inplace=True),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ResBlock(nn.Module):
    """Standard pre-activation residual block with squeeze-excitation (Mish activation)."""

    def __init__(self, channels: int, se_ratio: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.GroupNorm(_gn_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.GroupNorm(_gn_groups(channels), channels)

        # Squeeze-Excitation
        se_ch = max(channels // se_ratio, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, se_ch),
            nn.Mish(inplace=True),
            nn.Linear(se_ch, channels * 2),  # scale + bias
        )

    def forward(self, x):
        residual = x
        out = F.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # SE gating
        se = self.se(out)                                   # (B, C*2)
        scale, bias = se.chunk(2, dim=1)
        scale = torch.sigmoid(scale).view(-1, out.size(1), 1, 1)
        bias  = bias.view(-1, out.size(1), 1, 1)
        out   = out * scale + bias

        return F.mish(out + residual)


class CapablancaNet(nn.Module):
    """
    AlphaZero-style network for Capablanca Chess (10×8 board).

    Args:
        num_channels:   Filters per residual block (128 is good for local training)
        num_res_blocks: Number of residual blocks   (10 is a solid baseline)
    """

    # Каноническая форма с историей (LC0-style, см. boards_to_tensor в lib.rs):
    # Layout: 8 history slots × 17 planes/board + 3 meta = 139 планов.
    #   per history slot h ∈ 0..8 (newest=0):
    #     h*17 + 0..7   OUR pieces (P, N, B, R, Q, A, C, K) [canonical-flipped if side=1]
    #     h*17 + 8..15  THEIR pieces
    #     h*17 + 16     repetition flag
    #   136  castling (4 зоны × 20 клеток)
    #   137  halfmove / 100
    #   138  all-ones (CNN edge helper)
    HISTORY_LEN = 8
    PLANES_PER_BOARD = 17
    META_PLANES = 3
    INPUT_PLANES = HISTORY_LEN * PLANES_PER_BOARD + META_PLANES  # 139
    BOARD_H = 8
    BOARD_W = 10

    # MLH normalization constant: позиция с N оставшихся полуходов → mlh_target = N/MLH_PLY_NORM ∈ [0,1].
    # Среднестатистическая партия Капабланки длится ~150-300 ply. 200 = разумная середина.
    MLH_PLY_NORM = 200.0

    def __init__(self, num_channels: int = 128, num_res_blocks: int = 10,
                 enable_mlh: bool = True,
                 num_transformer_blocks: int = 2,
                 transformer_heads: int = 8,
                 enable_future: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.enable_mlh = enable_mlh
        self.enable_future = enable_future
        self.num_transformer_blocks = num_transformer_blocks

        # ── Input tower ─────────────────────────────────────────────────────
        self.input_conv = ConvBnRelu(self.INPUT_PLANES, num_channels, kernel=3, padding=1)

        # ── Residual tower ───────────────────────────────────────────────────
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # ── Transformer head с RPB (LC0 BT3+ inspired) ──────────────────────
        # Глобальное "понимание позиции" — связывает любые две клетки за 1 шаг.
        # Особенно полезно для архиепископа и канцлера (гибридная геометрия:
        # длинные диагонали/линии + локальные прыжки конём).
        # RPB вместо Smolgen — 2280 параметров на блок вместо 1.7M.
        if num_transformer_blocks > 0:
            assert num_channels % transformer_heads == 0, \
                f"num_channels={num_channels} должно делиться на transformer_heads={transformer_heads}"
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(num_channels, heads=transformer_heads,
                                 board_h=self.BOARD_H, board_w=self.BOARD_W)
                for _ in range(num_transformer_blocks)
            ])
        else:
            self.transformer_blocks = nn.ModuleList([])

        # ── Policy head ──────────────────────────────────────────────────────
        # Outputs POLICY_SIZE logits (7000).
        # Bottleneck = POLICY_HEAD_CHANNELS каналов перед финальным Linear.
        # ВАЖНО про ранг: Linear(C_pol*80, 7000) имеет ранг ≤ C_pol*80.
        # При 8 каналах → 640 фичей → policy физически low-rank (max rank 640 на 7000
        # выходов) → сеть не может выдавать независимые острые вероятности по всем ходам.
        # 32 канала → 2560 фичей: тактическая чёткость заметно растёт.
        # Цена: Linear(2560,7000)=17.9M против (640,7000)=4.5M параметров.
        # LC0 использует 32-128 каналов в policy bottleneck.
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, POLICY_HEAD_CHANNELS, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(POLICY_HEAD_CHANNELS), POLICY_HEAD_CHANNELS),
            nn.Mish(inplace=True),
            nn.Flatten(),
            nn.Linear(POLICY_HEAD_CHANNELS * self.BOARD_H * self.BOARD_W, POLICY_SIZE),
        )

        # ── Value head (WDL) ─────────────────────────────────────────────────
        # Outputs 3 логита: [Win, Draw, Loss].
        # Ожидаемое значение Q = P(Win) - P(Loss) вычисляется в inference().
        # WDL даёт лучший градиент чем скалярный Tanh:
        #   - сеть явно учится различать "острая позиция" vs "мёртвая ничья"
        #   - cross-entropy loss вместо MSE — стабильнее обучение
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(8), 8),
            nn.Mish(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * self.BOARD_H * self.BOARD_W, 256),
            nn.Mish(inplace=True),
            nn.Linear(256, 3),   # [Win, Draw, Loss] логиты
        )

        # ── Moves-Left Head (LC0 MLH) ─────────────────────────────────────────
        # Скалярный выход — оставшихся полуходов / MLH_PLY_NORM ∈ [0, 1].
        # Используется в MCTS для предпочтения коротких выигрышей / длинных проигрышей.
        # 4 канала достаточно — задача проще чем policy/value.
        if enable_mlh:
            self.mlh_head = nn.Sequential(
                nn.Conv2d(num_channels, 4, kernel_size=1, bias=False),
                nn.GroupNorm(_gn_groups(4), 4),
                nn.Mish(inplace=True),
                nn.Flatten(),
                nn.Linear(4 * self.BOARD_H * self.BOARD_W, 64),
                nn.Mish(inplace=True),
                nn.Linear(64, 1),  # raw output, sigmoid в inference()
            )

        # ── Future Move Head (LC0 BT4 "future heads" inspired) ───────────────
        # Предсказывает НАШ следующий ход — тот что будет сыгран через 2 полухода
        # (k+2: та же сторона, та же каноническая ориентация policy-индексов).
        # Auxiliary task: заставляет trunk "симулировать" продолжение партии
        # ещё ДО запуска MCTS → более планирующие представления.
        # Используется ТОЛЬКО при обучении (форма trunk), в inference не нужен.
        # Архитектура как у policy head: 32-канальный bottleneck → 7000 logits.
        if enable_future:
            self.future_head = nn.Sequential(
                nn.Conv2d(num_channels, POLICY_HEAD_CHANNELS, kernel_size=1, bias=False),
                nn.GroupNorm(_gn_groups(POLICY_HEAD_CHANNELS), POLICY_HEAD_CHANNELS),
                nn.Mish(inplace=True),
                nn.Flatten(),
                nn.Linear(POLICY_HEAD_CHANNELS * self.BOARD_H * self.BOARD_W, POLICY_SIZE),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Policy head финальный Linear: gain=0.01 → почти равномерный softmax на старте
        # Это только для свежей сети — на уже обученных весах не влияет
        policy_linear = list(self.policy_head.children())[-1]
        if isinstance(policy_linear, nn.Linear):
            nn.init.xavier_uniform_(policy_linear.weight, gain=0.01)
            nn.init.zeros_(policy_linear.bias)

        # Value head финальный Linear (WDL, 3 выхода): gain=0.01 →
        # логиты ≈ 0 на старте → softmax даёт [0.33, 0.33, 0.33]
        # Сеть не предпочитает ни победу, ни поражение до обучения
        value_linear = list(self.value_head.children())[-1]
        if isinstance(value_linear, nn.Linear):
            nn.init.xavier_uniform_(value_linear.weight, gain=0.01)
            nn.init.zeros_(value_linear.bias)

        # MLH финальный Linear: gain=0.01 + bias≈0 → output ≈ 0 → sigmoid(0)=0.5
        # → стартовая оценка "осталось 100 ply" (MLH_PLY_NORM/2). Разумно для middle-game.
        if self.enable_mlh:
            mlh_linear = list(self.mlh_head.children())[-1]
            if isinstance(mlh_linear, nn.Linear):
                nn.init.xavier_uniform_(mlh_linear.weight, gain=0.01)
                nn.init.zeros_(mlh_linear.bias)

        # Future head финальный Linear: gain=0.01 → почти равномерный softmax на старте.
        if self.enable_future:
            future_linear = list(self.future_head.children())[-1]
            if isinstance(future_linear, nn.Linear):
                nn.init.xavier_uniform_(future_linear.weight, gain=0.01)
                nn.init.zeros_(future_linear.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, 139, 8, 10) float tensor (8 history × 17 + 3 meta)
        Returns:
            policy_logits: (batch, 7000)  — raw logits
            wdl_logits:    (batch, 3)     — [Win, Draw, Loss] сырые логиты
            mlh_raw:       (batch, 1)     — raw, sigmoid в inference (None если enable_mlh=False)
            future_logits: (batch, 7000)  — raw logits хода на k+2 (None если enable_future=False)
        """
        x = self.input_conv(x)
        for block in self.res_blocks:
            x = block(x)

        # Transformer "head": (B, C, 8, 10) → (B, 80, C) → блоки → обратно.
        if len(self.transformer_blocks) > 0:
            B, C, H, W = x.shape
            tokens = x.flatten(2).transpose(1, 2).contiguous()  # (B, 80, C)
            for tb in self.transformer_blocks:
                tokens = tb(tokens)
            x = tokens.transpose(1, 2).contiguous().view(B, C, H, W)

        policy     = self.policy_head(x)
        wdl_logits = self.value_head(x)
        mlh_raw    = self.mlh_head(x) if self.enable_mlh else None
        future     = self.future_head(x) if self.enable_future else None
        return policy, wdl_logits, mlh_raw, future

    def inference(self, x: torch.Tensor):
        """
        Для MCTS: возвращает (policy_softmax, Q, D, M).
          Q = P(Win) - P(Loss) ∈ [-1, 1]
          D = P(Draw) ∈ [0, 1]
          M = sigmoid(mlh_raw) ∈ [0, 1] — нормализованное число оставшихся полуходов
              (умножь на MLH_PLY_NORM для PLY).
              Если MLH отключён → нули.
        future-голова в inference не используется (только train-time aux signal).
        """
        logits, wdl_logits, mlh_raw, _ = self(x)
        wdl = F.softmax(wdl_logits, dim=1)
        q   = (wdl[:, 0] - wdl[:, 2]).unsqueeze(1)
        d   = wdl[:, 1].unsqueeze(1)
        if mlh_raw is not None:
            m = torch.sigmoid(mlh_raw)
        else:
            m = torch.zeros_like(q)
        return F.softmax(logits, dim=1), q, d, m
