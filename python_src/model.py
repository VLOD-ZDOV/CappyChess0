# model.py — Neural network for Capablanca Chess (10×8 board)
# Architecture: AlphaZero-style residual network
# Input:  (batch, 139, 8, 10)  — 139 feature planes (8 history × 17 + 3 meta)
# Output: policy (batch, 7000), wdl (batch, 3)  — Win/Draw/Loss logits

import torch
import torch.nn as nn
import torch.nn.functional as F

# Policy vector layout (must match Rust engine):
#   0..6400        : from_sq * 80 + to_sq  (normal moves)
#   6400..6880     : promotions (6 types × 80 to-squares)
POLICY_SIZE = 7000  # FIX: was 6880 — max promotion index = 6400+99*6+5 = 6999

# Number of channels in the policy/future head bottleneck before the final Linear.
# Determines the rank of the mapping to the 7000-dim policy. 8 → low-rank, 32 → sufficient.
POLICY_HEAD_CHANNELS = 32


def _gn_groups(channels: int) -> int:
    """GroupNorm: 8 channels per group (standard heuristic).
    GroupNorm has no running stats → works correctly at batch=1 (MCTS inference)
    and does not accumulate stale statistics when the data distribution shifts (curriculum).
    Weights (weight/bias) are shape-compatible with BatchNorm checkpoints — strict=False loads correctly.
    """
    return max(1, channels // 8)


class ConvBnRelu(nn.Module):
    """Conv → GroupNorm → Mish.
    Mish (LC0 BT3+, Misra 2019): f(x) = x * tanh(softplus(x)).
    Smooth and self-gated — more stable than ReLU in deep stacks (~10-30 Elo for free).
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
    """2D learnable position bias over board geometry (Swin/T5-style).

    Alternative to Smolgen from LC0 BT3+. Idea: a knight move is ALWAYS a delta (±1,±2)
    or (±2,±1) regardless of where the piece stands. We learn one parameter for each
    offset (Δrank, Δfile) and apply it to all square pairs with the same offset.

    For an 8×10 board:
      Δrank ∈ [-7, +7] = 15 values
      Δfile ∈ [-9, +9] = 19 values
    Total: heads × 15 × 19 = 2280 parameters (vs 1.7M in Smolgen).

    Speed: one table lookup instead of 4 GEMMs. Wall-clock impact ~2-3%.
    Geometric baseline: from the first step the network understands that a "knight move"
    is a specific offset, not a random pair of squares.
    """
    def __init__(self, heads: int, board_h: int = 8, board_w: int = 10):
        super().__init__()
        self.heads = heads
        self.board_h = board_h
        self.board_w = board_w
        n_dr = 2 * board_h - 1   # 15
        n_df = 2 * board_w - 1   # 19
        # Zero initialization → at start the transformer behaves like plain MHA,
        # then gradually learns geometric biases.
        self.bias_table = nn.Parameter(torch.zeros(heads, n_dr * n_df))

        # Precomputed index map (n_sq, n_sq) for O(1) lookup at forward time.
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
        # (heads, n_sq, n_sq) — broadcast over the batch in attention.
        # Index lookup: bias_table[:, indices] dims (heads, n_sq, n_sq)
        return self.bias_table[:, self.relative_indices]


class MultiHeadAttentionRPB(nn.Module):
    """MHA with Relative Position Bias (no Smolgen). Pre-LN style.

    `qkv_bias=False` matches the BT5 finding: dropping QKV biases gives
    ~10% faster training and ~5% faster inference with no quality loss.
    Default True for backward compatibility with existing checkpoints —
    new training runs should set it to False.
    """
    def __init__(self, d_model: int, heads: int = 8,
                 board_h: int = 8, board_w: int = 10,
                 qkv_bias: bool = True):
        super().__init__()
        assert d_model % heads == 0, f"d_model={d_model} must be divisible by heads={heads}"
        self.heads = heads
        self.head_dim = d_model // heads
        # No explicit scale stored: F.scaled_dot_product_attention applies its own
        # 1/sqrt(head_dim) internally, so a self.scale here would be dead/misleading.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
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
        # SDPA picks the best backend (FlashAttention / mem-efficient / math)
        # based on shapes and hardware. attn_mask is added to scores before softmax,
        # so RPB rides in as an additive bias — broadcast across batch.
        bias = self.rpb().unsqueeze(0)                              # (1, h, S, S)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        # flatten(2) collapses the (h, hd) tail into a single D dim. The dynamo
        # ONNX exporter lowers it to an explicit Flatten/Reshape op (unlike
        # view/reshape which it tried to optimize through the transpose).
        out = out.transpose(1, 2).flatten(2)                        # (B, S, D)
        return self.out_proj(out)


def _make_norm(d_model: int, use_rmsnorm: bool):
    """LayerNorm or RMSNorm. RMSNorm drops the centering step (no mean
    subtraction) and the bias term — matches BT5's "no centering, no
    biases in normalization" change. Falls back to a hand-rolled impl
    on PyTorch versions without nn.RMSNorm."""
    if not use_rmsnorm:
        return nn.LayerNorm(d_model)
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(d_model)
    return _RMSNormFallback(d_model)


class _RMSNormFallback(nn.Module):
    """RMSNorm for PyTorch < 2.4 (no centering, only scale)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class TransformerBlock(nn.Module):
    """Pre-LN transformer encoder block: norm → MHA(RPB) → residual → norm → FFN → residual.
    Pre-LN is more stable than post-LN when training without a transformer warmup schedule.

    `use_rmsnorm=True` switches LayerNorm → RMSNorm (BT5: no centering, no
    bias). `qkv_bias=False` drops the QKV bias. Both default to the legacy
    layout so existing checkpoints load unchanged.
    """
    def __init__(self, d_model: int, heads: int = 8, ffn_mult: int = 2,
                 board_h: int = 8, board_w: int = 10,
                 qkv_bias: bool = True, use_rmsnorm: bool = False):
        super().__init__()
        self.ln1 = _make_norm(d_model, use_rmsnorm)
        self.attn = MultiHeadAttentionRPB(d_model, heads, board_h, board_w,
                                          qkv_bias=qkv_bias)
        self.ln2 = _make_norm(d_model, use_rmsnorm)
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

    # Canonical input layout with history (LC0-style, see boards_to_tensor in lib.rs):
    # Layout: 8 history slots × 17 planes/board + 3 meta = 139 planes.
    #   per history slot h ∈ 0..8 (newest=0):
    #     h*17 + 0..7   OUR pieces (P, N, B, R, Q, A, C, K) [canonical-flipped if side=1]
    #     h*17 + 8..15  THEIR pieces
    #     h*17 + 16     repetition flag
    #   136  castling (4 zones × 20 squares)
    #   137  halfmove / 100
    #   138  all-ones (CNN edge helper)
    HISTORY_LEN = 8
    PLANES_PER_BOARD = 17
    META_PLANES = 3
    INPUT_PLANES = HISTORY_LEN * PLANES_PER_BOARD + META_PLANES  # 139
    BOARD_H = 8
    BOARD_W = 10

    # MLH normalization constant: a position with N remaining half-moves → mlh_target = N/MLH_PLY_NORM ∈ [0,1].
    # An average Capablanca game lasts ~150-300 ply. 200 is a reasonable midpoint.
    MLH_PLY_NORM = 200.0

    def __init__(self, num_channels: int = 128, num_res_blocks: int = 10,
                 enable_mlh: bool = True,
                 num_transformer_blocks: int = 2,
                 transformer_heads: int = 8,
                 enable_future: bool = True,
                 qkv_bias: bool = True,
                 use_rmsnorm: bool = False,
                 piece_embed_dim: int = 0):
        super().__init__()
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.transformer_heads = transformer_heads
        self.enable_mlh = enable_mlh
        self.enable_future = enable_future
        self.num_transformer_blocks = num_transformer_blocks
        self.qkv_bias = qkv_bias
        self.use_rmsnorm = use_rmsnorm
        self.piece_embed_dim = piece_embed_dim

        # ── Piece embedding (BT3 trick) ─────────────────────────────────────────
        # Per-square linear projection of the newest "what piece sits here" vector
        # — 16 piece planes (8 our + 8 their). Concatenated to the raw input
        # BEFORE the input conv so the trunk sees the board state both as the
        # usual 139-plane stack AND as a learned dense per-square code. LC0 BT3
        # reports "model plays as if 15% larger with a 5% latency increase".
        # 0 = disabled (default, backward-compatible with existing checkpoints).
        if piece_embed_dim > 0:
            # 16 = current-position piece planes (planes 0..15 in our layout):
            #   planes 0..7   = our pieces (P N B R Q A C K)
            #   planes 8..15  = their pieces
            # Linear is shared across all 80 squares.
            self.piece_embed = nn.Linear(16, piece_embed_dim)
            input_planes = self.INPUT_PLANES + piece_embed_dim
        else:
            input_planes = self.INPUT_PLANES

        # ── Input tower ─────────────────────────────────────────────────────────
        self.input_conv = ConvBnRelu(input_planes, num_channels, kernel=3, padding=1)

        # ── Residual tower ───────────────────────────────────────────────────────
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # ── Transformer head with RPB (LC0 BT3+ inspired) ───────────────────────
        # Global "positional understanding" — connects any two squares in one step.
        # Especially useful for Archbishop and Chancellor (hybrid geometry:
        # long diagonals/lines + local knight jumps).
        # RPB instead of Smolgen — 2280 parameters per block vs 1.7M.
        if num_transformer_blocks > 0:
            assert num_channels % transformer_heads == 0, \
                f"num_channels={num_channels} must be divisible by transformer_heads={transformer_heads}"
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(num_channels, heads=transformer_heads,
                                 board_h=self.BOARD_H, board_w=self.BOARD_W,
                                 qkv_bias=qkv_bias, use_rmsnorm=use_rmsnorm)
                for _ in range(num_transformer_blocks)
            ])
        else:
            self.transformer_blocks = nn.ModuleList([])

        # ── Policy head ──────────────────────────────────────────────────────────
        # Outputs POLICY_SIZE logits (7000).
        # Bottleneck = POLICY_HEAD_CHANNELS channels before the final Linear.
        # Note on rank: Linear(C_pol*80, 7000) has rank ≤ C_pol*80.
        # With 8 channels → 640 features → policy is physically low-rank (max rank 640 for 7000
        # outputs) → the network cannot produce independent sharp probabilities for all moves.
        # 32 channels → 2560 features: tactical sharpness noticeably improves.
        # Cost: Linear(2560,7000)=17.9M vs (640,7000)=4.5M parameters.
        # LC0 uses 32-128 channels in the policy bottleneck.
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, POLICY_HEAD_CHANNELS, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(POLICY_HEAD_CHANNELS), POLICY_HEAD_CHANNELS),
            nn.Mish(inplace=True),
            nn.Flatten(),
            nn.Linear(POLICY_HEAD_CHANNELS * self.BOARD_H * self.BOARD_W, POLICY_SIZE),
        )

        # ── Value head (WDL) ─────────────────────────────────────────────────────
        # Outputs 3 logits: [Win, Draw, Loss].
        # Expected value Q = P(Win) - P(Loss) is computed in inference().
        # WDL gives better gradients than scalar Tanh:
        #   - the network explicitly learns to distinguish "sharp position" vs "dead draw"
        #   - cross-entropy loss instead of MSE — more stable training
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(8), 8),
            nn.Mish(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * self.BOARD_H * self.BOARD_W, 256),
            nn.Mish(inplace=True),
            nn.Linear(256, 3),   # [Win, Draw, Loss] logits
        )

        # ── Moves-Left Head (LC0 MLH) ─────────────────────────────────────────────
        # Scalar output — remaining half-moves / MLH_PLY_NORM ∈ [0, 1].
        # Used in MCTS to prefer shorter wins / longer losses.
        # 4 channels is sufficient — the task is simpler than policy/value.
        if enable_mlh:
            self.mlh_head = nn.Sequential(
                nn.Conv2d(num_channels, 4, kernel_size=1, bias=False),
                nn.GroupNorm(_gn_groups(4), 4),
                nn.Mish(inplace=True),
                nn.Flatten(),
                nn.Linear(4 * self.BOARD_H * self.BOARD_W, 64),
                nn.Mish(inplace=True),
                nn.Linear(64, 1),  # raw output, sigmoid applied in inference()
            )

        # ── Future Move Head (LC0 BT4 "future heads" inspired) ───────────────────
        # Predicts OUR next move — the one played two half-moves later
        # (k+2: same side, same canonical policy-index orientation).
        # Auxiliary task: forces the trunk to "simulate" the game continuation
        # BEFORE running MCTS → more planning-oriented representations.
        # Used ONLY during training (trunk shape), not needed at inference.
        # Architecture like policy head: 32-channel bottleneck → 7000 logits.
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

        # Policy head final Linear: gain=0.01 → nearly uniform softmax at init.
        # Only affects freshly initialised networks, not pre-trained weights.
        policy_linear = list(self.policy_head.children())[-1]
        if isinstance(policy_linear, nn.Linear):
            nn.init.xavier_uniform_(policy_linear.weight, gain=0.01)
            nn.init.zeros_(policy_linear.bias)

        # Value head final Linear (WDL, 3 outputs): gain=0.01 →
        # logits ≈ 0 at init → softmax gives [0.33, 0.33, 0.33].
        # Network has no prior preference for win or loss before training.
        value_linear = list(self.value_head.children())[-1]
        if isinstance(value_linear, nn.Linear):
            nn.init.xavier_uniform_(value_linear.weight, gain=0.01)
            nn.init.zeros_(value_linear.bias)

        # MLH final Linear: gain=0.01 + bias≈0 → output ≈ 0 → sigmoid(0)=0.5
        # → initial estimate of "100 ply remaining" (MLH_PLY_NORM/2). Reasonable for middlegame.
        if self.enable_mlh:
            mlh_linear = list(self.mlh_head.children())[-1]
            if isinstance(mlh_linear, nn.Linear):
                nn.init.xavier_uniform_(mlh_linear.weight, gain=0.01)
                nn.init.zeros_(mlh_linear.bias)

        # Future head final Linear: gain=0.01 → nearly uniform softmax at init.
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
            wdl_logits:    (batch, 3)     — [Win, Draw, Loss] raw logits
            mlh_raw:       (batch, 1)     — raw, sigmoid applied in inference (None if enable_mlh=False)
            future_logits: (batch, 7000)  — raw logits for move at k+2 (None if enable_future=False)
        """
        # Optional piece embedding: project the 16 current-position piece planes
        # per-square to piece_embed_dim and concatenate with the raw input.
        if self.piece_embed_dim > 0:
            # x is (B, 139, 8, 10). Take planes 0..15 (current our + their),
            # move to (B, 8, 10, 16), project to (B, 8, 10, E), put back.
            pieces = x[:, :16].permute(0, 2, 3, 1).contiguous()       # (B, 8, 10, 16)
            emb = self.piece_embed(pieces)                            # (B, 8, 10, E)
            emb = emb.permute(0, 3, 1, 2).contiguous()                # (B, E, 8, 10)
            x = torch.cat([x, emb], dim=1)                            # (B, 139+E, 8, 10)

        x = self.input_conv(x)
        for block in self.res_blocks:
            x = block(x)

        # Transformer "head": (B, C, 8, 10) → (B, 80, C) → blocks → back.
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
        For MCTS: returns (policy_softmax, Q, D, M).
          Q = P(Win) - P(Loss) ∈ [-1, 1]
          D = P(Draw) ∈ [0, 1]
          M = sigmoid(mlh_raw) ∈ [0, 1] — normalized remaining half-moves
              (multiply by MLH_PLY_NORM for PLY).
              If MLH is disabled → zeros.
        The future head is not used at inference (training-only auxiliary signal).
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


def build_net_from_state_dict(raw_sd: dict):
    """Inspect a checkpoint's state_dict and reconstruct CapablancaNet with the
    exact architecture that produced those weights — channels, residual blocks,
    transformer blocks, attention heads, plus mlh / future toggles.

    Returns (net, sd) where sd has had any incompatible-shape entries dropped,
    so caller can do `net.load_state_dict(sd, strict=False)` cleanly. Prevents
    the silent-mismatch trap where eval/play scripts hardcoded ch×bl and quietly
    skipped unfamiliar transformer/mlh/future weights through strict=False."""
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
          for k, v in raw_sd.items()}
    stem_key = next((k for k in sd if ("input_conv" in k or "input_block" in k)
                     and k.endswith(".weight") and "bn" not in k
                     and "bias" not in k), None)
    ch = sd[stem_key].shape[0] if stem_key else 128
    bl = sum(1 for k in sd if "res_blocks" in k and k.endswith("conv1.weight"))
    tb = len({k.split(".")[1] for k in sd if k.startswith("transformer_blocks.")})
    rpb = sd.get("transformer_blocks.0.attn.rpb.bias_table")
    heads = rpb.shape[0] if rpb is not None else 8
    enable_mlh    = any(k.startswith("mlh_head.")    for k in sd)
    enable_future = any(k.startswith("future_head.") for k in sd)
    # Detect BT5-style trim from the saved weights:
    # - missing transformer_blocks.0.attn.qkv.bias → trained with qkv_bias=False
    # - LN entries lack `.bias` (RMSNorm-only) → trained with use_rmsnorm=True
    # - presence of `piece_embed.weight` → trained with piece embedding
    qkv_bias = ("transformer_blocks.0.attn.qkv.bias" in sd) if tb > 0 else True
    use_rmsnorm = (tb > 0
                   and "transformer_blocks.0.ln1.weight" in sd
                   and "transformer_blocks.0.ln1.bias" not in sd)
    piece_embed_w = sd.get("piece_embed.weight")
    piece_embed_dim = piece_embed_w.shape[0] if piece_embed_w is not None else 0
    net = CapablancaNet(num_channels=ch, num_res_blocks=bl,
                        num_transformer_blocks=tb, transformer_heads=heads,
                        enable_mlh=enable_mlh, enable_future=enable_future,
                        qkv_bias=qkv_bias, use_rmsnorm=use_rmsnorm,
                        piece_embed_dim=piece_embed_dim)
    target = net.state_dict()
    sd = {k: v for k, v in sd.items() if k in target and v.shape == target[k].shape}
    return net, sd
