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

    `qk_norm=True` normalises Q and K over the head dimension before the dot
    product (Gemma 2 / Chameleon / ViT-22B). Attention logits grow like
    |q|·|k|, so at a high learning rate they can saturate the softmax into a
    near one-hot the gradient cannot recover from. Costs two head_dim-sized
    vectors and is what lets the stack tolerate the larger LR.
    """
    def __init__(self, d_model: int, heads: int = 8,
                 board_h: int = 8, board_w: int = 10,
                 qkv_bias: bool = True, qk_norm: bool = False,
                 value_residual: bool = False):
        super().__init__()
        assert d_model % heads == 0, f"d_model={d_model} must be divisible by heads={heads}"
        self.heads = heads
        self.head_dim = d_model // heads
        self.qk_norm = qk_norm
        # No explicit scale stored: F.scaled_dot_product_attention applies its own
        # 1/sqrt(head_dim) internally, so a self.scale here would be dead/misleading.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rpb = RelativePositionBias(heads, board_h, board_w)
        if qk_norm:
            self.q_norm = _make_norm(self.head_dim, use_rmsnorm=True)
            self.k_norm = _make_norm(self.head_dim, use_rmsnorm=True)
        self.value_residual = value_residual
        if value_residual:
            # One scalar per block, through a sigmoid → mixing weight in (0, 1).
            self.value_res_lambda = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, v_first: torch.Tensor = None):
        """Returns (out, v) — `v` is this block's value tensor so the stack can
        feed the FIRST block's values back into the later ones (ResFormer)."""
        # x: (B, S, D)
        B, S, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        v_out = v
        if self.value_residual and v_first is not None:
            # Value residual learning (ResFormer, arXiv:2410.17897): mix in the
            # first layer's values. Deep layers otherwise suffer "value-state
            # drain" — attention concentrates and the value stream degenerates.
            # The paper's key negative result: doing the same for Q or K does
            # NOT help, only V. lam starts at 0 so the block begins identical to
            # a plain one and learns how much of layer 0 it wants.
            lam = torch.sigmoid(self.value_res_lambda)
            v = (1.0 - lam) * v + lam * v_first
        # Keep the (B, S, D) form to hand upward: every block mixes against the
        # first block's values in this layout, before the head split.
        v_out = v
        # (B, S, h, hd) → (B, h, S, hd)
        q = q.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # SDPA picks the best backend (FlashAttention / mem-efficient / math)
        # based on shapes and hardware. attn_mask is added to scores before softmax,
        # so RPB rides in as an additive bias — broadcast across batch.
        bias = self.rpb().unsqueeze(0)                              # (1, h, 80, 80)
        if bias.shape[-1] != S:
            # Register tokens are prepended and have no board geometry, so they
            # get a zero relative-position bias; the 80×80 board block stays in
            # the bottom-right corner.
            pad = S - bias.shape[-1]
            bias = F.pad(bias, (pad, 0, pad, 0))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        # flatten(2) collapses the (h, hd) tail into a single D dim. The dynamo
        # ONNX exporter lowers it to an explicit Flatten/Reshape op (unlike
        # view/reshape which it tried to optimize through the transpose).
        out = out.transpose(1, 2).flatten(2)                        # (B, S, D)
        return self.out_proj(out), v_out


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


class SwiGLU(nn.Module):
    """Gated feed-forward: (SiLU(W1 x) * W3 x) W2 — the LLM-standard FFN
    (PaLM / LLaMA / Mistral).

    The gate multiplies two projections instead of applying a pointwise
    non-linearity to one, which buys a multiplicative interaction for free.
    `hidden` is scaled by 2/3 so the block keeps the same parameter count as the
    plain `Linear → Mish → Linear` it replaces (three matrices instead of two).
    """
    def __init__(self, d_model: int, ffn_mult: int = 2):
        super().__init__()
        hidden = int(d_model * ffn_mult * 2 / 3)
        hidden = max(8, (hidden + 7) // 8 * 8)   # keep it tensor-core friendly
        self.w_in = nn.Linear(d_model, 2 * hidden)   # gate and value in one GEMM
        self.w_out = nn.Linear(hidden, d_model)

    def forward(self, x):
        gate, value = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(gate) * value)


class HyperConnection(nn.Module):
    """Manifold-constrained hyper-connections (DeepSeek, arXiv:2512.24880),
    replacing one sub-layer's residual `x = x + f(x)`.

    Hyper-Connections widen the residual into `n` parallel streams and let the
    network learn how to route between them: a pre-mapping aggregates the
    streams into the sub-layer's input, a post-mapping scatters its output back,
    and a residual mapping mixes the streams with each other. Plain HC diverges
    at depth — the mixing matrix has no norm control, so signal energy grows
    layer over layer (the paper measures Amax gains up to ~3000).

    mHC is the fix: the mixing matrix is projected onto the Birkhoff polytope
    (doubly stochastic matrices) with Sinkhorn-Knopp. Doubly stochastic matrices
    have spectral norm 1 and are closed under multiplication, so stream mixing
    becomes a convex combination and cannot amplify — the paper reports the Amax
    gain dropping to ~1.6.

    Parameter cost is `2n + n²` scalars per sub-layer (24 at n=4); the real cost
    is activation memory, which is n× the residual stream.

    Initialised to be *exactly* a plain residual: streams start identical, the
    pre-mapping averages them, the mixing matrix starts at the identity and each
    stream receives the full sub-layer output.
    """

    def __init__(self, n_streams: int = 4, sinkhorn_iters: int = 8,
                 identity_init: float = 4.0):
        super().__init__()
        self.n = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.pre = nn.Parameter(torch.zeros(n_streams))           # → softmax
        self.post = nn.Parameter(torch.ones(n_streams))
        # Logits of the stream-mixing matrix. A strong diagonal makes the
        # Sinkhorn projection start ≈ identity, so layer 0 behaves like a
        # standard residual and the network *learns* to differentiate streams.
        self.res_logits = nn.Parameter(identity_init * torch.eye(n_streams))

    def mixing_matrix(self) -> torch.Tensor:
        """Sinkhorn-Knopp projection onto the Birkhoff polytope."""
        m = torch.exp(self.res_logits - self.res_logits.max())
        for _ in range(self.sinkhorn_iters):
            m = m / (m.sum(dim=1, keepdim=True) + 1e-8)
            m = m / (m.sum(dim=0, keepdim=True) + 1e-8)
        return m

    def aggregate(self, streams: torch.Tensor) -> torch.Tensor:
        """(B, S, n, D) → (B, S, D): convex combination of the streams."""
        w = torch.softmax(self.pre, dim=0).view(1, 1, -1, 1)
        return (streams * w).sum(dim=2)

    def scatter(self, streams: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Mix the streams, then add the sub-layer output back into each."""
        mixed = torch.einsum('ij,bsjd->bsid', self.mixing_matrix(), streams)
        return mixed + self.post.view(1, 1, -1, 1) * out.unsqueeze(2)


class TransformerBlock(nn.Module):
    """Pre-LN transformer encoder block: norm → MHA(RPB) → residual → norm → FFN → residual.
    Pre-LN is more stable than post-LN when training without a transformer warmup schedule.

    `use_rmsnorm=True` switches LayerNorm → RMSNorm (BT5: no centering, no
    bias). `qkv_bias=False` drops the QKV bias. `swiglu=True` swaps the FFN for
    a gated one, `qk_norm=True` normalises Q/K. All default to the legacy
    layout so existing checkpoints load unchanged.
    """
    def __init__(self, d_model: int, heads: int = 8, ffn_mult: int = 2,
                 board_h: int = 8, board_w: int = 10,
                 qkv_bias: bool = True, use_rmsnorm: bool = False,
                 qk_norm: bool = False, swiglu: bool = False,
                 value_residual: bool = False, hyper_streams: int = 0):
        super().__init__()
        # 0 = a plain residual; >0 = mHC with that many parallel streams.
        self.hyper_streams = hyper_streams
        if hyper_streams > 0:
            self.hc_attn = HyperConnection(hyper_streams)
            self.hc_ffn = HyperConnection(hyper_streams)
        self.ln1 = _make_norm(d_model, use_rmsnorm)
        self.attn = MultiHeadAttentionRPB(d_model, heads, board_h, board_w,
                                          qkv_bias=qkv_bias, qk_norm=qk_norm,
                                          value_residual=value_residual)
        self.ln2 = _make_norm(d_model, use_rmsnorm)
        self.ffn = SwiGLU(d_model, ffn_mult) if swiglu else nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.Mish(inplace=True),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, x: torch.Tensor, v_first: torch.Tensor = None):
        if self.hyper_streams == 0:
            attn_out, v = self.attn(self.ln1(x), v_first)
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x, v
        # x is the widened residual: (B, S, n, D)
        h = self.hc_attn.aggregate(x)
        attn_out, v = self.attn(self.ln1(h), v_first)
        x = self.hc_attn.scatter(x, attn_out)
        h = self.hc_ffn.aggregate(x)
        x = self.hc_ffn.scatter(x, self.ffn(self.ln2(h)))
        return x, v


class AttentionPolicyHead(nn.Module):
    """Bilinear from→to policy head (lc0 BT3+ "attention policy").

    The policy index layout is already an 80×80 matrix: a normal move encodes as
    `from_sq * 80 + to_sq` (indices 0..6399, see Board::move_to_idx in lib.rs).
    So the logits can be a scaled dot product between a per-square "from"
    projection and a per-square "to" projection of the trunk's own 80 tokens,
    instead of a `Linear(C*80 → 7000)`.

    Why it matters here: on a 256ch × 15 + 4tb net the policy and future heads
    are 17.9M parameters *each* — 63% of the whole 56.6M network — while the
    trunk is only 20.8M. This head is ~0.25M. It also generalises: one learned
    rule for "this kind of from-square attacks that kind of to-square", shared
    across all 6400 square pairs, rather than 7000 independent output rows.

    Promotions occupy 6400 + (from_file*10 + to_file)*6 + promo_idx, i.e. exactly
    600 slots (6400..6999). In canonical coordinates our pawns always promote
    from rank 6 to rank 7, so those logits come from the rank-6 and rank-7
    tokens through a second, smaller bilinear form.
    """

    N_PROMO = 6          # Q R B N A C — order fixed by Board::move_to_idx

    def __init__(self, channels: int, d_head: int = 128, d_promo: int = 64,
                 board_h: int = 8, board_w: int = 10):
        super().__init__()
        self.board_h, self.board_w = board_h, board_w
        self.d_head, self.d_promo = d_head, d_promo
        self.q_proj = nn.Linear(channels, d_head)
        self.k_proj = nn.Linear(channels, d_head)
        # Promotions: from-file token → one query per promotion piece.
        self.promo_q = nn.Linear(channels, self.N_PROMO * d_promo)
        self.promo_k = nn.Linear(channels, d_promo)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, 80, C) in row-major square order → (B, 7000) logits."""
        B = tokens.shape[0]
        W = self.board_w
        q = self.q_proj(tokens)                                   # (B, 80, d)
        k = self.k_proj(tokens)                                   # (B, 80, d)
        # (B, 80, 80) with [i, j] = from square i to square j, then row-major
        # flatten → index i*80 + j, exactly the normal-move encoding.
        normal = torch.matmul(q, k.transpose(1, 2)) * (self.d_head ** -0.5)
        normal = normal.flatten(1)                                # (B, 6400)

        # Canonical promotions: rank 6 → rank 7 (our pawns, after the flip).
        from_tok = tokens[:, (self.board_h - 2) * W: (self.board_h - 1) * W]   # (B, 10, C)
        to_tok   = tokens[:, (self.board_h - 1) * W: self.board_h * W]         # (B, 10, C)
        pq = self.promo_q(from_tok).view(B, W, self.N_PROMO, self.d_promo)
        pk = self.promo_k(to_tok)                                              # (B, 10, d)
        # → (B, from_file, to_file, promo); flattening gives
        #   f*60 + t*6 + p = (f*10 + t)*6 + p, matching the 6400+ layout.
        promo = torch.einsum('bfpd,btd->bftp', pq, pk) * (self.d_promo ** -0.5)
        promo = promo.flatten(1)                                  # (B, 600)
        return torch.cat([normal, promo], dim=1)                  # (B, 7000)


def reachable_policy_indices(board_h: int = 8, board_w: int = 10):
    """Indices of POLICY_SIZE that a legal move can ever occupy.

    Normal moves are encoded `from*80 + to`, which spans all 6400 square pairs —
    but on a 10x8 board only queen lines and knight jumps are reachable by any
    piece in this variant (archbishop = B+N, chancellor = R+N add no new
    geometry). Promotions live at `6400 + (from_file*10 + to_file)*6 + piece`
    and a pawn changes file by at most one.

    2672 of 7000 indices survive. The other 4328 rows of a dense
    `Linear(*, 7000)` can only ever learn to output -inf: 11.1M parameters in
    the policy head, and as many again in the future head, doing nothing but
    suppression. Measured leak onto illegal moves was 0.9-10.5% depending on the
    head, i.e. they never fully learn it either.
    """
    n_sq = board_h * board_w
    out = []
    for f in range(n_sq):
        fr, fc = divmod(f, board_w)
        for t in range(n_sq):
            if f == t:
                continue
            dr, dc = divmod(t, board_w)[0] - fr, divmod(t, board_w)[1] - fc
            straight = dr == 0 or dc == 0 or abs(dr) == abs(dc)
            knight = (abs(dr), abs(dc)) in ((1, 2), (2, 1))
            if straight or knight:
                out.append(f * n_sq + t)
    for ff in range(board_w):
        for tf in range(board_w):
            if abs(ff - tf) <= 1:
                for pc in range(AttentionPolicyHead.N_PROMO):
                    out.append(n_sq * n_sq + (ff * board_w + tf)
                               * AttentionPolicyHead.N_PROMO + pc)
    return out


class RestrictedPolicyHead(nn.Module):
    """Dense policy head that only carries the reachable rows.

    Same shape of computation as the plain dense head — conv, norm, flatten,
    one Linear — but the Linear emits 2672 logits instead of 7000, and they are
    scattered back into a 7000-wide vector whose remaining entries are a large
    negative constant. Downstream code (loss, MCTS, ONNX) sees an unchanged
    (B, 7000) output.
    """

    NEG = -1e4

    def __init__(self, channels: int, board_h: int = 8, board_w: int = 10):
        super().__init__()
        idx = reachable_policy_indices(board_h, board_w)
        self.register_buffer("index", torch.tensor(idx, dtype=torch.long),
                             persistent=False)
        self.n_out = len(idx)
        self.body = nn.Sequential(
            nn.Conv2d(channels, POLICY_HEAD_CHANNELS, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(POLICY_HEAD_CHANNELS), POLICY_HEAD_CHANNELS),
            nn.Mish(inplace=True),
            nn.Flatten(),
        )
        self.fc = nn.Linear(POLICY_HEAD_CHANNELS * board_h * board_w, self.n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(self.body(x))
        full = h.new_full((h.shape[0], POLICY_SIZE), self.NEG)
        return full.index_copy(1, self.index, h)


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
                 piece_embed_dim: int = 0,
                 qk_norm: bool = False,
                 swiglu: bool = False,
                 attn_policy: bool = False,
                 num_registers: int = 0,
                 value_residual: bool = False,
                 hyper_streams: int = 0,
                 restricted_policy: bool = False,
                 abs_pos_embed: bool = False,
                 wide_value: bool = False,
                 ffn_mult: int = 2):
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
        self.qk_norm = qk_norm
        self.swiglu = swiglu
        self.attn_policy = attn_policy
        self.num_registers = num_registers
        self.value_residual = value_residual
        self.hyper_streams = hyper_streams
        self.restricted_policy = restricted_policy
        self.abs_pos_embed = abs_pos_embed
        self.wide_value = wide_value
        self.ffn_mult = ffn_mult

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
                TransformerBlock(num_channels, heads=transformer_heads, ffn_mult=ffn_mult,
                                 board_h=self.BOARD_H, board_w=self.BOARD_W,
                                 qkv_bias=qkv_bias, use_rmsnorm=use_rmsnorm,
                                 qk_norm=qk_norm, swiglu=swiglu,
                                 value_residual=value_residual,
                                 hyper_streams=hyper_streams)
                for _ in range(num_transformer_blocks)
            ])
            if hyper_streams > 0:
                # Final aggregation of the widened residual back to one stream.
                self.hc_out = nn.Parameter(torch.zeros(hyper_streams))
            # Register tokens ("Vision Transformers Need Registers", Darcet 2023):
            # a few learnable non-square tokens the attention can use as scratch
            # space for global state, instead of hijacking a real board square to
            # store it. Dropped again before the tokens go back to (B, C, H, W).
            if num_registers > 0:
                self.registers = nn.Parameter(
                    torch.zeros(1, num_registers, num_channels))
                nn.init.normal_(self.registers, std=0.02)
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
        if attn_policy:
            self.policy_head = AttentionPolicyHead(
                num_channels, board_h=self.BOARD_H, board_w=self.BOARD_W)
        else:
            self.policy_head = (
                RestrictedPolicyHead(num_channels, self.BOARD_H, self.BOARD_W)
                if restricted_policy else
                nn.Sequential(
                    nn.Conv2d(num_channels, POLICY_HEAD_CHANNELS, kernel_size=1, bias=False),
                    nn.GroupNorm(_gn_groups(POLICY_HEAD_CHANNELS), POLICY_HEAD_CHANNELS),
                    nn.Mish(inplace=True),
                    nn.Flatten(),
                    nn.Linear(POLICY_HEAD_CHANNELS * self.BOARD_H * self.BOARD_W, POLICY_SIZE),
                ))

        # ── Value head (WDL) ─────────────────────────────────────────────────────
        # Outputs 3 logits: [Win, Draw, Loss].
        # Expected value Q = P(Win) - P(Loss) is computed in inference().
        # WDL gives better gradients than scalar Tanh:
        #   - the network explicitly learns to distinguish "sharp position" vs "dead draw"
        #   - cross-entropy loss instead of MSE — more stable training
        v_ch, v_hid = (32, 512) if wide_value else (8, 256)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, v_ch, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(v_ch), v_ch),
            nn.Mish(inplace=True),
            nn.Flatten(),
            nn.Linear(v_ch * self.BOARD_H * self.BOARD_W, v_hid),
            nn.Mish(inplace=True),
            nn.Linear(v_hid, 3),   # [Win, Draw, Loss] logits
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
            # Same shape of problem as the policy head, so the same head type.
            if attn_policy:
                self.future_head = AttentionPolicyHead(
                    num_channels, board_h=self.BOARD_H, board_w=self.BOARD_W)
            else:
                self.future_head = nn.Sequential(
                    nn.Conv2d(num_channels, POLICY_HEAD_CHANNELS, kernel_size=1, bias=False),
                    nn.GroupNorm(_gn_groups(POLICY_HEAD_CHANNELS), POLICY_HEAD_CHANNELS),
                    nn.Mish(inplace=True),
                    nn.Flatten(),
                    nn.Linear(POLICY_HEAD_CHANNELS * self.BOARD_H * self.BOARD_W, POLICY_SIZE),
                )

        # Absolute position on the token stream. The conv trunk only knows
        # where a square is through padding at the edges, and the attention
        # stack sees a bag of 80 tokens plus a RELATIVE bias — nothing says
        # "this token is e4". 80*C parameters, 1 MAC each.
        if abs_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.BOARD_H * self.BOARD_W, num_channels))
            nn.init.normal_(self.pos_embed, std=0.02)

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

        # Policy head: start near-uniform so the first searches are not biased by
        # random noise. For the Linear head that is a small gain on the last
        # layer; for the bilinear head it is a small gain on the q/k projections
        # (the logits are their dot product, so the scale enters twice).
        self._init_policy_like(self.policy_head)

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

        # Future head: same treatment as the policy head.
        if self.enable_future:
            self._init_policy_like(self.future_head)

    @staticmethod
    def _init_policy_like(head):
        if isinstance(head, AttentionPolicyHead):
            for lin in (head.q_proj, head.k_proj, head.promo_q, head.promo_k):
                nn.init.xavier_uniform_(lin.weight, gain=0.1)
                nn.init.zeros_(lin.bias)
            return
        last = list(head.children())[-1]
        if isinstance(last, nn.Linear):
            nn.init.xavier_uniform_(last.weight, gain=0.01)
            nn.init.zeros_(last.bias)

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
        B, C, H, W = x.shape
        tokens = None
        if len(self.transformer_blocks) > 0:
            tokens = x.flatten(2).transpose(1, 2).contiguous()  # (B, 80, C)
            if self.abs_pos_embed:
                tokens = tokens + self.pos_embed
            n_reg = self.num_registers
            if n_reg > 0:
                # Registers ride along through the blocks, then are dropped —
                # they exist only to give attention somewhere to park global
                # state that is not tied to a board square.
                tokens = torch.cat([self.registers.expand(B, -1, -1), tokens], dim=1)
            if self.hyper_streams > 0:
                # Widen: every stream starts as a copy, so the stack begins
                # exactly equivalent to a single-stream residual.
                tokens = tokens.unsqueeze(2).expand(
                    -1, -1, self.hyper_streams, -1).contiguous()
            v_first = None
            for tb in self.transformer_blocks:
                # v_first stays the FIRST block's values for the whole stack —
                # that is the point of ResFormer, not a running previous-layer V.
                tokens, v = tb(tokens, v_first)
                if v_first is None:
                    v_first = v
            if self.hyper_streams > 0:
                w = torch.softmax(self.hc_out, dim=0).view(1, 1, -1, 1)
                tokens = (tokens * w).sum(dim=2)
            if n_reg > 0:
                tokens = tokens[:, n_reg:]
            tokens = tokens.contiguous()
            x = tokens.transpose(1, 2).contiguous().view(B, C, H, W)

        if self.attn_policy and tokens is None:
            # Pure-ResNet trunk (no transformer blocks): the bilinear head still
            # wants per-square tokens, so read them straight off the feature map.
            tokens = x.flatten(2).transpose(1, 2).contiguous()

        policy     = self.policy_head(tokens if self.attn_policy else x)
        wdl_logits = self.value_head(x)
        mlh_raw    = self.mlh_head(x) if self.enable_mlh else None
        if self.enable_future:
            future = self.future_head(tokens if self.attn_policy else x)
        else:
            future = None
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


def describe_arch(net) -> str:
    """One-line architecture summary, used by every tool that loads a checkpoint.

    Lists the trunk shape and parameter count always, and then only the knobs
    that differ from the defaults — so a legacy net prints nothing extra and an
    unusual one is impossible to miss. It lives here because export_onnx.py,
    eval.py and game_stats.py each used to build their own string and each
    listed a different subset; adding a flag then silently failed to show up.
    """
    opts = []
    for name, attr, default in (
        ("attn-policy", "attn_policy", False),
        ("qk-norm", "qk_norm", False),
        ("swiglu", "swiglu", False),
        ("value-residual", "value_residual", False),
        ("rmsnorm", "use_rmsnorm", False),
        ("no-qkv-bias", "qkv_bias", True),
        ("no-mlh", "enable_mlh", True),
        ("no-future", "enable_future", True),
        ("restricted-policy", "restricted_policy", False),
        ("abs-pos", "abs_pos_embed", False),
        ("wide-value", "wide_value", False),
    ):
        if getattr(net, attr, default) != default:
            opts.append(name)
    if getattr(net, "ffn_mult", 2) != 2:
        opts.append(f"ffn×{net.ffn_mult}")
    for name, attr in (("registers", "num_registers"),
                       ("hyper-streams", "hyper_streams"),
                       ("piece-embed", "piece_embed_dim")):
        v = getattr(net, attr, 0)
        if v:
            opts.append(f"{name}={v}")
    n = sum(p.numel() for p in net.parameters())
    tb = (f" + {net.num_transformer_blocks}tb({net.transformer_heads}h)"
          if net.num_transformer_blocks else " (без трансформера)")
    return (f"{net.num_channels}ch × {net.num_res_blocks}bl{tb} · "
            f"{n / 1e6:.1f}M параметров · "
            + (", ".join(opts) if opts else "все флаги по умолчанию"))


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
    # Modern-stack toggles, all detectable from the saved tensors:
    #   attention policy head → policy_head.q_proj instead of policy_head.4
    #   QK-norm               → transformer_blocks.0.attn.q_norm.weight
    #   SwiGLU FFN            → ffn.w_in / ffn.w_out instead of ffn.0 / ffn.2
    #   register tokens       → a `registers` parameter, shape (1, N, C)
    attn_policy = "policy_head.q_proj.weight" in sd
    qk_norm = "transformer_blocks.0.attn.q_norm.weight" in sd
    swiglu = "transformer_blocks.0.ffn.w_in.weight" in sd
    reg = sd.get("registers")
    num_registers = reg.shape[1] if reg is not None else 0
    value_residual = "transformer_blocks.0.attn.value_res_lambda" in sd
    hc = sd.get("transformer_blocks.0.hc_attn.pre")
    hyper_streams = hc.shape[0] if hc is not None else 0
    restricted_policy = "policy_head.fc.weight" in sd
    abs_pos_embed = "pos_embed" in sd
    vw = sd.get("value_head.4.weight")
    wide_value = vw is not None and vw.shape[0] == 512
    fw = sd.get("transformer_blocks.0.ffn.w_in.weight")
    # SwiGLU: w_in = (2*hidden, d), hidden = d*mult*2/3
    ffn_mult = max(1, round(fw.shape[0] / 2 * 3 / (2 * ch))) if fw is not None else 2
    net = CapablancaNet(num_channels=ch, num_res_blocks=bl,
                        num_transformer_blocks=tb, transformer_heads=heads,
                        enable_mlh=enable_mlh, enable_future=enable_future,
                        qkv_bias=qkv_bias, use_rmsnorm=use_rmsnorm,
                        piece_embed_dim=piece_embed_dim,
                        qk_norm=qk_norm, swiglu=swiglu,
                        attn_policy=attn_policy, num_registers=num_registers,
                        value_residual=value_residual,
                        hyper_streams=hyper_streams,
                        restricted_policy=restricted_policy,
                        abs_pos_embed=abs_pos_embed,
                        wide_value=wide_value, ffn_mult=ffn_mult)
    target = net.state_dict()
    # Drop entries whose shape doesn't match the reconstructed arch. Since the
    # arch is inferred FROM this state_dict, drops should be ~empty; a non-trivial
    # drop means a genuine mismatch worth surfacing rather than loading silently
    # with default-initialised weights (e.g. a head that won't actually load).
    kept = {k: v for k, v in sd.items() if k in target and v.shape == target[k].shape}
    dropped = [k for k in sd if k not in kept]
    weighty = [k for k in dropped if k.endswith(".weight") or k.endswith(".bias")]
    if weighty:
        import warnings
        warnings.warn(
            f"build_net_from_state_dict: dropped {len(weighty)} weight tensor(s) "
            f"on shape/key mismatch — these stay default-initialised. "
            f"First few: {weighty[:5]}", stacklevel=2)
    return net, kept
