# model.py — Neural network for Capablanca Chess (10×8 board)
# Architecture: AlphaZero-style residual network
# Input:  (batch, 20, 8, 10)  — 20 feature planes
# Output: policy (batch, 6720), value (batch, 1)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Policy vector layout (must match Rust engine):
#   0..6400        : from_sq * 80 + to_sq  (normal moves)
#   6400..6880     : promotions (6 types × 80 to-squares)
POLICY_SIZE = 6880 # Было 6720


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    """Standard pre-activation residual block with squeeze-excitation."""

    def __init__(self, channels: int, se_ratio: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

        # Squeeze-Excitation
        se_ch = max(channels // se_ratio, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, se_ch),
            nn.ReLU(inplace=True),
            nn.Linear(se_ch, channels * 2),  # scale + bias
        )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        # SE gating
        se = self.se(out)                                   # (B, C*2)
        scale, bias = se.chunk(2, dim=1)
        scale = torch.sigmoid(scale).view(-1, out.size(1), 1, 1)
        bias  = bias.view(-1, out.size(1), 1, 1)
        out   = out * scale + bias

        return F.relu(out + residual, inplace=True)


class CapablancaNet(nn.Module):
    """
    AlphaZero-style network for Capablanca Chess (10×8 board).

    Args:
        num_channels:   Filters per residual block (128 is good for local training)
        num_res_blocks: Number of residual blocks   (10 is a solid baseline)
    """

    INPUT_PLANES = 20   # 8 piece types × 2 colors + 4 meta planes
    BOARD_H = 8
    BOARD_W = 10

    def __init__(self, num_channels: int = 128, num_res_blocks: int = 10):
        super().__init__()
        self.num_channels = num_channels

        # ── Input tower ─────────────────────────────────────────────────────
        self.input_conv = ConvBnRelu(self.INPUT_PLANES, num_channels, kernel=3, padding=1)

        # ── Residual tower ───────────────────────────────────────────────────
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # ── Policy head ──────────────────────────────────────────────────────
        # Outputs POLICY_SIZE logits (6720)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * self.BOARD_H * self.BOARD_W, POLICY_SIZE),
        )

        # ── Value head ───────────────────────────────────────────────────────
        # Outputs scalar in [-1, 1]
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * self.BOARD_H * self.BOARD_W, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, 20, 8, 10) float tensor
        Returns:
            policy_logits: (batch, 6720)  — raw logits (use CrossEntropy in training)
            value:         (batch, 1)     — in [-1, 1]
        """
        x = self.input_conv(x)
        for block in self.res_blocks:
            x = block(x)

        policy = self.policy_head(x)  # raw logits
        value  = self.value_head(x)
        return policy, value

    def inference(self, x: torch.Tensor):
        """
        Like forward(), but returns softmax policy (for MCTS usage).
        """
        logits, value = self(x)
        return F.softmax(logits, dim=1), value
