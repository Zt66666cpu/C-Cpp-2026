import math
import struct
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GNN1_MAGIC = 0x314E4E47  # "GNN1"
GCV1_MAGIC = 0x31564347  # "GCV1"
BOARD_SIZE = 15
CNN_IN_CH = 6


class NnueNet(nn.Module):
    def __init__(self, in_dim: int = 32, h1: int = 64, h2: int = 32):
        super().__init__()
        self.in_dim = in_dim
        self.h1_dim = h1
        self.h2_dim = h2
        self.register_buffer("in_mean", torch.zeros(in_dim, dtype=torch.float32))
        self.register_buffer("in_std", torch.ones(in_dim, dtype=torch.float32))
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)

    def set_norm_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        safe_std = np.maximum(std, 1e-4).astype(np.float32)
        self.in_mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        self.in_std.copy_(torch.from_numpy(safe_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.in_mean) / self.in_std
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.tanh(self.fc3(h)).squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class PolicyValueCNN(nn.Module):
    def __init__(
        self,
        in_ch: int = CNN_IN_CH,
        ch: int = 32,
        num_blocks: int = 3,
        pol_mid: int = 16,
        val_mid: int = 2,
        val_fc: int = 32,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.ch = ch
        self.num_blocks = num_blocks
        self.pol_mid = pol_mid
        self.val_mid = val_mid
        self.val_fc = val_fc

        self.stem_conv = nn.Conv2d(in_ch, ch, kernel_size=3, padding=1, bias=True)
        self.stem_bn = nn.BatchNorm2d(ch)
        self.blocks = nn.ModuleList([ResBlock(ch) for _ in range(num_blocks)])

        self.pol_conv1 = nn.Conv2d(ch, pol_mid, kernel_size=1, bias=True)
        self.pol_bn1 = nn.BatchNorm2d(pol_mid)
        self.pol_conv2 = nn.Conv2d(pol_mid, 1, kernel_size=1, bias=True)

        self.val_conv1 = nn.Conv2d(ch, val_mid, kernel_size=1, bias=True)
        self.val_bn1 = nn.BatchNorm2d(val_mid)
        self.val_fc1 = nn.Linear(val_mid * BOARD_SIZE * BOARD_SIZE, val_fc)
        self.val_fc2 = nn.Linear(val_fc, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            h = block(h)

        p = F.relu(self.pol_bn1(self.pol_conv1(h)))
        p = self.pol_conv2(p).flatten(1)  # (B, 225)

        v = F.relu(self.val_bn1(self.val_conv1(h)))
        v = v.flatten(1)
        v = F.relu(self.val_fc1(v))
        v = torch.tanh(self.val_fc2(v)).squeeze(-1)  # (B,)
        return p, v


def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[np.ndarray, np.ndarray]:
    w = conv.weight.detach().cpu().to(torch.float64)
    b = (
        conv.bias.detach().cpu().to(torch.float64)
        if conv.bias is not None
        else torch.zeros(conv.out_channels, dtype=torch.float64)
    )
    mean = bn.running_mean.detach().cpu().to(torch.float64)
    var = bn.running_var.detach().cpu().to(torch.float64)
    gamma = bn.weight.detach().cpu().to(torch.float64)
    beta = bn.bias.detach().cpu().to(torch.float64)

    inv_std = torch.rsqrt(var + bn.eps)
    scale = gamma * inv_std
    w_folded = w * scale.view(-1, 1, 1, 1)
    b_folded = (b - mean) * scale + beta
    return w_folded.numpy().astype(np.float32), b_folded.numpy().astype(np.float32)


def _write_f32(fp, arr: np.ndarray) -> None:
    fp.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def export_nnue_bin(model: NnueNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fp:
        header = struct.pack(
            "<6I",
            GNN1_MAGIC,
            1,
            model.in_dim,
            model.h1_dim,
            model.h2_dim,
            1,
        )
        fp.write(header)
        _write_f32(fp, model.in_mean.detach().cpu().numpy())
        _write_f32(fp, model.in_std.detach().cpu().numpy())
        _write_f32(fp, model.fc1.weight.detach().cpu().numpy())
        _write_f32(fp, model.fc1.bias.detach().cpu().numpy())
        _write_f32(fp, model.fc2.weight.detach().cpu().numpy())
        _write_f32(fp, model.fc2.bias.detach().cpu().numpy())
        _write_f32(fp, model.fc3.weight.detach().cpu().numpy())
        _write_f32(fp, model.fc3.bias.detach().cpu().numpy())


def export_cnn_bin(model: PolicyValueCNN, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fp:
        header = struct.pack(
            "<8I",
            GCV1_MAGIC,
            1,
            model.in_ch,
            model.ch,
            model.num_blocks,
            model.pol_mid,
            model.val_mid,
            model.val_fc,
        )
        fp.write(header)

        w, b = fold_conv_bn(model.stem_conv, model.stem_bn)
        # (cout, cin, 3, 3) -> (cout, 3, 3, cin)
        _write_f32(fp, np.transpose(w, (0, 2, 3, 1)))
        _write_f32(fp, b)

        for block in model.blocks:
            w1, b1 = fold_conv_bn(block.conv1, block.bn1)
            w2, b2 = fold_conv_bn(block.conv2, block.bn2)
            _write_f32(fp, np.transpose(w1, (0, 2, 3, 1)))
            _write_f32(fp, b1)
            _write_f32(fp, np.transpose(w2, (0, 2, 3, 1)))
            _write_f32(fp, b2)

        pw1, pb1 = fold_conv_bn(model.pol_conv1, model.pol_bn1)
        _write_f32(fp, pw1.reshape(model.pol_mid, model.ch))
        _write_f32(fp, pb1)
        pw2 = model.pol_conv2.weight.detach().cpu().numpy().reshape(1, model.pol_mid)
        pb2 = model.pol_conv2.bias.detach().cpu().numpy().reshape(1)
        _write_f32(fp, pw2)
        _write_f32(fp, pb2)

        vw1, vb1 = fold_conv_bn(model.val_conv1, model.val_bn1)
        _write_f32(fp, vw1.reshape(model.val_mid, model.ch))
        _write_f32(fp, vb1)
        _write_f32(fp, model.val_fc1.weight.detach().cpu().numpy())
        _write_f32(fp, model.val_fc1.bias.detach().cpu().numpy())
        _write_f32(fp, model.val_fc2.weight.detach().cpu().numpy())
        _write_f32(fp, model.val_fc2.bias.detach().cpu().numpy())


def boards_to_cnn_tensor(
    boards: np.ndarray,
    players: np.ndarray,
    last_moves: np.ndarray,
    forbidden: np.ndarray,
) -> np.ndarray:
    """将 C 引擎采集的原始局面转为 (N, 6, 15, 15) float32 张量，与 C 端 nn_build_cnn_input 严格一致。"""
    n = boards.shape[0]
    b2d = boards.reshape(n, BOARD_SIZE, BOARD_SIZE)
    f2d = forbidden.reshape(n, BOARD_SIZE, BOARD_SIZE)

    x = np.zeros((n, CNN_IN_CH, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    p = players.reshape(n, 1, 1)
    opp = np.where(p == 1, 2, 1)

    x[:, 0] = (b2d == p).astype(np.float32)
    x[:, 1] = (b2d == opp).astype(np.float32)
    x[:, 2] = (b2d == 0).astype(np.float32)

    valid_last = last_moves >= 0
    if np.any(valid_last):
        idx = np.where(valid_last)[0]
        lm = last_moves[idx]
        x[idx, 3, lm // BOARD_SIZE, lm % BOARD_SIZE] = 1.0

    x[:, 4] = (players == 1).astype(np.float32).reshape(n, 1, 1)
    x[:, 5] = f2d.astype(np.float32)
    return x
