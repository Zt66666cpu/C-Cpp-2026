"""V3 single-network Pattern-NNUE (GPN1) canonical encoder, PyTorch model, and C11 exporter."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
PNNUE_DIRS = 4
PNNUE_NUM_RAW_PATTERNS = 43923
PNNUE_NUM_CANON = 22143
PNNUE_SUM_SCALE = 1.0 / 32.0
PNNUE_OCC_SCALE = 1.0 / 16.0
GPN1_MAGIC = 0x314E5047  # "GPN1" little-endian
GPN2_MAGIC = 0x324E5047  # "GPN2" little-endian

_DIR_DR = (0, 1, 1, 1)
_DIR_DC = (1, 0, 1, -1)
_POW3 = tuple(3**i for i in range(10))

_TABLES_CACHE: Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
] | None = None
_ZERO_PAT_CACHE: np.ndarray | None = None


def build_pattern_tables() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build exact canonical 9-cell 4-direction lookup and vectorized gather tables.

    Returns:
        raw_to_canon: (43923,) int64 mapping raw ternary code -> canonical id in [0, 22142].
        cell_dir_offsets: (225, 4, 9) int64 board flat indices (225 = dummy zero cell).
        cell_dir_pow3: (225, 4, 9) int64 ternary weights matching src/pnnue.c.
        cell_dir_base_off: (225, 4) int64 lr_offset[L][R].
    """
    global _TABLES_CACHE
    if _TABLES_CACHE is not None:
        return _TABLES_CACHE

    lr_offset = [[0] * 5 for _ in range(5)]
    raw_to_canon = np.zeros(PNNUE_NUM_RAW_PATTERNS, dtype=np.int64)

    raw_idx = 0
    canon_idx = 0
    for L in range(5):
        for R in range(5):
            length = L + 1 + R
            max_code = _POW3[length]
            lr_offset[L][R] = raw_idx
            for code in range(max_code):
                tmp = code
                rev = 0
                for _ in range(length):
                    rev = rev * 3 + (tmp % 3)
                    tmp //= 3
                if L < R or (L == R and code <= rev):
                    raw_to_canon[raw_idx] = canon_idx
                    canon_idx += 1
                else:
                    mirror_raw = lr_offset[R][L] + rev
                    raw_to_canon[raw_idx] = raw_to_canon[mirror_raw]
                raw_idx += 1

    assert raw_idx == PNNUE_NUM_RAW_PATTERNS, f"Expected 43923 raw patterns, got {raw_idx}"
    assert canon_idx == PNNUE_NUM_CANON, f"Expected 22143 canon patterns, got {canon_idx}"

    cell_dir_offsets = np.full((BOARD_CELLS, PNNUE_DIRS, 9), BOARD_CELLS, dtype=np.int64)
    cell_dir_pow3 = np.zeros((BOARD_CELLS, PNNUE_DIRS, 9), dtype=np.int64)
    cell_dir_base_off = np.zeros((BOARD_CELLS, PNNUE_DIRS), dtype=np.int64)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            idx = r * BOARD_SIZE + c
            for d in range(PNNUE_DIRS):
                dr = _DIR_DR[d]
                dc = _DIR_DC[d]
                L = 0
                while L < 4:
                    rr = r - (L + 1) * dr
                    cc = c - (L + 1) * dc
                    if rr < 0 or rr >= BOARD_SIZE or cc < 0 or cc >= BOARD_SIZE:
                        break
                    L += 1
                R = 0
                while R < 4:
                    rr = r + (R + 1) * dr
                    cc = c + (R + 1) * dc
                    if rr < 0 or rr >= BOARD_SIZE or cc < 0 or cc >= BOARD_SIZE:
                        break
                    R += 1

                cell_dir_base_off[idx, d] = lr_offset[L][R]
                length = L + 1 + R
                for slot, k in enumerate(range(-L, R + 1)):
                    rr = r + k * dr
                    cc = c + k * dc
                    cell_dir_offsets[idx, d, slot] = rr * BOARD_SIZE + cc
                    cell_dir_pow3[idx, d, slot] = _POW3[length - 1 - slot]

    _TABLES_CACHE = (raw_to_canon, cell_dir_offsets, cell_dir_pow3, cell_dir_base_off)
    return _TABLES_CACHE


def boards_to_pattern_ids(boards: np.ndarray) -> np.ndarray:
    """Convert (N, 15, 15) boards in {0, 1, 2} to (N, 225, 4) canonical pattern IDs."""
    arr = np.asarray(boards, dtype=np.int64)
    single = arr.ndim == 2
    if single:
        arr = arr[None, ...]
    n = arr.shape[0]
    raw_to_canon, offsets, pow3, base_off = build_pattern_tables()

    flat = np.zeros((n, BOARD_CELLS + 1), dtype=np.int64)
    flat[:, :BOARD_CELLS] = arr.reshape(n, BOARD_CELLS)

    gathered = flat[:, offsets]  # (N, 225, 4, 9)
    codes = (gathered * pow3[None, ...]).sum(axis=-1) + base_off[None, ...]
    pat_ids = raw_to_canon[codes]
    return pat_ids[0] if single else pat_ids


def get_zero_pattern_ids() -> np.ndarray:
    """Return canonical (225, 4) int64 pattern IDs for an all-empty board."""
    global _ZERO_PAT_CACHE
    if _ZERO_PAT_CACHE is None:
        empty = np.zeros((1, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        _ZERO_PAT_CACHE = boards_to_pattern_ids(empty)[0].copy()
    return _ZERO_PAT_CACHE


class PatternNnueNet(nn.Module):
    """Single-network shared-backbone Policy + Value Pattern-NNUE matching src/pnnue.c (GPN1)."""

    version = 1

    def __init__(self, dim: int = 64, v_hidden: int = 32) -> None:
        super().__init__()
        if dim <= 0 or dim > 64:
            raise ValueError(f"dim must be in [1, 64], got {dim}")
        if v_hidden <= 0 or v_hidden > 64:
            raise ValueError(f"v_hidden must be in [1, 64], got {v_hidden}")

        self.num_canon = PNNUE_NUM_CANON
        self.dim = dim
        self.v_hidden = v_hidden

        self.emb = nn.Embedding(PNNUE_NUM_CANON, dim)
        self.cell_bias = nn.Parameter(torch.zeros(dim))
        self.policy_w = nn.Parameter(torch.empty(2, dim))
        self.policy_b = nn.Parameter(torch.zeros(2))
        self.val_w1 = nn.Parameter(torch.empty(v_hidden, dim))
        self.val_b1 = nn.Parameter(torch.zeros(2, v_hidden))
        self.val_w2 = nn.Parameter(torch.empty(v_hidden))
        self.val_b2 = nn.Parameter(torch.zeros(2))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.04)
        nn.init.zeros_(self.cell_bias)
        nn.init.kaiming_uniform_(self.policy_w, a=0.0)
        nn.init.zeros_(self.policy_b)
        nn.init.kaiming_uniform_(self.val_w1, a=0.0)
        nn.init.zeros_(self.val_b1)
        nn.init.normal_(self.val_w2, mean=0.0, std=0.08)
        nn.init.zeros_(self.val_b2)

    def forward(
        self,
        pat_ids: torch.Tensor,
        player_idx: torch.Tensor,
        boards_flat: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            pat_ids: (B, 225, 4) int64 canonical pattern ids.
            player_idx: (B,) int64 (0 = CELL_BLACK, 1 = CELL_WHITE).

        Returns:
            policy_logits: (B, 225) float32 logits.
            value: (B,) float32 scalar value in [-1, 1] for side-to-move.
        """
        del boards_flat
        e = self.emb(pat_ids)  # (B, 225, 4, D)
        cell_sum = e.sum(dim=2)  # (B, 225, D)
        cell_feat = torch.relu(cell_sum + self.cell_bias)  # (B, 225, D)

        pw = self.policy_w[player_idx]  # (B, D)
        pb = self.policy_b[player_idx]  # (B,)
        policy_logits = torch.einsum("bcd,bd->bc", cell_feat, pw) + pb.unsqueeze(1)

        board_scaled = cell_feat.sum(dim=1) * PNNUE_SUM_SCALE  # (B, D)
        vb1 = self.val_b1[player_idx]  # (B, H)
        vh = torch.relu(torch.matmul(board_scaled, self.val_w1.t()) + vb1)  # (B, H)
        v_raw = torch.matmul(vh, self.val_w2) + self.val_b2[player_idx]  # (B,)
        sign = torch.where(player_idx == 0, 1.0, -1.0).to(v_raw.dtype)
        value = torch.tanh(v_raw) * sign
        return policy_logits, value


class PatternNnueV4Net(nn.Module):
    """V4 Pattern-NNUE (GPN2): 3-channel active-zone pooling + 3*dim -> 64 -> 32 -> 1 value head."""

    version = 2

    def __init__(self, dim: int = 64, v_h1: int = 64, v_h2: int = 32) -> None:
        super().__init__()
        if dim <= 0 or dim > 64:
            raise ValueError(f"dim must be in [1, 64], got {dim}")
        if v_h1 <= 0 or v_h1 > 64:
            raise ValueError(f"v_h1 must be in [1, 64], got {v_h1}")
        if v_h2 <= 0 or v_h2 > 64:
            raise ValueError(f"v_h2 must be in [1, 64], got {v_h2}")

        self.num_canon = PNNUE_NUM_CANON
        self.dim = dim
        self.v_h1 = v_h1
        self.v_h2 = v_h2

        self.emb = nn.Embedding(PNNUE_NUM_CANON, dim)
        self.cell_bias = nn.Parameter(torch.zeros(dim))
        self.policy_w = nn.Parameter(torch.empty(2, dim))
        self.policy_b = nn.Parameter(torch.zeros(2))

        self.v2_w1 = nn.Parameter(torch.empty(v_h1, 3 * dim))
        self.v2_b1 = nn.Parameter(torch.zeros(2, v_h1))
        self.v2_w2 = nn.Parameter(torch.empty(v_h2, v_h1))
        self.v2_b2 = nn.Parameter(torch.zeros(2, v_h2))
        self.v2_w3 = nn.Parameter(torch.empty(v_h2))
        self.v2_b3 = nn.Parameter(torch.zeros(2))

        zero_ids = torch.from_numpy(get_zero_pattern_ids()).long().unsqueeze(0)
        self.register_buffer("zero_pat_ids", zero_ids)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.04)
        nn.init.zeros_(self.cell_bias)
        nn.init.kaiming_uniform_(self.policy_w, a=0.0)
        nn.init.zeros_(self.policy_b)
        nn.init.kaiming_uniform_(self.v2_w1, a=0.0)
        nn.init.zeros_(self.v2_b1)
        nn.init.kaiming_uniform_(self.v2_w2, a=0.0)
        nn.init.zeros_(self.v2_b2)
        nn.init.normal_(self.v2_w3, mean=0.0, std=0.08)
        nn.init.zeros_(self.v2_b3)

    def forward(
        self,
        pat_ids: torch.Tensor,
        player_idx: torch.Tensor,
        boards_flat: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if boards_flat is None:
            raise ValueError("PatternNnueV4Net.forward requires boards_flat (B, 225)")

        e = self.emb(pat_ids)
        cell_sum = e.sum(dim=2)
        cell_feat = torch.relu(cell_sum + self.cell_bias)

        pw = self.policy_w[player_idx]
        pb = self.policy_b[player_idx]
        policy_logits = torch.einsum("bcd,bd->bc", cell_feat, pw) + pb.unsqueeze(1)

        dt = cell_feat.dtype
        is_b = (boards_flat == 1).unsqueeze(-1).to(dt)
        is_w = (boards_flat == 2).unsqueeze(-1).to(dt)
        is_act_e = (
            (boards_flat == 0) & (pat_ids != self.zero_pat_ids).any(dim=-1)
        ).unsqueeze(-1).to(dt)

        occ_b = (cell_feat * is_b).sum(dim=1) * PNNUE_OCC_SCALE
        occ_w = (cell_feat * is_w).sum(dim=1) * PNNUE_OCC_SCALE
        act_e = (cell_feat * is_act_e).sum(dim=1) * PNNUE_SUM_SCALE

        is_b_turn = (player_idx == 0).unsqueeze(-1)
        my_occ = torch.where(is_b_turn, occ_b, occ_w)
        opp_occ = torch.where(is_b_turn, occ_w, occ_b)
        x = torch.cat([my_occ, opp_occ, act_e], dim=-1)

        h1 = torch.relu(torch.matmul(x, self.v2_w1.t()) + self.v2_b1[player_idx])
        h2 = torch.relu(torch.matmul(h1, self.v2_w2.t()) + self.v2_b2[player_idx])
        v_raw = torch.matmul(h2, self.v2_w3) + self.v2_b3[player_idx]
        return policy_logits, torch.tanh(v_raw)


def load_backbone_from_v3_state(v4_model: PatternNnueV4Net, state_dict: dict) -> None:
    """Warm-start V4 PatternNnueV4Net emb + cell_bias + policy head from a GPN1 checkpoint."""
    with torch.no_grad():
        for key in ("emb.weight", "cell_bias", "policy_w", "policy_b"):
            if key in state_dict:
                dst = dict(v4_model.named_parameters())[key]
                src = state_dict[key]
                if dst.shape == src.shape:
                    dst.copy_(src.to(dst.device, dtype=dst.dtype))


def export_pnnue_bin(model: nn.Module, path: str | Path) -> Path:
    """Export PatternNnueNet (GPN1) or PatternNnueV4Net (GPN2) weights to C11 binary format."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    state = model.eval()
    version = getattr(state, "version", 1)
    with torch.no_grad():
        emb = state.emb.weight.detach().cpu().numpy().astype("<f4", copy=False)
        cell_bias = state.cell_bias.detach().cpu().numpy().astype("<f4", copy=False)
        policy_w = state.policy_w.detach().cpu().numpy().astype("<f4", copy=False)
        policy_b = state.policy_b.detach().cpu().numpy().astype("<f4", copy=False)

        if version == 1:
            val_w1 = state.val_w1.detach().cpu().numpy().astype("<f4", copy=False)
            val_b1 = state.val_b1.detach().cpu().numpy().astype("<f4", copy=False)
            val_w2 = state.val_w2.detach().cpu().numpy().astype("<f4", copy=False)
            val_b2 = state.val_b2.detach().cpu().numpy().astype("<f4", copy=False)
        else:
            v2_w1 = state.v2_w1.detach().cpu().numpy().astype("<f4", copy=False)
            v2_b1 = state.v2_b1.detach().cpu().numpy().astype("<f4", copy=False)
            v2_w2 = state.v2_w2.detach().cpu().numpy().astype("<f4", copy=False)
            v2_b2 = state.v2_b2.detach().cpu().numpy().astype("<f4", copy=False)
            v2_w3 = state.v2_w3.detach().cpu().numpy().astype("<f4", copy=False)
            v2_b3 = state.v2_b3.detach().cpu().numpy().astype("<f4", copy=False)

    with open(tmp_path, "wb") as fp:
        if version == 1:
            fp.write(
                struct.pack(
                    "<IIII",
                    GPN1_MAGIC,
                    int(state.num_canon),
                    int(state.dim),
                    int(state.v_hidden),
                )
            )
        else:
            fp.write(
                struct.pack(
                    "<IIIII",
                    GPN2_MAGIC,
                    int(state.num_canon),
                    int(state.dim),
                    int(state.v_h1),
                    int(state.v_h2),
                )
            )

        fp.write(np.ascontiguousarray(emb).tobytes())
        fp.write(np.ascontiguousarray(cell_bias).tobytes())
        fp.write(np.ascontiguousarray(policy_w[0]).tobytes())
        fp.write(np.ascontiguousarray(policy_w[1]).tobytes())
        fp.write(np.ascontiguousarray(policy_b).tobytes())

        if version == 1:
            fp.write(np.ascontiguousarray(val_w1).tobytes())
            fp.write(np.ascontiguousarray(val_b1[0]).tobytes())
            fp.write(np.ascontiguousarray(val_b1[1]).tobytes())
            fp.write(np.ascontiguousarray(val_w2).tobytes())
            fp.write(np.ascontiguousarray(val_b2).tobytes())
        else:
            fp.write(np.ascontiguousarray(v2_w1).tobytes())
            fp.write(np.ascontiguousarray(v2_b1[0]).tobytes())
            fp.write(np.ascontiguousarray(v2_b1[1]).tobytes())
            fp.write(np.ascontiguousarray(v2_w2).tobytes())
            fp.write(np.ascontiguousarray(v2_b2[0]).tobytes())
            fp.write(np.ascontiguousarray(v2_b2[1]).tobytes())
            fp.write(np.ascontiguousarray(v2_w3).tobytes())
            fp.write(np.ascontiguousarray(v2_b3).tobytes())

    tmp_path.replace(out_path)
    return out_path
