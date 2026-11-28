#!/usr/bin/env python3
"""Pretrain GPN2 (V4) on 157k balanced V2+Rapfi dataset from V3_R4 backbone, benchmark, and promote."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from engine_bridge import (
    find_engine_lib,
    run_pnnue_arena_parallel,
    verify_pnnue_parity,
)
from pnnue_model import PatternNnueV4Net, export_pnnue_bin, load_backbone_from_v3_state
from train_pnnue import load_npz_dataset, merge_replay_buffer, select_device, train_pnnue_epoch_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain GPN2 (V4) from V3 backbone")
    parser.add_argument("--rule", type=str, default="renju", choices=("renju", "freestyle"))
    args = parser.parse_args()
    rule = args.rule
    is_free = rule == "freestyle"

    lib_path = find_engine_lib(PROJECT_ROOT)
    device = select_device("auto")
    run_dir = PROJECT_ROOT / "training" / ("runs_freestyle" if is_free else "runs_v3")
    run_dir.mkdir(parents=True, exist_ok=True)
    renju_dir = PROJECT_ROOT / "training" / "runs_v3"
    models_dir = PROJECT_ROOT / "models"
    r4_bin = renju_dir / "v3_r4_backup.bin"
    r4_pt = renju_dir / "v3_r4_backup.pt"
    v2_nnue = models_dir / "nnue.bin"
    v2_cnn = models_dir / "cnn.bin"
    v2_buf = PROJECT_ROOT / "training" / "data" / "replay_buffer.npz"
    rapfi_buf = run_dir / ("rapfi_distill_freestyle.npz" if is_free else "rapfi_distill.npz")
    if not rapfi_buf.exists():
        rapfi_buf = renju_dir / "rapfi_distill.npz"
    buf_path = run_dir / "pnnue_buffer.npz"
    prod_name = "pnnue_freestyle.bin" if is_free else "pnnue.bin"
    phase_name = "v3_freestyle" if is_free else "v3_pnnue"

    assert r4_bin.exists() and r4_pt.exists(), f"Missing {r4_bin} / {r4_pt}"
    assert v2_buf.exists() and rapfi_buf.exists(), f"Missing {v2_buf} / {rapfi_buf}"

    if buf_path.exists():
        buf_path.unlink()
    tmp_buf = buf_path.with_suffix(".tmp.npz")
    if tmp_buf.exists():
        tmp_buf.unlink()

    seed = 2026112704
    t0 = time.time()
    print(f"[Bootstrap V4] device={device} lib={lib_path}", flush=True)

    d_v2 = load_npz_dataset(v2_buf)
    print(f"[Bootstrap V4] Loaded V2 tactical buffer: {len(d_v2['boards'])} pos", flush=True)
    merged = merge_replay_buffer(buf_path, d_v2, max_positions=180000)

    d_rapfi = load_npz_dataset(rapfi_buf)
    print(f"[Bootstrap V4] Loaded Rapfi anchor buffer: {len(d_rapfi['boards'])} pos", flush=True)
    merged = merge_replay_buffer(buf_path, d_rapfi, max_positions=180000)

    outs = merged["outcomes"]
    pos_cnt = int((outs > 0).sum())
    neg_cnt = int((outs < 0).sum())
    draw_cnt = int((outs == 0).sum())
    print(
        f"[Bootstrap V4] Clean balanced buffer ready: N={len(outs)} "
        f"(+1={pos_cnt}, -1={neg_cnt}, 0={draw_cnt}) in {time.time() - t0:.1f}s",
        flush=True,
    )

    v4_model = PatternNnueV4Net(dim=64, v_h1=64, v_h2=32)
    r4_ckpt = torch.load(r4_pt, map_location="cpu", weights_only=True)
    load_backbone_from_v3_state(v4_model, r4_ckpt["model"])

    t_trn = time.time()
    stats = train_pnnue_epoch_loop(
        v4_model,
        merged,
        device,
        epochs=24,
        batch_size=1024,
        lr=6e-4,
        seed=seed + 404,
        weight_decay=1e-3,
        grad_clip=1.0,
    )
    cand_bin = run_dir / "v4_init_cand.bin"
    cand_pt = run_dir / "v4_init_cand.pt"
    export_pnnue_bin(v4_model, cand_bin)
    diffs = verify_pnnue_parity(lib_path, v4_model, cand_bin)
    print(f"[Bootstrap V4 Train] {time.time() - t_trn:.1f}s | {stats} | parity={diffs}", flush=True)
    assert diffs["max_val_diff"] < 1e-4 and diffs["max_pol_diff"] < 1e-4

    metrics_path = PROJECT_ROOT / "training" / "runs" / "metrics.jsonl"
    max_logged_round = 0 if is_free else 22
    if metrics_path.exists():
        for ln in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(ln.strip())
                if row.get("phase") == phase_name:
                    max_logged_round = max(max_logged_round, int(row.get("round") or 0))
            except Exception:
                pass

    torch.save({"model": v4_model.state_dict(), "round": max_logged_round, "arch": "v4"}, cand_pt)

    matches = [
        (f"1. V4_init ({rule}) vs Heuristic (64g d5 t=1.0s)", None, None, None, 0.0),
        (f"2. V4_init ({rule}) vs V3_R4     (64g d5 t=1.0s)", r4_bin, None, None, 0.0),
        (f"3. V4_init ({rule}) vs V2+Rapfi  (64g d5 t=1.0s)", None, v2_nnue, v2_cnn, 0.35),
    ]
    vs_heur = None
    for label, pb, bn, bc, bcw in matches:
        res = run_pnnue_arena_parallel(
            lib_path=lib_path,
            pnnue_a_path=cand_bin,
            pnnue_b_path=pb,
            v2_b_nnue_path=bn,
            v2_b_cnn_path=bc,
            num_games=64,
            workers=12,
            a_depth=5,
            a_vcf=14,
            a_time=1.0,
            b_depth=5,
            b_vcf=14,
            b_time=1.0,
            b_cnn_w=bcw,
            seed=seed + 505,
            rule=rule,
        )
        print(f"{label}: {res}", flush=True)
        if pb is None and bn is None:
            vs_heur = res

    if vs_heur and vs_heur["score"] < 0.60:
        print(f"ABORT_PROMOTE: V4_init ({rule}) vs Heuristic={vs_heur['score']:.3f} < 0.60", flush=True)
        sys.exit(2)

    shutil.copy2(cand_bin, run_dir / "pnnue_champ.bin")
    shutil.copy2(cand_pt, run_dir / "pnnue_champ.pt")
    shutil.copy2(cand_bin, models_dir / prod_name)
    print(f"[Bootstrap V4] PROMOTED -> {run_dir}/pnnue_champ.{{bin,pt}} & models/{prod_name} (round={max_logged_round})", flush=True)


if __name__ == "__main__":
    main()
