#!/usr/bin/env python3
"""Bootstrap GPN2 (V4) from V3_R4 backbone + clean tactical sparring buffer, benchmark, and promote."""

from __future__ import annotations

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
    generate_pnnue_games_parallel,
    run_pnnue_arena_parallel,
    sparring_pnnue_vs_v2,
    verify_pnnue_parity,
)
from pnnue_model import PatternNnueV4Net, export_pnnue_bin, load_backbone_from_v3_state
from rapfi_teacher import find_rapfi_bin, sparring_pnnue_vs_rapfi
from train_pnnue import merge_replay_buffer, select_device, train_pnnue_epoch_loop


def main() -> None:
    lib_path = find_engine_lib(PROJECT_ROOT)
    device = select_device("auto")
    run_dir = PROJECT_ROOT / "training" / "runs_v3"
    models_dir = PROJECT_ROOT / "models"
    r4_bin = run_dir / "v3_r4_backup.bin"
    r4_pt = run_dir / "v3_r4_backup.pt"
    v2_nnue = models_dir / "nnue.bin"
    v2_cnn = models_dir / "cnn.bin"
    rapfi_bin = find_rapfi_bin(PROJECT_ROOT)
    buf_path = run_dir / "pnnue_buffer.npz"

    assert r4_bin.exists() and r4_pt.exists(), f"Missing {r4_bin} / {r4_pt}"
    if buf_path.exists():
        buf_path.unlink()
    tmp_buf = buf_path.with_suffix(".tmp.npz")
    if tmp_buf.exists():
        tmp_buf.unlink()

    seed = 2026112704
    t0 = time.time()
    print(f"[Bootstrap V4] device={device} lib={lib_path}", flush=True)

    sp_data = generate_pnnue_games_parallel(
        lib_path=lib_path,
        pnnue_path=r4_bin,
        num_games=240,
        workers=16,
        max_depth=6,
        vcf_depth=14,
        time_limit=0.6,
        random_opening_plies=2,
        explore_temp=0.30,
        seed=seed + 101,
    )
    print(f"[Bootstrap V4] SelfPlay 240g -> {len(sp_data['boards'])} pos ({time.time() - t0:.1f}s)", flush=True)
    merged = merge_replay_buffer(buf_path, sp_data, max_positions=180000)

    v2_data, v2_stats = sparring_pnnue_vs_v2(
        lib_path=lib_path,
        pnnue_path=r4_bin,
        v2_nnue_path=v2_nnue,
        v2_cnn_path=v2_cnn,
        num_games=200,
        workers=16,
        our_depth=5,
        vcf_depth=14,
        our_time=0.5,
        seed=seed + 202,
    )
    if v2_data:
        merged = merge_replay_buffer(buf_path, v2_data, max_positions=180000)

    if rapfi_bin and rapfi_bin.exists():
        r_data, r_stats = sparring_pnnue_vs_rapfi(
            lib_path=lib_path,
            pnnue_path=r4_bin,
            rapfi_bin=rapfi_bin,
            num_games=120,
            workers=8,
            our_depth=5,
            vcf_depth=14,
            our_time=0.5,
            teacher_time_ms=60,
            seed=seed + 303,
        )
        if r_data:
            merged = merge_replay_buffer(buf_path, r_data, max_positions=180000)

    print(f"[Bootstrap V4] Clean tactical buffer ready: {len(merged['boards'])} pos", flush=True)

    v4_model = PatternNnueV4Net(dim=64, v_h1=64, v_h2=32)
    r4_ckpt = torch.load(r4_pt, map_location="cpu", weights_only=True)
    load_backbone_from_v3_state(v4_model, r4_ckpt["model"])

    t_trn = time.time()
    stats = train_pnnue_epoch_loop(
        v4_model,
        merged,
        device,
        epochs=12,
        batch_size=2048,
        lr=4e-4,
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
    max_logged_round = 22
    if metrics_path.exists():
        for ln in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(ln.strip())
                if row.get("phase") == "v3_pnnue":
                    max_logged_round = max(max_logged_round, int(row.get("round") or 0))
            except Exception:
                pass

    torch.save({"model": v4_model.state_dict(), "round": max_logged_round, "arch": "v4"}, cand_pt)

    matches = [
        ("1. V4_init vs Heuristic (32g d5 t=1.0s)", None, None, None, 0.0),
        ("2. V4_init vs V3_R4     (32g d5 t=1.0s)", r4_bin, None, None, 0.0),
        ("3. V4_init vs V2+Rapfi  (32g d5 t=1.0s)", None, v2_nnue, v2_cnn, 0.35),
    ]
    vs_heur = None
    for label, pb, bn, bc, bcw in matches:
        res = run_pnnue_arena_parallel(
            lib_path=lib_path,
            pnnue_a_path=cand_bin,
            pnnue_b_path=pb,
            v2_b_nnue_path=bn,
            v2_b_cnn_path=bc,
            num_games=32,
            workers=16,
            a_depth=5,
            a_vcf=14,
            a_time=1.0,
            b_depth=5,
            b_vcf=14,
            b_time=1.0,
            b_cnn_w=bcw,
            seed=seed + 505,
        )
        print(f"{label}: {res}", flush=True)
        if pb is None and bn is None:
            vs_heur = res

    if vs_heur and vs_heur["score"] < 0.50:
        print(f"ABORT_PROMOTE: V4_init vs Heuristic={vs_heur['score']:.3f} < 0.50", flush=True)
        sys.exit(2)

    shutil.copy2(cand_bin, run_dir / "pnnue_champ.bin")
    shutil.copy2(cand_pt, run_dir / "pnnue_champ.pt")
    shutil.copy2(cand_bin, models_dir / "pnnue.bin")
    print(f"[Bootstrap V4] PROMOTED -> pnnue_champ.{{bin,pt}} & models/pnnue.bin (round={max_logged_round})", flush=True)


if __name__ == "__main__":
    main()
