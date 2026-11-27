#!/usr/bin/env python3
"""16-worker 64-game equal-depth & equal-time A/B benchmark for V3 O(1) Pattern-NNUE."""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from engine_bridge import find_engine_lib, run_pnnue_arena_parallel


def main() -> None:
    lib_path = find_engine_lib(PROJECT_ROOT)
    r16 = PROJECT_ROOT / "training" / "runs_v3" / "pnnue_champ.bin"
    r4 = PROJECT_ROOT / "training" / "runs_v3" / "v3_r4_backup.bin"
    v2_nnue = PROJECT_ROOT / "models" / "nnue.bin"
    v2_cnn = PROJECT_ROOT / "models" / "cnn.bin"

    matches = [
        ("1. EqDepth d5 (t=1.5s) V3_R4  vs Heuristic", r4, None, None, None, 0.0, 5, 1.5),
        ("2. EqDepth d5 (t=1.5s) V3_R4  vs V2+Rapfi ", r4, None, v2_nnue, v2_cnn, 0.35, 5, 1.5),
        ("3. EqTime 0.5s (d8)    V3_R4  vs Heuristic", r4, None, None, None, 0.0, 8, 0.5),
        ("4. EqTime 0.5s (d8)    V3_R4  vs V2+Rapfi ", r4, None, v2_nnue, v2_cnn, 0.35, 8, 0.5),
        ("5. EqTime 0.5s (d8)    V3_R16 vs Heuristic", r16, None, None, None, 0.0, 8, 0.5),
        ("6. EqTime 0.5s (d8)    V3_R16 vs V2+Rapfi ", r16, None, v2_nnue, v2_cnn, 0.35, 8, 0.5),
    ]

    print(f"[Bench V3] lib={lib_path} (64g, 16 workers, vcf=14)", flush=True)
    for label, pa, pb, bn, bc, bcw, depth, tlim in matches:
        t0 = time.time()
        res = run_pnnue_arena_parallel(
            lib_path=lib_path,
            pnnue_a_path=pa,
            pnnue_b_path=pb,
            v2_b_nnue_path=bn,
            v2_b_cnn_path=bc,
            num_games=64,
            workers=16,
            a_depth=depth,
            a_vcf=14,
            a_time=tlim,
            b_depth=depth,
            b_vcf=14,
            b_time=tlim,
            b_cnn_w=bcw,
            seed=20261127,
        )
        dt = time.time() - t0
        print(f"{label}: {res} ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
