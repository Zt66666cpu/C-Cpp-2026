import argparse
import time
from pathlib import Path
from typing import Optional

from engine_bridge import find_engine_lib, run_arena_match_parallel


def run_duel(
    label: str,
    lib_path: Path,
    a_nnue: Optional[Path],
    a_cnn: Optional[Path],
    a_cnn_w: float,
    b_nnue: Optional[Path],
    b_cnn: Optional[Path],
    b_cnn_w: float,
    games: int,
    depth: int,
    vcf_depth: int,
    time_limit: float,
    workers: int,
    seed: int,
) -> dict:
    t0 = time.time()
    s = run_arena_match_parallel(
        lib_path,
        a_nnue, a_cnn, depth, vcf_depth, time_limit, a_cnn_w,
        b_nnue, b_cnn, depth, vcf_depth, time_limit, b_cnn_w,
        num_games=games,
        seed=seed,
        workers=workers,
    )
    total = s.a_wins + s.b_wins + s.draws
    score = (s.a_wins + 0.5 * s.draws) / max(1, total)
    dt = time.time() - t0
    print(
        f"[{label}] W={s.a_wins} L={s.b_wins} D={s.draws} "
        f"score={score:.3f} illegal={s.illegal_moves} "
        f"t_A={s.a_avg_time_sec*1000:.1f}ms t_B={s.b_avg_time_sec*1000:.1f}ms "
        f"(elapsed={dt:.1f}s)"
    )
    return {
        "label": label,
        "wins": s.a_wins,
        "losses": s.b_wins,
        "draws": s.draws,
        "score": round(score, 4),
        "illegal": s.illegal_moves,
        "t_a_ms": round(s.a_avg_time_sec * 1000, 2),
        "t_b_ms": round(s.b_avg_time_sec * 1000, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="4-way ablation & V2 vs V1 evaluator")
    parser.add_argument("--nnue", type=str, default="models/nnue.bin")
    parser.add_argument("--cnn", type=str, default="models/cnn.bin")
    parser.add_argument("--ref-nnue", type=str, default="")
    parser.add_argument("--ref-cnn", type=str, default="")
    parser.add_argument("--tag", type=str, default="Model")
    parser.add_argument("--ref-tag", type=str, default="Ref")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--vs-ref-games", type=int, default=60)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--vcf-depth", type=int, default=12)
    parser.add_argument("--time-limit", type=float, default=0.6)
    parser.add_argument("--cnn-weight", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    lib_path = find_engine_lib(project_root)

    nnue_p = (project_root / args.nnue) if not Path(args.nnue).is_absolute() else Path(args.nnue)
    cnn_p = (project_root / args.cnn) if not Path(args.cnn).is_absolute() else Path(args.cnn)

    print(f"=== Ablation Suite ({args.tag}, depth={args.depth}, games={args.games}) ===")
    run_duel(
        f"{args.tag} NNUE-only vs Heuristic",
        lib_path, nnue_p, None, 0.0, None, None, 0.0,
        args.games, args.depth, args.vcf_depth, args.time_limit, args.workers, 1001,
    )
    run_duel(
        f"{args.tag} CNN-only  vs Heuristic",
        lib_path, None, cnn_p, args.cnn_weight, None, None, 0.0,
        args.games, args.depth, args.vcf_depth, args.time_limit, args.workers, 2001,
    )
    run_duel(
        f"{args.tag} Dual-Net  vs Heuristic",
        lib_path, nnue_p, cnn_p, args.cnn_weight, None, None, 0.0,
        args.games, args.depth, args.vcf_depth, args.time_limit, args.workers, 3001,
    )
    run_duel(
        f"{args.tag} Dual-Net  vs NNUE-only",
        lib_path, nnue_p, cnn_p, args.cnn_weight, nnue_p, None, 0.0,
        args.games, args.depth, args.vcf_depth, args.time_limit, args.workers, 4001,
    )

    if args.ref_nnue and args.ref_cnn:
        ref_n = (project_root / args.ref_nnue) if not Path(args.ref_nnue).is_absolute() else Path(args.ref_nnue)
        ref_c = (project_root / args.ref_cnn) if not Path(args.ref_cnn).is_absolute() else Path(args.ref_cnn)
        run_duel(
            f"{args.tag} Dual-Net  vs {args.ref_tag} Dual-Net",
            lib_path, nnue_p, cnn_p, args.cnn_weight, ref_n, ref_c, args.cnn_weight,
            args.vs_ref_games, args.depth, args.vcf_depth, args.time_limit, args.workers, 5001,
        )


if __name__ == "__main__":
    main()
