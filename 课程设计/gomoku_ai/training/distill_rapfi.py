"""Offline Rapfi teacher self-play distillation generator for Pattern-NNUE V3."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine_bridge import BOARD_CELLS, BOARD_SIZE, _bind_lib, extract_state, find_engine_lib
from rapfi_teacher import (
    RapfiClient,
    _is_five_in_row,
    _make_opening_moves,
    _teacher_policy,
    find_rapfi_bin,
)


def _worker_distill(args: Tuple) -> Tuple[Dict[str, np.ndarray], int, int, int]:
    (
        lib_path_str,
        rapfi_bin_str,
        num_games,
        time_ms,
        opening_plies,
        base_seed,
        rule,
    ) = args

    freestyle = str(rule).strip().lower() in ("freestyle", "free", "1")
    rapfi_rule = 0 if freestyle else 4

    lib = _bind_lib(Path(lib_path_str))
    if hasattr(lib, "gomoku_set_rule"):
        lib.gomoku_set_rule(1 if freestyle else 0)
    rng = np.random.default_rng(base_seed)

    boards_list: List[np.ndarray] = []
    players_list: List[np.uint8] = []
    forb_list: List[np.ndarray] = []
    policies_list: List[np.ndarray] = []
    values_list: List[float] = []
    outcomes_list: List[float] = []

    b_wins = 0
    w_wins = 0
    draws = 0

    with RapfiClient(Path(rapfi_bin_str)) as teacher_b, RapfiClient(Path(rapfi_bin_str)) as teacher_w:
        teacher_b.start_game(rule=rapfi_rule, timeout_turn_ms=time_ms, max_memory_mb=192, threads=1)
        teacher_w.start_game(rule=rapfi_rule, timeout_turn_ms=time_ms, max_memory_mb=192, threads=1)

        for g in range(num_games):
            if g > 0:
                for client in (teacher_b, teacher_w):
                    client.restart_board()
                    client._send(f"INFO rule {rapfi_rule}")
                    client._send(f"INFO timeout_turn {time_ms}")
                    client._send("INFO thread_num 1")

            board = np.zeros(BOARD_CELLS, dtype=np.uint8)
            opening = _make_opening_moves(rng, lib, pair_idx=None, seed=int(base_seed + g))
            target_plies = int(rng.integers(max(3, opening_plies - 1), opening_plies + 2))
            moves_hist: List[Tuple[int, int, int]] = []
            for mr, mc, mp_side in opening[:target_plies]:
                board[mr * BOARD_SIZE + mc] = mp_side
                moves_hist.append((mr, mc, mp_side))

            while len(moves_hist) < target_plies:
                cur_p = 2 if (len(moves_hist) % 2 == 1) else 1
                forb, _ = extract_state(lib, board, cur_p)
                legal = np.where((board == 0) & ((forb == 0) if cur_p == 1 else True))[0]
                if legal.size == 0:
                    break
                near = [
                    int(idx)
                    for idx in legal
                    if any(
                        abs((idx // BOARD_SIZE) - mr) <= 2 and abs((idx % BOARD_SIZE) - mc) <= 2
                        for mr, mc, _ in moves_hist
                    )
                ]
                cand_pool = [
                    idx
                    for idx in (near if near else legal.tolist())
                    if not _is_five_in_row(
                        board, idx // BOARD_SIZE, idx % BOARD_SIZE, cur_p, freestyle=freestyle
                    )
                ]
                pool = cand_pool if cand_pool else (near if near else legal.tolist())
                pick = int(rng.choice(pool))
                pr, pc = pick // BOARD_SIZE, pick % BOARD_SIZE
                board[pick] = cur_p
                moves_hist.append((pr, pc, cur_p))

            player = 2 if (len(moves_hist) % 2 == 1) else 1
            winner = 0

            g_boards: List[np.ndarray] = []
            g_players: List[int] = []
            g_forb: List[np.ndarray] = []
            g_pol: List[np.ndarray] = []
            g_vals: List[float] = []

            for ply in range(len(moves_hist), BOARD_CELLS):
                forb, _ = extract_state(lib, board, player)
                client = teacher_b if player == 1 else teacher_w
                r, c, tval = client.play_board(moves_hist, player)

                idx = r * BOARD_SIZE + c
                if (
                    idx < 0
                    or idx >= BOARD_CELLS
                    or board[idx] != 0
                    or (not freestyle and player == 1 and forb[idx] != 0)
                ):
                    winner = 2 if player == 1 else 1
                    break

                # 白方或无禁手模式下，策略软标签只按空位约束；连珠黑方额外排除禁手点
                forb_for_pol = forb if (not freestyle and player == 1) else np.zeros_like(forb)
                g_boards.append(board.copy().reshape(BOARD_SIZE, BOARD_SIZE))
                g_players.append(player)
                g_forb.append(forb.copy())
                g_pol.append(_teacher_policy(board, forb_for_pol, r, c))
                g_vals.append(float(tval) if tval is not None else 0.0)

                board[idx] = player
                moves_hist.append((r, c, player))

                b2d = board.reshape(BOARD_SIZE, BOARD_SIZE)
                if _is_five_in_row(b2d, r, c, player, freestyle=freestyle):
                    winner = player
                    break
                if ply + 1 >= BOARD_CELLS:
                    winner = 0
                    break
                player = 2 if player == 1 else 1

            if winner == 1:
                b_wins += 1
            elif winner == 2:
                w_wins += 1
            else:
                draws += 1

            for step_idx in range(len(g_boards)):
                side = g_players[step_idx]
                out_v = 0.0 if winner == 0 else (1.0 if side == winner else -1.0)
                t_v = g_vals[step_idx]
                if t_v == 0.0:
                    t_v = out_v
                boards_list.append(g_boards[step_idx])
                players_list.append(side)
                forb_list.append(g_forb[step_idx])
                policies_list.append(g_pol[step_idx])
                values_list.append(t_v)
                outcomes_list.append(out_v)

    if not boards_list:
        return {}, b_wins, w_wins, draws

    data = {
        "boards": np.stack(boards_list, axis=0).astype(np.uint8),
        "players": np.array(players_list, dtype=np.uint8),
        "forbidden": np.stack(forb_list, axis=0).astype(np.uint8),
        "policies": np.stack(policies_list, axis=0).astype(np.float32),
        "values": np.array(values_list, dtype=np.float32),
        "outcomes": np.array(outcomes_list, dtype=np.float32),
    }
    return data, b_wins, w_wins, draws


def distill_rapfi_games(
    teacher_bin: Path,
    games: int,
    workers: int,
    time_ms: int,
    opening_plies: int,
    output_path: Path,
    seed: int = 202611,
    rule: str = "renju",
) -> Path:
    rule_norm = "freestyle" if str(rule).strip().lower() in ("freestyle", "free", "1") else "renju"
    lib_path = find_engine_lib(PROJECT_ROOT)
    eff_workers = max(1, min(workers, games))
    per_w = [games // eff_workers] * eff_workers
    for i in range(games % eff_workers):
        per_w[i] += 1

    tasks = []
    cur_seed = seed
    for wg in per_w:
        tasks.append((str(lib_path), str(teacher_bin), wg, time_ms, opening_plies, cur_seed, rule_norm))
        cur_seed += wg * 1009

    t0 = time.time()
    if eff_workers == 1:
        parts = [_worker_distill(tasks[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=eff_workers) as pool:
            parts = pool.map(_worker_distill, tasks)

    valid_data = [p[0] for p in parts if p[0]]
    total_b = sum(p[1] for p in parts)
    total_w = sum(p[2] for p in parts)
    total_d = sum(p[3] for p in parts)

    if not valid_data:
        raise RuntimeError("No Rapfi distillation samples generated.")

    merged = {k: np.concatenate([d[k] for d in valid_data], axis=0) for k in valid_data[0].keys()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **merged)
    tmp.replace(output_path)

    elapsed = time.time() - t0
    print(
        f"[Distill Rapfi] games={games} (B={total_b}, W={total_w}, D={total_d}), "
        f"positions={len(merged['boards'])}, time={elapsed:.1f}s -> {output_path}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Rapfi distillation dataset generator")
    parser.add_argument("--teacher-bin", type=str, default="")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--time-ms", type=int, default=60)
    parser.add_argument("--opening-plies", type=int, default=3)
    parser.add_argument("--output", type=str, default="training/runs/rapfi_distill.npz")
    parser.add_argument("--rule", type=str, default="renju", choices=("renju", "freestyle"))
    parser.add_argument("--seed", type=int, default=202611)
    args = parser.parse_args()

    tbin = Path(args.teacher_bin) if args.teacher_bin else find_rapfi_bin(PROJECT_ROOT)
    if tbin is None or not tbin.exists():
        raise FileNotFoundError("Cannot find Rapfi binary. Pass --teacher-bin PATH.")

    out = Path(args.output)
    if not out.is_absolute():
        out = PROJECT_ROOT / out

    distill_rapfi_games(
        teacher_bin=tbin,
        games=args.games,
        workers=args.workers,
        time_ms=args.time_ms,
        opening_plies=args.opening_plies,
        output_path=out,
        seed=args.seed,
        rule=args.rule,
    )


if __name__ == "__main__":
    main()
