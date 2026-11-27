#!/usr/bin/env python3
"""Rapfi (Renju 15x15) 外教对战与知识蒸馏桥接模块。"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine_bridge import (
    BOARD_CELLS,
    BOARD_SIZE,
    MatchStats,
    _bind_lib,
    extract_state,
    find_engine_lib,
    step_search,
)

OPEN_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
)
OPEN_M3: Tuple[Tuple[int, int], ...] = (
    (0, 2), (2, 0), (-2, 0), (0, -2),
)

_EVAL_MATE_RE = re.compile(r"\bEval\s+([+-])M(\d+)", re.IGNORECASE)
_EVAL_NUM_RE = re.compile(r"\bEval\s+([+-]?\d+)", re.IGNORECASE)
_WINRATE_RE = re.compile(r"\bwinrate\s+([\d.]+)%", re.IGNORECASE)
_COORD_RE = re.compile(r"^(\d+)\s*,\s*(\d+)$")


def find_rapfi_bin(project_root: Path) -> Optional[Path]:
    rapfi_dir = project_root / "teacher" / "rapfi"
    if not rapfi_dir.exists():
        return None
    for name in (
        "pbrain-rapfi-windows-avxvnni.exe",
        "pbrain-rapfi-windows-avx2.exe",
        "pbrain-rapfi-windows-sse.exe",
        "pbrain-rapfi-macos-apple-silicon",
    ):
        cand = rapfi_dir / name
        if cand.exists():
            return cand
    for cand in sorted(rapfi_dir.glob("pbrain-rapfi*")):
        if cand.is_file():
            return cand
    return None


def _parse_teacher_value(message_line: str) -> Optional[float]:
    m_mate = _EVAL_MATE_RE.search(message_line)
    if m_mate:
        return 1.0 if m_mate.group(1) == "+" else -1.0
    m_win = _WINRATE_RE.search(message_line)
    if m_win:
        p = float(m_win.group(1)) / 100.0
        return max(-1.0, min(1.0, 2.0 * p - 1.0))
    m_ev = _EVAL_NUM_RE.search(message_line)
    if m_ev:
        ev = int(m_ev.group(1))
        if ev >= 10000:
            return 1.0
        if ev <= -10000:
            return -1.0
        return float(math.tanh(ev / 1200.0))
    return None


class RapfiClient:
    """管理单个 Rapfi Piskvork/Renju 子进程。"""

    def __init__(self, exe_path: Path) -> None:
        self.exe_path = Path(exe_path).resolve()
        self.proc = subprocess.Popen(
            [str(self.exe_path)],
            cwd=str(self.exe_path.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def __enter__(self) -> "RapfiClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until_ok(self) -> None:
        assert self.proc.stdout is not None
        while True:
            raw = self.proc.stdout.readline()
            if not raw:
                raise RuntimeError("Rapfi exited before sending OK")
            line = raw.strip()
            if line == "OK":
                return
            if line.startswith("ERROR"):
                raise RuntimeError(f"Rapfi START error: {line}")

    def _read_move(self) -> Tuple[int, int, Optional[float]]:
        assert self.proc.stdout is not None
        last_val: Optional[float] = None
        while True:
            raw = self.proc.stdout.readline()
            if not raw:
                raise RuntimeError("Rapfi exited unexpectedly during move")
            line = raw.strip()
            if not line:
                continue
            if line.startswith("MESSAGE"):
                parsed = _parse_teacher_value(line)
                if parsed is not None:
                    last_val = parsed
                continue
            if line.startswith(("DEBUG", "UNKNOWN", "SUGGEST")):
                continue
            if line.startswith("ERROR"):
                raise RuntimeError(f"Rapfi move error: {line}")
            m = _COORD_RE.match(line)
            if m:
                col = int(m.group(1))
                row = int(m.group(2))
                return row, col, last_val

    def start_game(
        self,
        rule: int = 4,
        timeout_turn_ms: int = 100,
        max_memory_mb: int = 256,
        threads: int = 1,
    ) -> None:
        self._send(f"START {BOARD_SIZE}")
        self._read_until_ok()
        self._send(f"INFO rule {rule}")
        self._send(f"INFO timeout_turn {timeout_turn_ms}")
        self._send("INFO timeout_match 0")
        self._send(f"INFO max_memory {max_memory_mb * 1024 * 1024}")
        self._send(f"INFO thread_num {threads}")

    def play_board(
        self, moves_list: List[Tuple[int, int, int]], rapfi_player: int
    ) -> Tuple[int, int, Optional[float]]:
        self._send("BOARD")
        for r, c, p in moves_list:
            who = 1 if p == rapfi_player else 2
            self._send(f"{c},{r},{who}")
        self._send("DONE")
        return self._read_move()

    def play_begin(self) -> Tuple[int, int, Optional[float]]:
        self._send("BEGIN")
        return self._read_move()

    def play_turn(self, row: int, col: int) -> Tuple[int, int, Optional[float]]:
        self._send(f"TURN {col},{row}")
        return self._read_move()

    def restart_board(self) -> None:
        self._send("RESTART")
        self._read_until_ok()

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write("END\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=1.5)
        except Exception:
            try:
                self.proc.kill()
                self.proc.wait(timeout=1.0)
            except Exception:
                pass


def _is_five_in_row(board: np.ndarray, row: int, col: int, player: int) -> bool:
    b2d = board.reshape(BOARD_SIZE, BOARD_SIZE)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        cnt = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and b2d[r, c] == player:
                cnt += 1
                r += sign * dr
                c += sign * dc
        if player == 1:
            if cnt == 5:
                return True
        else:
            if cnt >= 5:
                return True
    return False


def _teacher_policy(
    board: np.ndarray, forbidden: np.ndarray, row: int, col: int
) -> np.ndarray:
    pol = np.zeros(BOARD_CELLS, dtype=np.float32)
    idx = row * BOARD_SIZE + col
    legal = (board == 0) & (forbidden == 0)
    neighbors: List[int] = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                nidx = nr * BOARD_SIZE + nc
                if legal[nidx]:
                    neighbors.append(nidx)
    if neighbors:
        pol[idx] = 0.94
        share = 0.06 / len(neighbors)
        for nidx in neighbors:
            pol[nidx] = share
    else:
        pol[idx] = 1.0
    return pol


def _make_opening_moves(
    rng: np.random.Generator, lib, pair_idx: Optional[int] = None, seed: int = 0
) -> List[Tuple[int, int, int]]:
    board = np.zeros(BOARD_CELLS, dtype=np.uint8)
    moves: List[Tuple[int, int, int]] = []
    if pair_idx is not None:
        r1, c1 = 7, 7
        board[r1 * BOARD_SIZE + c1] = 1
        moves.append((r1, c1, 1))
        off2 = (pair_idx + (seed % 16)) % 16
        r2 = 7 + OPEN_OFFSETS[off2][0]
        c2 = 7 + OPEN_OFFSETS[off2][1]
        board[r2 * BOARD_SIZE + c2] = 2
        moves.append((r2, c2, 2))

        m3_base = ((pair_idx // 16) + ((seed >> 4) % 4)) % 4
        forb, _ = extract_state(lib, board, 1)
        for t in range(4):
            idx3 = (m3_base + t) % 4
            r3 = 7 + OPEN_M3[idx3][0]
            c3 = 7 + OPEN_M3[idx3][1]
            p3 = r3 * BOARD_SIZE + c3
            if board[p3] == 0 and forb[p3] == 0:
                moves.append((r3, c3, 1))
                break
        return moves

    dr1 = int(rng.integers(-2, 3))
    dc1 = int(rng.integers(-2, 3))
    r1, c1 = 7 + dr1, 7 + dc1
    board[r1 * BOARD_SIZE + c1] = 1
    moves.append((r1, c1, 1))

    open_plies = int(rng.integers(2, 4))
    player = 2
    for _ in range(1, open_plies):
        forb, _ = extract_state(lib, board, player)
        legal = np.where((board == 0) & (forb == 0))[0]
        near = [
            int(idx)
            for idx in legal
            if any(
                abs((idx // BOARD_SIZE) - mr) <= 2 and abs((idx % BOARD_SIZE) - mc) <= 2
                for mr, mc, _ in moves
            )
        ]
        pool = near if near else legal.tolist()
        pick = int(rng.choice(pool))
        r, c = pick // BOARD_SIZE, pick % BOARD_SIZE
        board[pick] = player
        moves.append((r, c, player))
        player = 2 if player == 1 else 1
    return moves


def _worker_sparring(args: Tuple) -> Tuple[Dict[str, np.ndarray], int, int, int, int, float, float, int]:
    (
        lib_path_str,
        nnue_path_str,
        cnn_path_str,
        rapfi_bin_str,
        num_games,
        our_depth,
        vcf_depth,
        our_time,
        cnn_weight,
        teacher_time_ms,
        base_seed,
        record_data,
        deterministic_openings,
        pair_offset,
    ) = args

    lib = _bind_lib(Path(lib_path_str))
    handle = lib.gomoku_load_models(
        nnue_path_str.encode("utf-8") if nnue_path_str else b"",
        cnn_path_str.encode("utf-8") if cnn_path_str else b"",
    )
    rng = np.random.default_rng(base_seed)

    all_boards: List[np.ndarray] = []
    all_players: List[np.ndarray] = []
    all_last: List[np.ndarray] = []
    all_forb: List[np.ndarray] = []
    all_feat: List[np.ndarray] = []
    all_pol: List[np.ndarray] = []
    all_sval: List[np.ndarray] = []
    all_out: List[np.ndarray] = []

    a_wins = 0
    b_wins = 0
    draws = 0
    illegal = 0
    a_time_sum = 0.0
    b_time_sum = 0.0
    a_moves = 0
    b_moves = 0

    try:
        with RapfiClient(Path(rapfi_bin_str)) as rapfi:
            rapfi.start_game(rule=4, timeout_turn_ms=teacher_time_ms, max_memory_mb=256, threads=1)
            for g in range(num_games):
                if g > 0:
                    rapfi.restart_board()
                    rapfi._send("INFO rule 4")
                    rapfi._send(f"INFO timeout_turn {teacher_time_ms}")
                    rapfi._send("INFO thread_num 1")

                our_is_black = (g % 2 == 0)
                our_player = 1 if our_is_black else 2
                rapfi_player = 2 if our_is_black else 1

                p_idx = (pair_offset + g // 2) if deterministic_openings else None
                open_moves = _make_opening_moves(rng, lib, pair_idx=p_idx, seed=base_seed)
                moves_history: List[Tuple[int, int, int]] = list(open_moves)

                board = np.zeros(BOARD_CELLS, dtype=np.uint8)
                for r, c, p in open_moves:
                    board[r * BOARD_SIZE + c] = p
                last_r, last_c, last_p = open_moves[-1]
                last_move = last_r * BOARD_SIZE + last_c
                player = 2 if last_p == 1 else 1
                winner = 0

                game_records: List[Tuple[bool, np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray, float]] = []
                need_board_init = True

                for ply in range(len(open_moves), BOARD_CELLS):
                    if np.all(board != 0):
                        winner = 0
                        break

                    if player == our_player:
                        t0 = time.perf_counter()
                        move_idx, sval, pol, forb, feat = step_search(
                            lib,
                            handle,
                            board,
                            player,
                            last_move,
                            our_depth,
                            vcf_depth,
                            our_time,
                            0.0 if deterministic_openings else (0.15 if ply < 10 else 0.0),
                            cnn_weight,
                            int(base_seed + g * 1009 + ply),
                        )
                        a_time_sum += time.perf_counter() - t0
                        a_moves += 1

                        if move_idx < 0 or move_idx >= BOARD_CELLS or board[move_idx] != 0 or forb[move_idx] != 0:
                            illegal += 1
                            winner = rapfi_player
                            break

                        r, c = move_idx // BOARD_SIZE, move_idx % BOARD_SIZE
                        if record_data:
                            game_records.append(
                                (False, board.copy(), player, last_move, forb, feat, pol, sval)
                            )

                        board[move_idx] = player
                        moves_history.append((r, c, player))
                        last_move = move_idx
                        if _is_five_in_row(board, r, c, player):
                            winner = player
                            break
                        player = rapfi_player
                    else:
                        forb, feat = extract_state(lib, board, player)
                        t0 = time.perf_counter()
                        if need_board_init:
                            r, c, tval = rapfi.play_board(moves_history, rapfi_player)
                            need_board_init = False
                        else:
                            lr, lc = last_move // BOARD_SIZE, last_move % BOARD_SIZE
                            r, c, tval = rapfi.play_turn(lr, lc)
                        b_time_sum += time.perf_counter() - t0
                        b_moves += 1

                        move_idx = r * BOARD_SIZE + c
                        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE) or board[move_idx] != 0 or forb[move_idx] != 0:
                            winner = our_player
                            break

                        if record_data:
                            pol = _teacher_policy(board, forb, r, c)
                            game_records.append(
                                (True, board.copy(), player, last_move, forb, feat, pol, tval if tval is not None else 0.0)
                            )

                        board[move_idx] = player
                        moves_history.append((r, c, player))
                        last_move = move_idx
                        if _is_five_in_row(board, r, c, player):
                            winner = player
                            break
                        player = our_player

                if winner == 0:
                    draws += 1
                elif winner == our_player:
                    a_wins += 1
                else:
                    b_wins += 1

                if record_data and game_records:
                    for is_teacher, b_snap, p_snap, lm_snap, f_snap, feat_snap, pol_snap, sv_snap in game_records:
                        if not is_teacher and winner != our_player:
                            continue
                        out_val = 0.0 if winner == 0 else (1.0 if p_snap == winner else -1.0)
                        sval_use = out_val if (is_teacher and sv_snap == 0.0 and winner != 0) else sv_snap
                        all_boards.append(b_snap)
                        all_players.append(np.uint8(p_snap))
                        all_last.append(np.int32(lm_snap))
                        all_forb.append(f_snap)
                        all_feat.append(feat_snap)
                        all_pol.append(pol_snap)
                        all_sval.append(np.float32(sval_use))
                        all_out.append(np.float32(out_val))
    finally:
        lib.gomoku_free_models(handle)

    data: Dict[str, np.ndarray] = {}
    if all_boards:
        data = {
            "boards": np.stack(all_boards, axis=0).astype(np.uint8),
            "players": np.array(all_players, dtype=np.uint8),
            "last_moves": np.array(all_last, dtype=np.int32),
            "forbidden": np.stack(all_forb, axis=0).astype(np.uint8),
            "features": np.stack(all_feat, axis=0).astype(np.float32),
            "policies": np.stack(all_pol, axis=0).astype(np.float32),
            "search_values": np.array(all_sval, dtype=np.float32),
            "outcomes": np.array(all_out, dtype=np.float32),
        }
    return data, a_wins, b_wins, draws, illegal, a_time_sum, b_time_sum, a_moves, b_moves


def _run_rapfi_parallel(
    lib_path: Path,
    nnue_path: Optional[Path],
    cnn_path: Optional[Path],
    rapfi_bin: Path,
    num_games: int,
    workers: int,
    our_depth: int,
    vcf_depth: int,
    our_time: float,
    cnn_weight: float,
    teacher_time_ms: int,
    seed: int,
    record_data: bool,
    deterministic_openings: bool,
) -> Tuple[Dict[str, np.ndarray], MatchStats]:
    nnue_str = str(nnue_path) if (nnue_path and nnue_path.exists()) else ""
    cnn_str = str(cnn_path) if (cnn_path and cnn_path.exists()) else ""

    if deterministic_openings:
        total_pairs = max(1, num_games // 2)
        eff_workers = max(1, min(workers, total_pairs))
        counts = [total_pairs // eff_workers] * eff_workers
        for i in range(total_pairs % eff_workers):
            counts[i] += 1
        game_counts = [c * 2 for c in counts]
    else:
        eff_workers = max(1, min(workers, num_games))
        game_counts = [num_games // eff_workers] * eff_workers
        for i in range(num_games % eff_workers):
            game_counts[i] += 1

    tasks = []
    pair_off = 0
    for idx, gc in enumerate(game_counts):
        tasks.append((
            str(lib_path),
            nnue_str,
            cnn_str,
            str(rapfi_bin),
            gc,
            our_depth,
            vcf_depth,
            our_time,
            cnn_weight,
            teacher_time_ms,
            seed + idx * 100003,
            record_data,
            deterministic_openings,
            pair_off,
        ))
        pair_off += gc // 2

    if eff_workers == 1:
        results = [_worker_sparring(tasks[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=eff_workers) as pool:
            results = pool.map(_worker_sparring, tasks)

    agg = MatchStats()
    a_t_sum = 0.0
    b_t_sum = 0.0
    a_m_sum = 0
    b_m_sum = 0
    data_parts: List[Dict[str, np.ndarray]] = []
    for d, aw, bw, dr, ill, at, bt, am, bm in results:
        agg.a_wins += aw
        agg.b_wins += bw
        agg.draws += dr
        agg.illegal_moves += ill
        a_t_sum += at
        b_t_sum += bt
        a_m_sum += am
        b_m_sum += bm
        if d:
            data_parts.append(d)

    agg.a_avg_time_sec = a_t_sum / max(1, a_m_sum)
    agg.b_avg_time_sec = b_t_sum / max(1, b_m_sum)

    merged: Dict[str, np.ndarray] = {}
    if data_parts:
        keys = data_parts[0].keys()
        merged = {k: np.concatenate([p[k] for p in data_parts], axis=0) for k in keys}
    return merged, agg


def generate_rapfi_sparring_parallel(
    lib_path: Path,
    nnue_path: Optional[Path],
    cnn_path: Optional[Path],
    rapfi_bin: Path,
    num_games: int,
    workers: int,
    our_depth: int,
    vcf_depth: int,
    our_time: float,
    cnn_weight: float,
    teacher_time_ms: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], MatchStats]:
    return _run_rapfi_parallel(
        lib_path=lib_path,
        nnue_path=nnue_path,
        cnn_path=cnn_path,
        rapfi_bin=rapfi_bin,
        num_games=num_games,
        workers=workers,
        our_depth=our_depth,
        vcf_depth=vcf_depth,
        our_time=our_time,
        cnn_weight=cnn_weight,
        teacher_time_ms=teacher_time_ms,
        seed=seed,
        record_data=True,
        deterministic_openings=False,
    )


def eval_vs_rapfi_parallel(
    lib_path: Path,
    nnue_path: Optional[Path],
    cnn_path: Optional[Path],
    rapfi_bin: Path,
    num_games: int,
    workers: int,
    eval_depth: int,
    vcf_depth: int,
    eval_time: float,
    cnn_weight: float,
    teacher_time_ms: int,
    seed: int,
) -> MatchStats:
    _, stats = _run_rapfi_parallel(
        lib_path=lib_path,
        nnue_path=nnue_path,
        cnn_path=cnn_path,
        rapfi_bin=rapfi_bin,
        num_games=num_games,
        workers=workers,
        our_depth=eval_depth,
        vcf_depth=vcf_depth,
        our_time=eval_time,
        cnn_weight=cnn_weight,
        teacher_time_ms=teacher_time_ms,
        seed=seed,
        record_data=False,
        deterministic_openings=True,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Rapfi Renju 外教实战与冒烟测试")
    parser.add_argument("--rapfi-bin", type=str, default="")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--teacher-time-ms", type=int, default=80)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    lib_path = find_engine_lib(project_root)
    rapfi_bin = Path(args.rapfi_bin) if args.rapfi_bin else find_rapfi_bin(project_root)
    if rapfi_bin is None or not rapfi_bin.exists():
        raise SystemExit("找不到 Rapfi 可执行文件")

    nnue_path = project_root / "models" / "nnue.bin"
    cnn_path = project_root / "models" / "cnn.bin"

    if args.smoke_test:
        t0 = time.time()
        data, stats = generate_rapfi_sparring_parallel(
            lib_path=lib_path,
            nnue_path=nnue_path if nnue_path.exists() else None,
            cnn_path=cnn_path if cnn_path.exists() else None,
            rapfi_bin=rapfi_bin,
            num_games=args.games,
            workers=args.workers,
            our_depth=4,
            vcf_depth=12,
            our_time=0.5,
            cnn_weight=0.35,
            teacher_time_ms=args.teacher_time_ms,
            seed=20261127,
        )
        elapsed = time.time() - t0
        n_pos = int(data["outcomes"].shape[0]) if data else 0
        assert stats.illegal_moves == 0, f"Illegal moves: {stats.illegal_moves}"
        assert n_pos > 0, "No positions collected from Rapfi sparring"
        pol_sums = data["policies"].sum(axis=1)
        assert np.allclose(pol_sums, 1.0, atol=1e-4), f"Policy sum mismatch: {pol_sums}"
        assert np.all(np.isfinite(data["features"])), "NaN/Inf in features"
        print(
            f"[Rapfi SmokeTest] PASSED in {elapsed:.2f}s | "
            f"W={stats.a_wins} L={stats.b_wins} D={stats.draws} illegal={stats.illegal_moves} | "
            f"positions={n_pos} shapes={{k: v.shape for k, v in data.items()}}"
        )


if __name__ == "__main__":
    main()
