import ctypes
import multiprocessing as mp
import os
import platform
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BOARD_SIZE = 15
BOARD_CELLS = 225
EVAL_FEATURE_DIM = 32
CNN_IN_CH = 6
_DLL_DIR_HANDLES: List[object] = []


class MatchStats(ctypes.Structure):
    _fields_ = [
        ("a_wins", ctypes.c_int),
        ("b_wins", ctypes.c_int),
        ("draws", ctypes.c_int),
        ("illegal_moves", ctypes.c_int),
        ("a_avg_time_sec", ctypes.c_double),
        ("b_avg_time_sec", ctypes.c_double),
    ]


def find_engine_lib(project_root: Path) -> Path:
    system = platform.system()
    if system == "Darwin":
        names = ["libgomoku_engine.dylib"]
    elif system == "Windows":
        names = ["gomoku_engine.dll", "libgomoku_engine.dll"]
    else:
        names = ["libgomoku_engine.so"]

    search_dirs = [
        project_root / "build-win",
        project_root / "build-win" / "Release",
        project_root / "cmake-build-release",
        project_root / "cmake-build-release" / "Release",
        project_root / "cmake-build-debug",
        project_root / "build",
        project_root / "build" / "Release",
    ]
    for d in search_dirs:
        for name in names:
            p = d / name
            if p.exists():
                return p
    raise FileNotFoundError(
        f"Cannot find gomoku_engine shared library under {project_root}. Build Release first."
    )


def _bind_lib(lib_path: Path) -> ctypes.CDLL:
    if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(lib_path.parent.resolve())))
        except OSError:
            pass
    lib = ctypes.CDLL(str(lib_path))

    lib.gomoku_load_models.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.gomoku_load_models.restype = ctypes.c_void_p

    lib.gomoku_free_models.argtypes = [ctypes.c_void_p]
    lib.gomoku_free_models.restype = None

    lib.gomoku_generate_game.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.gomoku_generate_game.restype = ctypes.c_int

    lib.gomoku_run_match.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_float,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.POINTER(MatchStats),
    ]
    lib.gomoku_run_match.restype = ctypes.c_int

    lib.gomoku_eval_nn_debug.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.gomoku_eval_nn_debug.restype = ctypes.c_int

    if hasattr(lib, "gomoku_extract_state"):
        lib.gomoku_extract_state.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.gomoku_extract_state.restype = ctypes.c_int

    if hasattr(lib, "gomoku_step_search"):
        lib.gomoku_step_search.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.gomoku_step_search.restype = ctypes.c_int

    if hasattr(lib, "gomoku_load_pnnue"):
        lib.gomoku_load_pnnue.argtypes = [ctypes.c_char_p]
        lib.gomoku_load_pnnue.restype = ctypes.c_void_p

        lib.gomoku_free_pnnue.argtypes = [ctypes.c_void_p]
        lib.gomoku_free_pnnue.restype = None

        lib.gomoku_eval_pnnue_debug.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.gomoku_eval_pnnue_debug.restype = ctypes.c_int

        lib.gomoku_generate_game_pnnue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.gomoku_generate_game_pnnue.restype = ctypes.c_int

        lib.gomoku_run_match_pnnue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.POINTER(MatchStats),
        ]
        lib.gomoku_run_match_pnnue.restype = ctypes.c_int

    if hasattr(lib, "gomoku_step_search_pnnue"):
        lib.gomoku_step_search_pnnue.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_float,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        lib.gomoku_step_search_pnnue.restype = ctypes.c_int
    return lib


def extract_state(
    lib: ctypes.CDLL, board: np.ndarray, player: int
) -> Tuple[np.ndarray, np.ndarray]:
    b = np.ascontiguousarray(board, dtype=np.uint8)
    forb = np.zeros(BOARD_CELLS, dtype=np.uint8)
    feat = np.zeros(EVAL_FEATURE_DIM, dtype=np.float32)
    rc = lib.gomoku_extract_state(
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(player),
        forb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        feat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        raise RuntimeError(f"gomoku_extract_state failed with rc={rc}")
    return forb, feat


def step_search(
    lib: ctypes.CDLL,
    handle: ctypes.c_void_p,
    board: np.ndarray,
    player: int,
    last_move: int,
    max_depth: int,
    vcf_depth: int,
    time_limit: float,
    explore_temp: float,
    cnn_weight: float,
    seed: int,
) -> Tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
    b = np.ascontiguousarray(board, dtype=np.uint8)
    best_move = ctypes.c_int(-1)
    sval = ctypes.c_float(0.0)
    pol = np.zeros(BOARD_CELLS, dtype=np.float32)
    forb = np.zeros(BOARD_CELLS, dtype=np.uint8)
    feat = np.zeros(EVAL_FEATURE_DIM, dtype=np.float32)
    rc = lib.gomoku_step_search(
        handle,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(player),
        int(last_move),
        int(max_depth),
        int(vcf_depth),
        float(time_limit),
        float(explore_temp),
        float(cnn_weight),
        int(seed),
        ctypes.byref(best_move),
        ctypes.byref(sval),
        pol.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        forb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        feat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if rc != 0:
        return -1, 0.0, pol, forb, feat
    return int(best_move.value), float(sval.value), pol, forb, feat


def step_search_pnnue(
    lib: ctypes.CDLL,
    pnnue_handle: ctypes.c_void_p,
    board: np.ndarray,
    player: int,
    last_move: int,
    max_depth: int,
    vcf_depth: int,
    time_limit: float,
    explore_temp: float,
    seed: int,
) -> Tuple[int, float, np.ndarray, np.ndarray]:
    b = np.ascontiguousarray(board, dtype=np.uint8)
    best_move = ctypes.c_int(-1)
    sval = ctypes.c_float(0.0)
    pol = np.zeros(BOARD_CELLS, dtype=np.float32)
    forb = np.zeros(BOARD_CELLS, dtype=np.uint8)
    rc = lib.gomoku_step_search_pnnue(
        pnnue_handle,
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(player),
        int(last_move),
        int(max_depth),
        int(vcf_depth),
        float(time_limit),
        float(explore_temp),
        int(seed),
        ctypes.byref(best_move),
        ctypes.byref(sval),
        pol.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        forb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    if rc != 0:
        return -1, 0.0, pol, forb
    return int(best_move.value), float(sval.value), pol, forb


def _worker_generate_games(args: Tuple) -> Dict[str, np.ndarray]:
    (
        lib_path_str,
        nnue_path_str,
        cnn_path_str,
        num_games,
        max_depth,
        vcf_depth,
        time_limit,
        explore_temp,
        explore_plies,
        cnn_weight,
        base_seed,
    ) = args

    lib = _bind_lib(Path(lib_path_str))
    handle = lib.gomoku_load_models(
        nnue_path_str.encode("utf-8") if nnue_path_str else b"",
        cnn_path_str.encode("utf-8") if cnn_path_str else b"",
    )

    max_steps = BOARD_CELLS
    boards_buf = np.zeros((max_steps, BOARD_CELLS), dtype=np.uint8)
    players_buf = np.zeros(max_steps, dtype=np.uint8)
    last_buf = np.zeros(max_steps, dtype=np.int32)
    forb_buf = np.zeros((max_steps, BOARD_CELLS), dtype=np.uint8)
    feat_buf = np.zeros((max_steps, EVAL_FEATURE_DIM), dtype=np.float32)
    pol_buf = np.zeros((max_steps, BOARD_CELLS), dtype=np.float32)
    sval_buf = np.zeros(max_steps, dtype=np.float32)
    out_buf = np.zeros(max_steps, dtype=np.float32)

    all_boards: List[np.ndarray] = []
    all_players: List[np.ndarray] = []
    all_last: List[np.ndarray] = []
    all_forb: List[np.ndarray] = []
    all_feat: List[np.ndarray] = []
    all_pol: List[np.ndarray] = []
    all_sval: List[np.ndarray] = []
    all_out: List[np.ndarray] = []

    try:
        for g in range(num_games):
            seed = base_seed + g * 10007
            open_plies = 2 + (g % 3)  # 2~4 步随机开局
            steps = lib.gomoku_generate_game(
                handle,
                max_depth,
                vcf_depth,
                time_limit,
                open_plies,
                explore_temp,
                explore_plies,
                cnn_weight,
                seed,
                max_steps,
                boards_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                players_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                last_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                forb_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                feat_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                pol_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                sval_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            if steps > 0:
                all_boards.append(boards_buf[:steps].copy())
                all_players.append(players_buf[:steps].copy())
                all_last.append(last_buf[:steps].copy())
                all_forb.append(forb_buf[:steps].copy())
                all_feat.append(feat_buf[:steps].copy())
                all_pol.append(pol_buf[:steps].copy())
                all_sval.append(sval_buf[:steps].copy())
                all_out.append(out_buf[:steps].copy())
    finally:
        lib.gomoku_free_models(handle)

    if not all_boards:
        return {}

    return {
        "boards": np.concatenate(all_boards, axis=0),
        "players": np.concatenate(all_players, axis=0),
        "last_moves": np.concatenate(all_last, axis=0),
        "forbidden": np.concatenate(all_forb, axis=0),
        "features": np.concatenate(all_feat, axis=0),
        "policies": np.concatenate(all_pol, axis=0),
        "search_values": np.concatenate(all_sval, axis=0),
        "outcomes": np.concatenate(all_out, axis=0),
    }


def generate_games_parallel(
    lib_path: Path,
    nnue_path: Optional[Path],
    cnn_path: Optional[Path],
    num_games: int,
    workers: int,
    max_depth: int,
    vcf_depth: int,
    time_limit: float,
    explore_temp: float,
    explore_plies: int,
    cnn_weight: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    nnue_str = str(nnue_path) if (nnue_path and nnue_path.exists()) else ""
    cnn_str = str(cnn_path) if (cnn_path and cnn_path.exists()) else ""

    eff_workers = max(1, min(workers, num_games))
    per_worker = [num_games // eff_workers] * eff_workers
    for i in range(num_games % eff_workers):
        per_worker[i] += 1

    tasks = []
    offset_seed = seed
    for w_games in per_worker:
        tasks.append((
            str(lib_path),
            nnue_str,
            cnn_str,
            w_games,
            max_depth,
            vcf_depth,
            time_limit,
            explore_temp,
            explore_plies,
            cnn_weight,
            offset_seed,
        ))
        offset_seed += w_games * 100003

    if eff_workers == 1:
        results = [_worker_generate_games(tasks[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=eff_workers) as pool:
            results = pool.map(_worker_generate_games, tasks)

    results = [r for r in results if r]
    keys = results[0].keys()
    return {k: np.concatenate([r[k] for r in results], axis=0) for k in keys}


def run_arena_match(
    lib_path: Path,
    a_nnue: Optional[Path],
    a_cnn: Optional[Path],
    a_depth: int,
    a_vcf_depth: int,
    a_time: float,
    a_cnn_weight: float,
    b_nnue: Optional[Path],
    b_cnn: Optional[Path],
    b_depth: int,
    b_vcf_depth: int,
    b_time: float,
    b_cnn_weight: float,
    num_games: int,
    seed: int,
) -> MatchStats:
    lib = _bind_lib(lib_path)
    ha = lib.gomoku_load_models(
        str(a_nnue).encode("utf-8") if (a_nnue and a_nnue.exists()) else b"",
        str(a_cnn).encode("utf-8") if (a_cnn and a_cnn.exists()) else b"",
    )
    hb = lib.gomoku_load_models(
        str(b_nnue).encode("utf-8") if (b_nnue and b_nnue.exists()) else b"",
        str(b_cnn).encode("utf-8") if (b_cnn and b_cnn.exists()) else b"",
    )
    stats = MatchStats()
    try:
        rc = lib.gomoku_run_match(
            ha, a_depth, a_vcf_depth, a_time, a_cnn_weight,
            hb, b_depth, b_vcf_depth, b_time, b_cnn_weight,
            num_games, seed, ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"gomoku_run_match failed with rc={rc}")
    finally:
        lib.gomoku_free_models(ha)
        lib.gomoku_free_models(hb)
    return stats


def _worker_arena_match(args: Tuple) -> MatchStats:
    (
        lib_path_str,
        a_nnue_str,
        a_cnn_str,
        a_depth,
        a_vcf_depth,
        a_time,
        a_cnn_weight,
        b_nnue_str,
        b_cnn_str,
        b_depth,
        b_vcf_depth,
        b_time,
        b_cnn_weight,
        num_games,
        seed,
    ) = args

    lib = _bind_lib(Path(lib_path_str))
    ha = lib.gomoku_load_models(
        a_nnue_str.encode("utf-8") if a_nnue_str else b"",
        a_cnn_str.encode("utf-8") if a_cnn_str else b"",
    )
    hb = lib.gomoku_load_models(
        b_nnue_str.encode("utf-8") if b_nnue_str else b"",
        b_cnn_str.encode("utf-8") if b_cnn_str else b"",
    )
    stats = MatchStats()
    try:
        rc = lib.gomoku_run_match(
            ha, a_depth, a_vcf_depth, a_time, a_cnn_weight,
            hb, b_depth, b_vcf_depth, b_time, b_cnn_weight,
            num_games, seed, ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"gomoku_run_match failed with rc={rc}")
    finally:
        lib.gomoku_free_models(ha)
        lib.gomoku_free_models(hb)
    return stats


def run_arena_match_parallel(
    lib_path: Path,
    a_nnue: Optional[Path],
    a_cnn: Optional[Path],
    a_depth: int,
    a_vcf_depth: int,
    a_time: float,
    a_cnn_weight: float,
    b_nnue: Optional[Path],
    b_cnn: Optional[Path],
    b_depth: int,
    b_vcf_depth: int,
    b_time: float,
    b_cnn_weight: float,
    num_games: int,
    seed: int,
    workers: int = 8,
) -> MatchStats:
    """按成对开局切分给多进程并行评测，汇总完全公平的对局统计。"""
    if workers <= 1 or num_games < 4:
        return run_arena_match(
            lib_path,
            a_nnue, a_cnn, a_depth, a_vcf_depth, a_time, a_cnn_weight,
            b_nnue, b_cnn, b_depth, b_vcf_depth, b_time, b_cnn_weight,
            num_games, seed,
        )

    total_pairs = max(1, num_games // 2)
    eff_workers = max(1, min(workers, total_pairs))
    pair_counts = [total_pairs // eff_workers] * eff_workers
    for i in range(total_pairs % eff_workers):
        pair_counts[i] += 1

    a_nnue_str = str(a_nnue) if (a_nnue and a_nnue.exists()) else ""
    a_cnn_str = str(a_cnn) if (a_cnn and a_cnn.exists()) else ""
    b_nnue_str = str(b_nnue) if (b_nnue and b_nnue.exists()) else ""
    b_cnn_str = str(b_cnn) if (b_cnn and b_cnn.exists()) else ""

    tasks = []
    pair_offset = 0
    for pc in pair_counts:
        tasks.append((
            str(lib_path),
            a_nnue_str, a_cnn_str, a_depth, a_vcf_depth, a_time, a_cnn_weight,
            b_nnue_str, b_cnn_str, b_depth, b_vcf_depth, b_time, b_cnn_weight,
            pc * 2,
            seed + pair_offset,
        ))
        pair_offset += pc

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=eff_workers) as pool:
        results = pool.map(_worker_arena_match, tasks)

    agg = MatchStats()
    a_time_sum = 0.0
    b_time_sum = 0.0
    total_g = 0
    for r in results:
        g = r.a_wins + r.b_wins + r.draws
        agg.a_wins += r.a_wins
        agg.b_wins += r.b_wins
        agg.draws += r.draws
        agg.illegal_moves += r.illegal_moves
        a_time_sum += r.a_avg_time_sec * g
        b_time_sum += r.b_avg_time_sec * g
        total_g += g

    agg.a_avg_time_sec = a_time_sum / max(1, total_g)
    agg.b_avg_time_sec = b_time_sum / max(1, total_g)
    return agg


def verify_c_parity(lib_path: Path, nnue_model, cnn_model) -> Dict[str, float]:
    """验证 PyTorch 模型输出与纯 C11 前向推理误差 < 1e-4。"""
    import torch
    from models import boards_to_cnn_tensor, export_cnn_bin, export_nnue_bin

    lib = _bind_lib(lib_path)
    rng = np.random.default_rng(2026)

    nnue_model.eval()
    cnn_model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        nnue_bin = Path(tmpdir) / "test_nnue.bin"
        cnn_bin = Path(tmpdir) / "test_cnn.bin"
        export_nnue_bin(nnue_model, nnue_bin)
        export_cnn_bin(cnn_model, cnn_bin)

        max_inp_diff = 0.0
        max_nnue_diff = 0.0
        max_cnn_val_diff = 0.0
        max_cnn_pol_diff = 0.0

        for idx in range(12):
            board = np.zeros(BOARD_CELLS, dtype=np.uint8)
            n_stones = 6 + idx * 3
            coords = rng.choice(BOARD_CELLS, size=n_stones, replace=False)
            for k, pos in enumerate(coords):
                board[pos] = 1 if (k % 2 == 0) else 2
            player = 1 if (n_stones % 2 == 0) else 2
            last_move = int(coords[-1])

            c_cnn_in = np.zeros((BOARD_SIZE, BOARD_SIZE, CNN_IN_CH), dtype=np.float32)
            c_feat = np.zeros(EVAL_FEATURE_DIM, dtype=np.float32)
            c_nnue_val = ctypes.c_float(0.0)
            c_cnn_val = ctypes.c_float(0.0)
            c_logits = np.zeros(BOARD_CELLS, dtype=np.float32)

            rc = lib.gomoku_eval_nn_debug(
                board.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                player,
                last_move,
                str(nnue_bin).encode("utf-8"),
                str(cnn_bin).encode("utf-8"),
                c_cnn_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                c_feat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(c_nnue_val),
                ctypes.byref(c_cnn_val),
                c_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            if rc != 0:
                raise RuntimeError(f"gomoku_eval_nn_debug failed with rc={rc}")

            # 取出 C 端第 5 通道作为 forbidden mask 校验 Python 张量构建一致性
            forb_plane = c_cnn_in[:, :, 5].reshape(1, BOARD_CELLS)
            py_cnn_in = boards_to_cnn_tensor(
                board.reshape(1, BOARD_CELLS),
                np.array([player], dtype=np.uint8),
                np.array([last_move], dtype=np.int32),
                forb_plane,
            )
            c_cnn_nchw = np.transpose(c_cnn_in, (2, 0, 1))[None, ...]
            max_inp_diff = max(max_inp_diff, float(np.max(np.abs(py_cnn_in - c_cnn_nchw))))

            with torch.no_grad():
                pt_nnue_val = nnue_model(
                    torch.from_numpy(c_feat).unsqueeze(0).cpu()
                ).item()
                pt_logits, pt_cnn_val = cnn_model(torch.from_numpy(py_cnn_in).cpu())
                pt_logits_np = pt_logits.squeeze(0).numpy()
                pt_cnn_val_f = float(pt_cnn_val.item())

            max_nnue_diff = max(max_nnue_diff, abs(pt_nnue_val - c_nnue_val.value))
            max_cnn_val_diff = max(max_cnn_val_diff, abs(pt_cnn_val_f - c_cnn_val.value))
            max_cnn_pol_diff = max(
                max_cnn_pol_diff, float(np.max(np.abs(pt_logits_np - c_logits)))
            )

        return {
            "max_input_diff": max_inp_diff,
            "max_nnue_diff": max_nnue_diff,
            "max_cnn_val_diff": max_cnn_val_diff,
            "max_cnn_pol_diff": max_cnn_pol_diff,
        }


def verify_pnnue_parity(
    lib_path: Path, pnnue_model, pnnue_bin_path: Optional[Path] = None, num_boards: int = 8
) -> Dict[str, float]:
    """验证 V3 PyTorch PatternNnueNet 与纯 C11 GPN1 前向推理数值一致 (< 1e-4)。"""
    import torch
    from pnnue_model import boards_to_pattern_ids, export_pnnue_bin

    lib = _bind_lib(lib_path)
    rng = np.random.default_rng(202611)
    pnnue_model.eval()

    def _run_with_path(bin_path: Path) -> Dict[str, float]:
        max_val_diff = 0.0
        max_pol_diff = 0.0
        for idx in range(num_boards):
            board = np.zeros(BOARD_CELLS, dtype=np.uint8)
            n_stones = 5 + idx * 4
            coords = rng.choice(BOARD_CELLS, size=n_stones, replace=False)
            for k, pos in enumerate(coords):
                board[pos] = 1 if (k % 2 == 0) else 2
            player = 1 if (idx % 2 == 0) else 2

            c_val = ctypes.c_float(0.0)
            c_logits = np.zeros(BOARD_CELLS, dtype=np.float32)
            rc = lib.gomoku_eval_pnnue_debug(
                board.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                player,
                str(bin_path).encode("utf-8"),
                ctypes.byref(c_val),
                c_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            if rc != 0:
                raise RuntimeError(f"gomoku_eval_pnnue_debug failed with rc={rc}")

            pat_ids = boards_to_pattern_ids(board.reshape(1, BOARD_SIZE, BOARD_SIZE))
            p_idx = 0 if player == 1 else 1
            with torch.no_grad():
                pt_logits, pt_val = pnnue_model(
                    torch.from_numpy(pat_ids).long(),
                    torch.tensor([p_idx], dtype=torch.long),
                )
                pt_logits_np = pt_logits.squeeze(0).numpy()
                pt_val_f = float(pt_val.item())

            max_val_diff = max(max_val_diff, abs(pt_val_f - c_val.value))
            max_pol_diff = max(max_pol_diff, float(np.max(np.abs(pt_logits_np - c_logits))))

        return {"max_val_diff": max_val_diff, "max_pol_diff": max_pol_diff}

    if pnnue_bin_path is not None:
        return _run_with_path(Path(pnnue_bin_path))
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_bin = Path(tmpdir) / "test_pnnue.bin"
        export_pnnue_bin(pnnue_model, tmp_bin)
        return _run_with_path(tmp_bin)


def _worker_generate_pnnue_games(args: Tuple) -> Dict[str, np.ndarray]:
    (
        lib_path_str,
        pnnue_path_str,
        num_games,
        max_depth,
        vcf_depth,
        time_limit,
        open_plies_base,
        explore_temp,
        base_seed,
    ) = args

    lib = _bind_lib(Path(lib_path_str))
    handle = lib.gomoku_load_pnnue(pnnue_path_str.encode("utf-8")) if pnnue_path_str else None

    all_boards: List[np.ndarray] = []
    all_players: List[np.ndarray] = []
    all_forb: List[np.ndarray] = []
    all_pol: List[np.ndarray] = []
    all_sval: List[np.ndarray] = []
    all_out: List[np.ndarray] = []

    boards_buf = np.zeros((BOARD_CELLS, BOARD_CELLS), dtype=np.uint8)
    players_buf = np.zeros(BOARD_CELLS, dtype=np.uint8)
    forb_buf = np.zeros((BOARD_CELLS, BOARD_CELLS), dtype=np.uint8)
    pol_buf = np.zeros((BOARD_CELLS, BOARD_CELLS), dtype=np.float32)
    sval_buf = np.zeros(BOARD_CELLS, dtype=np.float32)
    out_buf = np.zeros(BOARD_CELLS, dtype=np.float32)

    try:
        for g in range(num_games):
            seed = base_seed + g * 10007
            open_plies = open_plies_base + (g % 3)
            steps = lib.gomoku_generate_game_pnnue(
                handle,
                max_depth,
                vcf_depth,
                time_limit,
                open_plies,
                explore_temp,
                seed,
                boards_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                players_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                forb_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                pol_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                sval_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            if steps > 0:
                all_boards.append(boards_buf[:steps].copy().reshape(steps, BOARD_SIZE, BOARD_SIZE))
                all_players.append(players_buf[:steps].copy())
                all_forb.append(forb_buf[:steps].copy())
                all_pol.append(pol_buf[:steps].copy())
                all_sval.append(sval_buf[:steps].copy())
                all_out.append(out_buf[:steps].copy())
    finally:
        if handle:
            lib.gomoku_free_pnnue(handle)

    if not all_boards:
        return {}

    return {
        "boards": np.concatenate(all_boards, axis=0),
        "players": np.concatenate(all_players, axis=0),
        "forbidden": np.concatenate(all_forb, axis=0),
        "policies": np.concatenate(all_pol, axis=0),
        "values": np.concatenate(all_sval, axis=0),
        "outcomes": np.concatenate(all_out, axis=0),
    }


def generate_pnnue_games_parallel(
    lib_path: Path,
    pnnue_path: Optional[Path],
    num_games: int,
    workers: int,
    max_depth: int,
    vcf_depth: int,
    time_limit: float,
    random_opening_plies: int,
    explore_temp: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    p_str = str(pnnue_path) if (pnnue_path and Path(pnnue_path).exists()) else ""
    eff_workers = max(1, min(workers, num_games))
    per_worker = [num_games // eff_workers] * eff_workers
    for i in range(num_games % eff_workers):
        per_worker[i] += 1

    tasks = []
    offset_seed = seed
    for w_games in per_worker:
        tasks.append((
            str(lib_path),
            p_str,
            w_games,
            max_depth,
            vcf_depth,
            time_limit,
            random_opening_plies,
            explore_temp,
            offset_seed,
        ))
        offset_seed += w_games * 100003

    if eff_workers == 1:
        results = [_worker_generate_pnnue_games(tasks[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=eff_workers) as pool:
            results = pool.map(_worker_generate_pnnue_games, tasks)

    results = [r for r in results if r]
    if not results:
        return {}
    keys = results[0].keys()
    return {k: np.concatenate([r[k] for r in results], axis=0) for k in keys}


def _worker_pnnue_arena(args: Tuple) -> MatchStats:
    (
        lib_path_str,
        pa_str,
        pb_str,
        v2_nnue_str,
        v2_cnn_str,
        a_depth,
        a_vcf,
        a_time,
        b_depth,
        b_vcf,
        b_time,
        b_cnn_w,
        num_games,
        seed,
    ) = args

    lib = _bind_lib(Path(lib_path_str))
    pa = lib.gomoku_load_pnnue(pa_str.encode("utf-8")) if pa_str else None
    pb = lib.gomoku_load_pnnue(pb_str.encode("utf-8")) if pb_str else None
    v2_b = (
        lib.gomoku_load_models(v2_nnue_str.encode("utf-8"), v2_cnn_str.encode("utf-8"))
        if (not pb and (v2_nnue_str or v2_cnn_str))
        else None
    )

    stats = MatchStats()
    try:
        rc = lib.gomoku_run_match_pnnue(
            pa,
            a_depth,
            a_vcf,
            a_time,
            pb,
            v2_b,
            b_depth,
            b_vcf,
            b_time,
            b_cnn_w,
            num_games,
            seed,
            ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"gomoku_run_match_pnnue failed with rc={rc}")
    finally:
        if pa:
            lib.gomoku_free_pnnue(pa)
        if pb:
            lib.gomoku_free_pnnue(pb)
        if v2_b:
            lib.gomoku_free_models(v2_b)
    return stats


def run_pnnue_arena_parallel(
    lib_path: Path,
    pnnue_a_path: Optional[Path],
    pnnue_b_path: Optional[Path],
    v2_b_nnue_path: Optional[Path],
    v2_b_cnn_path: Optional[Path],
    num_games: int,
    workers: int,
    a_depth: int,
    a_vcf: int,
    a_time: float,
    b_depth: int,
    b_vcf: int,
    b_time: float,
    b_cnn_w: float,
    seed: int,
) -> Dict[str, float]:
    pa_str = str(pnnue_a_path) if (pnnue_a_path and Path(pnnue_a_path).exists()) else ""
    pb_str = str(pnnue_b_path) if (pnnue_b_path and Path(pnnue_b_path).exists()) else ""
    v2_n_str = str(v2_b_nnue_path) if (v2_b_nnue_path and Path(v2_b_nnue_path).exists()) else ""
    v2_c_str = str(v2_b_cnn_path) if (v2_b_cnn_path and Path(v2_b_cnn_path).exists()) else ""

    eff_workers = max(1, min(workers, max(1, num_games // 2)))
    base_pairs = (num_games // 2) // eff_workers
    rem_pairs = (num_games // 2) % eff_workers

    tasks = []
    cur_seed = seed
    for w in range(eff_workers):
        pairs = base_pairs + (1 if w < rem_pairs else 0)
        g = pairs * 2
        if w == eff_workers - 1 and (num_games % 2 == 1):
            g += 1
        if g <= 0:
            continue
        tasks.append((
            str(lib_path),
            pa_str,
            pb_str,
            v2_n_str,
            v2_c_str,
            a_depth,
            a_vcf,
            a_time,
            b_depth,
            b_vcf,
            b_time,
            b_cnn_w,
            g,
            cur_seed,
        ))
        cur_seed += 97

    if len(tasks) <= 1:
        parts = [_worker_pnnue_arena(tasks[0])] if tasks else []
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(tasks)) as pool:
            parts = pool.map(_worker_pnnue_arena, tasks)

    a_wins = sum(p.a_wins for p in parts)
    b_wins = sum(p.b_wins for p in parts)
    draws = sum(p.draws for p in parts)
    illegal = sum(p.illegal_moves for p in parts)
    total = max(1, a_wins + b_wins + draws)
    cand_ms = float(np.mean([p.a_avg_time_sec * 1000.0 for p in parts])) if parts else 0.0
    base_ms = float(np.mean([p.b_avg_time_sec * 1000.0 for p in parts])) if parts else 0.0

    return {
        "wins": int(a_wins),
        "losses": int(b_wins),
        "draws": int(draws),
        "illegal": int(illegal),
        "score": float(a_wins + 0.5 * draws) / float(total),
        "cand_ms": cand_ms,
        "base_ms": base_ms,
    }
