import ctypes
import multiprocessing as mp
import platform
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BOARD_SIZE = 15
BOARD_CELLS = 225
EVAL_FEATURE_DIM = 32
CNN_IN_CH = 6


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
        project_root / "cmake-build-release",
        project_root / "cmake-build-debug",
        project_root / "build",
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
    return lib


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
