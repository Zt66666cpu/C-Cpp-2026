#!/usr/bin/env python3
"""Zero-dependency HTTP bridge exposing the pure C11 Gomoku/Renju engine to web_gui."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
EVAL_FEATURE_DIM = 32
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
_DLL_HANDLES: List[object] = []


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
        project_root / "cmake-build-release" / "Release",
        project_root / "build-win",
        project_root / "build-win" / "Release",
        project_root / "cmake-build-debug",
        project_root / "build",
        project_root / "build" / "Release",
    ]
    for d in search_dirs:
        for name in names:
            cand = d / name
            if cand.exists():
                return cand
    raise FileNotFoundError(
        f"Cannot find gomoku_engine shared library under {project_root}. Build Release first."
    )


def _bind_lib(lib_path: Path) -> ctypes.CDLL:
    if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
        try:
            _DLL_HANDLES.append(os.add_dll_directory(str(lib_path.parent.resolve())))
        except OSError:
            pass

    lib = ctypes.CDLL(str(lib_path))

    lib.gomoku_load_models.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.gomoku_load_models.restype = ctypes.c_void_p

    lib.gomoku_free_models.argtypes = [ctypes.c_void_p]
    lib.gomoku_free_models.restype = None

    lib.gomoku_extract_state.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.gomoku_extract_state.restype = ctypes.c_int

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


def check_winner(board: List[int]) -> Tuple[int, List[List[int]]]:
    """Check Renju win condition: Black exact 5, White >= 5, or full-board draw (3)."""
    dirs = ((0, 1), (1, 0), (1, 1), (1, -1))
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = board[r * BOARD_SIZE + c]
            if p not in (1, 2):
                continue
            for dr, dc in dirs:
                pr, pc = r - dr, c - dc
                if 0 <= pr < BOARD_SIZE and 0 <= pc < BOARD_SIZE and board[pr * BOARD_SIZE + pc] == p:
                    continue
                line: List[List[int]] = [[r, c]]
                nr, nc = r + dr, c + dc
                while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr * BOARD_SIZE + nc] == p:
                    line.append([nr, nc])
                    nr += dr
                    nc += dc
                if p == 1 and len(line) == 5:
                    return 1, line
                if p == 2 and len(line) >= 5:
                    return 2, line[:5]
    if all(cell != 0 for cell in board):
        return 3, []
    return 0, []


def _to_black_winrate(side_val: float, player: int) -> float:
    b_val = side_val if player == 1 else -side_val
    return round(max(0.0, min(100.0, (b_val + 1.0) * 50.0)), 2)


def _top_probs_from_vec(probs: List[float], limit: int = 20) -> List[Dict[str, Any]]:
    ranked = sorted(
        ((idx, p) for idx, p in enumerate(probs) if p > 1e-4),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    return [
        {
            "idx": idx,
            "row": idx // BOARD_SIZE,
            "col": idx % BOARD_SIZE,
            "prob": round(float(p), 4),
        }
        for idx, p in ranked
    ]


class GomokuEngineBridge:
    """Thread-safe ctypes bridge managing V3, V2, and heuristic C11 engine handles."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.lib_path = find_engine_lib(project_root)
        self.lib = _bind_lib(self.lib_path)
        self.lock = threading.Lock()

        self.pnnue_path = project_root / "models" / "pnnue.bin"
        self.nnue_path = project_root / "models" / "nnue.bin"
        self.cnn_path = project_root / "models" / "cnn.bin"

        self.v3_handle = (
            self.lib.gomoku_load_pnnue(str(self.pnnue_path).encode("utf-8"))
            if self.pnnue_path.exists()
            else None
        )
        self.pnnue_mtime: float = (
            self.pnnue_path.stat().st_mtime if self.v3_handle is not None else 0.0
        )
        v2_n = str(self.nnue_path).encode("utf-8") if self.nnue_path.exists() else b""
        v2_c = str(self.cnn_path).encode("utf-8") if self.cnn_path.exists() else b""
        self.v2_handle = self.lib.gomoku_load_models(v2_n, v2_c) if (v2_n or v2_c) else None
        self.heur_handle = self.lib.gomoku_load_models(b"", b"")

    def _maybe_reload_pnnue(self) -> None:
        if not self.pnnue_path.exists():
            return
        try:
            st = self.pnnue_path.stat()
        except OSError:
            return
        if st.st_size <= 0 or st.st_mtime <= self.pnnue_mtime + 0.5:
            return
        new_h = self.lib.gomoku_load_pnnue(str(self.pnnue_path).encode("utf-8"))
        if new_h:
            old_h = self.v3_handle
            self.v3_handle = new_h
            self.pnnue_mtime = st.st_mtime
            if old_h:
                self.lib.gomoku_free_pnnue(old_h)
            print(f"[WebGUI] Hot-reloaded models/pnnue.bin (mtime={st.st_mtime:.1f})", flush=True)

    def close(self) -> None:
        with self.lock:
            if self.v3_handle:
                self.lib.gomoku_free_pnnue(self.v3_handle)
                self.v3_handle = None
            if self.v2_handle:
                self.lib.gomoku_free_models(self.v2_handle)
                self.v2_handle = None
            if self.heur_handle:
                self.lib.gomoku_free_models(self.heur_handle)
                self.heur_handle = None

    def info(self) -> Dict[str, Any]:
        with self.lock:
            self._maybe_reload_pnnue()
            v3_ok = self.v3_handle is not None
            updated = (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.pnnue_mtime))
                if self.pnnue_mtime > 0
                else None
            )
        return {
            "lib_path": str(self.lib_path),
            "board_size": BOARD_SIZE,
            "models": {
                "v3": v3_ok,
                "v2": self.v2_handle is not None,
                "heuristic": self.heur_handle is not None,
            },
            "pnnue_updated_at": updated,
        }

    def extract_forbidden(self, board: List[int], player: int) -> List[int]:
        c_board = (ctypes.c_uint8 * BOARD_CELLS)(*board)
        c_forb = (ctypes.c_uint8 * BOARD_CELLS)()
        c_feat = (ctypes.c_float * EVAL_FEATURE_DIM)()
        with self.lock:
            rc = self.lib.gomoku_extract_state(c_board, int(player), c_forb, c_feat)
        if rc != 0:
            raise RuntimeError(f"gomoku_extract_state failed with rc={rc}")
        return [int(c_forb[i]) for i in range(BOARD_CELLS)]

    def eval_v3_prior(
        self, board: List[int], player: int, forbidden: Optional[List[int]] = None
    ) -> Tuple[Optional[float], List[Dict[str, Any]]]:
        if not self.pnnue_path.exists():
            return None, []
        if forbidden is None:
            forbidden = self.extract_forbidden(board, player)
        c_board = (ctypes.c_uint8 * BOARD_CELLS)(*board)
        c_val = ctypes.c_float(0.0)
        c_logits = (ctypes.c_float * BOARD_CELLS)()
        with self.lock:
            self._maybe_reload_pnnue()
            rc = self.lib.gomoku_eval_pnnue_debug(
                c_board,
                int(player),
                str(self.pnnue_path).encode("utf-8"),
                ctypes.byref(c_val),
                c_logits,
            )
        if rc != 0:
            return None, []

        legal_indices = [
            i for i in range(BOARD_CELLS) if board[i] == 0 and (player == 2 or forbidden[i] == 0)
        ]
        if not legal_indices:
            return _to_black_winrate(float(c_val.value), player), []

        max_lg = max(float(c_logits[i]) for i in legal_indices)
        exps = [0.0] * BOARD_CELLS
        total = 0.0
        for i in legal_indices:
            w = math.exp(max(-30.0, float(c_logits[i]) - max_lg))
            exps[i] = w
            total += w
        if total > 0.0:
            probs = [w / total for w in exps]
        else:
            probs = [0.0] * BOARD_CELLS
        return _to_black_winrate(float(c_val.value), player), _top_probs_from_vec(probs, limit=12)

    def search_move(
        self,
        board: List[int],
        player: int,
        last_move: int,
        engine: str,
        depth: int,
        time_limit: float,
        vcf_depth: int,
    ) -> Dict[str, Any]:
        c_board = (ctypes.c_uint8 * BOARD_CELLS)(*board)
        c_best = ctypes.c_int(-1)
        c_sval = ctypes.c_float(0.0)
        c_pol = (ctypes.c_float * BOARD_CELLS)()
        c_forb = (ctypes.c_uint8 * BOARD_CELLS)()
        c_feat = (ctypes.c_float * EVAL_FEATURE_DIM)()

        chosen = engine
        if chosen == "v3" and self.v3_handle is None:
            chosen = "v2" if self.v2_handle is not None else "heuristic"
        if chosen == "v2" and self.v2_handle is None:
            chosen = "heuristic"

        t0 = time.perf_counter()
        with self.lock:
            self._maybe_reload_pnnue()
            if chosen == "v3" and self.v3_handle is None:
                chosen = "v2" if self.v2_handle is not None else "heuristic"
            if chosen == "v3":
                rc = self.lib.gomoku_step_search_pnnue(
                    self.v3_handle,
                    c_board,
                    int(player),
                    int(last_move),
                    int(depth),
                    int(vcf_depth),
                    float(time_limit),
                    0.0,
                    int(time.time() * 1000) & 0xFFFFFFFFFFFFFFFF,
                    ctypes.byref(c_best),
                    ctypes.byref(c_sval),
                    c_pol,
                    c_forb,
                )
            elif chosen == "v2":
                rc = self.lib.gomoku_step_search(
                    self.v2_handle,
                    c_board,
                    int(player),
                    int(last_move),
                    int(depth),
                    int(vcf_depth),
                    float(time_limit),
                    0.0,
                    0.35 if self.cnn_path.exists() else 0.0,
                    int(time.time() * 1000) & 0xFFFFFFFFFFFFFFFF,
                    ctypes.byref(c_best),
                    ctypes.byref(c_sval),
                    c_pol,
                    c_forb,
                    c_feat,
                )
            else:
                chosen = "heuristic"
                rc = self.lib.gomoku_step_search(
                    self.heur_handle,
                    c_board,
                    int(player),
                    int(last_move),
                    int(depth),
                    int(vcf_depth),
                    float(time_limit),
                    0.0,
                    0.0,
                    int(time.time() * 1000) & 0xFFFFFFFFFFFFFFFF,
                    ctypes.byref(c_best),
                    ctypes.byref(c_sval),
                    c_pol,
                    c_forb,
                    c_feat,
                )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if rc != 0 or c_best.value < 0 or c_best.value >= BOARD_CELLS:
            raise RuntimeError(f"Engine search failed (engine={chosen}, rc={rc}, best={c_best.value})")

        idx = int(c_best.value)
        pol_list = [float(c_pol[i]) for i in range(BOARD_CELLS)]
        sval = float(c_sval.value)
        return {
            "engine_used": chosen,
            "move_idx": idx,
            "row": idx // BOARD_SIZE,
            "col": idx % BOARD_SIZE,
            "search_value": round(sval, 4),
            "black_winrate": _to_black_winrate(sval, player),
            "elapsed_ms": elapsed_ms,
            "policy": _top_probs_from_vec(pol_list, limit=20),
        }


def _validate_board_payload(payload: Dict[str, Any]) -> Tuple[List[int], int]:
    board = payload.get("board")
    if not isinstance(board, list) or len(board) != BOARD_CELLS:
        raise ValueError("`board` must be a list of 225 integers")
    norm_board: List[int] = []
    for cell in board:
        if not isinstance(cell, int) or cell not in (0, 1, 2):
            raise ValueError("Each cell in `board` must be 0, 1, or 2")
        norm_board.append(cell)
    player = int(payload.get("player", 1))
    if player not in (1, 2):
        raise ValueError("`player` must be 1 (Black) or 2 (White)")
    return norm_board, player


def make_handler(bridge: GomokuEngineBridge):
    index_path = WEB_DIR / "index.html"

    class GomokuHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, data: Dict[str, Any]) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                html = index_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            if path == "/api/info":
                self._send_json(HTTPStatus.OK, bridge.info())
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                payload = self._read_json()
                board, player = _validate_board_payload(payload)

                if path == "/api/state":
                    winner, win_line = check_winner(board)
                    forb = bridge.extract_forbidden(board, player)
                    prior_wr = None
                    prior_pol: List[Dict[str, Any]] = []
                    if payload.get("need_prior", False) and winner == 0:
                        prior_wr, prior_pol = bridge.eval_v3_prior(board, player, forb)
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "forbidden": forb,
                            "winner": winner,
                            "win_line": win_line,
                            "black_winrate": prior_wr,
                            "prior_policy": prior_pol,
                        },
                    )
                    return

                if path == "/api/move":
                    winner, win_line = check_winner(board)
                    if winner != 0:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": "Game is already over", "winner": winner, "win_line": win_line},
                        )
                        return
                    last_move = int(payload.get("last_move", -1))
                    engine = str(payload.get("engine", "v3")).lower()
                    depth = max(1, min(10, int(payload.get("depth", 6))))
                    vcf_depth = max(0, min(20, int(payload.get("vcf_depth", 14))))
                    time_limit = max(0.05, min(10.0, float(payload.get("time_limit", 0.8))))

                    res = bridge.search_move(
                        board, player, last_move, engine, depth, time_limit, vcf_depth
                    )
                    idx = res["move_idx"]
                    if board[idx] != 0:
                        raise RuntimeError(f"Engine returned occupied cell idx={idx}")
                    after_board = list(board)
                    after_board[idx] = player
                    next_player = 3 - player
                    win_after, line_after = check_winner(after_board)
                    forb_after = bridge.extract_forbidden(after_board, next_player)
                    res["winner"] = win_after
                    res["win_line"] = line_after
                    res["forbidden_after"] = forb_after
                    self._send_json(HTTPStatus.OK, res)
                    return

                if path == "/api/analyze":
                    winner, win_line = check_winner(board)
                    if winner != 0:
                        self._send_json(
                            HTTPStatus.OK,
                            {
                                "best_row": -1,
                                "best_col": -1,
                                "best_idx": -1,
                                "black_winrate": 100.0 if winner == 1 else (0.0 if winner == 2 else 50.0),
                                "elapsed_ms": 0.0,
                                "policy": [],
                                "winner": winner,
                                "win_line": win_line,
                            },
                        )
                        return
                    last_move = int(payload.get("last_move", -1))
                    engine = str(payload.get("engine", "v3")).lower()
                    depth = max(1, min(10, int(payload.get("depth", 6))))
                    vcf_depth = max(0, min(20, int(payload.get("vcf_depth", 14))))
                    time_limit = max(0.05, min(10.0, float(payload.get("time_limit", 0.8))))

                    res = bridge.search_move(
                        board, player, last_move, engine, depth, time_limit, vcf_depth
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "engine_used": res["engine_used"],
                            "best_row": res["row"],
                            "best_col": res["col"],
                            "best_idx": res["move_idx"],
                            "black_winrate": res["black_winrate"],
                            "elapsed_ms": res["elapsed_ms"],
                            "policy": res["policy"][:12],
                        },
                    )
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    return GomokuHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Gomoku/Renju Web GUI server backed by C11 engine")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()

    bridge = GomokuEngineBridge(PROJECT_ROOT)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(bridge))
    print(
        f"[WebGUI] http://{args.host}:{args.port} | lib={bridge.lib_path} | "
        f"models={bridge.info()['models']}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        bridge.close()


if __name__ == "__main__":
    main()
