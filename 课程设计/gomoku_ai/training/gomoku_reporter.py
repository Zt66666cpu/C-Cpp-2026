#!/usr/bin/env python3
"""将 Windows 上的五子棋训练指标与系统资源推送到 Mac 监控台。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


def _load_token(explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    env_tok = os.environ.get("MONITOR_TOKEN") or os.environ.get("TINYSTORIES_MONITOR_TOKEN") or ""
    if env_tok.strip():
        return env_tok.strip()
    default_file = Path(r"D:\TinyStories\monitor_token.txt")
    if default_file.exists():
        return default_file.read_text(encoding="utf-8").strip()
    return ""


def _read_jsonl(metrics_path: Path, phase: str) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            raw = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        row_phase = str(raw.get("phase") or "v1")
        if row_phase == phase:
            rows.append(raw)
    return rows


def _is_v3(phase: str) -> bool:
    return phase.startswith("v3")


def _parse_cli_args(command: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "resume": bool(re.search(r"(?:^|\s)--resume(?:\s|$)", command)),
    }
    for key, flag in (
        ("rounds", "--rounds"),
        ("workers", "--workers"),
        ("depth", "--depth"),
        ("nnue_h1", "--nnue-h1"),
        ("nnue_h2", "--nnue-h2"),
        ("ch", "--ch"),
        ("blocks", "--blocks"),
        ("dim", "--dim"),
        ("v_hidden", "--v-hidden"),
    ):
        m = re.search(rf"{re.escape(flag)}\s+(\d+)", command)
        if m:
            parsed[key] = int(m.group(1))
    for key, flag in (("cnn_lr", "--cnn-lr"), ("lr", "--lr")):
        m_lr = re.search(rf"{re.escape(flag)}\s+([\d.eE+-]+)", command)
        if m_lr:
            try:
                parsed[key] = float(m_lr.group(1))
            except ValueError:
                pass
    return parsed


def _detect_train_process(project_dir: Path, script: str = "train_loop.py") -> dict[str, Any] | None:
    if psutil is None:
        return None
    proj_resolved = str(project_dir.resolve()).lower()
    main_proc = None
    for proc in psutil.process_iter(["pid", "ppid", "cmdline", "cwd", "create_time"]):
        try:
            info = proc.info
            cmdline = info.get("cmdline") or []
            cmd_str = " ".join(cmdline)
            if script not in cmd_str or "gomoku_reporter.py" in cmd_str:
                continue
            cwd = (info.get("cwd") or "").lower()
            if proj_resolved in cmd_str.lower() or cwd == proj_resolved:
                main_proc = (proc, info, cmd_str)
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    if main_proc is None:
        return None

    proc, info, cmd_str = main_proc
    worker_count = 0
    try:
        for child in proc.children(recursive=False):
            try:
                child_cmd = " ".join(child.cmdline())
                if "multiprocessing" in child_cmd or "spawn_main" in child_cmd:
                    worker_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        pass

    elapsed_s = max(0.0, time.time() - float(info.get("create_time") or time.time()))
    return {
        "pid": info["pid"],
        "worker_count": worker_count,
        "elapsed_seconds": elapsed_s,
        "command": cmd_str,
        "cli_args": _parse_cli_args(cmd_str),
    }


def _gpu_stats() -> dict[str, Any]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        if not output:
            return {}
        vals = [part.strip() for part in output.splitlines()[0].split(",")]
        return {
            "gpu_utilization": float(vals[0]),
            "gpu_memory_used_mb": float(vals[1]),
            "gpu_memory_total_mb": float(vals[2]),
            "gpu_temperature_c": float(vals[3]),
        }
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


def _system_stats(root: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    if psutil is not None:
        try:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage(str(root))
            stats.update(
                {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_used_percent": mem.percent,
                    "memory_total_mb": round(mem.total / 1024 / 1024, 1),
                    "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 1),
                }
            )
        except (OSError, RuntimeError):
            pass
    stats.update(_gpu_stats())
    return stats


def _tail_lines(path: Path, limit: int = 80) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as fp:
            fp.seek(0, os.SEEK_END)
            size = fp.tell()
            fp.seek(max(0, size - 100_000), os.SEEK_SET)
            raw = fp.read()
        text = raw.decode("utf-8", errors="replace")
        return text.splitlines()[-limit:]
    except OSError:
        return []


def _build_payload(
    machine: str,
    run: str,
    phase: str,
    project_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    runs_dir = project_dir / "training" / "runs"
    data_dir = project_dir / "training" / "data"
    v3_dir = project_dir / "training" / "runs_v3"
    metrics_path = runs_dir / "metrics.jsonl"
    v3 = _is_v3(phase)
    raw_rows = _read_jsonl(metrics_path, phase)
    proc = _detect_train_process(project_dir, "train_pnnue.py" if v3 else "train_loop.py")
    cli = (proc or {}).get("cli_args") or {}
    base_lr = float(cli.get("lr") or 8e-4) if v3 else float(cli.get("cnn_lr") or 1.0e-3)

    rows: list[dict[str, Any]] = []
    cum_sec = 0.0
    for raw in raw_rows:
        rnd = int(raw.get("round") or 1)
        cum_sec += float(raw.get("total_sec") or 0.0)
        if v3:
            # V3 单网：val_mse/val_top1 复用监控台 “NNUE MSE / Top-1” 槽位，vs_v2 复用 “vs V1” 槽位
            rows.append(
                {
                    "iter": rnd,
                    "round": rnd,
                    "phase": phase,
                    "train_loss": float(raw.get("train_loss") or raw.get("val_loss") or 0.0),
                    "val_loss": float(raw.get("val_loss") or 0.0),
                    "lr": float(raw.get("lr") or base_lr),
                    "elapsed_s": round(cum_sec, 2),
                    "tokens_seen": int(raw.get("buffer_positions") or 0),
                    "new_positions": int(raw.get("new_positions") or 0),
                    "buffer_positions": int(raw.get("buffer_positions") or 0),
                    "selfplay_sec": float(raw.get("selfplay_sec") or 0.0),
                    "train_sec": float(raw.get("train_sec") or 0.0),
                    "eval_sec": float(raw.get("eval_sec") or 0.0),
                    "total_sec": float(raw.get("total_sec") or 0.0),
                    "nnue_val_mse": float(raw.get("val_mse") or 0.0),
                    "nnue_val_loss": float(raw.get("val_loss") or 0.0),
                    "cnn_val_top1": float(raw.get("val_top1") or 0.0),
                    "best_ep": raw.get("best_ep"),
                    "gate_score": raw.get("gate_score"),
                    "nnue_gate_score": raw.get("gate_score"),
                    "nnue_promoted": bool(raw.get("promoted")),
                    "promoted": bool(raw.get("promoted")),
                    "prod_deployed": bool(raw.get("prod_deployed")),
                    "vs_heuristic": raw.get("vs_heuristic") or {},
                    "vs_v2": raw.get("vs_v2") or {},
                    "vs_v1": raw.get("vs_v2") or {},
                }
            )
            continue
        decay = max(0.4, 1.0 - 0.08 * (rnd - 1)) if phase == "v2" else 1.0
        nnue_train = raw.get("nnue_train_loss")
        if nnue_train is None:
            nnue_train = raw.get("nnue_train_mse") or 0.0
        nnue_val = raw.get("nnue_val_loss")
        if nnue_val is None:
            nnue_val = raw.get("nnue_val_mse") or 0.0
        rows.append(
            {
                "iter": rnd,
                "round": rnd,
                "phase": phase,
                "train_loss": float(raw.get("cnn_train_pol") or 0.0),
                "val_loss": float(raw.get("cnn_val_pol") or 0.0),
                "lr": float(raw.get("lr") or (base_lr * decay)),
                "elapsed_s": round(cum_sec, 2),
                "tokens_seen": int(raw.get("buffer_positions") or 0),
                "new_positions": int(raw.get("new_positions") or 0),
                "buffer_positions": int(raw.get("buffer_positions") or 0),
                "selfplay_sec": float(raw.get("selfplay_sec") or 0.0),
                "train_sec": float(raw.get("train_sec") or 0.0),
                "eval_sec": float(raw.get("eval_sec") or 0.0),
                "total_sec": float(raw.get("total_sec") or 0.0),
                "nnue_train_loss": float(nnue_train),
                "nnue_val_loss": float(nnue_val),
                "nnue_val_mse": float(raw.get("nnue_val_mse") or nnue_val),
                "nnue_best_ep": raw.get("nnue_best_ep"),
                "cnn_train_pol": float(raw.get("cnn_train_pol") or 0.0),
                "cnn_train_val": float(raw.get("cnn_train_val") or 0.0),
                "cnn_val_pol": float(raw.get("cnn_val_pol") or 0.0),
                "cnn_val_mse": float(raw.get("cnn_val_mse") or 0.0),
                "cnn_val_top1": float(raw.get("cnn_val_top1") or 0.0),
                "cnn_best_ep": raw.get("cnn_best_ep"),
                "nnue_gate_score": raw.get("nnue_gate_score"),
                "cnn_gate_score": raw.get("cnn_gate_score"),
                "nnue_promoted": bool(raw.get("nnue_promoted")),
                "cnn_promoted": bool(raw.get("cnn_promoted")),
                "prod_deployed": bool(raw.get("prod_deployed")),
                "promoted": bool(raw.get("promoted")),
                "vs_champion_score": raw.get("vs_champion_score"),
                "vs_heuristic": raw.get("vs_heuristic") or {},
                "vs_v1": raw.get("vs_v1") or {},
            }
        )

    log_tail = _tail_lines(log_path, limit=80)
    resumed_from = None
    for line in log_tail:
        m_res = re.search(r"Resumed (?:champion states|checkpoint) from round (\d+)", line)
        if m_res:
            resumed_from = int(m_res.group(1))
    in_pretrain = v3 and any("[Pretrain V3] Loading" in ln for ln in log_tail) and not any(
        ("V3 Round" in ln) or ("[Pretrain V3]" in ln and "epochs in" in ln) for ln in log_tail
    )

    completed_rounds = len(rows)
    is_infinite = bool(proc and ("rounds" in cli) and int(cli["rounds"]) == 0)
    if is_infinite:
        max_rounds = max(completed_rounds + 1, (rows[-1]["round"] + 1) if rows else 1)
    elif proc and cli.get("rounds"):
        if cli.get("resume") and resumed_from is not None:
            max_rounds = resumed_from + int(cli["rounds"])
        elif cli.get("resume"):
            max_rounds = max(int(cli["rounds"]), completed_rounds)
        else:
            max_rounds = max(int(cli["rounds"]), completed_rounds)
    else:
        max_rounds = max(1, completed_rounds, rows[-1]["round"] if rows else 1)

    now = time.time()
    metrics_mtime = metrics_path.stat().st_mtime if metrics_path.exists() else now
    recent_sp = [r["selfplay_sec"] for r in rows[-2:] if r["selfplay_sec"] > 0]
    recent_tr = [r["train_sec"] for r in rows[-2:] if r["train_sec"] > 0]
    recent_ev = [r["eval_sec"] for r in rows[-2:] if r["eval_sec"] > 0]
    recent_tot = [r["total_sec"] for r in rows[-2:] if r["total_sec"] > 0]
    est_sp = (sum(recent_sp) / len(recent_sp)) if recent_sp else 180.0
    est_tr = (sum(recent_tr) / len(recent_tr)) if recent_tr else 90.0
    est_ev = (sum(recent_ev) / len(recent_ev)) if recent_ev else 60.0
    est_round = max(30.0, est_sp + est_tr + est_ev)
    avg_round_sec = (sum(recent_tot) / len(recent_tot)) if recent_tot else est_round

    stage = "idle"
    stage_label = "已完成" if (completed_rounds >= max_rounds and rows) else "待命"
    stage_detail = "当前未在运行"
    stage_elapsed_s = 0.0
    current_round = completed_rounds
    progress = min(1.0, completed_rounds / max(1, max_rounds)) if rows else 0.0
    eta_hours = None
    worker_count = int((proc or {}).get("worker_count") or cli.get("workers") or 0)

    if proc is not None and completed_rounds < max_rounds:
        next_round = (rows[-1]["round"] + 1) if rows else 1
        current_round = next_round
        if v3:
            champ_pt = v3_dir / "pnnue_champ.pt"
            champ_ts = champ_pt.stat().st_mtime if champ_pt.exists() else 0.0
            base_ts = metrics_mtime if rows else 0.0
            last_round_ts = max(base_ts, champ_ts) or max(0.0, now - float(proc["elapsed_seconds"]))
            buf_path = v3_dir / "pnnue_buffer.npz"
            cand_paths = [v3_dir / "pnnue_cand.bin"]
            train_label = "CUDA 训练中"
            train_detail = f"第 {next_round} 轮 · Pattern-NNUE 单网训练（RTX 4070 CUDA）"
        else:
            last_round_ts = metrics_mtime if rows else max(0.0, now - float(proc["elapsed_seconds"]))
            buf_path = data_dir / "replay_buffer.npz"
            cand_paths = [runs_dir / f"cand_cnn_r{next_round}.bin", runs_dir / f"cand_nnue_r{next_round}.bin"]
            train_label = "CUDA 训练中"
            train_detail = f"第 {next_round} 轮 · NNUE + CNN 神经网络训练（RTX 4070 CUDA）"
        buf_ts = buf_path.stat().st_mtime if buf_path.exists() else 0.0
        cand_ts = next((p.stat().st_mtime for p in cand_paths if p.exists()), 0.0)
        w_disp = worker_count or int(cli.get("workers") or 16)
        if in_pretrain:
            stage = "train"
            stage_label = "预训练中"
            stage_detail = "V3 冷启动 · 加载 V2 样本库 + Rapfi 蒸馏样本预训练 Pattern-NNUE"
            stage_elapsed_s = float(proc["elapsed_seconds"])
            intra_raw = 0.02
            rem_cur = est_round
        elif cand_ts > last_round_ts + 1.0:
            stage = "eval"
            stage_label = "评测门控中"
            stage_detail = f"第 {next_round} 轮 · 对弈评测与门控筛选（{w_disp} workers）"
            stage_elapsed_s = max(0.0, now - cand_ts)
            intra_raw = (est_sp + est_tr + stage_elapsed_s) / est_round
            rem_cur = max(5.0, est_ev - stage_elapsed_s)
        elif buf_ts > last_round_ts + 1.0:
            stage = "train"
            stage_label = train_label
            stage_detail = train_detail
            stage_elapsed_s = max(0.0, now - buf_ts)
            intra_raw = (est_sp + stage_elapsed_s) / est_round
            rem_cur = max(10.0, est_tr - stage_elapsed_s) + est_ev
        else:
            stage = "selfplay"
            stage_label = "自博弈中"
            stage_detail = f"第 {next_round} 轮 · C 引擎自博弈生成局面（{w_disp} workers）"
            stage_elapsed_s = max(0.0, now - last_round_ts)
            intra_raw = stage_elapsed_s / est_round
            rem_cur = max(15.0, est_sp - stage_elapsed_s) + est_tr + est_ev

        intra_frac = max(0.02, min(0.96, intra_raw))
        progress = min(0.995, (completed_rounds + intra_frac) / max(1, max_rounds))
        if is_infinite:
            eta_hours = None
        else:
            rem_after = max(0, max_rounds - completed_rounds - 1) * est_round
            eta_hours = (rem_cur + rem_after) / 3600.0

    heur_scores = [
        float(r["vs_heuristic"]["score"])
        for r in rows
        if isinstance(r.get("vs_heuristic"), dict) and r["vs_heuristic"].get("score") is not None
    ]
    v1_scores = [
        float(r["vs_v1"]["score"])
        for r in rows
        if isinstance(r.get("vs_v1"), dict) and r["vs_v1"].get("score") is not None
    ]
    latest = rows[-1] if rows else None
    buf_pos = latest["buffer_positions"] if latest else 0
    if v3:
        ver_tag = "V4-GPN2" if (cli.get("arch") == "v4" or " --arch v4" in str((proc or {}).get("command") or "")) else "Pattern-NNUE"
        arch = (
            f"{ver_tag} 22143×{cli.get('dim') or 64}→V({cli.get('v_hidden') or 32})+P · "
            f"depth={cli.get('depth') or 6} · 「vs V1」栏 = vs V2 基线"
            + (" · 持续自进化" if is_infinite else "")
        )
    else:
        h1 = cli.get("nnue_h1") or (128 if phase == "v2" else 64)
        h2 = cli.get("nnue_h2") or (64 if phase == "v2" else 32)
        ch = cli.get("ch") or (48 if phase == "v2" else 32)
        blocks = cli.get("blocks") or (4 if phase == "v2" else 3)
        arch = f"NNUE 32→{h1}→{h2}→1 · CNN {ch}ch×{blocks}blocks"

    w_label = f"{worker_count or int(cli.get('workers') or 0)}w"
    speed_text = f"{buf_pos / 1000:.1f}K 局面 · {w_label}" if buf_pos else f"4070 CUDA · {w_label}"
    elapsed_min = (float(proc["elapsed_seconds"]) / 60.0) if proc else (cum_sec / 60.0)

    if not log_tail:
        log_tail = [
            f"[Windows-4070 Reporter] run={run} | phase={phase} | arch={arch}",
            f"[Stage] {stage_label} · {stage_detail}",
            f"[Metrics] completed_rounds={completed_rounds}/{max_rounds} | buffer={buf_pos}",
        ]

    return {
        "machine": machine,
        "run": run,
        "preset": f"gomoku-{phase}",
        "kind": "gomoku",
        "unit_label": "轮",
        "running": proc is not None,
        "progress": round(progress, 5),
        "max_iters": int(max_rounds),
        "metrics": rows,
        "log_progress": {
            "iter": completed_rounds,
            "max_iters": int(max_rounds),
            "loss": latest["val_loss"] if latest else None,
            "lr": latest["lr"] if latest else base_lr,
            "elapsed_minutes": round(elapsed_min, 2),
            "eta_hours": round(eta_hours, 4) if eta_hours is not None else None,
            "speed_text": speed_text,
            "stage_label": stage_label,
            "unit_label": "轮",
        },
        "gomoku": {
            "phase": phase,
            "completed_rounds": completed_rounds,
            "current_round": current_round,
            "max_rounds": int(max_rounds),
            "stage": stage,
            "stage_label": stage_label,
            "stage_detail": stage_detail,
            "stage_elapsed_s": round(stage_elapsed_s, 1),
            "worker_count": worker_count,
            "arch": arch,
            "latest_heuristic_score": heur_scores[-1] if heur_scores else None,
            "best_heuristic_score": max(heur_scores) if heur_scores else None,
            "latest_v1_score": v1_scores[-1] if v1_scores else None,
            "best_v1_score": max(v1_scores) if v1_scores else None,
            "latest_nnue_val_mse": latest["nnue_val_mse"] if latest else None,
            "latest_cnn_val_top1": latest["cnn_val_top1"] if latest else None,
            "buffer_positions": buf_pos,
            "avg_round_sec": round(avg_round_sec, 1),
        },
        "log_tail": log_tail,
        "system": _system_stats(project_dir),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _post(url: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {token}",
            "User-Agent": "Gomoku-Windows-Reporter/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read(4096).decode("utf-8", errors="replace"))


def main() -> None:
    default_project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Windows 五子棋训练指标上报器")
    parser.add_argument("--monitor", default="http://192.168.137.4:8766")
    parser.add_argument("--token", default="")
    parser.add_argument("--machine", default="windows-4070")
    parser.add_argument("--run", default="gomoku-v2")
    parser.add_argument("--phase", default="v2")
    parser.add_argument("--project-dir", default=str(default_project))
    parser.add_argument("--log-file", default="")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    token = _load_token(args.token)
    if not token:
        raise SystemExit("缺少 token：请用 --token、MONITOR_TOKEN 或 D:\\TinyStories\\monitor_token.txt 提供")

    project_dir = Path(args.project_dir).resolve()
    log_path = (
        Path(args.log_file)
        if args.log_file
        else project_dir / "training" / "runs" / "train_win4070.log"
    )
    url = args.monitor.rstrip("/") + "/api/ingest"

    if args.once:
        payload = _build_payload(args.machine, args.run, args.phase, project_dir, log_path)
        res = _post(url, token, payload)
        print(f"[Reporter] once OK: {res}", flush=True)
        return

    failures = 0
    print(f"[Reporter] {args.machine}/{args.run} -> {url} (every {args.interval:g}s)", flush=True)
    while True:
        payload = _build_payload(args.machine, args.run, args.phase, project_dir, log_path)
        try:
            _post(url, token, payload)
            failures = 0
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failures += 1
            if failures == 1 or failures % 10 == 0:
                print(f"[Reporter] post failed ({failures}): {exc}", flush=True)
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
