#!/usr/bin/env python3
"""V3 single-network Pattern-NNUE trainer, Rapfi/buffer pretrainer, parity verifier & self-play loop."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from engine_bridge import (
    BOARD_CELLS,
    BOARD_SIZE,
    find_engine_lib,
    generate_pnnue_games_parallel,
    run_pnnue_arena_parallel,
    verify_pnnue_parity,
)
from pnnue_model import PatternNnueNet, boards_to_pattern_ids, export_pnnue_bin


def select_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_dataset(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    boards = np.asarray(raw["boards"], dtype=np.uint8).reshape(-1, BOARD_SIZE, BOARD_SIZE)
    players = np.asarray(raw["players"], dtype=np.uint8).reshape(-1)
    forbidden = np.asarray(raw["forbidden"], dtype=np.uint8).reshape(-1, BOARD_CELLS)
    policies = np.asarray(raw["policies"], dtype=np.float32).reshape(-1, BOARD_CELLS)
    vals_src = raw["values"] if "values" in raw else raw["search_values"]
    values = np.asarray(vals_src, dtype=np.float32).reshape(-1)
    outcomes = np.asarray(raw["outcomes"], dtype=np.float32).reshape(-1)
    return {
        "boards": boards,
        "players": players,
        "forbidden": forbidden,
        "policies": policies,
        "values": values,
        "outcomes": outcomes,
    }


def load_npz_dataset(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as zf:
        raw = {k: zf[k] for k in zf.files}
    return normalize_dataset(raw)


def merge_replay_buffer(
    buf_path: Path, new_data: Dict[str, np.ndarray], max_positions: int
) -> Dict[str, np.ndarray]:
    norm_new = normalize_dataset(new_data)
    if buf_path.exists():
        old_data = load_npz_dataset(buf_path)
        merged = {
            k: np.concatenate([old_data[k], norm_new[k]], axis=0)
            for k in norm_new.keys()
        }
    else:
        merged = norm_new

    total = merged["outcomes"].shape[0]
    if total > max_positions:
        merged = {k: v[total - max_positions :] for k, v in merged.items()}

    buf_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = buf_path.with_suffix(".tmp.npz")
    np.savez(tmp, **merged)
    tmp.replace(buf_path)
    return merged


def apply_dihedral_batch(
    boards: np.ndarray,
    forbidden: np.ndarray,
    policies: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply random D4 symmetry per mini-batch to preserve board geometry."""
    sym = int(rng.integers(0, 8))
    if sym == 0:
        return boards, forbidden, policies
    k_rot = sym & 3
    do_flip = bool(sym & 4)

    b2 = boards
    f2 = forbidden.reshape(-1, BOARD_SIZE, BOARD_SIZE)
    p2 = policies.reshape(-1, BOARD_SIZE, BOARD_SIZE)
    if k_rot:
        b2 = np.rot90(b2, k=k_rot, axes=(1, 2))
        f2 = np.rot90(f2, k=k_rot, axes=(1, 2))
        p2 = np.rot90(p2, k=k_rot, axes=(1, 2))
    if do_flip:
        b2 = np.flip(b2, axis=2)
        f2 = np.flip(f2, axis=2)
        p2 = np.flip(p2, axis=2)
    return (
        np.ascontiguousarray(b2),
        np.ascontiguousarray(f2.reshape(-1, BOARD_CELLS)),
        np.ascontiguousarray(p2.reshape(-1, BOARD_CELLS)),
    )


def train_pnnue_epoch_loop(
    model: PatternNnueNet,
    data: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Dict[str, float]:
    boards = data["boards"]
    players = data["players"]
    forbidden = data["forbidden"]
    policies = data["policies"]
    val_targets = (0.50 * data["outcomes"] + 0.50 * data["values"]).astype(np.float32)

    n = boards.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_n = max(1, min(8192, int(n * 0.1)))
    val_idx = perm[:val_n]
    trn_idx = perm[val_n:] if n > val_n else perm

    val_pat = torch.from_numpy(boards_to_pattern_ids(boards[val_idx])).long().to(device)
    val_pidx = torch.from_numpy((players[val_idx] == 2).astype(np.int64)).to(device)
    val_legal = torch.from_numpy(
        ((boards[val_idx].reshape(-1, BOARD_CELLS) == 0) & (forbidden[val_idx] == 0))
    ).to(device)
    val_pol_t = torch.from_numpy(policies[val_idx]).float().to(device)
    val_y_t = torch.from_numpy(val_targets[val_idx]).float().to(device)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_ep = 0
    best_val_mse = 0.0
    best_val_top1 = 0.0
    last_train_loss = 0.0

    def _compute_losses(
        pat_t: torch.Tensor,
        pidx_t: torch.Tensor,
        legal_t: torch.Tensor,
        pol_t: torch.Tensor,
        val_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, pred_v = model(pat_t, pidx_t)
        masked_logits = logits.masked_fill(~legal_t, -1e4 if use_amp else -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)

        clean_pol = pol_t * legal_t.float()
        pol_sum = clean_pol.sum(dim=-1, keepdim=True)
        valid_pol = pol_sum.squeeze(-1) > 1e-4
        if valid_pol.any():
            norm_pol = clean_pol[valid_pol] / pol_sum[valid_pol].clamp_min(1e-8)
            pol_loss = -(norm_pol * log_probs[valid_pol]).sum(dim=-1).mean()
            top1 = (masked_logits[valid_pol].argmax(dim=-1) == norm_pol.argmax(dim=-1)).float().mean()
        else:
            pol_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
            top1 = torch.zeros((), device=logits.device, dtype=logits.dtype)

        val_loss = F.smooth_l1_loss(pred_v, val_t, beta=0.2)
        val_mse = F.mse_loss(pred_v, val_t)
        total = pol_loss + 1.5 * val_loss
        return total, pol_loss, val_mse, top1

    for ep in range(1, epochs + 1):
        model.train()
        rng.shuffle(trn_idx)
        ep_loss = 0.0
        steps = 0

        for s in range(0, len(trn_idx), batch_size):
            b_idx = trn_idx[s : s + batch_size]
            bb, bf, bp = apply_dihedral_batch(
                boards[b_idx], forbidden[b_idx], policies[b_idx], rng
            )
            pat_np = boards_to_pattern_ids(bb)
            pat_t = torch.from_numpy(pat_np).long().to(device, non_blocking=True)
            pidx_t = torch.from_numpy((players[b_idx] == 2).astype(np.int64)).to(device, non_blocking=True)
            legal_t = torch.from_numpy(((bb.reshape(-1, BOARD_CELLS) == 0) & (bf == 0))).to(
                device, non_blocking=True
            )
            pol_t = torch.from_numpy(bp).float().to(device, non_blocking=True)
            val_t = torch.from_numpy(val_targets[b_idx]).float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss, _, _, _ = _compute_losses(pat_t, pidx_t, legal_t, pol_t, val_t)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, _, _, _ = _compute_losses(pat_t, pidx_t, legal_t, pol_t, val_t)
                loss.backward()
                optimizer.step()

            ep_loss += float(loss.item())
            steps += 1

        scheduler.step()
        last_train_loss = ep_loss / max(1, steps)

        model.eval()
        with torch.no_grad():
            v_losses: List[float] = []
            v_mses: List[float] = []
            v_top1s: List[float] = []
            for vs in range(0, val_n, batch_size):
                ve = min(val_n, vs + batch_size)
                tl, _, vm, t1 = _compute_losses(
                    val_pat[vs:ve],
                    val_pidx[vs:ve],
                    val_legal[vs:ve],
                    val_pol_t[vs:ve],
                    val_y_t[vs:ve],
                )
                w = float(ve - vs)
                v_losses.append(float(tl.item()) * w)
                v_mses.append(float(vm.item()) * w)
                v_top1s.append(float(t1.item()) * w)

            cur_val_loss = sum(v_losses) / max(1, val_n)
            cur_val_mse = sum(v_mses) / max(1, val_n)
            cur_val_top1 = sum(v_top1s) / max(1, val_n)

        if cur_val_loss < best_val_loss - 1e-6:
            best_val_loss = cur_val_loss
            best_val_mse = cur_val_mse
            best_val_top1 = cur_val_top1
            best_state = copy.deepcopy(model.state_dict())
            best_ep = ep

    model.load_state_dict(best_state)
    model.cpu()
    return {
        "train_loss": round(last_train_loss, 5),
        "val_loss": round(best_val_loss, 5),
        "val_mse": round(best_val_mse, 5),
        "val_top1": round(best_val_top1, 5),
        "best_ep": best_ep,
    }


def append_metrics(metrics_path: Path, record: Dict) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_verify_only(dim: int, v_hidden: int) -> None:
    lib_path = find_engine_lib(PROJECT_ROOT)
    model = PatternNnueNet(dim=dim, v_hidden=v_hidden)
    diffs = verify_pnnue_parity(lib_path, model, num_boards=10)
    print(f"[Parity V3] lib={lib_path} diffs={json.dumps(diffs)}")
    if diffs["max_val_diff"] >= 1e-4 or diffs["max_pol_diff"] >= 1e-4:
        raise AssertionError(f"Pattern-NNUE parity check failed: {diffs}")
    print("[Parity V3] PASSED (< 1e-4)")


def run_smoke_test(dim: int, v_hidden: int) -> None:
    lib_path = find_engine_lib(PROJECT_ROOT)
    model = PatternNnueNet(dim=dim, v_hidden=v_hidden)
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "smoke_pnnue.bin"
        export_pnnue_bin(model, bin_path)
        t0 = time.time()
        data = generate_pnnue_games_parallel(
            lib_path=lib_path,
            pnnue_path=bin_path,
            num_games=4,
            workers=2,
            max_depth=2,
            vcf_depth=6,
            time_limit=0.15,
            random_opening_plies=2,
            explore_temp=0.4,
            seed=777,
        )
        gen_sec = time.time() - t0
        assert data and len(data["boards"]) > 0, "Smoke self-play produced 0 samples"
        arena = run_pnnue_arena_parallel(
            lib_path=lib_path,
            pnnue_a_path=bin_path,
            pnnue_b_path=None,
            v2_b_nnue_path=None,
            v2_b_cnn_path=None,
            num_games=4,
            workers=2,
            a_depth=2,
            a_vcf=6,
            a_time=0.15,
            b_depth=2,
            b_vcf=6,
            b_time=0.15,
            b_cnn_w=0.0,
            seed=999,
        )
        assert arena["illegal"] == 0, f"Smoke arena illegal move detected: {arena}"
        print(
            f"[Smoke V3] selfplay={len(data['boards'])} positions in {gen_sec:.2f}s | "
            f"arena_vs_heuristic={arena}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 Single-Network Pattern-NNUE trainer")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--pretrain-buffer",
        action="append",
        default=[],
        help="预训练样本库，可重复传入或用逗号分隔多个 .npz（如 V2 replay_buffer + Rapfi 蒸馏库）",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=0, help="预训练轮数，0 表示沿用 --epochs")
    parser.add_argument(
        "--force-pretrain",
        action="store_true",
        help="即使已从 checkpoint 恢复也强制重新预训练（默认恢复时跳过，避免重复预训练）",
    )
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--games-per-round", type=int, default=480)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--vcf-depth", type=int, default=14)
    parser.add_argument("--time-limit", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--v-hidden", type=int, default=32)
    parser.add_argument("--max-buffer", type=int, default=200000)
    parser.add_argument("--eval-games", type=int, default=64)
    parser.add_argument("--eval-depth", type=int, default=5)
    parser.add_argument("--eval-time", type=float, default=0.7)
    parser.add_argument("--gate-thresh", type=float, default=0.515)
    parser.add_argument("--teacher-bin", type=str, default="")
    parser.add_argument("--teacher-games", type=int, default=0)
    parser.add_argument("--teacher-time-ms", type=int, default=80)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", type=str, default="training/runs_v3")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=20261127)
    args = parser.parse_args()

    if args.verify_only:
        run_verify_only(args.dim, args.v_hidden)
        return
    if args.smoke_test:
        run_smoke_test(args.dim, args.v_hidden)
        return

    lib_path = find_engine_lib(PROJECT_ROOT)
    device = select_device(args.device)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    prod_bin = PROJECT_ROOT / "models" / "pnnue.bin"
    champ_bin = run_dir / "pnnue_champ.bin"
    cand_bin = run_dir / "pnnue_cand.bin"
    ckpt_path = run_dir / "pnnue_champ.pt"
    buf_path = run_dir / "pnnue_buffer.npz"
    metrics_path = PROJECT_ROOT / "training" / "runs" / "metrics.jsonl"

    v2_nnue_bin = PROJECT_ROOT / "models" / "nnue.bin"
    v2_cnn_bin = PROJECT_ROOT / "models" / "cnn.bin"

    model = PatternNnueNet(dim=args.dim, v_hidden=args.v_hidden)
    start_round = 1
    resumed = False
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        start_round = int(ckpt.get("round", 0)) + 1
        resumed = True
        print(f"[Trainer V3] Resumed checkpoint from round {start_round - 1}")

    pretrain_paths: List[Path] = []
    for item in args.pretrain_buffer:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            p = Path(part)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                pretrain_paths.append(p)
            else:
                print(f"[Pretrain V3] WARNING: buffer not found, skipped: {p}")
    if resumed and pretrain_paths and not args.force_pretrain:
        print("[Pretrain V3] Resumed from checkpoint -> skip pretraining (use --force-pretrain to override)")
        pretrain_paths = []

    print(
        f"[Trainer V3] Device={device}, C_lib={lib_path}, "
        f"Arch=22143x{args.dim}->V({args.v_hidden})+P, Batch={args.batch_size}"
    )

    if pretrain_paths:
        merged = None
        for pre_path in pretrain_paths:
            print(f"[Pretrain V3] Loading buffer: {pre_path}")
            pre_data = load_npz_dataset(pre_path)
            print(f"[Pretrain V3]   +{len(pre_data['boards'])} positions")
            merged = merge_replay_buffer(buf_path, pre_data, args.max_buffer)
        assert merged is not None
        t0 = time.time()
        pre_epochs = args.pretrain_epochs if args.pretrain_epochs > 0 else args.epochs
        stats = train_pnnue_epoch_loop(
            model, merged, device, pre_epochs, args.batch_size, args.lr, args.seed
        )
        export_pnnue_bin(model, champ_bin)
        export_pnnue_bin(model, prod_bin)
        torch.save({"model": model.state_dict(), "round": start_round - 1}, ckpt_path)
        diffs = verify_pnnue_parity(lib_path, model, champ_bin)
        print(
            f"[Pretrain V3] {len(merged['boards'])} positions, {pre_epochs} epochs in "
            f"{time.time() - t0:.1f}s | {stats} | parity={diffs}"
        )

    if not champ_bin.exists():
        export_pnnue_bin(model, champ_bin)
    if not prod_bin.exists():
        export_pnnue_bin(model, prod_bin)

    for rnd in range(start_round, start_round + args.rounds):
        r_t0 = time.time()
        print(f"\n================ V3 Round {rnd} ================")

        t_gen = time.time()
        sp_data = generate_pnnue_games_parallel(
            lib_path=lib_path,
            pnnue_path=champ_bin if champ_bin.exists() else None,
            num_games=args.games_per_round,
            workers=args.workers,
            max_depth=args.depth,
            vcf_depth=args.vcf_depth,
            time_limit=args.time_limit,
            random_opening_plies=2,
            explore_temp=0.45 if rnd <= 3 else 0.25,
            seed=args.seed + rnd * 100003,
        )
        gen_sec = time.time() - t_gen
        new_pos = len(sp_data["boards"]) if sp_data else 0
        print(f"[SelfPlay V3] {args.games_per_round} games -> {new_pos} pos in {gen_sec:.1f}s")

        merged = merge_replay_buffer(buf_path, sp_data, args.max_buffer)

        if args.teacher_bin and args.teacher_games > 0:
            from distill_rapfi import distill_rapfi_games

            t_rapfi = Path(args.teacher_bin)
            dist_tmp = run_dir / "rapfi_step_tmp.npz"
            distill_rapfi_games(
                teacher_bin=t_rapfi,
                games=args.teacher_games,
                workers=min(8, args.workers),
                time_ms=args.teacher_time_ms,
                opening_plies=3,
                output_path=dist_tmp,
                seed=args.seed + rnd * 31337,
            )
            dist_data = load_npz_dataset(dist_tmp)
            merged = merge_replay_buffer(buf_path, dist_data, args.max_buffer)

        t_trn = time.time()
        cand_model = copy.deepcopy(model)
        trn_stats = train_pnnue_epoch_loop(
            cand_model,
            merged,
            device,
            args.epochs,
            args.batch_size,
            args.lr,
            args.seed + rnd,
        )
        train_sec = time.time() - t_trn
        export_pnnue_bin(cand_model, cand_bin)
        diffs = verify_pnnue_parity(lib_path, cand_model, cand_bin)
        print(f"[Train V3] {train_sec:.1f}s | {trn_stats} | parity={diffs}")

        t_ev = time.time()
        gate_champ = run_pnnue_arena_parallel(
            lib_path=lib_path,
            pnnue_a_path=cand_bin,
            pnnue_b_path=champ_bin,
            v2_b_nnue_path=None,
            v2_b_cnn_path=None,
            num_games=args.eval_games,
            workers=args.workers,
            a_depth=args.eval_depth,
            a_vcf=args.vcf_depth,
            a_time=args.eval_time,
            b_depth=args.eval_depth,
            b_vcf=args.vcf_depth,
            b_time=args.eval_time,
            b_cnn_w=0.0,
            seed=args.seed + rnd * 17,
        )
        promoted = gate_champ["score"] >= args.gate_thresh and gate_champ["illegal"] == 0
        print(
            f"[Gate V3 vs Champ] W={gate_champ['wins']} L={gate_champ['losses']} "
            f"D={gate_champ['draws']} score={gate_champ['score']:.3f} -> "
            f"{'PROMOTED' if promoted else 'ROLLBACK'}"
        )

        if promoted:
            model = cand_model
            export_pnnue_bin(model, champ_bin)
            export_pnnue_bin(model, prod_bin)
            torch.save({"model": model.state_dict(), "round": rnd}, ckpt_path)

        eval_target = champ_bin
        vs_heur = run_pnnue_arena_parallel(
            lib_path=lib_path,
            pnnue_a_path=eval_target,
            pnnue_b_path=None,
            v2_b_nnue_path=None,
            v2_b_cnn_path=None,
            num_games=args.eval_games,
            workers=args.workers,
            a_depth=args.eval_depth,
            a_vcf=args.vcf_depth,
            a_time=args.eval_time,
            b_depth=args.eval_depth,
            b_vcf=args.vcf_depth,
            b_time=args.eval_time,
            b_cnn_w=0.0,
            seed=args.seed + rnd * 29,
        )
        vs_v2 = None
        if v2_nnue_bin.exists() or v2_cnn_bin.exists():
            vs_v2 = run_pnnue_arena_parallel(
                lib_path=lib_path,
                pnnue_a_path=eval_target,
                pnnue_b_path=None,
                v2_b_nnue_path=v2_nnue_bin if v2_nnue_bin.exists() else None,
                v2_b_cnn_path=v2_cnn_bin if v2_cnn_bin.exists() else None,
                num_games=args.eval_games,
                workers=args.workers,
                a_depth=args.eval_depth,
                a_vcf=args.vcf_depth,
                a_time=args.eval_time,
                b_depth=max(4, args.eval_depth - 1),
                b_vcf=args.vcf_depth,
                b_time=args.eval_time,
                b_cnn_w=0.35,
                seed=args.seed + rnd * 41,
            )
        eval_sec = time.time() - t_ev
        total_sec = time.time() - r_t0

        record = {
            "phase": "v3_pnnue",
            "round": rnd,
            "depth": args.depth,
            "new_positions": new_pos,
            "buffer_positions": int(len(merged["boards"])),
            "selfplay_sec": round(gen_sec, 2),
            "train_sec": round(train_sec, 2),
            "eval_sec": round(eval_sec, 2),
            "total_sec": round(total_sec, 2),
            "lr": args.lr,
            "train_loss": trn_stats["train_loss"],
            "val_loss": trn_stats["val_loss"],
            "val_mse": trn_stats["val_mse"],
            "val_top1": trn_stats["val_top1"],
            "best_ep": trn_stats["best_ep"],
            "gate_score": round(gate_champ["score"], 4),
            "promoted": promoted,
            "prod_deployed": promoted,
            "vs_heuristic": vs_heur,
            "vs_v2": vs_v2,
        }
        append_metrics(metrics_path, record)
        print(
            f"[Round {rnd} Done] total={total_sec:.1f}s | vs_heuristic={vs_heur['score']:.3f} | "
            + (f"vs_v2={vs_v2['score']:.3f}" if vs_v2 else "vs_v2=N/A")
        )


if __name__ == "__main__":
    main()
