import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine_bridge import (
    find_engine_lib,
    generate_games_parallel,
    run_arena_match_parallel,
    verify_c_parity,
)
from models import (
    BOARD_SIZE,
    NnueNet,
    PolicyValueCNN,
    boards_to_cnn_tensor,
    export_cnn_bin,
    export_nnue_bin,
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def augment_dihedral_batch(
    x: np.ndarray,
    policy: np.ndarray,
    legal_mask: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对 mini-batch 进行随机 8 折二面体对称变换（旋转 + 镜像），同步变换棋盘、策略与合法掩码。"""
    bs = x.shape[0]
    p2d = policy.reshape(bs, BOARD_SIZE, BOARD_SIZE)
    m2d = legal_mask.reshape(bs, BOARD_SIZE, BOARD_SIZE)

    x_out = np.empty_like(x)
    p_out = np.empty_like(p2d)
    m_out = np.empty_like(m2d)

    ks = rng.integers(0, 4, size=bs)
    flips = rng.integers(0, 2, size=bs)
    for k in range(4):
        for fl in (0, 1):
            idx = np.where((ks == k) & (flips == fl))[0]
            if idx.size == 0:
                continue
            xi = x[idx]
            pi = p2d[idx]
            mi = m2d[idx]
            if k > 0:
                xi = np.rot90(xi, k=k, axes=(2, 3))
                pi = np.rot90(pi, k=k, axes=(1, 2))
                mi = np.rot90(mi, k=k, axes=(1, 2))
            if fl == 1:
                xi = np.flip(xi, axis=3)
                pi = np.flip(pi, axis=2)
                mi = np.flip(mi, axis=2)
            x_out[idx] = xi
            p_out[idx] = pi
            m_out[idx] = mi

    return (
        np.ascontiguousarray(x_out),
        np.ascontiguousarray(p_out.reshape(bs, -1)),
        np.ascontiguousarray(m_out.reshape(bs, -1)),
    )


def append_replay_buffer(
    buf_path: Path,
    new_data: Dict[str, np.ndarray],
    max_positions: int,
) -> Dict[str, np.ndarray]:
    if buf_path.exists():
        old = dict(np.load(buf_path))
        merged = {
            k: np.concatenate([old[k], new_data[k]], axis=0)
            for k in new_data.keys()
        }
    else:
        merged = new_data

    total = merged["outcomes"].shape[0]
    if total > max_positions:
        merged = {k: v[total - max_positions :] for k, v in merged.items()}

    buf_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(buf_path, **merged)
    return merged


def train_nnue(
    model: NnueNet,
    data: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    update_norm: bool,
    seed: int,
) -> Dict[str, float]:
    features = data["features"].astype(np.float32)
    targets = (0.65 * data["outcomes"] + 0.35 * data["search_values"]).astype(np.float32)

    if update_norm:
        mean = features.mean(axis=0)
        std = np.maximum(features.std(axis=0), 1e-3)
        model.set_norm_stats(mean, std)

    model.to(device)
    n = features.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_n = max(1, int(n * 0.1))
    val_idx = perm[:val_n]
    trn_idx = perm[val_n:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.SmoothL1Loss(beta=0.2)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_ep = 0
    last_train_loss = 0.0

    xv = torch.from_numpy(features[val_idx]).to(device)
    yv = torch.from_numpy(targets[val_idx]).to(device)

    for ep in range(1, epochs + 1):
        model.train()
        rng.shuffle(trn_idx)
        ep_loss = 0.0
        steps = 0
        for s in range(0, len(trn_idx), batch_size):
            b_idx = trn_idx[s : s + batch_size]
            xb = torch.from_numpy(features[b_idx]).to(device)
            yb = torch.from_numpy(targets[b_idx]).to(device)

            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += float(loss.item())
            steps += 1
        scheduler.step()
        last_train_loss = ep_loss / max(1, steps)

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(xv), yv).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_ep = ep

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_mse = float(F.mse_loss(model(xv), yv).item())

    model.cpu()
    return {
        "nnue_train_loss": round(last_train_loss, 5),
        "nnue_val_loss": round(best_val_loss, 5),
        "nnue_val_mse": round(val_mse, 5),
        "nnue_best_ep": best_ep,
    }


def train_cnn(
    model: PolicyValueCNN,
    data: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Dict[str, float]:
    x_all = boards_to_cnn_tensor(
        data["boards"], data["players"], data["last_moves"], data["forbidden"]
    )
    p_all = data["policies"].astype(np.float32)
    legal_all = (data["boards"] == 0) & (data["forbidden"] == 0)
    v_all = (0.60 * data["outcomes"] + 0.40 * data["search_values"]).astype(np.float32)

    n = x_all.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_n = max(1, int(n * 0.1))
    val_idx = perm[:val_n]
    trn_idx = perm[val_n:]

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    val_criterion = nn.SmoothL1Loss(beta=0.2, reduction="sum")

    best_score_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_metrics: Dict[str, float] = {}
    last_pol_loss = 0.0
    last_val_loss = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        rng.shuffle(trn_idx)
        ep_pol = 0.0
        ep_val = 0.0
        steps = 0

        for s in range(0, len(trn_idx), batch_size):
            b_idx = trn_idx[s : s + batch_size]
            xb, pb, mb = augment_dihedral_batch(
                x_all[b_idx], p_all[b_idx], legal_all[b_idx], rng
            )
            xt = torch.from_numpy(xb).to(device)
            pt = torch.from_numpy(pb).to(device)
            mt = torch.from_numpy(mb).to(device)
            vt = torch.from_numpy(v_all[b_idx]).to(device)

            logits, val_pred = model(xt)
            masked_logits = logits.masked_fill(~mt, -1e9)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            pol_loss = -(pt * log_probs).sum(dim=-1).mean()
            val_loss = F.smooth_l1_loss(val_pred, vt, beta=0.2)
            loss = pol_loss + 1.5 * val_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            ep_pol += float(pol_loss.item())
            ep_val += float(val_loss.item())
            steps += 1

        scheduler.step()
        last_pol_loss = ep_pol / max(1, steps)
        last_val_loss = ep_val / max(1, steps)

        model.eval()
        val_pol_sum = 0.0
        val_huber_sum = 0.0
        val_mse_sum = 0.0
        val_top1_hits = 0
        val_total = 0
        with torch.no_grad():
            for s in range(0, len(val_idx), batch_size):
                b_idx = val_idx[s : s + batch_size]
                xt = torch.from_numpy(x_all[b_idx]).to(device)
                pt = torch.from_numpy(p_all[b_idx]).to(device)
                mt = torch.from_numpy(legal_all[b_idx]).to(device)
                vt = torch.from_numpy(v_all[b_idx]).to(device)

                logits, val_pred = model(xt)
                masked_logits = logits.masked_fill(~mt, -1e9)
                log_probs = F.log_softmax(masked_logits, dim=-1)
                pol_loss = -(pt * log_probs).sum(dim=-1).sum()
                huber_loss = val_criterion(val_pred, vt)
                mse_loss = F.mse_loss(val_pred, vt, reduction="sum")

                val_pol_sum += float(pol_loss.item())
                val_huber_sum += float(huber_loss.item())
                val_mse_sum += float(mse_loss.item())
                val_top1_hits += int(
                    (masked_logits.argmax(dim=-1) == pt.argmax(dim=-1)).sum().item()
                )
                val_total += len(b_idx)

        vp = val_pol_sum / max(1, val_total)
        vh = val_huber_sum / max(1, val_total)
        vm = val_mse_sum / max(1, val_total)
        vt1 = val_top1_hits / max(1, val_total)
        comb = vp + 1.5 * vh
        if comb < best_score_loss:
            best_score_loss = comb
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = {
                "cnn_train_pol": round(last_pol_loss, 5),
                "cnn_train_val": round(last_val_loss, 5),
                "cnn_val_pol": round(vp, 5),
                "cnn_val_mse": round(vm, 5),
                "cnn_val_top1": round(vt1, 5),
                "cnn_best_ep": ep,
            }

    model.load_state_dict(best_state)
    model.cpu()
    return best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Gomoku Route-3 Dual-Net Self-Play Trainer (V2)")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--games-per-round", type=int, default=600)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--vcf-depth", type=int, default=12)
    parser.add_argument("--time-limit", type=float, default=0.6)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-depth", type=int, default=4)
    parser.add_argument("--eval-time", type=float, default=0.6)
    parser.add_argument("--cnn-weight", type=float, default=0.35)
    parser.add_argument("--nnue-epochs", type=int, default=35)
    parser.add_argument("--cnn-epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--nnue-lr", type=float, default=8e-4)
    parser.add_argument("--cnn-lr", type=float, default=1.0e-3)
    parser.add_argument("--max-buffer", type=int, default=200000)
    parser.add_argument("--nnue-h1", type=int, default=128)
    parser.add_argument("--nnue-h2", type=int, default=64)
    parser.add_argument("--ch", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--gate-thresh", type=float, default=0.525)
    parser.add_argument("--ckpt-name", type=str, default="checkpoint_v2.pt")
    parser.add_argument("--v1-nnue", type=str, default="training/runs/v1_nnue.bin")
    parser.add_argument("--v1-cnn", type=str, default="training/runs/v1_cnn.bin")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    training_dir = Path(__file__).resolve().parent
    project_root = training_dir.parent
    models_dir = project_root / "models"
    runs_dir = training_dir / "runs"
    data_dir = training_dir / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    lib_path = find_engine_lib(project_root)
    device = select_device()
    print(
        f"[Trainer] Device={device}, C_lib={lib_path}, "
        f"NNUE=32->{args.nnue_h1}->{args.nnue_h2}->1, CNN={args.ch}ch x {args.blocks}blocks"
    )

    nnue = NnueNet(in_dim=32, h1=args.nnue_h1, h2=args.nnue_h2)
    cnn = PolicyValueCNN(ch=args.ch, num_blocks=args.blocks)

    if args.verify_only:
        parity = verify_c_parity(lib_path, nnue, cnn)
        print(f"[Verify] PyTorch vs C11 parity: {parity}")
        for k, v in parity.items():
            assert v < 1e-4, f"Parity check failed for {k}: {v}"
        print("[Verify] PASSED (< 1e-4)")
        return

    ckpt_path = runs_dir / args.ckpt_name
    champ_nnue_bin = runs_dir / "champ_nnue_active.bin"
    champ_cnn_bin = runs_dir / "champ_cnn_active.bin"
    prod_nnue_bin = models_dir / "nnue.bin"
    prod_cnn_bin = models_dir / "cnn.bin"
    v1_nnue_bin = (project_root / args.v1_nnue) if not Path(args.v1_nnue).is_absolute() else Path(args.v1_nnue)
    v1_cnn_bin = (project_root / args.v1_cnn) if not Path(args.v1_cnn).is_absolute() else Path(args.v1_cnn)

    start_round = 1
    champ_nnue_state: Optional[dict] = None
    champ_cnn_state: Optional[dict] = None

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        champ_nnue_state = ckpt["champ_nnue"]
        champ_cnn_state = ckpt["champ_cnn"]
        nnue.load_state_dict(champ_nnue_state)
        cnn.load_state_dict(champ_cnn_state)
        start_round = int(ckpt["round"]) + 1
        export_nnue_bin(nnue, champ_nnue_bin)
        export_cnn_bin(cnn, champ_cnn_bin)
        print(f"[Trainer] Resumed champion states from round {start_round - 1}")

    replay_path = data_dir / "replay_buffer.npz"
    metrics_path = runs_dir / "metrics.jsonl"

    for rnd in range(start_round, start_round + args.rounds):
        t0 = time.time()
        print(f"\n================ Round {rnd} ================")

        # 自博弈优先级：当前架构冠军 > 生产冠军 > 纯搜索冷启动
        sp_nnue = champ_nnue_bin if champ_nnue_bin.exists() else (prod_nnue_bin if prod_nnue_bin.exists() else None)
        sp_cnn = champ_cnn_bin if champ_cnn_bin.exists() else (prod_cnn_bin if prod_cnn_bin.exists() else None)
        temp = 0.65 if rnd <= 2 else 0.45

        t_sp = time.time()
        new_data = generate_games_parallel(
            lib_path=lib_path,
            nnue_path=sp_nnue,
            cnn_path=sp_cnn,
            num_games=args.games_per_round,
            workers=args.workers,
            max_depth=args.depth,
            vcf_depth=args.vcf_depth,
            time_limit=args.time_limit,
            explore_temp=temp,
            explore_plies=12,
            cnn_weight=args.cnn_weight,
            seed=302600 + rnd * 99991,
        )
        sp_sec = time.time() - t_sp
        new_pos = int(new_data["outcomes"].shape[0])
        buf_data = append_replay_buffer(replay_path, new_data, args.max_buffer)
        total_pos = int(buf_data["outcomes"].shape[0])
        print(
            f"[SelfPlay] depth={args.depth}, {args.games_per_round} games -> {new_pos} positions "
            f"in {sp_sec:.1f}s (buffer={total_pos})"
        )

        # 确保从当前架构冠军权重起训，防止被拒权重残留
        if champ_nnue_state is not None:
            nnue.load_state_dict(champ_nnue_state)
        if champ_cnn_state is not None:
            cnn.load_state_dict(champ_cnn_state)

        t_tr = time.time()
        decay = max(0.4, 1.0 - 0.08 * (rnd - 1))
        nnue_metrics = train_nnue(
            nnue,
            buf_data,
            device=device,
            epochs=args.nnue_epochs,
            batch_size=args.batch_size,
            lr=args.nnue_lr * decay,
            update_norm=(champ_nnue_state is None),
            seed=1000 + rnd,
        )
        cnn_metrics = train_cnn(
            cnn,
            buf_data,
            device=device,
            epochs=args.cnn_epochs,
            batch_size=args.batch_size,
            lr=args.cnn_lr * decay,
            seed=5000 + rnd,
        )
        tr_sec = time.time() - t_tr
        print(f"[Train] {tr_sec:.1f}s | NNUE={nnue_metrics} | CNN={cnn_metrics}")

        parity = verify_c_parity(lib_path, nnue, cnn)
        for k, v in parity.items():
            if v >= 1e-4:
                raise RuntimeError(f"C parity exceeded 1e-4 on {k}: {v}")
        print(f"[Parity] max diffs < 1e-4: {parity}")

        cand_nnue = runs_dir / f"cand_nnue_r{rnd}.bin"
        cand_cnn = runs_dir / f"cand_cnn_r{rnd}.bin"
        export_nnue_bin(nnue, cand_nnue)
        export_cnn_bin(cnn, cand_cnn)

        t_ev = time.time()
        nnue_promoted = False
        cnn_promoted = False
        nnue_gate_score = 0.0
        cnn_gate_score = 0.0

        if champ_nnue_state is None or champ_cnn_state is None:
            # 本架构首轮初始化冠军态
            champ_nnue_state = copy.deepcopy(nnue.state_dict())
            champ_cnn_state = copy.deepcopy(cnn.state_dict())
            shutil.copy2(cand_nnue, champ_nnue_bin)
            shutil.copy2(cand_cnn, champ_cnn_bin)
            nnue_promoted = True
            cnn_promoted = True
            nnue_gate_score = 1.0
            cnn_gate_score = 1.0
            print("[Gate] Initialized active V2 champion weights.")
        else:
            # 独立门控 1：NNUE
            s_nnue = run_arena_match_parallel(
                lib_path,
                cand_nnue, champ_cnn_bin, args.eval_depth, args.vcf_depth, args.eval_time, args.cnn_weight,
                champ_nnue_bin, champ_cnn_bin, args.eval_depth, args.vcf_depth, args.eval_time, args.cnn_weight,
                num_games=args.eval_games, seed=7100 + rnd * 101, workers=args.workers,
            )
            nnue_gate_score = (s_nnue.a_wins + 0.5 * s_nnue.draws) / max(1, s_nnue.a_wins + s_nnue.b_wins + s_nnue.draws)
            if s_nnue.illegal_moves == 0 and nnue_gate_score >= args.gate_thresh:
                champ_nnue_state = copy.deepcopy(nnue.state_dict())
                shutil.copy2(cand_nnue, champ_nnue_bin)
                nnue_promoted = True
            else:
                nnue.load_state_dict(champ_nnue_state)
            print(
                f"[Gate NNUE] W={s_nnue.a_wins} L={s_nnue.b_wins} D={s_nnue.draws} "
                f"score={nnue_gate_score:.3f} -> {'PROMOTED' if nnue_promoted else 'ROLLBACK'}"
            )

            # 独立门控 2：CNN（使用最新活跃的 champ_nnue_bin）
            s_cnn = run_arena_match_parallel(
                lib_path,
                champ_nnue_bin, cand_cnn, args.eval_depth, args.vcf_depth, args.eval_time, args.cnn_weight,
                champ_nnue_bin, champ_cnn_bin, args.eval_depth, args.vcf_depth, args.eval_time, args.cnn_weight,
                num_games=args.eval_games, seed=8100 + rnd * 101, workers=args.workers,
            )
            cnn_gate_score = (s_cnn.a_wins + 0.5 * s_cnn.draws) / max(1, s_cnn.a_wins + s_cnn.b_wins + s_cnn.draws)
            if s_cnn.illegal_moves == 0 and cnn_gate_score >= args.gate_thresh:
                champ_cnn_state = copy.deepcopy(cnn.state_dict())
                shutil.copy2(cand_cnn, champ_cnn_bin)
                cnn_promoted = True
            else:
                cnn.load_state_dict(champ_cnn_state)
            print(
                f"[Gate CNN ] W={s_cnn.a_wins} L={s_cnn.b_wins} D={s_cnn.draws} "
                f"score={cnn_gate_score:.3f} -> {'PROMOTED' if cnn_promoted else 'ROLLBACK'}"
            )

        # 全量评测：当前活跃冠军 vs 纯启发式基线
        vs_base = run_arena_match_parallel(
            lib_path,
            champ_nnue_bin, champ_cnn_bin, args.eval_depth, args.vcf_depth, args.eval_time, args.cnn_weight,
            None, None, args.eval_depth, args.vcf_depth, args.eval_time, 0.0,
            num_games=args.eval_games, seed=9100 + rnd * 101, workers=args.workers,
        )
        base_score = (vs_base.a_wins + 0.5 * vs_base.draws) / max(1, vs_base.a_wins + vs_base.b_wins + vs_base.draws)
        print(
            f"[Arena vs Heuristic] W={vs_base.a_wins} L={vs_base.b_wins} D={vs_base.draws} "
            f"score={base_score:.3f} illegal={vs_base.illegal_moves} "
            f"t_cand={vs_base.a_avg_time_sec*1000:.1f}ms t_base={vs_base.b_avg_time_sec*1000:.1f}ms"
        )

        # 全量评测：当前活跃冠军 vs V1 快照（或当前 prod models）
        vs_v1_score = base_score
        v1_wins, v1_losses, v1_draws = 0, 0, 0
        if v1_nnue_bin.exists() and v1_cnn_bin.exists():
            vs_v1 = run_arena_match_parallel(
                lib_path,
                champ_nnue_bin, champ_cnn_bin, args.eval_depth, args.vcf_depth, args.eval_time, args.cnn_weight,
                v1_nnue_bin, v1_cnn_bin, args.eval_depth, args.vcf_depth, args.eval_time, args.cnn_weight,
                num_games=args.eval_games, seed=9600 + rnd * 101, workers=args.workers,
            )
            v1_wins, v1_losses, v1_draws = vs_v1.a_wins, vs_v1.b_wins, vs_v1.draws
            vs_v1_score = (v1_wins + 0.5 * v1_draws) / max(1, v1_wins + v1_losses + v1_draws)
            print(
                f"[Arena vs V1 Champ ] W={v1_wins} L={v1_losses} D={v1_draws} "
                f"score={vs_v1_score:.3f} illegal={vs_v1.illegal_moves}"
            )

        # 只有不弱于当前生产模型（且首次超越 V1 或本轮有晋级）才同步更新 models/*.bin
        prod_deployed = False
        if not prod_nnue_bin.exists() or not prod_cnn_bin.exists():
            shutil.copy2(champ_nnue_bin, prod_nnue_bin)
            shutil.copy2(champ_cnn_bin, prod_cnn_bin)
            prod_deployed = True
        elif nnue_promoted or cnn_promoted:
            vs_prod = run_arena_match_parallel(
                lib_path,
                champ_nnue_bin, champ_cnn_bin, args.eval_depth, args.vcf_depth, args.eval_time, args.cnn_weight,
                prod_nnue_bin, prod_cnn_bin, args.eval_depth, args.vcf_depth, args.eval_time, args.cnn_weight,
                num_games=args.eval_games, seed=9900 + rnd * 101, workers=args.workers,
            )
            prod_score = (vs_prod.a_wins + 0.5 * vs_prod.draws) / max(1, vs_prod.a_wins + vs_prod.b_wins + vs_prod.draws)
            if vs_prod.illegal_moves == 0 and prod_score >= 0.50:
                shutil.copy2(champ_nnue_bin, prod_nnue_bin)
                shutil.copy2(champ_cnn_bin, prod_cnn_bin)
                prod_deployed = True
            print(
                f"[Prod Gate] vs models/*.bin W={vs_prod.a_wins} L={vs_prod.b_wins} D={vs_prod.draws} "
                f"score={prod_score:.3f} -> {'DEPLOYED' if prod_deployed else 'KEPT PROD'}"
            )

        torch.save(
            {
                "round": rnd,
                "champ_nnue": champ_nnue_state,
                "champ_cnn": champ_cnn_state,
            },
            ckpt_path,
        )

        record = {
            "phase": "v2",
            "round": rnd,
            "depth": args.depth,
            "new_positions": new_pos,
            "buffer_positions": total_pos,
            "selfplay_sec": round(sp_sec, 2),
            "train_sec": round(tr_sec, 2),
            "eval_sec": round(time.time() - t_ev, 2),
            "total_sec": round(time.time() - t0, 2),
            **nnue_metrics,
            **cnn_metrics,
            "nnue_gate_score": round(nnue_gate_score, 4),
            "cnn_gate_score": round(cnn_gate_score, 4),
            "nnue_promoted": nnue_promoted,
            "cnn_promoted": cnn_promoted,
            "prod_deployed": prod_deployed,
            "vs_heuristic": {
                "wins": vs_base.a_wins,
                "losses": vs_base.b_wins,
                "draws": vs_base.draws,
                "score": round(base_score, 4),
                "illegal": vs_base.illegal_moves,
                "cand_ms": round(vs_base.a_avg_time_sec * 1000, 2),
                "base_ms": round(vs_base.b_avg_time_sec * 1000, 2),
            },
            "vs_v1": {
                "wins": v1_wins,
                "losses": v1_losses,
                "draws": v1_draws,
                "score": round(vs_v1_score, 4),
            },
        }
        with open(metrics_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
