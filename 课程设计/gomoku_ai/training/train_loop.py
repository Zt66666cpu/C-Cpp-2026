import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine_bridge import (
    find_engine_lib,
    generate_games_parallel,
    run_arena_match,
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
) -> Dict[str, float]:
    features = data["features"].astype(np.float32)
    targets = (0.65 * data["outcomes"] + 0.35 * data["search_values"]).astype(np.float32)

    if update_norm:
        mean = features.mean(axis=0)
        std = np.maximum(features.std(axis=0), 1e-3)
        model.set_norm_stats(mean, std)

    model.to(device)
    model.train()

    n = features.shape[0]
    rng = np.random.default_rng(1234)
    perm = rng.permutation(n)
    val_n = max(1, int(n * 0.1))
    val_idx = perm[:val_n]
    trn_idx = perm[val_n:]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_val_loss = float("inf")
    last_train_loss = 0.0

    for ep in range(epochs):
        rng.shuffle(trn_idx)
        ep_loss = 0.0
        steps = 0
        for s in range(0, len(trn_idx), batch_size):
            b_idx = trn_idx[s : s + batch_size]
            xb = torch.from_numpy(features[b_idx]).to(device)
            yb = torch.from_numpy(targets[b_idx]).to(device)

            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += float(loss.item())
            steps += 1
        scheduler.step()
        last_train_loss = ep_loss / max(1, steps)

    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(features[val_idx]).to(device)
        yv = torch.from_numpy(targets[val_idx]).to(device)
        best_val_loss = float(F.mse_loss(model(xv), yv).item())

    model.cpu()
    return {"nnue_train_mse": last_train_loss, "nnue_val_mse": best_val_loss}


def train_cnn(
    model: PolicyValueCNN,
    data: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Dict[str, float]:
    x_all = boards_to_cnn_tensor(
        data["boards"], data["players"], data["last_moves"], data["forbidden"]
    )
    p_all = data["policies"].astype(np.float32)
    # 合法位置 = 空位且非黑方禁手
    legal_all = (data["boards"] == 0) & (data["forbidden"] == 0)
    v_all = (0.60 * data["outcomes"] + 0.40 * data["search_values"]).astype(np.float32)

    n = x_all.shape[0]
    rng = np.random.default_rng(5678)
    perm = rng.permutation(n)
    val_n = max(1, int(n * 0.1))
    val_idx = perm[:val_n]
    trn_idx = perm[val_n:]

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    last_pol_loss = 0.0
    last_val_loss = 0.0

    for ep in range(epochs):
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
            val_loss = F.mse_loss(val_pred, vt)
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
            val_loss = F.mse_loss(val_pred, vt, reduction="sum")

            val_pol_sum += float(pol_loss.item())
            val_mse_sum += float(val_loss.item())
            val_top1_hits += int((masked_logits.argmax(dim=-1) == pt.argmax(dim=-1)).sum().item())
            val_total += len(b_idx)

    model.cpu()
    return {
        "cnn_train_pol": last_pol_loss,
        "cnn_train_val": last_val_loss,
        "cnn_val_pol": val_pol_sum / max(1, val_total),
        "cnn_val_mse": val_mse_sum / max(1, val_total),
        "cnn_val_top1": val_top1_hits / max(1, val_total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gomoku Route-3 Dual-Net Self-Play Trainer")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--games-per-round", type=int, default=800)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--vcf-depth", type=int, default=10)
    parser.add_argument("--time-limit", type=float, default=0.5)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-depth", type=int, default=3)
    parser.add_argument("--eval-time", type=float, default=0.4)
    parser.add_argument("--cnn-weight", type=float, default=0.35)
    parser.add_argument("--nnue-epochs", type=int, default=35)
    parser.add_argument("--cnn-epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--max-buffer", type=int, default=160000)
    parser.add_argument("--ch", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
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
    print(f"[Trainer] Device={device}, C_lib={lib_path}")

    nnue = NnueNet(in_dim=32, h1=64, h2=32)
    cnn = PolicyValueCNN(ch=args.ch, num_blocks=args.blocks)

    if args.verify_only:
        parity = verify_c_parity(lib_path, nnue, cnn)
        print(f"[Verify] PyTorch vs C11 parity: {parity}")
        for k, v in parity.items():
            assert v < 1e-4, f"Parity check failed for {k}: {v}"
        print("[Verify] PASSED (< 1e-4)")
        return

    ckpt_path = runs_dir / "checkpoint.pt"
    start_round = 1
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        nnue.load_state_dict(ckpt["nnue"])
        cnn.load_state_dict(ckpt["cnn"])
        start_round = int(ckpt["round"]) + 1
        print(f"[Trainer] Resumed from round {start_round - 1}")

    best_nnue_bin = models_dir / "nnue.bin"
    best_cnn_bin = models_dir / "cnn.bin"
    replay_path = data_dir / "replay_buffer.npz"
    metrics_path = runs_dir / "metrics.jsonl"

    for rnd in range(start_round, start_round + args.rounds):
        t0 = time.time()
        print(f"\n================ Round {rnd} ================")

        # Round 1 若尚无已晋级模型则用纯搜索冷启动，之后用当前冠军双网自博弈
        sp_nnue = best_nnue_bin if best_nnue_bin.exists() else None
        sp_cnn = best_cnn_bin if best_cnn_bin.exists() else None
        temp = 0.85 if rnd == 1 else 0.65

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
            seed=202600 + rnd * 99991,
        )
        sp_sec = time.time() - t_sp
        new_pos = int(new_data["outcomes"].shape[0])
        buf_data = append_replay_buffer(replay_path, new_data, args.max_buffer)
        total_pos = int(buf_data["outcomes"].shape[0])
        print(
            f"[SelfPlay] {args.games_per_round} games -> {new_pos} positions in {sp_sec:.1f}s "
            f"(replay buffer={total_pos})"
        )

        t_tr = time.time()
        nnue_metrics = train_nnue(
            nnue,
            buf_data,
            device=device,
            epochs=args.nnue_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            update_norm=(rnd == 1 and not args.resume),
        )
        cnn_metrics = train_cnn(
            cnn,
            buf_data,
            device=device,
            epochs=args.cnn_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
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

        # 评测 1：Candidate (NNUE+CNN) vs Pure Heuristic Baseline
        t_ev = time.time()
        vs_base = run_arena_match(
            lib_path,
            cand_nnue, cand_cnn, args.eval_depth, args.vcf_depth, args.eval_time, args.cnn_weight,
            None, None, args.eval_depth, args.vcf_depth, args.eval_time, 0.0,
            num_games=args.eval_games,
            seed=7000 + rnd * 101,
        )
        base_score = (vs_base.a_wins + 0.5 * vs_base.draws) / args.eval_games
        print(
            f"[Arena vs Heuristic] W={vs_base.a_wins} L={vs_base.b_wins} D={vs_base.draws} "
            f"score={base_score:.3f} illegal={vs_base.illegal_moves} "
            f"t_cand={vs_base.a_avg_time_sec*1000:.1f}ms t_base={vs_base.b_avg_time_sec*1000:.1f}ms"
        )

        # 评测 2：Candidate vs Current Champion（若已存在）
        promoted = False
        champ_score = base_score
        if best_nnue_bin.exists() and best_cnn_bin.exists():
            vs_champ = run_arena_match(
                lib_path,
                cand_nnue, cand_cnn, args.eval_depth, args.vcf_depth, args.eval_time, args.cnn_weight,
                best_nnue_bin, best_cnn_bin, args.eval_depth, args.vcf_depth, args.eval_time, args.cnn_weight,
                num_games=args.eval_games,
                seed=9000 + rnd * 101,
            )
            champ_score = (vs_champ.a_wins + 0.5 * vs_champ.draws) / args.eval_games
            print(
                f"[Arena vs Champion]  W={vs_champ.a_wins} L={vs_champ.b_wins} D={vs_champ.draws} "
                f"score={champ_score:.3f} illegal={vs_champ.illegal_moves}"
            )
            promoted = (vs_champ.illegal_moves == 0 and champ_score >= 0.50)
        else:
            promoted = (vs_base.illegal_moves == 0 and base_score >= 0.50)

        if promoted or not best_nnue_bin.exists():
            shutil.copy2(cand_nnue, best_nnue_bin)
            shutil.copy2(cand_cnn, best_cnn_bin)
            print(f"[Gate] PROMOTED Round {rnd} models -> models/nnue.bin & models/cnn.bin")
        else:
            print(f"[Gate] Kept previous champion (cand vs champ={champ_score:.3f})")

        torch.save(
            {"round": rnd, "nnue": nnue.state_dict(), "cnn": cnn.state_dict()},
            ckpt_path,
        )

        record = {
            "round": rnd,
            "new_positions": new_pos,
            "buffer_positions": total_pos,
            "selfplay_sec": round(sp_sec, 2),
            "train_sec": round(tr_sec, 2),
            "eval_sec": round(time.time() - t_ev, 2),
            "total_sec": round(time.time() - t0, 2),
            **nnue_metrics,
            **cnn_metrics,
            "vs_heuristic": {
                "wins": vs_base.a_wins,
                "losses": vs_base.b_wins,
                "draws": vs_base.draws,
                "score": round(base_score, 4),
                "illegal": vs_base.illegal_moves,
                "cand_ms": round(vs_base.a_avg_time_sec * 1000, 2),
                "base_ms": round(vs_base.b_avg_time_sec * 1000, 2),
            },
            "vs_champion_score": round(champ_score, 4),
            "promoted": promoted,
        }
        with open(metrics_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
