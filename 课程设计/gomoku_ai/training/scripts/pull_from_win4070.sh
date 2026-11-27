#!/usr/bin/env bash
# 从 Windows 4070 (D:\GomokuAI) 拉回最新模型、checkpoint 与指标日志到 Mac（自动备份）
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="training/runs/backup_${STAMP}"
mkdir -p "$BACKUP_DIR" models training/runs training/runs_v3

for f in models/*.bin training/runs/checkpoint_v2.pt training/runs/metrics.jsonl training/runs/champ_*_active.bin \
         training/runs_v3/pnnue_champ.pt training/runs_v3/pnnue_champ.bin; do
  [ -e "$f" ] && cp "$f" "$BACKUP_DIR/"
done
echo "[Pull] 已备份本机原有产物到 $BACKUP_DIR"

ssh win4070 'D:\TinyStories\.venv\Scripts\python.exe -' << 'PYEOF' | tar -xzf - -C "$PROJECT_DIR"
import sys
import tarfile
from pathlib import Path

root = Path(r"D:\GomokuAI")
patterns = [
    "models/*.bin",
    "training/runs/checkpoint_v2.pt",
    "training/runs/metrics.jsonl",
    "training/runs/champ_*_active.bin",
    "training/runs/train_win4070.log",
    "training/runs_v3/pnnue_champ.pt",
    "training/runs_v3/pnnue_champ.bin",
    "training/runs_v3/train_v3_win4070.log",
]
tf = tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz")
for pat in patterns:
    for p in sorted(root.glob(pat)):
        if p.is_file():
            tf.add(p, arcname=p.relative_to(root).as_posix())
tf.close()
PYEOF

echo "[Pull] 已同步最新产物回 Mac：$PROJECT_DIR"
