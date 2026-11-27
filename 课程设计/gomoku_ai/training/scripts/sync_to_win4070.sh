#!/usr/bin/env bash
# 将 Mac 端 gomoku_ai 源码（及可选训练数据）同步到 Windows 4070 (D:\GomokuAI)
# 用法：
#   bash training/scripts/sync_to_win4070.sh            # 仅同步源码与脚本
#   bash training/scripts/sync_to_win4070.sh --with-data # 连同最新模型、checkpoint 与 replay_buffer 一起同步
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

WITH_DATA=0
if [ "${1:-}" = "--with-data" ]; then
  WITH_DATA=1
fi

FILES=(
  "CMakeLists.txt"
  "include"
  "src"
  "tests"
)

for pyf in training/*.py; do
  [ -e "$pyf" ] && FILES+=("$pyf")
done

if [ -d "training/scripts" ]; then
  FILES+=("training/scripts")
fi

if [ "$WITH_DATA" -eq 1 ]; then
  echo "[Sync] 包含模型、checkpoint 与 replay_buffer..."
  for f in \
    models/*.bin \
    training/runs/checkpoint_v2.pt \
    training/runs/v1_*.bin \
    training/runs/champ_*_active.bin \
    training/runs/metrics.jsonl \
    training/data/replay_buffer.npz; do
    [ -e "$f" ] && FILES+=("$f")
  done
else
  echo "[Sync] 仅同步源码与脚本（不含大模型/数据文件）..."
fi

ssh win4070 "New-Item -ItemType Directory -Force -Path D:\GomokuAI | Out-Null"
COPYFILE_DISABLE=1 tar -czf - --exclude='__pycache__' --exclude='.DS_Store' "${FILES[@]}" \
  | ssh win4070 "tar -xzf - -C D:/GomokuAI"

echo "[Sync] 完成 -> win4070:D:\\GomokuAI"
