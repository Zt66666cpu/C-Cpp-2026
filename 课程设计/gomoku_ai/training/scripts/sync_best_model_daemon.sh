#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="/tmp/gomoku_model_sync.log"
MODELS=("pnnue.bin" "pnnue_freestyle.bin")

mkdir -p "$PROJECT_ROOT/models"

echo "[ModelSync] Started watching win4070:D:/GomokuAI/models/{pnnue.bin,pnnue_freestyle.bin} (every 120s)" >> "$LOG_FILE"

while true; do
  for MODEL_NAME in "${MODELS[@]}"; do
    LOCAL_MODEL="$PROJECT_ROOT/models/$MODEL_NAME"
    TMP_MODEL="$PROJECT_ROOT/models/${MODEL_NAME}.tmp"
    REMOTE_HASH="$(ssh -o ConnectTimeout=8 -o BatchMode=yes win4070 "D:\\TinyStories\\.venv\\Scripts\\python.exe -c \"import hashlib, pathlib; p = pathlib.Path(r'D:\\GomokuAI\\models\\${MODEL_NAME}'); print(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else '')\"" 2>/dev/null | tr -d '\r\n' || true)"

    if [[ -n "$REMOTE_HASH" ]]; then
      LOCAL_HASH=""
      if [[ -f "$LOCAL_MODEL" ]]; then
        LOCAL_HASH="$(shasum -a 256 "$LOCAL_MODEL" | awk '{print $1}')"
      fi

      if [[ "$REMOTE_HASH" != "$LOCAL_HASH" ]]; then
        if scp -o ConnectTimeout=10 -o BatchMode=yes "win4070:D:/GomokuAI/models/$MODEL_NAME" "$TMP_MODEL" >/dev/null 2>&1; then
          NEW_SIZE="$(wc -c < "$TMP_MODEL" | tr -d ' ')"
          if [[ "$NEW_SIZE" -gt 1048576 ]]; then
            mv -f "$TMP_MODEL" "$LOCAL_MODEL"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Synced new models/$MODEL_NAME (${NEW_SIZE}B, sha256=${REMOTE_HASH:0:12})" >> "$LOG_FILE"
          else
            rm -f "$TMP_MODEL"
          fi
        else
          rm -f "$TMP_MODEL"
        fi
      fi
    fi
  done

  sleep 120
done
