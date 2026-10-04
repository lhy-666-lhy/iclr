#!/usr/bin/env bash
set -euo pipefail
cd RLBench2/SPACE

unset COPPELIASIM_ROOT
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:?"set COPPELIASIM_ROOT"}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export DISPLAY=:99
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix || true

if [ ! -S /tmp/.X11-unix/X99 ]; then
  echo "[launch] starting Xvfb :99"
  rm -f /tmp/.X99-lock 2>/dev/null || true
  Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX -nolisten tcp >/tmp/xvfb_99.log 2>&1 &
  for i in $(seq 1 40); do
    [ -S /tmp/.X11-unix/X99 ] && break
    sleep 0.25
  done
  if [ ! -S /tmp/.X11-unix/X99 ]; then
    echo "[FATAL] Xvfb failed; /tmp/xvfb_99.log:"
    cat /tmp/xvfb_99.log 2>/dev/null || true
    exit 1
  fi
fi

OUT=/tmp/eval_laptop_epochs.out
echo "[launch] writing to $OUT"
nohup env TASK=laptop CKPT_EPOCHS="400 450 500 550" \
  COPPELIASIM_ROOT="$COPPELIASIM_ROOT" \
  bash scripts/inference/evaluate_ppi_rp1_epochs_4gpu.sh \
  >"$OUT" 2>&1 &
echo "LAUNCHER_PID=$!"
echo $! >/tmp/eval_laptop_epochs.pid
sleep 3
ls -la "$OUT"
tail -n 30 "$OUT" || true
