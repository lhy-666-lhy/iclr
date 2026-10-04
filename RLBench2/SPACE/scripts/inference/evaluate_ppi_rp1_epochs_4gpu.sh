#!/usr/bin/env bash
# Reliable sequential 4-GPU PPI_rp1 eval over multiple ckpt epochs.
# Fixes: Xvfb DISPLAY, CoppeliaSim LD_LIBRARY_PATH, unique LOGDIR, wait-between-epochs.
#
# Usage (run on an idle 4-GPU node):
#   TASK=drawer   CKPT_EPOCHS="400 450 500 550" bash scripts/inference/evaluate_ppi_rp1_epochs_4gpu.sh
#   TASK=laptop   CKPT_EPOCHS="400 450 500 550" bash scripts/inference/evaluate_ppi_rp1_epochs_4gpu.sh
#   TASK=handover_easy CKPT_EPOCHS="400 450 500 550" bash scripts/inference/evaluate_ppi_rp1_epochs_4gpu.sh
#   TASK=ball CKPT_EPOCHS="350 400 450" bash scripts/inference/evaluate_ppi_rp1_epochs_4gpu.sh
#   TASK=box  CKPT_EPOCHS="350 400 450" bash scripts/inference/evaluate_ppi_rp1_epochs_4gpu.sh
#   TASK=dustpan CKPT_EPOCHS="300 350 400 450 500 550 600 650" bash scripts/inference/evaluate_ppi_rp1_epochs_4gpu.sh
#   TASK=tray    CKPT_EPOCHS="300 350 400 450 500 550 600 650" bash scripts/inference/evaluate_ppi_rp1_epochs_4gpu.sh
set -euo pipefail

PPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PPI_ROOT"

TASK="${TASK:?Set TASK=drawer|laptop|handover_easy|ball|box|dustpan|sweep|tray}"
# normalize alias
if [ "$TASK" = "sweep" ]; then TASK=dustpan; fi
CKPT_EPOCHS="${CKPT_EPOCHS:-400 450 500 550}"
NUM_INFER="${NUM_INFER:-1000}"
TOTAL_EPS="${TOTAL_EPS:-100}"
NGPU="${NGPU:-4}"
DISPLAY_NUM="${DISPLAY_NUM:-99}"
# Prefer a known-good CoppeliaSim path; ignore stale/wrong env overrides without libs.
_DEFAULT_COPPELIASIM="${COPPELIASIM_ROOT:-}"
if [ -n "${COPPELIASIM_ROOT:-}" ] && { [ -e "${COPPELIASIM_ROOT}/libcoppeliaSim.so" ] || [ -e "${COPPELIASIM_ROOT}/libcoppeliaSim.so.1" ]; }; then
  :
else
  COPPELIASIM_ROOT="$_DEFAULT_COPPELIASIM"
fi
CONDA_BIN=""
DEMO_PATH="${DEMO_PATH:-${PPI_ROOT}/data/rlbench2_official/peract2_test_ppi}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$(hostname -s)}"

export PATH="${CONDA_BIN}:${PATH}"
export PYTHONPATH="${PPI_ROOT}:${PPI_ROOT}/inference-for-rlbench2:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export COPPELIASIM_ROOT
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export DISPLAY=":${DISPLAY_NUM}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MASTER_LOG="exp_logs/eval_runs/${TASK}_epochs_seq_${RUN_TAG}"
mkdir -p "$MASTER_LOG"
exec > >(tee -a "${MASTER_LOG}/launcher.log") 2>&1

# ---- helpers (no xdpyinfo dependency; many nodes lack x11-utils) ----
display_is_up() {
  local d="${1#:}"
  [ -S "/tmp/.X11-unix/X${d}" ]
}

ensure_xvfb() {
  local d="${1#:}"
  mkdir -p /tmp/.X11-unix
  chmod 1777 /tmp/.X11-unix 2>/dev/null || true
  if display_is_up "$d"; then
    echo "[setup] DISPLAY :${d} already up (socket OK)"
    return 0
  fi
  echo "[setup] starting Xvfb :${d}"
  rm -f "/tmp/.X${d}-lock" 2>/dev/null || true
  # Prefer a free display if :d is locked but socket missing
  Xvfb ":${d}" -screen 0 1024x768x24 -ac +extension GLX -nolisten tcp \
    >"/tmp/xvfb_${d}.log" 2>&1 &
  local xpid=$!
  echo "$xpid" > "${MASTER_LOG:-/tmp}/xvfb.pid"
  local i
  for i in $(seq 1 40); do
    if display_is_up "$d"; then
      echo "[setup] Xvfb :${d} ready (pid=$xpid)"
      return 0
    fi
    # process died
    if ! kill -0 "$xpid" 2>/dev/null; then
      echo "[setup] Xvfb :${d} exited early; log:"
      cat "/tmp/xvfb_${d}.log" 2>/dev/null || true
      break
    fi
    sleep 0.25
  done
  # fallback: try alternate displays 90-98
  local alt
  for alt in 90 91 92 93 94 95 96 97 98; do
    if display_is_up "$alt"; then
      DISPLAY_NUM="$alt"
      export DISPLAY=":${alt}"
      echo "[setup] reusing existing DISPLAY :${alt}"
      return 0
    fi
    if [ -e "/tmp/.X${alt}-lock" ]; then continue; fi
    echo "[setup] trying fallback Xvfb :${alt}"
    Xvfb ":${alt}" -screen 0 1024x768x24 -ac +extension GLX -nolisten tcp \
      >"/tmp/xvfb_${alt}.log" 2>&1 &
    xpid=$!
    for i in $(seq 1 40); do
      if display_is_up "$alt"; then
        DISPLAY_NUM="$alt"
        export DISPLAY=":${alt}"
        echo "$xpid" > "${MASTER_LOG:-/tmp}/xvfb.pid"
        echo "[setup] Xvfb :${alt} ready (pid=$xpid)"
        return 0
      fi
      kill -0 "$xpid" 2>/dev/null || break
      sleep 0.25
    done
  done
  echo "[FATAL] could not start any Xvfb display"
  return 1
}

echo "=============================================="
echo "HOST=$(hostname)  TASK=$TASK  EPOCHS=$CKPT_EPOCHS"
echo "RUN_TAG=$RUN_TAG  DISPLAY=$DISPLAY"
echo "COPPELIASIM_ROOT=$COPPELIASIM_ROOT"
echo "DEMO_PATH=$DEMO_PATH"
echo "MASTER_LOG=$MASTER_LOG"
echo "=============================================="

# ---- 1) Xvfb ----
ensure_xvfb "$DISPLAY_NUM" || exit 1
export DISPLAY=":${DISPLAY_NUM}"
echo "[setup] using DISPLAY=$DISPLAY"

# ---- 2) preflight ----
if [ ! -x "${CONDA_BIN}/python" ]; then
  echo "[FATAL] missing conda python: ${CONDA_BIN}/python"; exit 1
fi
if [ ! -e "${COPPELIASIM_ROOT}/libcoppeliaSim.so" ] && [ ! -e "${COPPELIASIM_ROOT}/libcoppeliaSim.so.1" ]; then
  echo "[FATAL] missing CoppeliaSim lib under $COPPELIASIM_ROOT"; exit 1
fi
if [ ! -d "$DEMO_PATH" ]; then
  echo "[FATAL] missing DEMO_PATH=$DEMO_PATH"; exit 1
fi
NGPU_AVAIL=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
if [ "${NGPU_AVAIL:-0}" -lt "$NGPU" ]; then
  echo "[FATAL] need $NGPU GPUs, nvidia-smi sees ${NGPU_AVAIL:-0}"; exit 1
fi

echo "[preflight] import pyrep/sim ..."
python - <<'PY'
from pyrep.backend import sim
print("pyrep/sim OK")
PY

case "$TASK" in
  drawer)
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_put_item_in_drawer"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_drawer_rp1_ppi_rp1_20260831-drawer-fullmesh_seed0}"
    RL_TASK="bimanual_put_item_in_drawer"
    ;;
  laptop)
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_pick_laptop"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_laptop_rp1_ppi_rp1_20260831-laptop-fullmesh_seed0}"
    RL_TASK="bimanual_pick_laptop"
    ;;
  handover_easy)
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_handover_item_easy"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_handover_easy_rp1_ppi_rp1_20260831-handover_easy-fullmesh_seed0}"
    RL_TASK="bimanual_handover_item_easy"
    ;;
  ball)
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_lift_ball"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_ball_rp1_ppi_rp1_20260902-ball-fullmesh_seed0}"
    RL_TASK="bimanual_lift_ball"
    ;;
  box)
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_push_box"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_box_rp1_ppi_rp1_20260902-box-fullmesh_seed0}"
    RL_TASK="bimanual_push_box"
    ;;
  dustpan)
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_sweep_to_dustpan"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_dustpan_rp1_ppi_rp1_20260902-dustpan-fullmesh_seed0}"
    RL_TASK="bimanual_sweep_to_dustpan"
    ;;
  tray)
    WEIGHTS_DIR="exp_logs/ckpt/bimanual_lift_tray"
    WEIGHT_NAME="${WEIGHT_NAME:-train_ppi_rp1_8gpus_lift_tray_rp1_ppi_rp1_20260902-tray-fullmesh_seed0}"
    RL_TASK="bimanual_lift_tray"
    ;;
  *)
    echo "[FATAL] TASK must be drawer|laptop|handover_easy|ball|box|dustpan|sweep|tray (got $TASK)"; exit 1
    ;;
esac

if [ ! -d "${DEMO_PATH}/${RL_TASK}/all_variations/episodes" ]; then
  echo "[FATAL] missing seed demos: ${DEMO_PATH}/${RL_TASK}/all_variations/episodes"; exit 1
fi

for ep in $CKPT_EPOCHS; do
  ckpt="${WEIGHTS_DIR}/${WEIGHT_NAME}/checkpoints/epoch${ep}_model.pth.tar"
  if [ ! -f "$ckpt" ]; then
    echo "[FATAL] missing ckpt: $ckpt"; exit 1
  fi
  echo "[preflight] ckpt ok: epoch${ep}"
done

echo "[preflight] ALL CHECKS PASSED"
echo

# ---- 3) sequential epochs ----
FOURGPU="${PPI_ROOT}/scripts/inference/evaluate_ppi_rp1_task_4gpu.sh"
for ep in $CKPT_EPOCHS; do
  CKPT_NAME="epoch${ep}_model"
  LOGDIR="exp_logs/eval_runs/${TASK}_4gpu_${CKPT_NAME}_peract2seeds_infer${NUM_INFER}_${RUN_TAG}"
  echo "############################################################"
  echo "[run] TASK=$TASK CKPT=$CKPT_NAME LOGDIR=$LOGDIR"
  echo "############################################################"

  # export so child script picks unique logdir + env
  export TASK CKPT_NAME NUM_INFER TOTAL_EPS NGPU DEMO_PATH WEIGHT_NAME
  export DISPLAY=":${DISPLAY_NUM}"
  export COPPELIASIM_ROOT LD_LIBRARY_PATH QT_QPA_PLATFORM_PLUGIN_PATH
  export LOGDIR_OVERRIDE="$LOGDIR"
  export WAIT_FOR_JOBS=1

  bash "$FOURGPU"

  # belt-and-suspenders: wait on pids file if child forgot
  if [ -f "${LOGDIR}/pids.txt" ]; then
    mapfile -t pids < "${LOGDIR}/pids.txt"
    echo "[wait] waiting for PIDs: ${pids[*]}"
    fail=0
    for p in "${pids[@]}"; do
      if ! wait "$p" 2>/dev/null; then
        # process may not be child of this shell if launched in subshell; poll instead
        while kill -0 "$p" 2>/dev/null; do sleep 30; done
      fi
      # detect hard crash: no Score lines and log mentions Qt/ImportError
      if ! grep -q 'Score:' "${LOGDIR}"/gpu*.log 2>/dev/null; then
        if grep -qiE 'ImportError|could not connect to display|FATAL|Traceback' "${LOGDIR}"/gpu*.log 2>/dev/null; then
          echo "[FATAL] epoch${ep} produced no Score and has errors; aborting sequence"
          fail=1
        fi
      fi
    done
    if [ "$fail" -eq 1 ]; then
      exit 1
    fi
  fi

  # summarize this epoch
  python - <<PY
import re, glob
from pathlib import Path
logdir = Path("${LOGDIR}")
pat = re.compile(r"Evaluating .*\| Episode (\\d+) \\| Score:\\s*([0-9.]+)")
s=f=0
for fp in sorted(logdir.glob("gpu*.log")):
    epmap={}
    for line in open(fp, errors="ignore"):
        m=pat.search(line)
        if m: epmap[int(m.group(1))]=float(m.group(2))
    s += sum(1 for v in epmap.values() if v>=100)
    f += sum(1 for v in epmap.values() if v<100)
n=s+f
print(f"[summary] {logdir.name}: done={n}/100 succ={s} fail={f} rate={s/n if n else 0:.3f}")
PY
  echo
done

echo "[done] all epochs finished for TASK=$TASK RUN_TAG=$RUN_TAG"
echo "MASTER_LOG=$MASTER_LOG"
