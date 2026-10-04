#!/usr/bin/env bash
# PPI_rp1 train: bimanual_lift_ball | full DualPanda link mesh | 8GPU
set -euo pipefail

PPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PPI_ROOT"

_DEFAULT_COPPELIASIM="${COPPELIASIM_ROOT:-}"
if [ -n "${COPPELIASIM_ROOT:-}" ] && { [ -e "${COPPELIASIM_ROOT}/libcoppeliaSim.so" ] || [ -e "${COPPELIASIM_ROOT}/libcoppeliaSim.so.1" ]; }; then
  :
else
  COPPELIASIM_ROOT="$_DEFAULT_COPPELIASIM"
fi
export COPPELIASIM_ROOT
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=2
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

ngpus=8
mkdir -p Logs exp_logs/ckpt
LOG="Logs/train_ppi_rp1_ball_8gpu.log"

echo "[$(date)] start PPI_rp1 ball full-link mesh ngpus=${ngpus}" | tee -a "$LOG"
torchrun --nnodes 1 --nproc_per_node "$ngpus" --master_port 10005 ddp_train.py \
  --config-name ppi_rp1 \
  task=ball_rp1 \
  name='train_ppi_rp1_8gpus' \
  addition_info="20260906-ball-fullmesh-retrain" \
  wandb_name="ppi_rp1_ball_retrain" \
  n_obs_steps=1 \
  n_action_steps=54 \
  policy.use_lang=true \
  policy.what_condition='ppi' \
  policy.predict_point_flow=true \
  policy.robot_allow_ee_proxy=false \
  policy.robot_points_per_link=16 \
  task.dataset.pcd_fps=6144 \
  task.dataset.pcd_type='rgb_pcd_rps6144' \
  task.dataset.point_flow_type='world_ordered_rps200' \
  task.dataset.kp_num=10 \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=64 \
  training.num_epochs=1000 \
  task.dataset.prediction_type='keyframe_continuous' \
  horizon_keyframe=4 \
  horizon_continuous=50 \
  2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "[$(date)] DONE PPI_rp1 ball exit=$ec" | tee -a "$LOG"
exit "$ec"
