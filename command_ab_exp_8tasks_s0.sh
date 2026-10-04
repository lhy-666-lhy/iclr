# Ablation-exp (ab_exp1..6) on 8 dual-arm tasks | seed=0 | demo_clean | n=100
# Tasks:
#   handover_block, hanging_mug, place_object_basket, scan_object,
#   pick_dual_bottles, place_bread_skillet, place_bread_basket, place_cans_plasticbox
# Layout: 12 platform jobs = 6 encoders × 2 task-batches (4 tasks serial / job)
# Trainings: 8×6 = 48 | 1 GPU / job (gpu_id=0)
# Each block = one platform job: copy job-name into name field, paste script body.
# ckpt: policy/SPACE/checkpoints/<task>-train_<enc>-100_0/
# log:  Logs/train/<task>_<enc>_s0.log

# ===== Batch A tasks: handover_block hanging_mug place_object_basket scan_object =====
# ===== Batch B tasks: pick_dual_bottles place_bread_skillet place_bread_basket place_cans_plasticbox =====

# ----- [0/11] abexp1_batchA_s0  (GPU 0)  ab_exp1 × 4 tasks -----
abexp1_batchA_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in handover_block hanging_mug place_object_basket scan_object; do
  echo "[$(date)] start ${TASK} ab_exp1 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/${TASK}_ab_exp1_s0.log
  echo "[$(date)] done  ${TASK} ab_exp1 seed=0"
done

# ----- [1/11] abexp1_batchB_s0  (GPU 0)  ab_exp1 × 4 tasks -----
abexp1_batchB_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in pick_dual_bottles place_bread_skillet place_bread_basket place_cans_plasticbox; do
  echo "[$(date)] start ${TASK} ab_exp1 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/${TASK}_ab_exp1_s0.log
  echo "[$(date)] done  ${TASK} ab_exp1 seed=0"
done

# ----- [2/11] abexp2_batchA_s0  (GPU 0) -----
abexp2_batchA_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in handover_block hanging_mug place_object_basket scan_object; do
  echo "[$(date)] start ${TASK} ab_exp2 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/${TASK}_ab_exp2_s0.log
  echo "[$(date)] done  ${TASK} ab_exp2 seed=0"
done

# ----- [3/11] abexp2_batchB_s0  (GPU 0) -----
abexp2_batchB_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in pick_dual_bottles place_bread_skillet place_bread_basket place_cans_plasticbox; do
  echo "[$(date)] start ${TASK} ab_exp2 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/${TASK}_ab_exp2_s0.log
  echo "[$(date)] done  ${TASK} ab_exp2 seed=0"
done

# ----- [4/11] abexp3_batchA_s0  (GPU 0) -----
abexp3_batchA_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in handover_block hanging_mug place_object_basket scan_object; do
  echo "[$(date)] start ${TASK} ab_exp3 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/${TASK}_ab_exp3_s0.log
  echo "[$(date)] done  ${TASK} ab_exp3 seed=0"
done

# ----- [5/11] abexp3_batchB_s0  (GPU 0) -----
abexp3_batchB_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in pick_dual_bottles place_bread_skillet place_bread_basket place_cans_plasticbox; do
  echo "[$(date)] start ${TASK} ab_exp3 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/${TASK}_ab_exp3_s0.log
  echo "[$(date)] done  ${TASK} ab_exp3 seed=0"
done

# ----- [6/11] abexp4_batchA_s0  (GPU 0) -----
abexp4_batchA_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in handover_block hanging_mug place_object_basket scan_object; do
  echo "[$(date)] start ${TASK} ab_exp4 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp4 2>&1 | tee ../../Logs/train/${TASK}_ab_exp4_s0.log
  echo "[$(date)] done  ${TASK} ab_exp4 seed=0"
done

# ----- [7/11] abexp4_batchB_s0  (GPU 0) -----
abexp4_batchB_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in pick_dual_bottles place_bread_skillet place_bread_basket place_cans_plasticbox; do
  echo "[$(date)] start ${TASK} ab_exp4 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp4 2>&1 | tee ../../Logs/train/${TASK}_ab_exp4_s0.log
  echo "[$(date)] done  ${TASK} ab_exp4 seed=0"
done

# ----- [8/11] abexp5_batchA_s0  (GPU 0) -----
abexp5_batchA_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in handover_block hanging_mug place_object_basket scan_object; do
  echo "[$(date)] start ${TASK} ab_exp5 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/${TASK}_ab_exp5_s0.log
  echo "[$(date)] done  ${TASK} ab_exp5 seed=0"
done

# ----- [9/11] abexp5_batchB_s0  (GPU 0) -----
abexp5_batchB_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in pick_dual_bottles place_bread_skillet place_bread_basket place_cans_plasticbox; do
  echo "[$(date)] start ${TASK} ab_exp5 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/${TASK}_ab_exp5_s0.log
  echo "[$(date)] done  ${TASK} ab_exp5 seed=0"
done

# ----- [10/11] abexp6_batchA_s0  (GPU 0) -----
abexp6_batchA_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in handover_block hanging_mug place_object_basket scan_object; do
  echo "[$(date)] start ${TASK} ab_exp6 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp6 2>&1 | tee ../../Logs/train/${TASK}_ab_exp6_s0.log
  echo "[$(date)] done  ${TASK} ab_exp6 seed=0"
done

# ----- [11/11] abexp6_batchB_s0  (GPU 0) -----
abexp6_batchB_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in pick_dual_bottles place_bread_skillet place_bread_basket place_cans_plasticbox; do
  echo "[$(date)] start ${TASK} ab_exp6 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp6 2>&1 | tee ../../Logs/train/${TASK}_ab_exp6_s0.log
  echo "[$(date)] done  ${TASK} ab_exp6 seed=0"
done
