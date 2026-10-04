# Ablation-exp remain | seed=0 | demo_clean | n=100
# Skip DONE(11): handover{2,3,4,6} pick_dual{2,3,4,5} skillet{2,3,5}
# Remain: 37 trainings → 13 platform jobs (1 GPU / job, gpu_id=0)
# Load: 11×3 + 2×2 = 37  (jobs 2,4,6 = 2 tasks; rest ≈3; job12 = 4)
# Each block = one job: copy job-name → name field, paste script body.
# ckpt: policy/SPACE/checkpoints/<task>-train_<enc>-100_0/
# log:  Logs/train/<task>_<enc>_s0.log

# ----- [0/12] abexp1_r0_s0  (3) ab_exp1 -----
abexp1_r0_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in handover_block hanging_mug place_object_basket; do
  echo "[$(date)] start ${TASK} ab_exp1 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/${TASK}_ab_exp1_s0.log
  echo "[$(date)] done  ${TASK} ab_exp1 seed=0"
done

# ----- [1/12] abexp1_r1_s0  (3) ab_exp1 -----
abexp1_r1_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in scan_object pick_dual_bottles place_bread_skillet; do
  echo "[$(date)] start ${TASK} ab_exp1 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/${TASK}_ab_exp1_s0.log
  echo "[$(date)] done  ${TASK} ab_exp1 seed=0"
done

# ----- [2/12] abexp1_r2_s0  (2) ab_exp1 -----
abexp1_r2_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in place_bread_basket place_cans_plasticbox; do
  echo "[$(date)] start ${TASK} ab_exp1 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/${TASK}_ab_exp1_s0.log
  echo "[$(date)] done  ${TASK} ab_exp1 seed=0"
done

# ----- [3/12] abexp2_r0_s0  (3) ab_exp2 -----
abexp2_r0_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in hanging_mug place_object_basket scan_object; do
  echo "[$(date)] start ${TASK} ab_exp2 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/${TASK}_ab_exp2_s0.log
  echo "[$(date)] done  ${TASK} ab_exp2 seed=0"
done

# ----- [4/12] abexp2_r1_s0  (2) ab_exp2 -----
abexp2_r1_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in place_bread_basket place_cans_plasticbox; do
  echo "[$(date)] start ${TASK} ab_exp2 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/${TASK}_ab_exp2_s0.log
  echo "[$(date)] done  ${TASK} ab_exp2 seed=0"
done

# ----- [5/12] abexp3_r0_s0  (3) ab_exp3 -----
abexp3_r0_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in hanging_mug place_object_basket scan_object; do
  echo "[$(date)] start ${TASK} ab_exp3 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/${TASK}_ab_exp3_s0.log
  echo "[$(date)] done  ${TASK} ab_exp3 seed=0"
done

# ----- [6/12] abexp3_r1_s0  (2) ab_exp3 -----
abexp3_r1_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in place_bread_basket place_cans_plasticbox; do
  echo "[$(date)] start ${TASK} ab_exp3 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/${TASK}_ab_exp3_s0.log
  echo "[$(date)] done  ${TASK} ab_exp3 seed=0"
done

# ----- [7/12] abexp4_r0_s0  (3) ab_exp4 -----
abexp4_r0_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in hanging_mug place_object_basket scan_object; do
  echo "[$(date)] start ${TASK} ab_exp4 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp4 2>&1 | tee ../../Logs/train/${TASK}_ab_exp4_s0.log
  echo "[$(date)] done  ${TASK} ab_exp4 seed=0"
done

# ----- [8/12] abexp4_r1_s0  (3) ab_exp4 -----
abexp4_r1_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in place_bread_skillet place_bread_basket place_cans_plasticbox; do
  echo "[$(date)] start ${TASK} ab_exp4 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp4 2>&1 | tee ../../Logs/train/${TASK}_ab_exp4_s0.log
  echo "[$(date)] done  ${TASK} ab_exp4 seed=0"
done

# ----- [9/12] abexp5_r0_s0  (3) ab_exp5 -----
abexp5_r0_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in handover_block hanging_mug place_object_basket; do
  echo "[$(date)] start ${TASK} ab_exp5 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/${TASK}_ab_exp5_s0.log
  echo "[$(date)] done  ${TASK} ab_exp5 seed=0"
done

# ----- [10/12] abexp5_r1_s0  (3) ab_exp5 -----
abexp5_r1_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in scan_object place_bread_basket place_cans_plasticbox; do
  echo "[$(date)] start ${TASK} ab_exp5 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/${TASK}_ab_exp5_s0.log
  echo "[$(date)] done  ${TASK} ab_exp5 seed=0"
done

# ----- [11/12] abexp6_r0_s0  (3) ab_exp6 -----
abexp6_r0_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in hanging_mug place_object_basket scan_object; do
  echo "[$(date)] start ${TASK} ab_exp6 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp6 2>&1 | tee ../../Logs/train/${TASK}_ab_exp6_s0.log
  echo "[$(date)] done  ${TASK} ab_exp6 seed=0"
done

# ----- [12/12] abexp6_r1_s0  (4) ab_exp6 -----
abexp6_r1_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
for TASK in pick_dual_bottles place_bread_skillet place_bread_basket place_cans_plasticbox; do
  echo "[$(date)] start ${TASK} ab_exp6 seed=0 gpu=0"
  bash train.sh ${TASK} demo_clean 100 0 0 ab_exp6 2>&1 | tee ../../Logs/train/${TASK}_ab_exp6_s0.log
  echo "[$(date)] done  ${TASK} ab_exp6 seed=0"
done
