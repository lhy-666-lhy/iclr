# Ablation-2 for 4 seed0-done tasks × ab2_ex2/ab2_ex3 = 8 trainings
# Skip ab2_ex4 (same as main); seed=0; 4 GPUs: one task per card
# Each job: ab2_ex2 → ab2_ex3 serial on that GPU
# ckpt: policy/SPACE/checkpoints/<task>-train_<enc>-100_0/

# ----- [0/3] beat_block_hammer_ab2_s0  (GPU 0) -----
beat_block_hammer_ab2_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start beat_block_hammer ab2_ex2 seed=0 gpu=0"
bash train.sh beat_block_hammer demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/beat_block_hammer_ab2_ex2_s0.log
echo "[$(date)] done  beat_block_hammer ab2_ex2 seed=0"
echo "[$(date)] start beat_block_hammer ab2_ex3 seed=0 gpu=0"
bash train.sh beat_block_hammer demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/beat_block_hammer_ab2_ex3_s0.log
echo "[$(date)] done  beat_block_hammer ab2_ex3 seed=0"

# ----- [1/3] place_cans_plasticbox_ab2_s0  (GPU 1) -----
place_cans_plasticbox_ab2_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_cans_plasticbox ab2_ex2 seed=0 gpu=1"
bash train.sh place_cans_plasticbox demo_clean 100 0 1 ab2_ex2 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab2_ex2_s0.log
echo "[$(date)] done  place_cans_plasticbox ab2_ex2 seed=0"
echo "[$(date)] start place_cans_plasticbox ab2_ex3 seed=0 gpu=1"
bash train.sh place_cans_plasticbox demo_clean 100 0 1 ab2_ex3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab2_ex3_s0.log
echo "[$(date)] done  place_cans_plasticbox ab2_ex3 seed=0"

# ----- [2/3] place_fan_ab2_s0  (GPU 2) -----
place_fan_ab2_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_fan ab2_ex2 seed=0 gpu=2"
bash train.sh place_fan demo_clean 100 0 2 ab2_ex2 2>&1 | tee ../../Logs/train/place_fan_ab2_ex2_s0.log
echo "[$(date)] done  place_fan ab2_ex2 seed=0"
echo "[$(date)] start place_fan ab2_ex3 seed=0 gpu=2"
bash train.sh place_fan demo_clean 100 0 2 ab2_ex3 2>&1 | tee ../../Logs/train/place_fan_ab2_ex3_s0.log
echo "[$(date)] done  place_fan ab2_ex3 seed=0"

# ----- [3/3] scan_object_ab2_s0  (GPU 3) -----
scan_object_ab2_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start scan_object ab2_ex2 seed=0 gpu=3"
bash train.sh scan_object demo_clean 100 0 3 ab2_ex2 2>&1 | tee ../../Logs/train/scan_object_ab2_ex2_s0.log
echo "[$(date)] done  scan_object ab2_ex2 seed=0"
echo "[$(date)] start scan_object ab2_ex3 seed=0 gpu=3"
bash train.sh scan_object demo_clean 100 0 3 ab2_ex3 2>&1 | tee ../../Logs/train/scan_object_ab2_ex3_s0.log
echo "[$(date)] done  scan_object ab2_ex3 seed=0"

