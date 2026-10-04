# 4 tasks × 11 encoders = 44 single-GPU jobs
# Tasks: beat_block_hammer, place_cans_plasticbox, place_fan, scan_object
# Include: main + Ablation-1 (ex1-3) + Ablation-3 (ex1,3-6) + Ablation-4 (ex2-3)
# Skip Ablation-2 (geo/topo on-off): ab2_ex2, ab2_ex3, ab2_ex4
# Skip same-as-main duplicates: ab1_ex4, ab3_ex2, ab4_ex1
# Each block = one platform job: copy job-name line into name field, paste script body.
# ckpt: policy/SPACE/checkpoints/<task>-train_<encoder>-100_0/

# ----- [0/44] beat_block_hammer_main -----
beat_block_hammer_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/beat_block_hammer_main.log

# ----- [1/44] beat_block_hammer_ab1_ex1 -----
beat_block_hammer_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/beat_block_hammer_ab1_ex1.log

# ----- [2/44] beat_block_hammer_ab1_ex2 -----
beat_block_hammer_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/beat_block_hammer_ab1_ex2.log

# ----- [3/44] beat_block_hammer_ab1_ex3 -----
beat_block_hammer_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/beat_block_hammer_ab1_ex3.log

# ----- [4/44] beat_block_hammer_ab3_ex1 -----
beat_block_hammer_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/beat_block_hammer_ab3_ex1.log

# ----- [5/44] beat_block_hammer_ab3_ex3 -----
beat_block_hammer_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/beat_block_hammer_ab3_ex3.log

# ----- [6/44] beat_block_hammer_ab3_ex4 -----
beat_block_hammer_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/beat_block_hammer_ab3_ex4.log

# ----- [7/44] beat_block_hammer_ab3_ex5 -----
beat_block_hammer_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/beat_block_hammer_ab3_ex5.log

# ----- [8/44] beat_block_hammer_ab3_ex6 -----
beat_block_hammer_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/beat_block_hammer_ab3_ex6.log

# ----- [9/44] beat_block_hammer_ab4_ex2 -----
beat_block_hammer_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/beat_block_hammer_ab4_ex2.log

# ----- [10/44] beat_block_hammer_ab4_ex3 -----
beat_block_hammer_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/beat_block_hammer_ab4_ex3.log

# ----- [11/44] place_cans_plasticbox_main -----
place_cans_plasticbox_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_cans_plasticbox_main.log

# ----- [12/44] place_cans_plasticbox_ab1_ex1 -----
place_cans_plasticbox_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab1_ex1.log

# ----- [13/44] place_cans_plasticbox_ab1_ex2 -----
place_cans_plasticbox_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab1_ex2.log

# ----- [14/44] place_cans_plasticbox_ab1_ex3 -----
place_cans_plasticbox_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab1_ex3.log

# ----- [15/44] place_cans_plasticbox_ab3_ex1 -----
place_cans_plasticbox_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab3_ex1.log

# ----- [16/44] place_cans_plasticbox_ab3_ex3 -----
place_cans_plasticbox_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab3_ex3.log

# ----- [17/44] place_cans_plasticbox_ab3_ex4 -----
place_cans_plasticbox_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab3_ex4.log

# ----- [18/44] place_cans_plasticbox_ab3_ex5 -----
place_cans_plasticbox_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab3_ex5.log

# ----- [19/44] place_cans_plasticbox_ab3_ex6 -----
place_cans_plasticbox_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab3_ex6.log

# ----- [20/44] place_cans_plasticbox_ab4_ex2 -----
place_cans_plasticbox_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab4_ex2.log

# ----- [21/44] place_cans_plasticbox_ab4_ex3 -----
place_cans_plasticbox_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab4_ex3.log

# ----- [22/44] place_fan_main -----
place_fan_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_fan_main.log

# ----- [23/44] place_fan_ab1_ex1 -----
place_fan_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_fan_ab1_ex1.log

# ----- [24/44] place_fan_ab1_ex2 -----
place_fan_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_fan_ab1_ex2.log

# ----- [25/44] place_fan_ab1_ex3 -----
place_fan_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_fan_ab1_ex3.log

# ----- [26/44] place_fan_ab3_ex1 -----
place_fan_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_fan_ab3_ex1.log

# ----- [27/44] place_fan_ab3_ex3 -----
place_fan_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_fan_ab3_ex3.log

# ----- [28/44] place_fan_ab3_ex4 -----
place_fan_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_fan_ab3_ex4.log

# ----- [29/44] place_fan_ab3_ex5 -----
place_fan_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_fan_ab3_ex5.log

# ----- [30/44] place_fan_ab3_ex6 -----
place_fan_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_fan_ab3_ex6.log

# ----- [31/44] place_fan_ab4_ex2 -----
place_fan_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_fan_ab4_ex2.log

# ----- [32/44] place_fan_ab4_ex3 -----
place_fan_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_fan_ab4_ex3.log

# ----- [33/44] scan_object_main -----
scan_object_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/scan_object_main.log

# ----- [34/44] scan_object_ab1_ex1 -----
scan_object_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/scan_object_ab1_ex1.log

# ----- [35/44] scan_object_ab1_ex2 -----
scan_object_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/scan_object_ab1_ex2.log

# ----- [36/44] scan_object_ab1_ex3 -----
scan_object_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/scan_object_ab1_ex3.log

# ----- [37/44] scan_object_ab3_ex1 -----
scan_object_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/scan_object_ab3_ex1.log

# ----- [38/44] scan_object_ab3_ex3 -----
scan_object_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/scan_object_ab3_ex3.log

# ----- [39/44] scan_object_ab3_ex4 -----
scan_object_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/scan_object_ab3_ex4.log

# ----- [40/44] scan_object_ab3_ex5 -----
scan_object_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/scan_object_ab3_ex5.log

# ----- [41/44] scan_object_ab3_ex6 -----
scan_object_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/scan_object_ab3_ex6.log

# ----- [42/44] scan_object_ab4_ex2 -----
scan_object_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/scan_object_ab4_ex2.log

# ----- [43/44] scan_object_ab4_ex3 -----
scan_object_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/scan_object_ab4_ex3.log
