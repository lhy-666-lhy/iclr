# Ablation seed0 only | 31 tasks (NO turn_switch) | skip main & same-as-main
# Encoders: ab1_ex{1,2,3} ab2_ex{2,3} ab3_ex{1,3,4,5,6} ab4_ex{2,3}  (12)
# Skip done ckpts; beat_block_hammer already complete → omitted
# Policy:
#   - Normal tasks: 1 platform job / task, serial ablations on 1 GPU (gpu_id=0)
#   - Slow tasks (main still incomplete): 2 platform jobs / task, split missing ablations
# Slow tasks: open_microwave, put_bottles_dustbin, stack_blocks_three, stack_blocks_two, stack_bowls_three, stack_bowls_two
# Jobs: 36  |  trainings: 329  |  1GPU-jobs: 24  |  SLOW-part-jobs: 12
# Each block = one platform job: copy job-name into name field, paste script body.
# ckpt: policy/SPACE/checkpoints/<task>-train_<enc>-100_0/
# log:  Logs/train/<task>_<enc>_s0.log

# ----- [0/35] handover_block_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
handover_block_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start handover_block ab1_ex1 seed=0"
bash train.sh handover_block demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/handover_block_ab1_ex1_s0.log
echo "[$(date)] done  handover_block ab1_ex1 seed=0"
echo "[$(date)] start handover_block ab1_ex2 seed=0"
bash train.sh handover_block demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/handover_block_ab1_ex2_s0.log
echo "[$(date)] done  handover_block ab1_ex2 seed=0"
echo "[$(date)] start handover_block ab1_ex3 seed=0"
bash train.sh handover_block demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/handover_block_ab1_ex3_s0.log
echo "[$(date)] done  handover_block ab1_ex3 seed=0"
echo "[$(date)] start handover_block ab2_ex2 seed=0"
bash train.sh handover_block demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/handover_block_ab2_ex2_s0.log
echo "[$(date)] done  handover_block ab2_ex2 seed=0"
echo "[$(date)] start handover_block ab2_ex3 seed=0"
bash train.sh handover_block demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/handover_block_ab2_ex3_s0.log
echo "[$(date)] done  handover_block ab2_ex3 seed=0"
echo "[$(date)] start handover_block ab3_ex1 seed=0"
bash train.sh handover_block demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/handover_block_ab3_ex1_s0.log
echo "[$(date)] done  handover_block ab3_ex1 seed=0"
echo "[$(date)] start handover_block ab3_ex3 seed=0"
bash train.sh handover_block demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/handover_block_ab3_ex3_s0.log
echo "[$(date)] done  handover_block ab3_ex3 seed=0"
echo "[$(date)] start handover_block ab3_ex4 seed=0"
bash train.sh handover_block demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/handover_block_ab3_ex4_s0.log
echo "[$(date)] done  handover_block ab3_ex4 seed=0"
echo "[$(date)] start handover_block ab3_ex5 seed=0"
bash train.sh handover_block demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/handover_block_ab3_ex5_s0.log
echo "[$(date)] done  handover_block ab3_ex5 seed=0"
echo "[$(date)] start handover_block ab3_ex6 seed=0"
bash train.sh handover_block demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/handover_block_ab3_ex6_s0.log
echo "[$(date)] done  handover_block ab3_ex6 seed=0"
echo "[$(date)] start handover_block ab4_ex2 seed=0"
bash train.sh handover_block demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/handover_block_ab4_ex2_s0.log
echo "[$(date)] done  handover_block ab4_ex2 seed=0"
echo "[$(date)] start handover_block ab4_ex3 seed=0"
bash train.sh handover_block demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/handover_block_ab4_ex3_s0.log
echo "[$(date)] done  handover_block ab4_ex3 seed=0"

# ----- [1/35] hanging_mug_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
hanging_mug_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start hanging_mug ab1_ex1 seed=0"
bash train.sh hanging_mug demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/hanging_mug_ab1_ex1_s0.log
echo "[$(date)] done  hanging_mug ab1_ex1 seed=0"
echo "[$(date)] start hanging_mug ab1_ex2 seed=0"
bash train.sh hanging_mug demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/hanging_mug_ab1_ex2_s0.log
echo "[$(date)] done  hanging_mug ab1_ex2 seed=0"
echo "[$(date)] start hanging_mug ab1_ex3 seed=0"
bash train.sh hanging_mug demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/hanging_mug_ab1_ex3_s0.log
echo "[$(date)] done  hanging_mug ab1_ex3 seed=0"
echo "[$(date)] start hanging_mug ab2_ex2 seed=0"
bash train.sh hanging_mug demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/hanging_mug_ab2_ex2_s0.log
echo "[$(date)] done  hanging_mug ab2_ex2 seed=0"
echo "[$(date)] start hanging_mug ab2_ex3 seed=0"
bash train.sh hanging_mug demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/hanging_mug_ab2_ex3_s0.log
echo "[$(date)] done  hanging_mug ab2_ex3 seed=0"
echo "[$(date)] start hanging_mug ab3_ex1 seed=0"
bash train.sh hanging_mug demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/hanging_mug_ab3_ex1_s0.log
echo "[$(date)] done  hanging_mug ab3_ex1 seed=0"
echo "[$(date)] start hanging_mug ab3_ex3 seed=0"
bash train.sh hanging_mug demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/hanging_mug_ab3_ex3_s0.log
echo "[$(date)] done  hanging_mug ab3_ex3 seed=0"
echo "[$(date)] start hanging_mug ab3_ex4 seed=0"
bash train.sh hanging_mug demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/hanging_mug_ab3_ex4_s0.log
echo "[$(date)] done  hanging_mug ab3_ex4 seed=0"
echo "[$(date)] start hanging_mug ab3_ex5 seed=0"
bash train.sh hanging_mug demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/hanging_mug_ab3_ex5_s0.log
echo "[$(date)] done  hanging_mug ab3_ex5 seed=0"
echo "[$(date)] start hanging_mug ab3_ex6 seed=0"
bash train.sh hanging_mug demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/hanging_mug_ab3_ex6_s0.log
echo "[$(date)] done  hanging_mug ab3_ex6 seed=0"
echo "[$(date)] start hanging_mug ab4_ex2 seed=0"
bash train.sh hanging_mug demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/hanging_mug_ab4_ex2_s0.log
echo "[$(date)] done  hanging_mug ab4_ex2 seed=0"
echo "[$(date)] start hanging_mug ab4_ex3 seed=0"
bash train.sh hanging_mug demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/hanging_mug_ab4_ex3_s0.log
echo "[$(date)] done  hanging_mug ab4_ex3 seed=0"

# ----- [2/35] move_can_pot_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
move_can_pot_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start move_can_pot ab1_ex1 seed=0"
bash train.sh move_can_pot demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/move_can_pot_ab1_ex1_s0.log
echo "[$(date)] done  move_can_pot ab1_ex1 seed=0"
echo "[$(date)] start move_can_pot ab1_ex2 seed=0"
bash train.sh move_can_pot demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/move_can_pot_ab1_ex2_s0.log
echo "[$(date)] done  move_can_pot ab1_ex2 seed=0"
echo "[$(date)] start move_can_pot ab1_ex3 seed=0"
bash train.sh move_can_pot demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/move_can_pot_ab1_ex3_s0.log
echo "[$(date)] done  move_can_pot ab1_ex3 seed=0"
echo "[$(date)] start move_can_pot ab2_ex2 seed=0"
bash train.sh move_can_pot demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/move_can_pot_ab2_ex2_s0.log
echo "[$(date)] done  move_can_pot ab2_ex2 seed=0"
echo "[$(date)] start move_can_pot ab2_ex3 seed=0"
bash train.sh move_can_pot demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/move_can_pot_ab2_ex3_s0.log
echo "[$(date)] done  move_can_pot ab2_ex3 seed=0"
echo "[$(date)] start move_can_pot ab3_ex1 seed=0"
bash train.sh move_can_pot demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/move_can_pot_ab3_ex1_s0.log
echo "[$(date)] done  move_can_pot ab3_ex1 seed=0"
echo "[$(date)] start move_can_pot ab3_ex3 seed=0"
bash train.sh move_can_pot demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/move_can_pot_ab3_ex3_s0.log
echo "[$(date)] done  move_can_pot ab3_ex3 seed=0"
echo "[$(date)] start move_can_pot ab3_ex4 seed=0"
bash train.sh move_can_pot demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/move_can_pot_ab3_ex4_s0.log
echo "[$(date)] done  move_can_pot ab3_ex4 seed=0"
echo "[$(date)] start move_can_pot ab3_ex5 seed=0"
bash train.sh move_can_pot demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/move_can_pot_ab3_ex5_s0.log
echo "[$(date)] done  move_can_pot ab3_ex5 seed=0"
echo "[$(date)] start move_can_pot ab3_ex6 seed=0"
bash train.sh move_can_pot demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/move_can_pot_ab3_ex6_s0.log
echo "[$(date)] done  move_can_pot ab3_ex6 seed=0"
echo "[$(date)] start move_can_pot ab4_ex2 seed=0"
bash train.sh move_can_pot demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/move_can_pot_ab4_ex2_s0.log
echo "[$(date)] done  move_can_pot ab4_ex2 seed=0"
echo "[$(date)] start move_can_pot ab4_ex3 seed=0"
bash train.sh move_can_pot demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/move_can_pot_ab4_ex3_s0.log
echo "[$(date)] done  move_can_pot ab4_ex3 seed=0"

# ----- [3/35] open_laptop_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
open_laptop_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start open_laptop ab1_ex1 seed=0"
bash train.sh open_laptop demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/open_laptop_ab1_ex1_s0.log
echo "[$(date)] done  open_laptop ab1_ex1 seed=0"
echo "[$(date)] start open_laptop ab1_ex2 seed=0"
bash train.sh open_laptop demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/open_laptop_ab1_ex2_s0.log
echo "[$(date)] done  open_laptop ab1_ex2 seed=0"
echo "[$(date)] start open_laptop ab1_ex3 seed=0"
bash train.sh open_laptop demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/open_laptop_ab1_ex3_s0.log
echo "[$(date)] done  open_laptop ab1_ex3 seed=0"
echo "[$(date)] start open_laptop ab2_ex2 seed=0"
bash train.sh open_laptop demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/open_laptop_ab2_ex2_s0.log
echo "[$(date)] done  open_laptop ab2_ex2 seed=0"
echo "[$(date)] start open_laptop ab2_ex3 seed=0"
bash train.sh open_laptop demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/open_laptop_ab2_ex3_s0.log
echo "[$(date)] done  open_laptop ab2_ex3 seed=0"
echo "[$(date)] start open_laptop ab3_ex1 seed=0"
bash train.sh open_laptop demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/open_laptop_ab3_ex1_s0.log
echo "[$(date)] done  open_laptop ab3_ex1 seed=0"
echo "[$(date)] start open_laptop ab3_ex3 seed=0"
bash train.sh open_laptop demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/open_laptop_ab3_ex3_s0.log
echo "[$(date)] done  open_laptop ab3_ex3 seed=0"
echo "[$(date)] start open_laptop ab3_ex4 seed=0"
bash train.sh open_laptop demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/open_laptop_ab3_ex4_s0.log
echo "[$(date)] done  open_laptop ab3_ex4 seed=0"
echo "[$(date)] start open_laptop ab3_ex5 seed=0"
bash train.sh open_laptop demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/open_laptop_ab3_ex5_s0.log
echo "[$(date)] done  open_laptop ab3_ex5 seed=0"
echo "[$(date)] start open_laptop ab3_ex6 seed=0"
bash train.sh open_laptop demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/open_laptop_ab3_ex6_s0.log
echo "[$(date)] done  open_laptop ab3_ex6 seed=0"
echo "[$(date)] start open_laptop ab4_ex2 seed=0"
bash train.sh open_laptop demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/open_laptop_ab4_ex2_s0.log
echo "[$(date)] done  open_laptop ab4_ex2 seed=0"
echo "[$(date)] start open_laptop ab4_ex3 seed=0"
bash train.sh open_laptop demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/open_laptop_ab4_ex3_s0.log
echo "[$(date)] done  open_laptop ab4_ex3 seed=0"

# ----- [4/35] open_microwave_ab_s0_g0  [SLOW 2GPU partA]  ab1_ex1..ab3_ex1(6) -----
open_microwave_ab_s0_g0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start open_microwave ab1_ex1 seed=0"
bash train.sh open_microwave demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/open_microwave_ab1_ex1_s0.log
echo "[$(date)] done  open_microwave ab1_ex1 seed=0"
echo "[$(date)] start open_microwave ab1_ex2 seed=0"
bash train.sh open_microwave demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/open_microwave_ab1_ex2_s0.log
echo "[$(date)] done  open_microwave ab1_ex2 seed=0"
echo "[$(date)] start open_microwave ab1_ex3 seed=0"
bash train.sh open_microwave demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/open_microwave_ab1_ex3_s0.log
echo "[$(date)] done  open_microwave ab1_ex3 seed=0"
echo "[$(date)] start open_microwave ab2_ex2 seed=0"
bash train.sh open_microwave demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/open_microwave_ab2_ex2_s0.log
echo "[$(date)] done  open_microwave ab2_ex2 seed=0"
echo "[$(date)] start open_microwave ab2_ex3 seed=0"
bash train.sh open_microwave demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/open_microwave_ab2_ex3_s0.log
echo "[$(date)] done  open_microwave ab2_ex3 seed=0"
echo "[$(date)] start open_microwave ab3_ex1 seed=0"
bash train.sh open_microwave demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/open_microwave_ab3_ex1_s0.log
echo "[$(date)] done  open_microwave ab3_ex1 seed=0"

# ----- [5/35] open_microwave_ab_s0_g1  [SLOW 2GPU partB]  ab3_ex3..ab4_ex3(6) -----
open_microwave_ab_s0_g1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start open_microwave ab3_ex3 seed=0"
bash train.sh open_microwave demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/open_microwave_ab3_ex3_s0.log
echo "[$(date)] done  open_microwave ab3_ex3 seed=0"
echo "[$(date)] start open_microwave ab3_ex4 seed=0"
bash train.sh open_microwave demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/open_microwave_ab3_ex4_s0.log
echo "[$(date)] done  open_microwave ab3_ex4 seed=0"
echo "[$(date)] start open_microwave ab3_ex5 seed=0"
bash train.sh open_microwave demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/open_microwave_ab3_ex5_s0.log
echo "[$(date)] done  open_microwave ab3_ex5 seed=0"
echo "[$(date)] start open_microwave ab3_ex6 seed=0"
bash train.sh open_microwave demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/open_microwave_ab3_ex6_s0.log
echo "[$(date)] done  open_microwave ab3_ex6 seed=0"
echo "[$(date)] start open_microwave ab4_ex2 seed=0"
bash train.sh open_microwave demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/open_microwave_ab4_ex2_s0.log
echo "[$(date)] done  open_microwave ab4_ex2 seed=0"
echo "[$(date)] start open_microwave ab4_ex3 seed=0"
bash train.sh open_microwave demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/open_microwave_ab4_ex3_s0.log
echo "[$(date)] done  open_microwave ab4_ex3 seed=0"

# ----- [6/35] pick_diverse_bottles_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
pick_diverse_bottles_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start pick_diverse_bottles ab1_ex1 seed=0"
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab1_ex1_s0.log
echo "[$(date)] done  pick_diverse_bottles ab1_ex1 seed=0"
echo "[$(date)] start pick_diverse_bottles ab1_ex2 seed=0"
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab1_ex2_s0.log
echo "[$(date)] done  pick_diverse_bottles ab1_ex2 seed=0"
echo "[$(date)] start pick_diverse_bottles ab1_ex3 seed=0"
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab1_ex3_s0.log
echo "[$(date)] done  pick_diverse_bottles ab1_ex3 seed=0"
echo "[$(date)] start pick_diverse_bottles ab2_ex2 seed=0"
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab2_ex2_s0.log
echo "[$(date)] done  pick_diverse_bottles ab2_ex2 seed=0"
echo "[$(date)] start pick_diverse_bottles ab2_ex3 seed=0"
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab2_ex3_s0.log
echo "[$(date)] done  pick_diverse_bottles ab2_ex3 seed=0"
echo "[$(date)] start pick_diverse_bottles ab3_ex1 seed=0"
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab3_ex1_s0.log
echo "[$(date)] done  pick_diverse_bottles ab3_ex1 seed=0"
echo "[$(date)] start pick_diverse_bottles ab3_ex3 seed=0"
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab3_ex3_s0.log
echo "[$(date)] done  pick_diverse_bottles ab3_ex3 seed=0"
echo "[$(date)] start pick_diverse_bottles ab3_ex4 seed=0"
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab3_ex4_s0.log
echo "[$(date)] done  pick_diverse_bottles ab3_ex4 seed=0"
echo "[$(date)] start pick_diverse_bottles ab3_ex5 seed=0"
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab3_ex5_s0.log
echo "[$(date)] done  pick_diverse_bottles ab3_ex5 seed=0"
echo "[$(date)] start pick_diverse_bottles ab3_ex6 seed=0"
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab3_ex6_s0.log
echo "[$(date)] done  pick_diverse_bottles ab3_ex6 seed=0"
echo "[$(date)] start pick_diverse_bottles ab4_ex2 seed=0"
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab4_ex2_s0.log
echo "[$(date)] done  pick_diverse_bottles ab4_ex2 seed=0"
echo "[$(date)] start pick_diverse_bottles ab4_ex3 seed=0"
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab4_ex3_s0.log
echo "[$(date)] done  pick_diverse_bottles ab4_ex3 seed=0"

# ----- [7/35] pick_dual_bottles_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
pick_dual_bottles_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start pick_dual_bottles ab1_ex1 seed=0"
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab1_ex1_s0.log
echo "[$(date)] done  pick_dual_bottles ab1_ex1 seed=0"
echo "[$(date)] start pick_dual_bottles ab1_ex2 seed=0"
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab1_ex2_s0.log
echo "[$(date)] done  pick_dual_bottles ab1_ex2 seed=0"
echo "[$(date)] start pick_dual_bottles ab1_ex3 seed=0"
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab1_ex3_s0.log
echo "[$(date)] done  pick_dual_bottles ab1_ex3 seed=0"
echo "[$(date)] start pick_dual_bottles ab2_ex2 seed=0"
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab2_ex2_s0.log
echo "[$(date)] done  pick_dual_bottles ab2_ex2 seed=0"
echo "[$(date)] start pick_dual_bottles ab2_ex3 seed=0"
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab2_ex3_s0.log
echo "[$(date)] done  pick_dual_bottles ab2_ex3 seed=0"
echo "[$(date)] start pick_dual_bottles ab3_ex1 seed=0"
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab3_ex1_s0.log
echo "[$(date)] done  pick_dual_bottles ab3_ex1 seed=0"
echo "[$(date)] start pick_dual_bottles ab3_ex3 seed=0"
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab3_ex3_s0.log
echo "[$(date)] done  pick_dual_bottles ab3_ex3 seed=0"
echo "[$(date)] start pick_dual_bottles ab3_ex4 seed=0"
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab3_ex4_s0.log
echo "[$(date)] done  pick_dual_bottles ab3_ex4 seed=0"
echo "[$(date)] start pick_dual_bottles ab3_ex5 seed=0"
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab3_ex5_s0.log
echo "[$(date)] done  pick_dual_bottles ab3_ex5 seed=0"
echo "[$(date)] start pick_dual_bottles ab3_ex6 seed=0"
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab3_ex6_s0.log
echo "[$(date)] done  pick_dual_bottles ab3_ex6 seed=0"
echo "[$(date)] start pick_dual_bottles ab4_ex2 seed=0"
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab4_ex2_s0.log
echo "[$(date)] done  pick_dual_bottles ab4_ex2 seed=0"
echo "[$(date)] start pick_dual_bottles ab4_ex3 seed=0"
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab4_ex3_s0.log
echo "[$(date)] done  pick_dual_bottles ab4_ex3 seed=0"

# ----- [8/35] place_a2b_left_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
place_a2b_left_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_a2b_left ab1_ex1 seed=0"
bash train.sh place_a2b_left demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_a2b_left_ab1_ex1_s0.log
echo "[$(date)] done  place_a2b_left ab1_ex1 seed=0"
echo "[$(date)] start place_a2b_left ab1_ex2 seed=0"
bash train.sh place_a2b_left demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_a2b_left_ab1_ex2_s0.log
echo "[$(date)] done  place_a2b_left ab1_ex2 seed=0"
echo "[$(date)] start place_a2b_left ab1_ex3 seed=0"
bash train.sh place_a2b_left demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_a2b_left_ab1_ex3_s0.log
echo "[$(date)] done  place_a2b_left ab1_ex3 seed=0"
echo "[$(date)] start place_a2b_left ab2_ex2 seed=0"
bash train.sh place_a2b_left demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_a2b_left_ab2_ex2_s0.log
echo "[$(date)] done  place_a2b_left ab2_ex2 seed=0"
echo "[$(date)] start place_a2b_left ab2_ex3 seed=0"
bash train.sh place_a2b_left demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_a2b_left_ab2_ex3_s0.log
echo "[$(date)] done  place_a2b_left ab2_ex3 seed=0"
echo "[$(date)] start place_a2b_left ab3_ex1 seed=0"
bash train.sh place_a2b_left demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_a2b_left_ab3_ex1_s0.log
echo "[$(date)] done  place_a2b_left ab3_ex1 seed=0"
echo "[$(date)] start place_a2b_left ab3_ex3 seed=0"
bash train.sh place_a2b_left demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_a2b_left_ab3_ex3_s0.log
echo "[$(date)] done  place_a2b_left ab3_ex3 seed=0"
echo "[$(date)] start place_a2b_left ab3_ex4 seed=0"
bash train.sh place_a2b_left demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_a2b_left_ab3_ex4_s0.log
echo "[$(date)] done  place_a2b_left ab3_ex4 seed=0"
echo "[$(date)] start place_a2b_left ab3_ex5 seed=0"
bash train.sh place_a2b_left demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_a2b_left_ab3_ex5_s0.log
echo "[$(date)] done  place_a2b_left ab3_ex5 seed=0"
echo "[$(date)] start place_a2b_left ab3_ex6 seed=0"
bash train.sh place_a2b_left demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_a2b_left_ab3_ex6_s0.log
echo "[$(date)] done  place_a2b_left ab3_ex6 seed=0"
echo "[$(date)] start place_a2b_left ab4_ex2 seed=0"
bash train.sh place_a2b_left demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_a2b_left_ab4_ex2_s0.log
echo "[$(date)] done  place_a2b_left ab4_ex2 seed=0"
echo "[$(date)] start place_a2b_left ab4_ex3 seed=0"
bash train.sh place_a2b_left demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_a2b_left_ab4_ex3_s0.log
echo "[$(date)] done  place_a2b_left ab4_ex3 seed=0"

# ----- [9/35] place_a2b_right_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
place_a2b_right_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_a2b_right ab1_ex1 seed=0"
bash train.sh place_a2b_right demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_a2b_right_ab1_ex1_s0.log
echo "[$(date)] done  place_a2b_right ab1_ex1 seed=0"
echo "[$(date)] start place_a2b_right ab1_ex2 seed=0"
bash train.sh place_a2b_right demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_a2b_right_ab1_ex2_s0.log
echo "[$(date)] done  place_a2b_right ab1_ex2 seed=0"
echo "[$(date)] start place_a2b_right ab1_ex3 seed=0"
bash train.sh place_a2b_right demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_a2b_right_ab1_ex3_s0.log
echo "[$(date)] done  place_a2b_right ab1_ex3 seed=0"
echo "[$(date)] start place_a2b_right ab2_ex2 seed=0"
bash train.sh place_a2b_right demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_a2b_right_ab2_ex2_s0.log
echo "[$(date)] done  place_a2b_right ab2_ex2 seed=0"
echo "[$(date)] start place_a2b_right ab2_ex3 seed=0"
bash train.sh place_a2b_right demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_a2b_right_ab2_ex3_s0.log
echo "[$(date)] done  place_a2b_right ab2_ex3 seed=0"
echo "[$(date)] start place_a2b_right ab3_ex1 seed=0"
bash train.sh place_a2b_right demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_a2b_right_ab3_ex1_s0.log
echo "[$(date)] done  place_a2b_right ab3_ex1 seed=0"
echo "[$(date)] start place_a2b_right ab3_ex3 seed=0"
bash train.sh place_a2b_right demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_a2b_right_ab3_ex3_s0.log
echo "[$(date)] done  place_a2b_right ab3_ex3 seed=0"
echo "[$(date)] start place_a2b_right ab3_ex4 seed=0"
bash train.sh place_a2b_right demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_a2b_right_ab3_ex4_s0.log
echo "[$(date)] done  place_a2b_right ab3_ex4 seed=0"
echo "[$(date)] start place_a2b_right ab3_ex5 seed=0"
bash train.sh place_a2b_right demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_a2b_right_ab3_ex5_s0.log
echo "[$(date)] done  place_a2b_right ab3_ex5 seed=0"
echo "[$(date)] start place_a2b_right ab3_ex6 seed=0"
bash train.sh place_a2b_right demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_a2b_right_ab3_ex6_s0.log
echo "[$(date)] done  place_a2b_right ab3_ex6 seed=0"
echo "[$(date)] start place_a2b_right ab4_ex2 seed=0"
bash train.sh place_a2b_right demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_a2b_right_ab4_ex2_s0.log
echo "[$(date)] done  place_a2b_right ab4_ex2 seed=0"
echo "[$(date)] start place_a2b_right ab4_ex3 seed=0"
bash train.sh place_a2b_right demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_a2b_right_ab4_ex3_s0.log
echo "[$(date)] done  place_a2b_right ab4_ex3 seed=0"

# ----- [10/35] place_bread_basket_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
place_bread_basket_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_basket ab1_ex1 seed=0"
bash train.sh place_bread_basket demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_bread_basket_ab1_ex1_s0.log
echo "[$(date)] done  place_bread_basket ab1_ex1 seed=0"
echo "[$(date)] start place_bread_basket ab1_ex2 seed=0"
bash train.sh place_bread_basket demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_bread_basket_ab1_ex2_s0.log
echo "[$(date)] done  place_bread_basket ab1_ex2 seed=0"
echo "[$(date)] start place_bread_basket ab1_ex3 seed=0"
bash train.sh place_bread_basket demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_bread_basket_ab1_ex3_s0.log
echo "[$(date)] done  place_bread_basket ab1_ex3 seed=0"
echo "[$(date)] start place_bread_basket ab2_ex2 seed=0"
bash train.sh place_bread_basket demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_bread_basket_ab2_ex2_s0.log
echo "[$(date)] done  place_bread_basket ab2_ex2 seed=0"
echo "[$(date)] start place_bread_basket ab2_ex3 seed=0"
bash train.sh place_bread_basket demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_bread_basket_ab2_ex3_s0.log
echo "[$(date)] done  place_bread_basket ab2_ex3 seed=0"
echo "[$(date)] start place_bread_basket ab3_ex1 seed=0"
bash train.sh place_bread_basket demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_bread_basket_ab3_ex1_s0.log
echo "[$(date)] done  place_bread_basket ab3_ex1 seed=0"
echo "[$(date)] start place_bread_basket ab3_ex3 seed=0"
bash train.sh place_bread_basket demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_bread_basket_ab3_ex3_s0.log
echo "[$(date)] done  place_bread_basket ab3_ex3 seed=0"
echo "[$(date)] start place_bread_basket ab3_ex4 seed=0"
bash train.sh place_bread_basket demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_bread_basket_ab3_ex4_s0.log
echo "[$(date)] done  place_bread_basket ab3_ex4 seed=0"
echo "[$(date)] start place_bread_basket ab3_ex5 seed=0"
bash train.sh place_bread_basket demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_bread_basket_ab3_ex5_s0.log
echo "[$(date)] done  place_bread_basket ab3_ex5 seed=0"
echo "[$(date)] start place_bread_basket ab3_ex6 seed=0"
bash train.sh place_bread_basket demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_bread_basket_ab3_ex6_s0.log
echo "[$(date)] done  place_bread_basket ab3_ex6 seed=0"
echo "[$(date)] start place_bread_basket ab4_ex2 seed=0"
bash train.sh place_bread_basket demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_bread_basket_ab4_ex2_s0.log
echo "[$(date)] done  place_bread_basket ab4_ex2 seed=0"
echo "[$(date)] start place_bread_basket ab4_ex3 seed=0"
bash train.sh place_bread_basket demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_bread_basket_ab4_ex3_s0.log
echo "[$(date)] done  place_bread_basket ab4_ex3 seed=0"

# ----- [11/35] place_bread_skillet_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
place_bread_skillet_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_skillet ab1_ex1 seed=0"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_bread_skillet_ab1_ex1_s0.log
echo "[$(date)] done  place_bread_skillet ab1_ex1 seed=0"
echo "[$(date)] start place_bread_skillet ab1_ex2 seed=0"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_bread_skillet_ab1_ex2_s0.log
echo "[$(date)] done  place_bread_skillet ab1_ex2 seed=0"
echo "[$(date)] start place_bread_skillet ab1_ex3 seed=0"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_bread_skillet_ab1_ex3_s0.log
echo "[$(date)] done  place_bread_skillet ab1_ex3 seed=0"
echo "[$(date)] start place_bread_skillet ab2_ex2 seed=0"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_bread_skillet_ab2_ex2_s0.log
echo "[$(date)] done  place_bread_skillet ab2_ex2 seed=0"
echo "[$(date)] start place_bread_skillet ab2_ex3 seed=0"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_bread_skillet_ab2_ex3_s0.log
echo "[$(date)] done  place_bread_skillet ab2_ex3 seed=0"
echo "[$(date)] start place_bread_skillet ab3_ex1 seed=0"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_bread_skillet_ab3_ex1_s0.log
echo "[$(date)] done  place_bread_skillet ab3_ex1 seed=0"
echo "[$(date)] start place_bread_skillet ab3_ex3 seed=0"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_bread_skillet_ab3_ex3_s0.log
echo "[$(date)] done  place_bread_skillet ab3_ex3 seed=0"
echo "[$(date)] start place_bread_skillet ab3_ex4 seed=0"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_bread_skillet_ab3_ex4_s0.log
echo "[$(date)] done  place_bread_skillet ab3_ex4 seed=0"
echo "[$(date)] start place_bread_skillet ab3_ex5 seed=0"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_bread_skillet_ab3_ex5_s0.log
echo "[$(date)] done  place_bread_skillet ab3_ex5 seed=0"
echo "[$(date)] start place_bread_skillet ab3_ex6 seed=0"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_bread_skillet_ab3_ex6_s0.log
echo "[$(date)] done  place_bread_skillet ab3_ex6 seed=0"
echo "[$(date)] start place_bread_skillet ab4_ex2 seed=0"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_bread_skillet_ab4_ex2_s0.log
echo "[$(date)] done  place_bread_skillet ab4_ex2 seed=0"
echo "[$(date)] start place_bread_skillet ab4_ex3 seed=0"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_bread_skillet_ab4_ex3_s0.log
echo "[$(date)] done  place_bread_skillet ab4_ex3 seed=0"

# ----- [12/35] place_burger_fries_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
place_burger_fries_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_burger_fries ab1_ex1 seed=0"
bash train.sh place_burger_fries demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_burger_fries_ab1_ex1_s0.log
echo "[$(date)] done  place_burger_fries ab1_ex1 seed=0"
echo "[$(date)] start place_burger_fries ab1_ex2 seed=0"
bash train.sh place_burger_fries demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_burger_fries_ab1_ex2_s0.log
echo "[$(date)] done  place_burger_fries ab1_ex2 seed=0"
echo "[$(date)] start place_burger_fries ab1_ex3 seed=0"
bash train.sh place_burger_fries demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_burger_fries_ab1_ex3_s0.log
echo "[$(date)] done  place_burger_fries ab1_ex3 seed=0"
echo "[$(date)] start place_burger_fries ab2_ex2 seed=0"
bash train.sh place_burger_fries demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_burger_fries_ab2_ex2_s0.log
echo "[$(date)] done  place_burger_fries ab2_ex2 seed=0"
echo "[$(date)] start place_burger_fries ab2_ex3 seed=0"
bash train.sh place_burger_fries demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_burger_fries_ab2_ex3_s0.log
echo "[$(date)] done  place_burger_fries ab2_ex3 seed=0"
echo "[$(date)] start place_burger_fries ab3_ex1 seed=0"
bash train.sh place_burger_fries demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_burger_fries_ab3_ex1_s0.log
echo "[$(date)] done  place_burger_fries ab3_ex1 seed=0"
echo "[$(date)] start place_burger_fries ab3_ex3 seed=0"
bash train.sh place_burger_fries demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_burger_fries_ab3_ex3_s0.log
echo "[$(date)] done  place_burger_fries ab3_ex3 seed=0"
echo "[$(date)] start place_burger_fries ab3_ex4 seed=0"
bash train.sh place_burger_fries demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_burger_fries_ab3_ex4_s0.log
echo "[$(date)] done  place_burger_fries ab3_ex4 seed=0"
echo "[$(date)] start place_burger_fries ab3_ex5 seed=0"
bash train.sh place_burger_fries demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_burger_fries_ab3_ex5_s0.log
echo "[$(date)] done  place_burger_fries ab3_ex5 seed=0"
echo "[$(date)] start place_burger_fries ab3_ex6 seed=0"
bash train.sh place_burger_fries demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_burger_fries_ab3_ex6_s0.log
echo "[$(date)] done  place_burger_fries ab3_ex6 seed=0"
echo "[$(date)] start place_burger_fries ab4_ex2 seed=0"
bash train.sh place_burger_fries demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_burger_fries_ab4_ex2_s0.log
echo "[$(date)] done  place_burger_fries ab4_ex2 seed=0"
echo "[$(date)] start place_burger_fries ab4_ex3 seed=0"
bash train.sh place_burger_fries demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_burger_fries_ab4_ex3_s0.log
echo "[$(date)] done  place_burger_fries ab4_ex3 seed=0"

# ----- [13/35] place_cans_plasticbox_ab_s0  [1GPU]  ab2_ex2+ab2_ex3 -----
place_cans_plasticbox_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_cans_plasticbox ab2_ex2 seed=0"
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab2_ex2_s0.log
echo "[$(date)] done  place_cans_plasticbox ab2_ex2 seed=0"
echo "[$(date)] start place_cans_plasticbox ab2_ex3 seed=0"
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab2_ex3_s0.log
echo "[$(date)] done  place_cans_plasticbox ab2_ex3 seed=0"

# ----- [14/35] place_container_plate_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
place_container_plate_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_container_plate ab1_ex1 seed=0"
bash train.sh place_container_plate demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_container_plate_ab1_ex1_s0.log
echo "[$(date)] done  place_container_plate ab1_ex1 seed=0"
echo "[$(date)] start place_container_plate ab1_ex2 seed=0"
bash train.sh place_container_plate demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_container_plate_ab1_ex2_s0.log
echo "[$(date)] done  place_container_plate ab1_ex2 seed=0"
echo "[$(date)] start place_container_plate ab1_ex3 seed=0"
bash train.sh place_container_plate demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_container_plate_ab1_ex3_s0.log
echo "[$(date)] done  place_container_plate ab1_ex3 seed=0"
echo "[$(date)] start place_container_plate ab2_ex2 seed=0"
bash train.sh place_container_plate demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_container_plate_ab2_ex2_s0.log
echo "[$(date)] done  place_container_plate ab2_ex2 seed=0"
echo "[$(date)] start place_container_plate ab2_ex3 seed=0"
bash train.sh place_container_plate demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_container_plate_ab2_ex3_s0.log
echo "[$(date)] done  place_container_plate ab2_ex3 seed=0"
echo "[$(date)] start place_container_plate ab3_ex1 seed=0"
bash train.sh place_container_plate demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_container_plate_ab3_ex1_s0.log
echo "[$(date)] done  place_container_plate ab3_ex1 seed=0"
echo "[$(date)] start place_container_plate ab3_ex3 seed=0"
bash train.sh place_container_plate demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_container_plate_ab3_ex3_s0.log
echo "[$(date)] done  place_container_plate ab3_ex3 seed=0"
echo "[$(date)] start place_container_plate ab3_ex4 seed=0"
bash train.sh place_container_plate demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_container_plate_ab3_ex4_s0.log
echo "[$(date)] done  place_container_plate ab3_ex4 seed=0"
echo "[$(date)] start place_container_plate ab3_ex5 seed=0"
bash train.sh place_container_plate demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_container_plate_ab3_ex5_s0.log
echo "[$(date)] done  place_container_plate ab3_ex5 seed=0"
echo "[$(date)] start place_container_plate ab3_ex6 seed=0"
bash train.sh place_container_plate demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_container_plate_ab3_ex6_s0.log
echo "[$(date)] done  place_container_plate ab3_ex6 seed=0"
echo "[$(date)] start place_container_plate ab4_ex2 seed=0"
bash train.sh place_container_plate demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_container_plate_ab4_ex2_s0.log
echo "[$(date)] done  place_container_plate ab4_ex2 seed=0"
echo "[$(date)] start place_container_plate ab4_ex3 seed=0"
bash train.sh place_container_plate demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_container_plate_ab4_ex3_s0.log
echo "[$(date)] done  place_container_plate ab4_ex3 seed=0"

# ----- [15/35] place_dual_shoes_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
place_dual_shoes_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_dual_shoes ab1_ex1 seed=0"
bash train.sh place_dual_shoes demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_dual_shoes_ab1_ex1_s0.log
echo "[$(date)] done  place_dual_shoes ab1_ex1 seed=0"
echo "[$(date)] start place_dual_shoes ab1_ex2 seed=0"
bash train.sh place_dual_shoes demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_dual_shoes_ab1_ex2_s0.log
echo "[$(date)] done  place_dual_shoes ab1_ex2 seed=0"
echo "[$(date)] start place_dual_shoes ab1_ex3 seed=0"
bash train.sh place_dual_shoes demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_dual_shoes_ab1_ex3_s0.log
echo "[$(date)] done  place_dual_shoes ab1_ex3 seed=0"
echo "[$(date)] start place_dual_shoes ab2_ex2 seed=0"
bash train.sh place_dual_shoes demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_dual_shoes_ab2_ex2_s0.log
echo "[$(date)] done  place_dual_shoes ab2_ex2 seed=0"
echo "[$(date)] start place_dual_shoes ab2_ex3 seed=0"
bash train.sh place_dual_shoes demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_dual_shoes_ab2_ex3_s0.log
echo "[$(date)] done  place_dual_shoes ab2_ex3 seed=0"
echo "[$(date)] start place_dual_shoes ab3_ex1 seed=0"
bash train.sh place_dual_shoes demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_dual_shoes_ab3_ex1_s0.log
echo "[$(date)] done  place_dual_shoes ab3_ex1 seed=0"
echo "[$(date)] start place_dual_shoes ab3_ex3 seed=0"
bash train.sh place_dual_shoes demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_dual_shoes_ab3_ex3_s0.log
echo "[$(date)] done  place_dual_shoes ab3_ex3 seed=0"
echo "[$(date)] start place_dual_shoes ab3_ex4 seed=0"
bash train.sh place_dual_shoes demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_dual_shoes_ab3_ex4_s0.log
echo "[$(date)] done  place_dual_shoes ab3_ex4 seed=0"
echo "[$(date)] start place_dual_shoes ab3_ex5 seed=0"
bash train.sh place_dual_shoes demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_dual_shoes_ab3_ex5_s0.log
echo "[$(date)] done  place_dual_shoes ab3_ex5 seed=0"
echo "[$(date)] start place_dual_shoes ab3_ex6 seed=0"
bash train.sh place_dual_shoes demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_dual_shoes_ab3_ex6_s0.log
echo "[$(date)] done  place_dual_shoes ab3_ex6 seed=0"
echo "[$(date)] start place_dual_shoes ab4_ex2 seed=0"
bash train.sh place_dual_shoes demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_dual_shoes_ab4_ex2_s0.log
echo "[$(date)] done  place_dual_shoes ab4_ex2 seed=0"
echo "[$(date)] start place_dual_shoes ab4_ex3 seed=0"
bash train.sh place_dual_shoes demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_dual_shoes_ab4_ex3_s0.log
echo "[$(date)] done  place_dual_shoes ab4_ex3 seed=0"

# ----- [16/35] place_empty_cup_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
place_empty_cup_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_empty_cup ab1_ex1 seed=0"
bash train.sh place_empty_cup demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_empty_cup_ab1_ex1_s0.log
echo "[$(date)] done  place_empty_cup ab1_ex1 seed=0"
echo "[$(date)] start place_empty_cup ab1_ex2 seed=0"
bash train.sh place_empty_cup demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_empty_cup_ab1_ex2_s0.log
echo "[$(date)] done  place_empty_cup ab1_ex2 seed=0"
echo "[$(date)] start place_empty_cup ab1_ex3 seed=0"
bash train.sh place_empty_cup demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_empty_cup_ab1_ex3_s0.log
echo "[$(date)] done  place_empty_cup ab1_ex3 seed=0"
echo "[$(date)] start place_empty_cup ab2_ex2 seed=0"
bash train.sh place_empty_cup demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_empty_cup_ab2_ex2_s0.log
echo "[$(date)] done  place_empty_cup ab2_ex2 seed=0"
echo "[$(date)] start place_empty_cup ab2_ex3 seed=0"
bash train.sh place_empty_cup demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_empty_cup_ab2_ex3_s0.log
echo "[$(date)] done  place_empty_cup ab2_ex3 seed=0"
echo "[$(date)] start place_empty_cup ab3_ex1 seed=0"
bash train.sh place_empty_cup demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_empty_cup_ab3_ex1_s0.log
echo "[$(date)] done  place_empty_cup ab3_ex1 seed=0"
echo "[$(date)] start place_empty_cup ab3_ex3 seed=0"
bash train.sh place_empty_cup demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_empty_cup_ab3_ex3_s0.log
echo "[$(date)] done  place_empty_cup ab3_ex3 seed=0"
echo "[$(date)] start place_empty_cup ab3_ex4 seed=0"
bash train.sh place_empty_cup demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_empty_cup_ab3_ex4_s0.log
echo "[$(date)] done  place_empty_cup ab3_ex4 seed=0"
echo "[$(date)] start place_empty_cup ab3_ex5 seed=0"
bash train.sh place_empty_cup demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_empty_cup_ab3_ex5_s0.log
echo "[$(date)] done  place_empty_cup ab3_ex5 seed=0"
echo "[$(date)] start place_empty_cup ab3_ex6 seed=0"
bash train.sh place_empty_cup demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_empty_cup_ab3_ex6_s0.log
echo "[$(date)] done  place_empty_cup ab3_ex6 seed=0"
echo "[$(date)] start place_empty_cup ab4_ex2 seed=0"
bash train.sh place_empty_cup demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_empty_cup_ab4_ex2_s0.log
echo "[$(date)] done  place_empty_cup ab4_ex2 seed=0"
echo "[$(date)] start place_empty_cup ab4_ex3 seed=0"
bash train.sh place_empty_cup demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_empty_cup_ab4_ex3_s0.log
echo "[$(date)] done  place_empty_cup ab4_ex3 seed=0"

# ----- [17/35] place_fan_ab_s0  [1GPU]  ab2_ex2+ab2_ex3 -----
place_fan_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_fan ab2_ex2 seed=0"
bash train.sh place_fan demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_fan_ab2_ex2_s0.log
echo "[$(date)] done  place_fan ab2_ex2 seed=0"
echo "[$(date)] start place_fan ab2_ex3 seed=0"
bash train.sh place_fan demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_fan_ab2_ex3_s0.log
echo "[$(date)] done  place_fan ab2_ex3 seed=0"

# ----- [18/35] place_object_basket_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
place_object_basket_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_object_basket ab1_ex1 seed=0"
bash train.sh place_object_basket demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_object_basket_ab1_ex1_s0.log
echo "[$(date)] done  place_object_basket ab1_ex1 seed=0"
echo "[$(date)] start place_object_basket ab1_ex2 seed=0"
bash train.sh place_object_basket demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_object_basket_ab1_ex2_s0.log
echo "[$(date)] done  place_object_basket ab1_ex2 seed=0"
echo "[$(date)] start place_object_basket ab1_ex3 seed=0"
bash train.sh place_object_basket demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_object_basket_ab1_ex3_s0.log
echo "[$(date)] done  place_object_basket ab1_ex3 seed=0"
echo "[$(date)] start place_object_basket ab2_ex2 seed=0"
bash train.sh place_object_basket demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_object_basket_ab2_ex2_s0.log
echo "[$(date)] done  place_object_basket ab2_ex2 seed=0"
echo "[$(date)] start place_object_basket ab2_ex3 seed=0"
bash train.sh place_object_basket demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_object_basket_ab2_ex3_s0.log
echo "[$(date)] done  place_object_basket ab2_ex3 seed=0"
echo "[$(date)] start place_object_basket ab3_ex1 seed=0"
bash train.sh place_object_basket demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_object_basket_ab3_ex1_s0.log
echo "[$(date)] done  place_object_basket ab3_ex1 seed=0"
echo "[$(date)] start place_object_basket ab3_ex3 seed=0"
bash train.sh place_object_basket demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_object_basket_ab3_ex3_s0.log
echo "[$(date)] done  place_object_basket ab3_ex3 seed=0"
echo "[$(date)] start place_object_basket ab3_ex4 seed=0"
bash train.sh place_object_basket demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_object_basket_ab3_ex4_s0.log
echo "[$(date)] done  place_object_basket ab3_ex4 seed=0"
echo "[$(date)] start place_object_basket ab3_ex5 seed=0"
bash train.sh place_object_basket demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_object_basket_ab3_ex5_s0.log
echo "[$(date)] done  place_object_basket ab3_ex5 seed=0"
echo "[$(date)] start place_object_basket ab3_ex6 seed=0"
bash train.sh place_object_basket demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_object_basket_ab3_ex6_s0.log
echo "[$(date)] done  place_object_basket ab3_ex6 seed=0"
echo "[$(date)] start place_object_basket ab4_ex2 seed=0"
bash train.sh place_object_basket demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_object_basket_ab4_ex2_s0.log
echo "[$(date)] done  place_object_basket ab4_ex2 seed=0"
echo "[$(date)] start place_object_basket ab4_ex3 seed=0"
bash train.sh place_object_basket demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_object_basket_ab4_ex3_s0.log
echo "[$(date)] done  place_object_basket ab4_ex3 seed=0"

# ----- [19/35] place_object_stand_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
place_object_stand_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_object_stand ab1_ex1 seed=0"
bash train.sh place_object_stand demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_object_stand_ab1_ex1_s0.log
echo "[$(date)] done  place_object_stand ab1_ex1 seed=0"
echo "[$(date)] start place_object_stand ab1_ex2 seed=0"
bash train.sh place_object_stand demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_object_stand_ab1_ex2_s0.log
echo "[$(date)] done  place_object_stand ab1_ex2 seed=0"
echo "[$(date)] start place_object_stand ab1_ex3 seed=0"
bash train.sh place_object_stand demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_object_stand_ab1_ex3_s0.log
echo "[$(date)] done  place_object_stand ab1_ex3 seed=0"
echo "[$(date)] start place_object_stand ab2_ex2 seed=0"
bash train.sh place_object_stand demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_object_stand_ab2_ex2_s0.log
echo "[$(date)] done  place_object_stand ab2_ex2 seed=0"
echo "[$(date)] start place_object_stand ab2_ex3 seed=0"
bash train.sh place_object_stand demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_object_stand_ab2_ex3_s0.log
echo "[$(date)] done  place_object_stand ab2_ex3 seed=0"
echo "[$(date)] start place_object_stand ab3_ex1 seed=0"
bash train.sh place_object_stand demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_object_stand_ab3_ex1_s0.log
echo "[$(date)] done  place_object_stand ab3_ex1 seed=0"
echo "[$(date)] start place_object_stand ab3_ex3 seed=0"
bash train.sh place_object_stand demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_object_stand_ab3_ex3_s0.log
echo "[$(date)] done  place_object_stand ab3_ex3 seed=0"
echo "[$(date)] start place_object_stand ab3_ex4 seed=0"
bash train.sh place_object_stand demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_object_stand_ab3_ex4_s0.log
echo "[$(date)] done  place_object_stand ab3_ex4 seed=0"
echo "[$(date)] start place_object_stand ab3_ex5 seed=0"
bash train.sh place_object_stand demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_object_stand_ab3_ex5_s0.log
echo "[$(date)] done  place_object_stand ab3_ex5 seed=0"
echo "[$(date)] start place_object_stand ab3_ex6 seed=0"
bash train.sh place_object_stand demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_object_stand_ab3_ex6_s0.log
echo "[$(date)] done  place_object_stand ab3_ex6 seed=0"
echo "[$(date)] start place_object_stand ab4_ex2 seed=0"
bash train.sh place_object_stand demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_object_stand_ab4_ex2_s0.log
echo "[$(date)] done  place_object_stand ab4_ex2 seed=0"
echo "[$(date)] start place_object_stand ab4_ex3 seed=0"
bash train.sh place_object_stand demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_object_stand_ab4_ex3_s0.log
echo "[$(date)] done  place_object_stand ab4_ex3 seed=0"

# ----- [20/35] place_phone_stand_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
place_phone_stand_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_phone_stand ab1_ex1 seed=0"
bash train.sh place_phone_stand demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_phone_stand_ab1_ex1_s0.log
echo "[$(date)] done  place_phone_stand ab1_ex1 seed=0"
echo "[$(date)] start place_phone_stand ab1_ex2 seed=0"
bash train.sh place_phone_stand demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_phone_stand_ab1_ex2_s0.log
echo "[$(date)] done  place_phone_stand ab1_ex2 seed=0"
echo "[$(date)] start place_phone_stand ab1_ex3 seed=0"
bash train.sh place_phone_stand demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_phone_stand_ab1_ex3_s0.log
echo "[$(date)] done  place_phone_stand ab1_ex3 seed=0"
echo "[$(date)] start place_phone_stand ab2_ex2 seed=0"
bash train.sh place_phone_stand demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_phone_stand_ab2_ex2_s0.log
echo "[$(date)] done  place_phone_stand ab2_ex2 seed=0"
echo "[$(date)] start place_phone_stand ab2_ex3 seed=0"
bash train.sh place_phone_stand demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_phone_stand_ab2_ex3_s0.log
echo "[$(date)] done  place_phone_stand ab2_ex3 seed=0"
echo "[$(date)] start place_phone_stand ab3_ex1 seed=0"
bash train.sh place_phone_stand demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_phone_stand_ab3_ex1_s0.log
echo "[$(date)] done  place_phone_stand ab3_ex1 seed=0"
echo "[$(date)] start place_phone_stand ab3_ex3 seed=0"
bash train.sh place_phone_stand demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_phone_stand_ab3_ex3_s0.log
echo "[$(date)] done  place_phone_stand ab3_ex3 seed=0"
echo "[$(date)] start place_phone_stand ab3_ex4 seed=0"
bash train.sh place_phone_stand demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_phone_stand_ab3_ex4_s0.log
echo "[$(date)] done  place_phone_stand ab3_ex4 seed=0"
echo "[$(date)] start place_phone_stand ab3_ex5 seed=0"
bash train.sh place_phone_stand demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_phone_stand_ab3_ex5_s0.log
echo "[$(date)] done  place_phone_stand ab3_ex5 seed=0"
echo "[$(date)] start place_phone_stand ab3_ex6 seed=0"
bash train.sh place_phone_stand demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_phone_stand_ab3_ex6_s0.log
echo "[$(date)] done  place_phone_stand ab3_ex6 seed=0"
echo "[$(date)] start place_phone_stand ab4_ex2 seed=0"
bash train.sh place_phone_stand demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_phone_stand_ab4_ex2_s0.log
echo "[$(date)] done  place_phone_stand ab4_ex2 seed=0"
echo "[$(date)] start place_phone_stand ab4_ex3 seed=0"
bash train.sh place_phone_stand demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_phone_stand_ab4_ex3_s0.log
echo "[$(date)] done  place_phone_stand ab4_ex3 seed=0"

# ----- [21/35] place_shoe_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
place_shoe_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_shoe ab1_ex1 seed=0"
bash train.sh place_shoe demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_shoe_ab1_ex1_s0.log
echo "[$(date)] done  place_shoe ab1_ex1 seed=0"
echo "[$(date)] start place_shoe ab1_ex2 seed=0"
bash train.sh place_shoe demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_shoe_ab1_ex2_s0.log
echo "[$(date)] done  place_shoe ab1_ex2 seed=0"
echo "[$(date)] start place_shoe ab1_ex3 seed=0"
bash train.sh place_shoe demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_shoe_ab1_ex3_s0.log
echo "[$(date)] done  place_shoe ab1_ex3 seed=0"
echo "[$(date)] start place_shoe ab2_ex2 seed=0"
bash train.sh place_shoe demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_shoe_ab2_ex2_s0.log
echo "[$(date)] done  place_shoe ab2_ex2 seed=0"
echo "[$(date)] start place_shoe ab2_ex3 seed=0"
bash train.sh place_shoe demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_shoe_ab2_ex3_s0.log
echo "[$(date)] done  place_shoe ab2_ex3 seed=0"
echo "[$(date)] start place_shoe ab3_ex1 seed=0"
bash train.sh place_shoe demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_shoe_ab3_ex1_s0.log
echo "[$(date)] done  place_shoe ab3_ex1 seed=0"
echo "[$(date)] start place_shoe ab3_ex3 seed=0"
bash train.sh place_shoe demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_shoe_ab3_ex3_s0.log
echo "[$(date)] done  place_shoe ab3_ex3 seed=0"
echo "[$(date)] start place_shoe ab3_ex4 seed=0"
bash train.sh place_shoe demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_shoe_ab3_ex4_s0.log
echo "[$(date)] done  place_shoe ab3_ex4 seed=0"
echo "[$(date)] start place_shoe ab3_ex5 seed=0"
bash train.sh place_shoe demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_shoe_ab3_ex5_s0.log
echo "[$(date)] done  place_shoe ab3_ex5 seed=0"
echo "[$(date)] start place_shoe ab3_ex6 seed=0"
bash train.sh place_shoe demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_shoe_ab3_ex6_s0.log
echo "[$(date)] done  place_shoe ab3_ex6 seed=0"
echo "[$(date)] start place_shoe ab4_ex2 seed=0"
bash train.sh place_shoe demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_shoe_ab4_ex2_s0.log
echo "[$(date)] done  place_shoe ab4_ex2 seed=0"
echo "[$(date)] start place_shoe ab4_ex3 seed=0"
bash train.sh place_shoe demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_shoe_ab4_ex3_s0.log
echo "[$(date)] done  place_shoe ab4_ex3 seed=0"

# ----- [22/35] press_stapler_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
press_stapler_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start press_stapler ab1_ex1 seed=0"
bash train.sh press_stapler demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/press_stapler_ab1_ex1_s0.log
echo "[$(date)] done  press_stapler ab1_ex1 seed=0"
echo "[$(date)] start press_stapler ab1_ex2 seed=0"
bash train.sh press_stapler demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/press_stapler_ab1_ex2_s0.log
echo "[$(date)] done  press_stapler ab1_ex2 seed=0"
echo "[$(date)] start press_stapler ab1_ex3 seed=0"
bash train.sh press_stapler demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/press_stapler_ab1_ex3_s0.log
echo "[$(date)] done  press_stapler ab1_ex3 seed=0"
echo "[$(date)] start press_stapler ab2_ex2 seed=0"
bash train.sh press_stapler demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/press_stapler_ab2_ex2_s0.log
echo "[$(date)] done  press_stapler ab2_ex2 seed=0"
echo "[$(date)] start press_stapler ab2_ex3 seed=0"
bash train.sh press_stapler demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/press_stapler_ab2_ex3_s0.log
echo "[$(date)] done  press_stapler ab2_ex3 seed=0"
echo "[$(date)] start press_stapler ab3_ex1 seed=0"
bash train.sh press_stapler demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/press_stapler_ab3_ex1_s0.log
echo "[$(date)] done  press_stapler ab3_ex1 seed=0"
echo "[$(date)] start press_stapler ab3_ex3 seed=0"
bash train.sh press_stapler demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/press_stapler_ab3_ex3_s0.log
echo "[$(date)] done  press_stapler ab3_ex3 seed=0"
echo "[$(date)] start press_stapler ab3_ex4 seed=0"
bash train.sh press_stapler demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/press_stapler_ab3_ex4_s0.log
echo "[$(date)] done  press_stapler ab3_ex4 seed=0"
echo "[$(date)] start press_stapler ab3_ex5 seed=0"
bash train.sh press_stapler demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/press_stapler_ab3_ex5_s0.log
echo "[$(date)] done  press_stapler ab3_ex5 seed=0"
echo "[$(date)] start press_stapler ab3_ex6 seed=0"
bash train.sh press_stapler demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/press_stapler_ab3_ex6_s0.log
echo "[$(date)] done  press_stapler ab3_ex6 seed=0"
echo "[$(date)] start press_stapler ab4_ex2 seed=0"
bash train.sh press_stapler demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/press_stapler_ab4_ex2_s0.log
echo "[$(date)] done  press_stapler ab4_ex2 seed=0"
echo "[$(date)] start press_stapler ab4_ex3 seed=0"
bash train.sh press_stapler demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/press_stapler_ab4_ex3_s0.log
echo "[$(date)] done  press_stapler ab4_ex3 seed=0"

# ----- [23/35] put_bottles_dustbin_ab_s0_g0  [SLOW 2GPU partA]  ab1_ex1..ab3_ex1(6) -----
put_bottles_dustbin_ab_s0_g0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start put_bottles_dustbin ab1_ex1 seed=0"
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab1_ex1_s0.log
echo "[$(date)] done  put_bottles_dustbin ab1_ex1 seed=0"
echo "[$(date)] start put_bottles_dustbin ab1_ex2 seed=0"
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab1_ex2_s0.log
echo "[$(date)] done  put_bottles_dustbin ab1_ex2 seed=0"
echo "[$(date)] start put_bottles_dustbin ab1_ex3 seed=0"
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab1_ex3_s0.log
echo "[$(date)] done  put_bottles_dustbin ab1_ex3 seed=0"
echo "[$(date)] start put_bottles_dustbin ab2_ex2 seed=0"
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab2_ex2_s0.log
echo "[$(date)] done  put_bottles_dustbin ab2_ex2 seed=0"
echo "[$(date)] start put_bottles_dustbin ab2_ex3 seed=0"
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab2_ex3_s0.log
echo "[$(date)] done  put_bottles_dustbin ab2_ex3 seed=0"
echo "[$(date)] start put_bottles_dustbin ab3_ex1 seed=0"
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab3_ex1_s0.log
echo "[$(date)] done  put_bottles_dustbin ab3_ex1 seed=0"

# ----- [24/35] put_bottles_dustbin_ab_s0_g1  [SLOW 2GPU partB]  ab3_ex3..ab4_ex3(6) -----
put_bottles_dustbin_ab_s0_g1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start put_bottles_dustbin ab3_ex3 seed=0"
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab3_ex3_s0.log
echo "[$(date)] done  put_bottles_dustbin ab3_ex3 seed=0"
echo "[$(date)] start put_bottles_dustbin ab3_ex4 seed=0"
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab3_ex4_s0.log
echo "[$(date)] done  put_bottles_dustbin ab3_ex4 seed=0"
echo "[$(date)] start put_bottles_dustbin ab3_ex5 seed=0"
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab3_ex5_s0.log
echo "[$(date)] done  put_bottles_dustbin ab3_ex5 seed=0"
echo "[$(date)] start put_bottles_dustbin ab3_ex6 seed=0"
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab3_ex6_s0.log
echo "[$(date)] done  put_bottles_dustbin ab3_ex6 seed=0"
echo "[$(date)] start put_bottles_dustbin ab4_ex2 seed=0"
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab4_ex2_s0.log
echo "[$(date)] done  put_bottles_dustbin ab4_ex2 seed=0"
echo "[$(date)] start put_bottles_dustbin ab4_ex3 seed=0"
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab4_ex3_s0.log
echo "[$(date)] done  put_bottles_dustbin ab4_ex3 seed=0"

# ----- [25/35] rotate_qrcode_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(11) -----
rotate_qrcode_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start rotate_qrcode ab1_ex1 seed=0"
bash train.sh rotate_qrcode demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/rotate_qrcode_ab1_ex1_s0.log
echo "[$(date)] done  rotate_qrcode ab1_ex1 seed=0"
echo "[$(date)] start rotate_qrcode ab1_ex3 seed=0"
bash train.sh rotate_qrcode demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/rotate_qrcode_ab1_ex3_s0.log
echo "[$(date)] done  rotate_qrcode ab1_ex3 seed=0"
echo "[$(date)] start rotate_qrcode ab2_ex2 seed=0"
bash train.sh rotate_qrcode demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/rotate_qrcode_ab2_ex2_s0.log
echo "[$(date)] done  rotate_qrcode ab2_ex2 seed=0"
echo "[$(date)] start rotate_qrcode ab2_ex3 seed=0"
bash train.sh rotate_qrcode demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/rotate_qrcode_ab2_ex3_s0.log
echo "[$(date)] done  rotate_qrcode ab2_ex3 seed=0"
echo "[$(date)] start rotate_qrcode ab3_ex1 seed=0"
bash train.sh rotate_qrcode demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/rotate_qrcode_ab3_ex1_s0.log
echo "[$(date)] done  rotate_qrcode ab3_ex1 seed=0"
echo "[$(date)] start rotate_qrcode ab3_ex3 seed=0"
bash train.sh rotate_qrcode demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/rotate_qrcode_ab3_ex3_s0.log
echo "[$(date)] done  rotate_qrcode ab3_ex3 seed=0"
echo "[$(date)] start rotate_qrcode ab3_ex4 seed=0"
bash train.sh rotate_qrcode demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/rotate_qrcode_ab3_ex4_s0.log
echo "[$(date)] done  rotate_qrcode ab3_ex4 seed=0"
echo "[$(date)] start rotate_qrcode ab3_ex5 seed=0"
bash train.sh rotate_qrcode demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/rotate_qrcode_ab3_ex5_s0.log
echo "[$(date)] done  rotate_qrcode ab3_ex5 seed=0"
echo "[$(date)] start rotate_qrcode ab3_ex6 seed=0"
bash train.sh rotate_qrcode demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/rotate_qrcode_ab3_ex6_s0.log
echo "[$(date)] done  rotate_qrcode ab3_ex6 seed=0"
echo "[$(date)] start rotate_qrcode ab4_ex2 seed=0"
bash train.sh rotate_qrcode demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/rotate_qrcode_ab4_ex2_s0.log
echo "[$(date)] done  rotate_qrcode ab4_ex2 seed=0"
echo "[$(date)] start rotate_qrcode ab4_ex3 seed=0"
bash train.sh rotate_qrcode demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/rotate_qrcode_ab4_ex3_s0.log
echo "[$(date)] done  rotate_qrcode ab4_ex3 seed=0"

# ----- [26/35] scan_object_ab_s0  [1GPU]  ab2_ex2+ab2_ex3 -----
scan_object_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start scan_object ab2_ex2 seed=0"
bash train.sh scan_object demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/scan_object_ab2_ex2_s0.log
echo "[$(date)] done  scan_object ab2_ex2 seed=0"
echo "[$(date)] start scan_object ab2_ex3 seed=0"
bash train.sh scan_object demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/scan_object_ab2_ex3_s0.log
echo "[$(date)] done  scan_object ab2_ex3 seed=0"

# ----- [27/35] shake_bottle_ab_s0  [1GPU]  ab1_ex1..ab4_ex3(12) -----
shake_bottle_ab_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start shake_bottle ab1_ex1 seed=0"
bash train.sh shake_bottle demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/shake_bottle_ab1_ex1_s0.log
echo "[$(date)] done  shake_bottle ab1_ex1 seed=0"
echo "[$(date)] start shake_bottle ab1_ex2 seed=0"
bash train.sh shake_bottle demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/shake_bottle_ab1_ex2_s0.log
echo "[$(date)] done  shake_bottle ab1_ex2 seed=0"
echo "[$(date)] start shake_bottle ab1_ex3 seed=0"
bash train.sh shake_bottle demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/shake_bottle_ab1_ex3_s0.log
echo "[$(date)] done  shake_bottle ab1_ex3 seed=0"
echo "[$(date)] start shake_bottle ab2_ex2 seed=0"
bash train.sh shake_bottle demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/shake_bottle_ab2_ex2_s0.log
echo "[$(date)] done  shake_bottle ab2_ex2 seed=0"
echo "[$(date)] start shake_bottle ab2_ex3 seed=0"
bash train.sh shake_bottle demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/shake_bottle_ab2_ex3_s0.log
echo "[$(date)] done  shake_bottle ab2_ex3 seed=0"
echo "[$(date)] start shake_bottle ab3_ex1 seed=0"
bash train.sh shake_bottle demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/shake_bottle_ab3_ex1_s0.log
echo "[$(date)] done  shake_bottle ab3_ex1 seed=0"
echo "[$(date)] start shake_bottle ab3_ex3 seed=0"
bash train.sh shake_bottle demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/shake_bottle_ab3_ex3_s0.log
echo "[$(date)] done  shake_bottle ab3_ex3 seed=0"
echo "[$(date)] start shake_bottle ab3_ex4 seed=0"
bash train.sh shake_bottle demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/shake_bottle_ab3_ex4_s0.log
echo "[$(date)] done  shake_bottle ab3_ex4 seed=0"
echo "[$(date)] start shake_bottle ab3_ex5 seed=0"
bash train.sh shake_bottle demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/shake_bottle_ab3_ex5_s0.log
echo "[$(date)] done  shake_bottle ab3_ex5 seed=0"
echo "[$(date)] start shake_bottle ab3_ex6 seed=0"
bash train.sh shake_bottle demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/shake_bottle_ab3_ex6_s0.log
echo "[$(date)] done  shake_bottle ab3_ex6 seed=0"
echo "[$(date)] start shake_bottle ab4_ex2 seed=0"
bash train.sh shake_bottle demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/shake_bottle_ab4_ex2_s0.log
echo "[$(date)] done  shake_bottle ab4_ex2 seed=0"
echo "[$(date)] start shake_bottle ab4_ex3 seed=0"
bash train.sh shake_bottle demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/shake_bottle_ab4_ex3_s0.log
echo "[$(date)] done  shake_bottle ab4_ex3 seed=0"

# ----- [28/35] stack_blocks_three_ab_s0_g0  [SLOW 2GPU partA]  ab1_ex1..ab3_ex1(6) -----
stack_blocks_three_ab_s0_g0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start stack_blocks_three ab1_ex1 seed=0"
bash train.sh stack_blocks_three demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/stack_blocks_three_ab1_ex1_s0.log
echo "[$(date)] done  stack_blocks_three ab1_ex1 seed=0"
echo "[$(date)] start stack_blocks_three ab1_ex2 seed=0"
bash train.sh stack_blocks_three demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/stack_blocks_three_ab1_ex2_s0.log
echo "[$(date)] done  stack_blocks_three ab1_ex2 seed=0"
echo "[$(date)] start stack_blocks_three ab1_ex3 seed=0"
bash train.sh stack_blocks_three demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/stack_blocks_three_ab1_ex3_s0.log
echo "[$(date)] done  stack_blocks_three ab1_ex3 seed=0"
echo "[$(date)] start stack_blocks_three ab2_ex2 seed=0"
bash train.sh stack_blocks_three demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/stack_blocks_three_ab2_ex2_s0.log
echo "[$(date)] done  stack_blocks_three ab2_ex2 seed=0"
echo "[$(date)] start stack_blocks_three ab2_ex3 seed=0"
bash train.sh stack_blocks_three demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/stack_blocks_three_ab2_ex3_s0.log
echo "[$(date)] done  stack_blocks_three ab2_ex3 seed=0"
echo "[$(date)] start stack_blocks_three ab3_ex1 seed=0"
bash train.sh stack_blocks_three demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/stack_blocks_three_ab3_ex1_s0.log
echo "[$(date)] done  stack_blocks_three ab3_ex1 seed=0"

# ----- [29/35] stack_blocks_three_ab_s0_g1  [SLOW 2GPU partB]  ab3_ex3..ab4_ex3(6) -----
stack_blocks_three_ab_s0_g1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start stack_blocks_three ab3_ex3 seed=0"
bash train.sh stack_blocks_three demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/stack_blocks_three_ab3_ex3_s0.log
echo "[$(date)] done  stack_blocks_three ab3_ex3 seed=0"
echo "[$(date)] start stack_blocks_three ab3_ex4 seed=0"
bash train.sh stack_blocks_three demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/stack_blocks_three_ab3_ex4_s0.log
echo "[$(date)] done  stack_blocks_three ab3_ex4 seed=0"
echo "[$(date)] start stack_blocks_three ab3_ex5 seed=0"
bash train.sh stack_blocks_three demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/stack_blocks_three_ab3_ex5_s0.log
echo "[$(date)] done  stack_blocks_three ab3_ex5 seed=0"
echo "[$(date)] start stack_blocks_three ab3_ex6 seed=0"
bash train.sh stack_blocks_three demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/stack_blocks_three_ab3_ex6_s0.log
echo "[$(date)] done  stack_blocks_three ab3_ex6 seed=0"
echo "[$(date)] start stack_blocks_three ab4_ex2 seed=0"
bash train.sh stack_blocks_three demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/stack_blocks_three_ab4_ex2_s0.log
echo "[$(date)] done  stack_blocks_three ab4_ex2 seed=0"
echo "[$(date)] start stack_blocks_three ab4_ex3 seed=0"
bash train.sh stack_blocks_three demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/stack_blocks_three_ab4_ex3_s0.log
echo "[$(date)] done  stack_blocks_three ab4_ex3 seed=0"

# ----- [30/35] stack_blocks_two_ab_s0_g0  [SLOW 2GPU partA]  ab1_ex1..ab3_ex1(6) -----
stack_blocks_two_ab_s0_g0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start stack_blocks_two ab1_ex1 seed=0"
bash train.sh stack_blocks_two demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/stack_blocks_two_ab1_ex1_s0.log
echo "[$(date)] done  stack_blocks_two ab1_ex1 seed=0"
echo "[$(date)] start stack_blocks_two ab1_ex2 seed=0"
bash train.sh stack_blocks_two demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/stack_blocks_two_ab1_ex2_s0.log
echo "[$(date)] done  stack_blocks_two ab1_ex2 seed=0"
echo "[$(date)] start stack_blocks_two ab1_ex3 seed=0"
bash train.sh stack_blocks_two demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/stack_blocks_two_ab1_ex3_s0.log
echo "[$(date)] done  stack_blocks_two ab1_ex3 seed=0"
echo "[$(date)] start stack_blocks_two ab2_ex2 seed=0"
bash train.sh stack_blocks_two demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/stack_blocks_two_ab2_ex2_s0.log
echo "[$(date)] done  stack_blocks_two ab2_ex2 seed=0"
echo "[$(date)] start stack_blocks_two ab2_ex3 seed=0"
bash train.sh stack_blocks_two demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/stack_blocks_two_ab2_ex3_s0.log
echo "[$(date)] done  stack_blocks_two ab2_ex3 seed=0"
echo "[$(date)] start stack_blocks_two ab3_ex1 seed=0"
bash train.sh stack_blocks_two demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/stack_blocks_two_ab3_ex1_s0.log
echo "[$(date)] done  stack_blocks_two ab3_ex1 seed=0"

# ----- [31/35] stack_blocks_two_ab_s0_g1  [SLOW 2GPU partB]  ab3_ex3..ab4_ex3(6) -----
stack_blocks_two_ab_s0_g1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start stack_blocks_two ab3_ex3 seed=0"
bash train.sh stack_blocks_two demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/stack_blocks_two_ab3_ex3_s0.log
echo "[$(date)] done  stack_blocks_two ab3_ex3 seed=0"
echo "[$(date)] start stack_blocks_two ab3_ex4 seed=0"
bash train.sh stack_blocks_two demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/stack_blocks_two_ab3_ex4_s0.log
echo "[$(date)] done  stack_blocks_two ab3_ex4 seed=0"
echo "[$(date)] start stack_blocks_two ab3_ex5 seed=0"
bash train.sh stack_blocks_two demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/stack_blocks_two_ab3_ex5_s0.log
echo "[$(date)] done  stack_blocks_two ab3_ex5 seed=0"
echo "[$(date)] start stack_blocks_two ab3_ex6 seed=0"
bash train.sh stack_blocks_two demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/stack_blocks_two_ab3_ex6_s0.log
echo "[$(date)] done  stack_blocks_two ab3_ex6 seed=0"
echo "[$(date)] start stack_blocks_two ab4_ex2 seed=0"
bash train.sh stack_blocks_two demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/stack_blocks_two_ab4_ex2_s0.log
echo "[$(date)] done  stack_blocks_two ab4_ex2 seed=0"
echo "[$(date)] start stack_blocks_two ab4_ex3 seed=0"
bash train.sh stack_blocks_two demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/stack_blocks_two_ab4_ex3_s0.log
echo "[$(date)] done  stack_blocks_two ab4_ex3 seed=0"

# ----- [32/35] stack_bowls_three_ab_s0_g0  [SLOW 2GPU partA]  ab1_ex1..ab3_ex1(6) -----
stack_bowls_three_ab_s0_g0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start stack_bowls_three ab1_ex1 seed=0"
bash train.sh stack_bowls_three demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/stack_bowls_three_ab1_ex1_s0.log
echo "[$(date)] done  stack_bowls_three ab1_ex1 seed=0"
echo "[$(date)] start stack_bowls_three ab1_ex2 seed=0"
bash train.sh stack_bowls_three demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/stack_bowls_three_ab1_ex2_s0.log
echo "[$(date)] done  stack_bowls_three ab1_ex2 seed=0"
echo "[$(date)] start stack_bowls_three ab1_ex3 seed=0"
bash train.sh stack_bowls_three demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/stack_bowls_three_ab1_ex3_s0.log
echo "[$(date)] done  stack_bowls_three ab1_ex3 seed=0"
echo "[$(date)] start stack_bowls_three ab2_ex2 seed=0"
bash train.sh stack_bowls_three demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/stack_bowls_three_ab2_ex2_s0.log
echo "[$(date)] done  stack_bowls_three ab2_ex2 seed=0"
echo "[$(date)] start stack_bowls_three ab2_ex3 seed=0"
bash train.sh stack_bowls_three demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/stack_bowls_three_ab2_ex3_s0.log
echo "[$(date)] done  stack_bowls_three ab2_ex3 seed=0"
echo "[$(date)] start stack_bowls_three ab3_ex1 seed=0"
bash train.sh stack_bowls_three demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/stack_bowls_three_ab3_ex1_s0.log
echo "[$(date)] done  stack_bowls_three ab3_ex1 seed=0"

# ----- [33/35] stack_bowls_three_ab_s0_g1  [SLOW 2GPU partB]  ab3_ex3..ab4_ex3(6) -----
stack_bowls_three_ab_s0_g1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start stack_bowls_three ab3_ex3 seed=0"
bash train.sh stack_bowls_three demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/stack_bowls_three_ab3_ex3_s0.log
echo "[$(date)] done  stack_bowls_three ab3_ex3 seed=0"
echo "[$(date)] start stack_bowls_three ab3_ex4 seed=0"
bash train.sh stack_bowls_three demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/stack_bowls_three_ab3_ex4_s0.log
echo "[$(date)] done  stack_bowls_three ab3_ex4 seed=0"
echo "[$(date)] start stack_bowls_three ab3_ex5 seed=0"
bash train.sh stack_bowls_three demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/stack_bowls_three_ab3_ex5_s0.log
echo "[$(date)] done  stack_bowls_three ab3_ex5 seed=0"
echo "[$(date)] start stack_bowls_three ab3_ex6 seed=0"
bash train.sh stack_bowls_three demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/stack_bowls_three_ab3_ex6_s0.log
echo "[$(date)] done  stack_bowls_three ab3_ex6 seed=0"
echo "[$(date)] start stack_bowls_three ab4_ex2 seed=0"
bash train.sh stack_bowls_three demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/stack_bowls_three_ab4_ex2_s0.log
echo "[$(date)] done  stack_bowls_three ab4_ex2 seed=0"
echo "[$(date)] start stack_bowls_three ab4_ex3 seed=0"
bash train.sh stack_bowls_three demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/stack_bowls_three_ab4_ex3_s0.log
echo "[$(date)] done  stack_bowls_three ab4_ex3 seed=0"

# ----- [34/35] stack_bowls_two_ab_s0_g0  [SLOW 2GPU partA]  ab1_ex1..ab3_ex1(6) -----
stack_bowls_two_ab_s0_g0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start stack_bowls_two ab1_ex1 seed=0"
bash train.sh stack_bowls_two demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/stack_bowls_two_ab1_ex1_s0.log
echo "[$(date)] done  stack_bowls_two ab1_ex1 seed=0"
echo "[$(date)] start stack_bowls_two ab1_ex2 seed=0"
bash train.sh stack_bowls_two demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/stack_bowls_two_ab1_ex2_s0.log
echo "[$(date)] done  stack_bowls_two ab1_ex2 seed=0"
echo "[$(date)] start stack_bowls_two ab1_ex3 seed=0"
bash train.sh stack_bowls_two demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/stack_bowls_two_ab1_ex3_s0.log
echo "[$(date)] done  stack_bowls_two ab1_ex3 seed=0"
echo "[$(date)] start stack_bowls_two ab2_ex2 seed=0"
bash train.sh stack_bowls_two demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/stack_bowls_two_ab2_ex2_s0.log
echo "[$(date)] done  stack_bowls_two ab2_ex2 seed=0"
echo "[$(date)] start stack_bowls_two ab2_ex3 seed=0"
bash train.sh stack_bowls_two demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/stack_bowls_two_ab2_ex3_s0.log
echo "[$(date)] done  stack_bowls_two ab2_ex3 seed=0"
echo "[$(date)] start stack_bowls_two ab3_ex1 seed=0"
bash train.sh stack_bowls_two demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/stack_bowls_two_ab3_ex1_s0.log
echo "[$(date)] done  stack_bowls_two ab3_ex1 seed=0"

# ----- [35/35] stack_bowls_two_ab_s0_g1  [SLOW 2GPU partB]  ab3_ex3..ab4_ex3(6) -----
stack_bowls_two_ab_s0_g1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start stack_bowls_two ab3_ex3 seed=0"
bash train.sh stack_bowls_two demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/stack_bowls_two_ab3_ex3_s0.log
echo "[$(date)] done  stack_bowls_two ab3_ex3 seed=0"
echo "[$(date)] start stack_bowls_two ab3_ex4 seed=0"
bash train.sh stack_bowls_two demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/stack_bowls_two_ab3_ex4_s0.log
echo "[$(date)] done  stack_bowls_two ab3_ex4 seed=0"
echo "[$(date)] start stack_bowls_two ab3_ex5 seed=0"
bash train.sh stack_bowls_two demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/stack_bowls_two_ab3_ex5_s0.log
echo "[$(date)] done  stack_bowls_two ab3_ex5 seed=0"
echo "[$(date)] start stack_bowls_two ab3_ex6 seed=0"
bash train.sh stack_bowls_two demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/stack_bowls_two_ab3_ex6_s0.log
echo "[$(date)] done  stack_bowls_two ab3_ex6 seed=0"
echo "[$(date)] start stack_bowls_two ab4_ex2 seed=0"
bash train.sh stack_bowls_two demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/stack_bowls_two_ab4_ex2_s0.log
echo "[$(date)] done  stack_bowls_two ab4_ex2 seed=0"
echo "[$(date)] start stack_bowls_two ab4_ex3 seed=0"
bash train.sh stack_bowls_two demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/stack_bowls_two_ab4_ex3_s0.log
echo "[$(date)] done  stack_bowls_two ab4_ex3 seed=0"

