# Main encoder × 31 tasks (no turn_switch)
# Each job: seed2 → seed1; if seed0 missing, then → seed0
# Seeds set: 0, 1, 2 (aligned with common RoboTwin/GAP convention)
# seed0 already done (skip): beat_block_hammer, place_cans_plasticbox, place_fan, place_object_basket, scan_object
# Jobs: 31; trainings: 5×(s2+s1) + 26×(s2+s1+s0) = 88
# Skip: turn_switch; Skip all ablations (ab*)
# Each block = one platform job: copy job-name into name field, paste script body.
# ckpt: policy/SPACE/checkpoints/<task>-train_main-100_<seed>/

# ----- [0/30] beat_block_hammer_main_s21 -----
beat_block_hammer_main_s21
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start beat_block_hammer seed=2"
bash train.sh beat_block_hammer demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/beat_block_hammer_main_s2.log
echo "[$(date)] done  beat_block_hammer seed=2"
echo "[$(date)] start beat_block_hammer seed=1"
bash train.sh beat_block_hammer demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/beat_block_hammer_main_s1.log
echo "[$(date)] done  beat_block_hammer seed=1"

# ----- [1/30] handover_block_main_s210 -----
handover_block_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start handover_block seed=2"
bash train.sh handover_block demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/handover_block_main_s2.log
echo "[$(date)] done  handover_block seed=2"
echo "[$(date)] start handover_block seed=1"
bash train.sh handover_block demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/handover_block_main_s1.log
echo "[$(date)] done  handover_block seed=1"
echo "[$(date)] start handover_block seed=0"
bash train.sh handover_block demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/handover_block_main_s0.log
echo "[$(date)] done  handover_block seed=0"

# ----- [2/30] hanging_mug_main_s210 -----
hanging_mug_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start hanging_mug seed=2"
bash train.sh hanging_mug demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/hanging_mug_main_s2.log
echo "[$(date)] done  hanging_mug seed=2"
echo "[$(date)] start hanging_mug seed=1"
bash train.sh hanging_mug demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/hanging_mug_main_s1.log
echo "[$(date)] done  hanging_mug seed=1"
echo "[$(date)] start hanging_mug seed=0"
bash train.sh hanging_mug demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/hanging_mug_main_s0.log
echo "[$(date)] done  hanging_mug seed=0"

# ----- [3/30] move_can_pot_main_s210 -----
move_can_pot_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start move_can_pot seed=2"
bash train.sh move_can_pot demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/move_can_pot_main_s2.log
echo "[$(date)] done  move_can_pot seed=2"
echo "[$(date)] start move_can_pot seed=1"
bash train.sh move_can_pot demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/move_can_pot_main_s1.log
echo "[$(date)] done  move_can_pot seed=1"
echo "[$(date)] start move_can_pot seed=0"
bash train.sh move_can_pot demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/move_can_pot_main_s0.log
echo "[$(date)] done  move_can_pot seed=0"

# ----- [4/30] open_laptop_main_s210 -----
open_laptop_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start open_laptop seed=2"
bash train.sh open_laptop demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/open_laptop_main_s2.log
echo "[$(date)] done  open_laptop seed=2"
echo "[$(date)] start open_laptop seed=1"
bash train.sh open_laptop demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/open_laptop_main_s1.log
echo "[$(date)] done  open_laptop seed=1"
echo "[$(date)] start open_laptop seed=0"
bash train.sh open_laptop demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/open_laptop_main_s0.log
echo "[$(date)] done  open_laptop seed=0"

# ----- [5/30] open_microwave_main_s210 -----
open_microwave_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start open_microwave seed=2"
bash train.sh open_microwave demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/open_microwave_main_s2.log
echo "[$(date)] done  open_microwave seed=2"
echo "[$(date)] start open_microwave seed=1"
bash train.sh open_microwave demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/open_microwave_main_s1.log
echo "[$(date)] done  open_microwave seed=1"
echo "[$(date)] start open_microwave seed=0"
bash train.sh open_microwave demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/open_microwave_main_s0.log
echo "[$(date)] done  open_microwave seed=0"

# ----- [6/30] pick_diverse_bottles_main_s210 -----
pick_diverse_bottles_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start pick_diverse_bottles seed=2"
bash train.sh pick_diverse_bottles demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/pick_diverse_bottles_main_s2.log
echo "[$(date)] done  pick_diverse_bottles seed=2"
echo "[$(date)] start pick_diverse_bottles seed=1"
bash train.sh pick_diverse_bottles demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/pick_diverse_bottles_main_s1.log
echo "[$(date)] done  pick_diverse_bottles seed=1"
echo "[$(date)] start pick_diverse_bottles seed=0"
bash train.sh pick_diverse_bottles demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/pick_diverse_bottles_main_s0.log
echo "[$(date)] done  pick_diverse_bottles seed=0"

# ----- [7/30] pick_dual_bottles_main_s210 -----
pick_dual_bottles_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start pick_dual_bottles seed=2"
bash train.sh pick_dual_bottles demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/pick_dual_bottles_main_s2.log
echo "[$(date)] done  pick_dual_bottles seed=2"
echo "[$(date)] start pick_dual_bottles seed=1"
bash train.sh pick_dual_bottles demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/pick_dual_bottles_main_s1.log
echo "[$(date)] done  pick_dual_bottles seed=1"
echo "[$(date)] start pick_dual_bottles seed=0"
bash train.sh pick_dual_bottles demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/pick_dual_bottles_main_s0.log
echo "[$(date)] done  pick_dual_bottles seed=0"

# ----- [8/30] place_a2b_left_main_s210 -----
place_a2b_left_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_a2b_left seed=2"
bash train.sh place_a2b_left demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_a2b_left_main_s2.log
echo "[$(date)] done  place_a2b_left seed=2"
echo "[$(date)] start place_a2b_left seed=1"
bash train.sh place_a2b_left demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_a2b_left_main_s1.log
echo "[$(date)] done  place_a2b_left seed=1"
echo "[$(date)] start place_a2b_left seed=0"
bash train.sh place_a2b_left demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_a2b_left_main_s0.log
echo "[$(date)] done  place_a2b_left seed=0"

# ----- [9/30] place_a2b_right_main_s210 -----
place_a2b_right_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_a2b_right seed=2"
bash train.sh place_a2b_right demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_a2b_right_main_s2.log
echo "[$(date)] done  place_a2b_right seed=2"
echo "[$(date)] start place_a2b_right seed=1"
bash train.sh place_a2b_right demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_a2b_right_main_s1.log
echo "[$(date)] done  place_a2b_right seed=1"
echo "[$(date)] start place_a2b_right seed=0"
bash train.sh place_a2b_right demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_a2b_right_main_s0.log
echo "[$(date)] done  place_a2b_right seed=0"

# ----- [10/30] place_bread_basket_main_s210 -----
place_bread_basket_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_basket seed=2"
bash train.sh place_bread_basket demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_bread_basket_main_s2.log
echo "[$(date)] done  place_bread_basket seed=2"
echo "[$(date)] start place_bread_basket seed=1"
bash train.sh place_bread_basket demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_bread_basket_main_s1.log
echo "[$(date)] done  place_bread_basket seed=1"
echo "[$(date)] start place_bread_basket seed=0"
bash train.sh place_bread_basket demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_bread_basket_main_s0.log
echo "[$(date)] done  place_bread_basket seed=0"

# ----- [11/30] place_bread_skillet_main_s210 -----
place_bread_skillet_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_skillet seed=2"
bash train.sh place_bread_skillet demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_bread_skillet_main_s2.log
echo "[$(date)] done  place_bread_skillet seed=2"
echo "[$(date)] start place_bread_skillet seed=1"
bash train.sh place_bread_skillet demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_bread_skillet_main_s1.log
echo "[$(date)] done  place_bread_skillet seed=1"
echo "[$(date)] start place_bread_skillet seed=0"
bash train.sh place_bread_skillet demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_bread_skillet_main_s0.log
echo "[$(date)] done  place_bread_skillet seed=0"

# ----- [12/30] place_burger_fries_main_s210 -----
place_burger_fries_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_burger_fries seed=2"
bash train.sh place_burger_fries demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_burger_fries_main_s2.log
echo "[$(date)] done  place_burger_fries seed=2"
echo "[$(date)] start place_burger_fries seed=1"
bash train.sh place_burger_fries demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_burger_fries_main_s1.log
echo "[$(date)] done  place_burger_fries seed=1"
echo "[$(date)] start place_burger_fries seed=0"
bash train.sh place_burger_fries demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_burger_fries_main_s0.log
echo "[$(date)] done  place_burger_fries seed=0"

# ----- [13/30] place_cans_plasticbox_main_s21 -----
place_cans_plasticbox_main_s21
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_cans_plasticbox seed=2"
bash train.sh place_cans_plasticbox demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_cans_plasticbox_main_s2.log
echo "[$(date)] done  place_cans_plasticbox seed=2"
echo "[$(date)] start place_cans_plasticbox seed=1"
bash train.sh place_cans_plasticbox demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_cans_plasticbox_main_s1.log
echo "[$(date)] done  place_cans_plasticbox seed=1"

# ----- [14/30] place_container_plate_main_s210 -----
place_container_plate_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_container_plate seed=2"
bash train.sh place_container_plate demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_container_plate_main_s2.log
echo "[$(date)] done  place_container_plate seed=2"
echo "[$(date)] start place_container_plate seed=1"
bash train.sh place_container_plate demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_container_plate_main_s1.log
echo "[$(date)] done  place_container_plate seed=1"
echo "[$(date)] start place_container_plate seed=0"
bash train.sh place_container_plate demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_container_plate_main_s0.log
echo "[$(date)] done  place_container_plate seed=0"

# ----- [15/30] place_dual_shoes_main_s210 -----
place_dual_shoes_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_dual_shoes seed=2"
bash train.sh place_dual_shoes demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_dual_shoes_main_s2.log
echo "[$(date)] done  place_dual_shoes seed=2"
echo "[$(date)] start place_dual_shoes seed=1"
bash train.sh place_dual_shoes demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_dual_shoes_main_s1.log
echo "[$(date)] done  place_dual_shoes seed=1"
echo "[$(date)] start place_dual_shoes seed=0"
bash train.sh place_dual_shoes demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_dual_shoes_main_s0.log
echo "[$(date)] done  place_dual_shoes seed=0"

# ----- [16/30] place_empty_cup_main_s210 -----
place_empty_cup_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_empty_cup seed=2"
bash train.sh place_empty_cup demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_empty_cup_main_s2.log
echo "[$(date)] done  place_empty_cup seed=2"
echo "[$(date)] start place_empty_cup seed=1"
bash train.sh place_empty_cup demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_empty_cup_main_s1.log
echo "[$(date)] done  place_empty_cup seed=1"
echo "[$(date)] start place_empty_cup seed=0"
bash train.sh place_empty_cup demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_empty_cup_main_s0.log
echo "[$(date)] done  place_empty_cup seed=0"

# ----- [17/30] place_fan_main_s21 -----
place_fan_main_s21
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_fan seed=2"
bash train.sh place_fan demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_fan_main_s2.log
echo "[$(date)] done  place_fan seed=2"
echo "[$(date)] start place_fan seed=1"
bash train.sh place_fan demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_fan_main_s1.log
echo "[$(date)] done  place_fan seed=1"

# ----- [18/30] place_object_basket_main_s21 -----
place_object_basket_main_s21
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_object_basket seed=2"
bash train.sh place_object_basket demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_object_basket_main_s2.log
echo "[$(date)] done  place_object_basket seed=2"
echo "[$(date)] start place_object_basket seed=1"
bash train.sh place_object_basket demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_object_basket_main_s1.log
echo "[$(date)] done  place_object_basket seed=1"

# ----- [19/30] place_object_stand_main_s210 -----
place_object_stand_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_object_stand seed=2"
bash train.sh place_object_stand demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_object_stand_main_s2.log
echo "[$(date)] done  place_object_stand seed=2"
echo "[$(date)] start place_object_stand seed=1"
bash train.sh place_object_stand demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_object_stand_main_s1.log
echo "[$(date)] done  place_object_stand seed=1"
echo "[$(date)] start place_object_stand seed=0"
bash train.sh place_object_stand demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_object_stand_main_s0.log
echo "[$(date)] done  place_object_stand seed=0"

# ----- [20/30] place_phone_stand_main_s210 -----
place_phone_stand_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_phone_stand seed=2"
bash train.sh place_phone_stand demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_phone_stand_main_s2.log
echo "[$(date)] done  place_phone_stand seed=2"
echo "[$(date)] start place_phone_stand seed=1"
bash train.sh place_phone_stand demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_phone_stand_main_s1.log
echo "[$(date)] done  place_phone_stand seed=1"
echo "[$(date)] start place_phone_stand seed=0"
bash train.sh place_phone_stand demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_phone_stand_main_s0.log
echo "[$(date)] done  place_phone_stand seed=0"

# ----- [21/30] place_shoe_main_s210 -----
place_shoe_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_shoe seed=2"
bash train.sh place_shoe demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/place_shoe_main_s2.log
echo "[$(date)] done  place_shoe seed=2"
echo "[$(date)] start place_shoe seed=1"
bash train.sh place_shoe demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/place_shoe_main_s1.log
echo "[$(date)] done  place_shoe seed=1"
echo "[$(date)] start place_shoe seed=0"
bash train.sh place_shoe demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_shoe_main_s0.log
echo "[$(date)] done  place_shoe seed=0"

# ----- [22/30] press_stapler_main_s210 -----
press_stapler_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start press_stapler seed=2"
bash train.sh press_stapler demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/press_stapler_main_s2.log
echo "[$(date)] done  press_stapler seed=2"
echo "[$(date)] start press_stapler seed=1"
bash train.sh press_stapler demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/press_stapler_main_s1.log
echo "[$(date)] done  press_stapler seed=1"
echo "[$(date)] start press_stapler seed=0"
bash train.sh press_stapler demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/press_stapler_main_s0.log
echo "[$(date)] done  press_stapler seed=0"

# ----- [23/30] put_bottles_dustbin_main_s210 -----
put_bottles_dustbin_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start put_bottles_dustbin seed=2"
bash train.sh put_bottles_dustbin demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/put_bottles_dustbin_main_s2.log
echo "[$(date)] done  put_bottles_dustbin seed=2"
echo "[$(date)] start put_bottles_dustbin seed=1"
bash train.sh put_bottles_dustbin demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/put_bottles_dustbin_main_s1.log
echo "[$(date)] done  put_bottles_dustbin seed=1"
echo "[$(date)] start put_bottles_dustbin seed=0"
bash train.sh put_bottles_dustbin demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/put_bottles_dustbin_main_s0.log
echo "[$(date)] done  put_bottles_dustbin seed=0"

# ----- [24/30] rotate_qrcode_main_s210 -----
rotate_qrcode_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start rotate_qrcode seed=2"
bash train.sh rotate_qrcode demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/rotate_qrcode_main_s2.log
echo "[$(date)] done  rotate_qrcode seed=2"
echo "[$(date)] start rotate_qrcode seed=1"
bash train.sh rotate_qrcode demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/rotate_qrcode_main_s1.log
echo "[$(date)] done  rotate_qrcode seed=1"
echo "[$(date)] start rotate_qrcode seed=0"
bash train.sh rotate_qrcode demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/rotate_qrcode_main_s0.log
echo "[$(date)] done  rotate_qrcode seed=0"

# ----- [25/30] scan_object_main_s21 -----
scan_object_main_s21
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start scan_object seed=2"
bash train.sh scan_object demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/scan_object_main_s2.log
echo "[$(date)] done  scan_object seed=2"
echo "[$(date)] start scan_object seed=1"
bash train.sh scan_object demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/scan_object_main_s1.log
echo "[$(date)] done  scan_object seed=1"

# ----- [26/30] shake_bottle_main_s210 -----
shake_bottle_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start shake_bottle seed=2"
bash train.sh shake_bottle demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/shake_bottle_main_s2.log
echo "[$(date)] done  shake_bottle seed=2"
echo "[$(date)] start shake_bottle seed=1"
bash train.sh shake_bottle demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/shake_bottle_main_s1.log
echo "[$(date)] done  shake_bottle seed=1"
echo "[$(date)] start shake_bottle seed=0"
bash train.sh shake_bottle demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/shake_bottle_main_s0.log
echo "[$(date)] done  shake_bottle seed=0"

# ----- [27/30] stack_blocks_three_main_s210 -----
stack_blocks_three_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start stack_blocks_three seed=2"
bash train.sh stack_blocks_three demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/stack_blocks_three_main_s2.log
echo "[$(date)] done  stack_blocks_three seed=2"
echo "[$(date)] start stack_blocks_three seed=1"
bash train.sh stack_blocks_three demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/stack_blocks_three_main_s1.log
echo "[$(date)] done  stack_blocks_three seed=1"
echo "[$(date)] start stack_blocks_three seed=0"
bash train.sh stack_blocks_three demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/stack_blocks_three_main_s0.log
echo "[$(date)] done  stack_blocks_three seed=0"

# ----- [28/30] stack_blocks_two_main_s210 -----
stack_blocks_two_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start stack_blocks_two seed=2"
bash train.sh stack_blocks_two demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/stack_blocks_two_main_s2.log
echo "[$(date)] done  stack_blocks_two seed=2"
echo "[$(date)] start stack_blocks_two seed=1"
bash train.sh stack_blocks_two demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/stack_blocks_two_main_s1.log
echo "[$(date)] done  stack_blocks_two seed=1"
echo "[$(date)] start stack_blocks_two seed=0"
bash train.sh stack_blocks_two demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/stack_blocks_two_main_s0.log
echo "[$(date)] done  stack_blocks_two seed=0"

# ----- [29/30] stack_bowls_three_main_s210 -----
stack_bowls_three_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start stack_bowls_three seed=2"
bash train.sh stack_bowls_three demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/stack_bowls_three_main_s2.log
echo "[$(date)] done  stack_bowls_three seed=2"
echo "[$(date)] start stack_bowls_three seed=1"
bash train.sh stack_bowls_three demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/stack_bowls_three_main_s1.log
echo "[$(date)] done  stack_bowls_three seed=1"
echo "[$(date)] start stack_bowls_three seed=0"
bash train.sh stack_bowls_three demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/stack_bowls_three_main_s0.log
echo "[$(date)] done  stack_bowls_three seed=0"

# ----- [30/30] stack_bowls_two_main_s210 -----
stack_bowls_two_main_s210
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start stack_bowls_two seed=2"
bash train.sh stack_bowls_two demo_clean 100 2 0 main 2>&1 | tee ../../Logs/train/stack_bowls_two_main_s2.log
echo "[$(date)] done  stack_bowls_two seed=2"
echo "[$(date)] start stack_bowls_two seed=1"
bash train.sh stack_bowls_two demo_clean 100 1 0 main 2>&1 | tee ../../Logs/train/stack_bowls_two_main_s1.log
echo "[$(date)] done  stack_bowls_two seed=1"
echo "[$(date)] start stack_bowls_two seed=0"
bash train.sh stack_bowls_two demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/stack_bowls_two_main_s0.log
echo "[$(date)] done  stack_bowls_two seed=0"

