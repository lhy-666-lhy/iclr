# 32 tasks × 13 encoders = 416 copy-paste job scripts
# Skip same-as-main: ab1_ex4, ab2_ex4, ab3_ex2, ab4_ex1
# Each block: job name (<task>_<enc>) + independent launch script.

# ----- [0/415] beat_block_hammer_main -----
beat_block_hammer_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/beat_block_hammer_main.log

# ----- [1/415] beat_block_hammer_ab1_ex1 -----
beat_block_hammer_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/beat_block_hammer_ab1_ex1.log

# ----- [2/415] beat_block_hammer_ab1_ex2 -----
beat_block_hammer_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/beat_block_hammer_ab1_ex2.log

# ----- [3/415] beat_block_hammer_ab1_ex3 -----
beat_block_hammer_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/beat_block_hammer_ab1_ex3.log

# ----- [4/415] beat_block_hammer_ab2_ex2 -----
beat_block_hammer_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/beat_block_hammer_ab2_ex2.log

# ----- [5/415] beat_block_hammer_ab2_ex3 -----
beat_block_hammer_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/beat_block_hammer_ab2_ex3.log

# ----- [6/415] beat_block_hammer_ab3_ex1 -----
beat_block_hammer_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/beat_block_hammer_ab3_ex1.log

# ----- [7/415] beat_block_hammer_ab3_ex3 -----
beat_block_hammer_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/beat_block_hammer_ab3_ex3.log

# ----- [8/415] beat_block_hammer_ab3_ex4 -----
beat_block_hammer_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/beat_block_hammer_ab3_ex4.log

# ----- [9/415] beat_block_hammer_ab3_ex5 -----
beat_block_hammer_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/beat_block_hammer_ab3_ex5.log

# ----- [10/415] beat_block_hammer_ab3_ex6 -----
beat_block_hammer_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/beat_block_hammer_ab3_ex6.log

# ----- [11/415] beat_block_hammer_ab4_ex2 -----
beat_block_hammer_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/beat_block_hammer_ab4_ex2.log

# ----- [12/415] beat_block_hammer_ab4_ex3 -----
beat_block_hammer_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh beat_block_hammer demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/beat_block_hammer_ab4_ex3.log

# ----- [13/415] handover_block_main -----
handover_block_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh handover_block demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/handover_block_main.log

# ----- [14/415] handover_block_ab1_ex1 -----
handover_block_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh handover_block demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/handover_block_ab1_ex1.log

# ----- [15/415] handover_block_ab1_ex2 -----
handover_block_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh handover_block demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/handover_block_ab1_ex2.log

# ----- [16/415] handover_block_ab1_ex3 -----
handover_block_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh handover_block demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/handover_block_ab1_ex3.log

# ----- [17/415] handover_block_ab2_ex2 -----
handover_block_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh handover_block demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/handover_block_ab2_ex2.log

# ----- [18/415] handover_block_ab2_ex3 -----
handover_block_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh handover_block demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/handover_block_ab2_ex3.log

# ----- [19/415] handover_block_ab3_ex1 -----
handover_block_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh handover_block demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/handover_block_ab3_ex1.log

# ----- [20/415] handover_block_ab3_ex3 -----
handover_block_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh handover_block demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/handover_block_ab3_ex3.log

# ----- [21/415] handover_block_ab3_ex4 -----
handover_block_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh handover_block demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/handover_block_ab3_ex4.log

# ----- [22/415] handover_block_ab3_ex5 -----
handover_block_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh handover_block demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/handover_block_ab3_ex5.log

# ----- [23/415] handover_block_ab3_ex6 -----
handover_block_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh handover_block demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/handover_block_ab3_ex6.log

# ----- [24/415] handover_block_ab4_ex2 -----
handover_block_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh handover_block demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/handover_block_ab4_ex2.log

# ----- [25/415] handover_block_ab4_ex3 -----
handover_block_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh handover_block demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/handover_block_ab4_ex3.log

# ----- [26/415] hanging_mug_main -----
hanging_mug_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh hanging_mug demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/hanging_mug_main.log

# ----- [27/415] hanging_mug_ab1_ex1 -----
hanging_mug_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh hanging_mug demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/hanging_mug_ab1_ex1.log

# ----- [28/415] hanging_mug_ab1_ex2 -----
hanging_mug_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh hanging_mug demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/hanging_mug_ab1_ex2.log

# ----- [29/415] hanging_mug_ab1_ex3 -----
hanging_mug_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh hanging_mug demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/hanging_mug_ab1_ex3.log

# ----- [30/415] hanging_mug_ab2_ex2 -----
hanging_mug_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh hanging_mug demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/hanging_mug_ab2_ex2.log

# ----- [31/415] hanging_mug_ab2_ex3 -----
hanging_mug_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh hanging_mug demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/hanging_mug_ab2_ex3.log

# ----- [32/415] hanging_mug_ab3_ex1 -----
hanging_mug_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh hanging_mug demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/hanging_mug_ab3_ex1.log

# ----- [33/415] hanging_mug_ab3_ex3 -----
hanging_mug_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh hanging_mug demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/hanging_mug_ab3_ex3.log

# ----- [34/415] hanging_mug_ab3_ex4 -----
hanging_mug_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh hanging_mug demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/hanging_mug_ab3_ex4.log

# ----- [35/415] hanging_mug_ab3_ex5 -----
hanging_mug_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh hanging_mug demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/hanging_mug_ab3_ex5.log

# ----- [36/415] hanging_mug_ab3_ex6 -----
hanging_mug_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh hanging_mug demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/hanging_mug_ab3_ex6.log

# ----- [37/415] hanging_mug_ab4_ex2 -----
hanging_mug_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh hanging_mug demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/hanging_mug_ab4_ex2.log

# ----- [38/415] hanging_mug_ab4_ex3 -----
hanging_mug_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh hanging_mug demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/hanging_mug_ab4_ex3.log

# ----- [39/415] move_can_pot_main -----
move_can_pot_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh move_can_pot demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/move_can_pot_main.log

# ----- [40/415] move_can_pot_ab1_ex1 -----
move_can_pot_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh move_can_pot demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/move_can_pot_ab1_ex1.log

# ----- [41/415] move_can_pot_ab1_ex2 -----
move_can_pot_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh move_can_pot demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/move_can_pot_ab1_ex2.log

# ----- [42/415] move_can_pot_ab1_ex3 -----
move_can_pot_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh move_can_pot demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/move_can_pot_ab1_ex3.log

# ----- [43/415] move_can_pot_ab2_ex2 -----
move_can_pot_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh move_can_pot demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/move_can_pot_ab2_ex2.log

# ----- [44/415] move_can_pot_ab2_ex3 -----
move_can_pot_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh move_can_pot demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/move_can_pot_ab2_ex3.log

# ----- [45/415] move_can_pot_ab3_ex1 -----
move_can_pot_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh move_can_pot demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/move_can_pot_ab3_ex1.log

# ----- [46/415] move_can_pot_ab3_ex3 -----
move_can_pot_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh move_can_pot demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/move_can_pot_ab3_ex3.log

# ----- [47/415] move_can_pot_ab3_ex4 -----
move_can_pot_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh move_can_pot demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/move_can_pot_ab3_ex4.log

# ----- [48/415] move_can_pot_ab3_ex5 -----
move_can_pot_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh move_can_pot demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/move_can_pot_ab3_ex5.log

# ----- [49/415] move_can_pot_ab3_ex6 -----
move_can_pot_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh move_can_pot demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/move_can_pot_ab3_ex6.log

# ----- [50/415] move_can_pot_ab4_ex2 -----
move_can_pot_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh move_can_pot demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/move_can_pot_ab4_ex2.log

# ----- [51/415] move_can_pot_ab4_ex3 -----
move_can_pot_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh move_can_pot demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/move_can_pot_ab4_ex3.log

# ----- [52/415] open_laptop_main -----
open_laptop_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_laptop demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/open_laptop_main.log

# ----- [53/415] open_laptop_ab1_ex1 -----
open_laptop_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_laptop demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/open_laptop_ab1_ex1.log

# ----- [54/415] open_laptop_ab1_ex2 -----
open_laptop_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_laptop demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/open_laptop_ab1_ex2.log

# ----- [55/415] open_laptop_ab1_ex3 -----
open_laptop_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_laptop demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/open_laptop_ab1_ex3.log

# ----- [56/415] open_laptop_ab2_ex2 -----
open_laptop_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_laptop demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/open_laptop_ab2_ex2.log

# ----- [57/415] open_laptop_ab2_ex3 -----
open_laptop_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_laptop demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/open_laptop_ab2_ex3.log

# ----- [58/415] open_laptop_ab3_ex1 -----
open_laptop_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_laptop demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/open_laptop_ab3_ex1.log

# ----- [59/415] open_laptop_ab3_ex3 -----
open_laptop_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_laptop demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/open_laptop_ab3_ex3.log

# ----- [60/415] open_laptop_ab3_ex4 -----
open_laptop_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_laptop demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/open_laptop_ab3_ex4.log

# ----- [61/415] open_laptop_ab3_ex5 -----
open_laptop_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_laptop demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/open_laptop_ab3_ex5.log

# ----- [62/415] open_laptop_ab3_ex6 -----
open_laptop_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_laptop demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/open_laptop_ab3_ex6.log

# ----- [63/415] open_laptop_ab4_ex2 -----
open_laptop_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_laptop demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/open_laptop_ab4_ex2.log

# ----- [64/415] open_laptop_ab4_ex3 -----
open_laptop_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_laptop demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/open_laptop_ab4_ex3.log

# ----- [65/415] open_microwave_main -----
open_microwave_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_microwave demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/open_microwave_main.log

# ----- [66/415] open_microwave_ab1_ex1 -----
open_microwave_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_microwave demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/open_microwave_ab1_ex1.log

# ----- [67/415] open_microwave_ab1_ex2 -----
open_microwave_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_microwave demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/open_microwave_ab1_ex2.log

# ----- [68/415] open_microwave_ab1_ex3 -----
open_microwave_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_microwave demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/open_microwave_ab1_ex3.log

# ----- [69/415] open_microwave_ab2_ex2 -----
open_microwave_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_microwave demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/open_microwave_ab2_ex2.log

# ----- [70/415] open_microwave_ab2_ex3 -----
open_microwave_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_microwave demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/open_microwave_ab2_ex3.log

# ----- [71/415] open_microwave_ab3_ex1 -----
open_microwave_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_microwave demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/open_microwave_ab3_ex1.log

# ----- [72/415] open_microwave_ab3_ex3 -----
open_microwave_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_microwave demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/open_microwave_ab3_ex3.log

# ----- [73/415] open_microwave_ab3_ex4 -----
open_microwave_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_microwave demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/open_microwave_ab3_ex4.log

# ----- [74/415] open_microwave_ab3_ex5 -----
open_microwave_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_microwave demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/open_microwave_ab3_ex5.log

# ----- [75/415] open_microwave_ab3_ex6 -----
open_microwave_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_microwave demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/open_microwave_ab3_ex6.log

# ----- [76/415] open_microwave_ab4_ex2 -----
open_microwave_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_microwave demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/open_microwave_ab4_ex2.log

# ----- [77/415] open_microwave_ab4_ex3 -----
open_microwave_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh open_microwave demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/open_microwave_ab4_ex3.log

# ----- [78/415] pick_diverse_bottles_main -----
pick_diverse_bottles_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_diverse_bottles demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/pick_diverse_bottles_main.log

# ----- [79/415] pick_diverse_bottles_ab1_ex1 -----
pick_diverse_bottles_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab1_ex1.log

# ----- [80/415] pick_diverse_bottles_ab1_ex2 -----
pick_diverse_bottles_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab1_ex2.log

# ----- [81/415] pick_diverse_bottles_ab1_ex3 -----
pick_diverse_bottles_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab1_ex3.log

# ----- [82/415] pick_diverse_bottles_ab2_ex2 -----
pick_diverse_bottles_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab2_ex2.log

# ----- [83/415] pick_diverse_bottles_ab2_ex3 -----
pick_diverse_bottles_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab2_ex3.log

# ----- [84/415] pick_diverse_bottles_ab3_ex1 -----
pick_diverse_bottles_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab3_ex1.log

# ----- [85/415] pick_diverse_bottles_ab3_ex3 -----
pick_diverse_bottles_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab3_ex3.log

# ----- [86/415] pick_diverse_bottles_ab3_ex4 -----
pick_diverse_bottles_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab3_ex4.log

# ----- [87/415] pick_diverse_bottles_ab3_ex5 -----
pick_diverse_bottles_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab3_ex5.log

# ----- [88/415] pick_diverse_bottles_ab3_ex6 -----
pick_diverse_bottles_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab3_ex6.log

# ----- [89/415] pick_diverse_bottles_ab4_ex2 -----
pick_diverse_bottles_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab4_ex2.log

# ----- [90/415] pick_diverse_bottles_ab4_ex3 -----
pick_diverse_bottles_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_diverse_bottles demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/pick_diverse_bottles_ab4_ex3.log

# ----- [91/415] pick_dual_bottles_main -----
pick_dual_bottles_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_dual_bottles demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/pick_dual_bottles_main.log

# ----- [92/415] pick_dual_bottles_ab1_ex1 -----
pick_dual_bottles_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab1_ex1.log

# ----- [93/415] pick_dual_bottles_ab1_ex2 -----
pick_dual_bottles_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab1_ex2.log

# ----- [94/415] pick_dual_bottles_ab1_ex3 -----
pick_dual_bottles_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab1_ex3.log

# ----- [95/415] pick_dual_bottles_ab2_ex2 -----
pick_dual_bottles_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab2_ex2.log

# ----- [96/415] pick_dual_bottles_ab2_ex3 -----
pick_dual_bottles_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab2_ex3.log

# ----- [97/415] pick_dual_bottles_ab3_ex1 -----
pick_dual_bottles_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab3_ex1.log

# ----- [98/415] pick_dual_bottles_ab3_ex3 -----
pick_dual_bottles_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab3_ex3.log

# ----- [99/415] pick_dual_bottles_ab3_ex4 -----
pick_dual_bottles_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab3_ex4.log

# ----- [100/415] pick_dual_bottles_ab3_ex5 -----
pick_dual_bottles_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab3_ex5.log

# ----- [101/415] pick_dual_bottles_ab3_ex6 -----
pick_dual_bottles_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab3_ex6.log

# ----- [102/415] pick_dual_bottles_ab4_ex2 -----
pick_dual_bottles_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab4_ex2.log

# ----- [103/415] pick_dual_bottles_ab4_ex3 -----
pick_dual_bottles_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh pick_dual_bottles demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab4_ex3.log

# ----- [104/415] place_a2b_left_main -----
place_a2b_left_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_left demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_a2b_left_main.log

# ----- [105/415] place_a2b_left_ab1_ex1 -----
place_a2b_left_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_left demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_a2b_left_ab1_ex1.log

# ----- [106/415] place_a2b_left_ab1_ex2 -----
place_a2b_left_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_left demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_a2b_left_ab1_ex2.log

# ----- [107/415] place_a2b_left_ab1_ex3 -----
place_a2b_left_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_left demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_a2b_left_ab1_ex3.log

# ----- [108/415] place_a2b_left_ab2_ex2 -----
place_a2b_left_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_left demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_a2b_left_ab2_ex2.log

# ----- [109/415] place_a2b_left_ab2_ex3 -----
place_a2b_left_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_left demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_a2b_left_ab2_ex3.log

# ----- [110/415] place_a2b_left_ab3_ex1 -----
place_a2b_left_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_left demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_a2b_left_ab3_ex1.log

# ----- [111/415] place_a2b_left_ab3_ex3 -----
place_a2b_left_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_left demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_a2b_left_ab3_ex3.log

# ----- [112/415] place_a2b_left_ab3_ex4 -----
place_a2b_left_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_left demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_a2b_left_ab3_ex4.log

# ----- [113/415] place_a2b_left_ab3_ex5 -----
place_a2b_left_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_left demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_a2b_left_ab3_ex5.log

# ----- [114/415] place_a2b_left_ab3_ex6 -----
place_a2b_left_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_left demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_a2b_left_ab3_ex6.log

# ----- [115/415] place_a2b_left_ab4_ex2 -----
place_a2b_left_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_left demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_a2b_left_ab4_ex2.log

# ----- [116/415] place_a2b_left_ab4_ex3 -----
place_a2b_left_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_left demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_a2b_left_ab4_ex3.log

# ----- [117/415] place_a2b_right_main -----
place_a2b_right_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_right demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_a2b_right_main.log

# ----- [118/415] place_a2b_right_ab1_ex1 -----
place_a2b_right_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_right demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_a2b_right_ab1_ex1.log

# ----- [119/415] place_a2b_right_ab1_ex2 -----
place_a2b_right_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_right demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_a2b_right_ab1_ex2.log

# ----- [120/415] place_a2b_right_ab1_ex3 -----
place_a2b_right_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_right demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_a2b_right_ab1_ex3.log

# ----- [121/415] place_a2b_right_ab2_ex2 -----
place_a2b_right_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_right demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_a2b_right_ab2_ex2.log

# ----- [122/415] place_a2b_right_ab2_ex3 -----
place_a2b_right_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_right demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_a2b_right_ab2_ex3.log

# ----- [123/415] place_a2b_right_ab3_ex1 -----
place_a2b_right_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_right demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_a2b_right_ab3_ex1.log

# ----- [124/415] place_a2b_right_ab3_ex3 -----
place_a2b_right_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_right demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_a2b_right_ab3_ex3.log

# ----- [125/415] place_a2b_right_ab3_ex4 -----
place_a2b_right_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_right demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_a2b_right_ab3_ex4.log

# ----- [126/415] place_a2b_right_ab3_ex5 -----
place_a2b_right_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_right demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_a2b_right_ab3_ex5.log

# ----- [127/415] place_a2b_right_ab3_ex6 -----
place_a2b_right_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_right demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_a2b_right_ab3_ex6.log

# ----- [128/415] place_a2b_right_ab4_ex2 -----
place_a2b_right_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_right demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_a2b_right_ab4_ex2.log

# ----- [129/415] place_a2b_right_ab4_ex3 -----
place_a2b_right_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_a2b_right demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_a2b_right_ab4_ex3.log

# ----- [130/415] place_bread_basket_main -----
place_bread_basket_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_basket demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_bread_basket_main.log

# ----- [131/415] place_bread_basket_ab1_ex1 -----
place_bread_basket_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_basket demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_bread_basket_ab1_ex1.log

# ----- [132/415] place_bread_basket_ab1_ex2 -----
place_bread_basket_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_basket demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_bread_basket_ab1_ex2.log

# ----- [133/415] place_bread_basket_ab1_ex3 -----
place_bread_basket_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_basket demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_bread_basket_ab1_ex3.log

# ----- [134/415] place_bread_basket_ab2_ex2 -----
place_bread_basket_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_basket demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_bread_basket_ab2_ex2.log

# ----- [135/415] place_bread_basket_ab2_ex3 -----
place_bread_basket_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_basket demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_bread_basket_ab2_ex3.log

# ----- [136/415] place_bread_basket_ab3_ex1 -----
place_bread_basket_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_basket demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_bread_basket_ab3_ex1.log

# ----- [137/415] place_bread_basket_ab3_ex3 -----
place_bread_basket_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_basket demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_bread_basket_ab3_ex3.log

# ----- [138/415] place_bread_basket_ab3_ex4 -----
place_bread_basket_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_basket demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_bread_basket_ab3_ex4.log

# ----- [139/415] place_bread_basket_ab3_ex5 -----
place_bread_basket_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_basket demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_bread_basket_ab3_ex5.log

# ----- [140/415] place_bread_basket_ab3_ex6 -----
place_bread_basket_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_basket demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_bread_basket_ab3_ex6.log

# ----- [141/415] place_bread_basket_ab4_ex2 -----
place_bread_basket_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_basket demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_bread_basket_ab4_ex2.log

# ----- [142/415] place_bread_basket_ab4_ex3 -----
place_bread_basket_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_basket demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_bread_basket_ab4_ex3.log

# ----- [143/415] place_bread_skillet_main -----
place_bread_skillet_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_skillet demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_bread_skillet_main.log

# ----- [144/415] place_bread_skillet_ab1_ex1 -----
place_bread_skillet_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_skillet demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_bread_skillet_ab1_ex1.log

# ----- [145/415] place_bread_skillet_ab1_ex2 -----
place_bread_skillet_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_skillet demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_bread_skillet_ab1_ex2.log

# ----- [146/415] place_bread_skillet_ab1_ex3 -----
place_bread_skillet_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_skillet demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_bread_skillet_ab1_ex3.log

# ----- [147/415] place_bread_skillet_ab2_ex2 -----
place_bread_skillet_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_skillet demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_bread_skillet_ab2_ex2.log

# ----- [148/415] place_bread_skillet_ab2_ex3 -----
place_bread_skillet_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_skillet demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_bread_skillet_ab2_ex3.log

# ----- [149/415] place_bread_skillet_ab3_ex1 -----
place_bread_skillet_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_skillet demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_bread_skillet_ab3_ex1.log

# ----- [150/415] place_bread_skillet_ab3_ex3 -----
place_bread_skillet_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_skillet demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_bread_skillet_ab3_ex3.log

# ----- [151/415] place_bread_skillet_ab3_ex4 -----
place_bread_skillet_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_skillet demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_bread_skillet_ab3_ex4.log

# ----- [152/415] place_bread_skillet_ab3_ex5 -----
place_bread_skillet_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_skillet demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_bread_skillet_ab3_ex5.log

# ----- [153/415] place_bread_skillet_ab3_ex6 -----
place_bread_skillet_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_skillet demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_bread_skillet_ab3_ex6.log

# ----- [154/415] place_bread_skillet_ab4_ex2 -----
place_bread_skillet_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_skillet demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_bread_skillet_ab4_ex2.log

# ----- [155/415] place_bread_skillet_ab4_ex3 -----
place_bread_skillet_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_bread_skillet demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_bread_skillet_ab4_ex3.log

# ----- [156/415] place_burger_fries_main -----
place_burger_fries_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_burger_fries demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_burger_fries_main.log

# ----- [157/415] place_burger_fries_ab1_ex1 -----
place_burger_fries_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_burger_fries demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_burger_fries_ab1_ex1.log

# ----- [158/415] place_burger_fries_ab1_ex2 -----
place_burger_fries_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_burger_fries demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_burger_fries_ab1_ex2.log

# ----- [159/415] place_burger_fries_ab1_ex3 -----
place_burger_fries_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_burger_fries demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_burger_fries_ab1_ex3.log

# ----- [160/415] place_burger_fries_ab2_ex2 -----
place_burger_fries_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_burger_fries demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_burger_fries_ab2_ex2.log

# ----- [161/415] place_burger_fries_ab2_ex3 -----
place_burger_fries_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_burger_fries demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_burger_fries_ab2_ex3.log

# ----- [162/415] place_burger_fries_ab3_ex1 -----
place_burger_fries_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_burger_fries demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_burger_fries_ab3_ex1.log

# ----- [163/415] place_burger_fries_ab3_ex3 -----
place_burger_fries_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_burger_fries demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_burger_fries_ab3_ex3.log

# ----- [164/415] place_burger_fries_ab3_ex4 -----
place_burger_fries_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_burger_fries demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_burger_fries_ab3_ex4.log

# ----- [165/415] place_burger_fries_ab3_ex5 -----
place_burger_fries_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_burger_fries demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_burger_fries_ab3_ex5.log

# ----- [166/415] place_burger_fries_ab3_ex6 -----
place_burger_fries_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_burger_fries demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_burger_fries_ab3_ex6.log

# ----- [167/415] place_burger_fries_ab4_ex2 -----
place_burger_fries_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_burger_fries demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_burger_fries_ab4_ex2.log

# ----- [168/415] place_burger_fries_ab4_ex3 -----
place_burger_fries_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_burger_fries demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_burger_fries_ab4_ex3.log

# ----- [169/415] place_cans_plasticbox_main -----
place_cans_plasticbox_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_cans_plasticbox_main.log

# ----- [170/415] place_cans_plasticbox_ab1_ex1 -----
place_cans_plasticbox_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab1_ex1.log

# ----- [171/415] place_cans_plasticbox_ab1_ex2 -----
place_cans_plasticbox_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab1_ex2.log

# ----- [172/415] place_cans_plasticbox_ab1_ex3 -----
place_cans_plasticbox_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab1_ex3.log

# ----- [173/415] place_cans_plasticbox_ab2_ex2 -----
place_cans_plasticbox_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab2_ex2.log

# ----- [174/415] place_cans_plasticbox_ab2_ex3 -----
place_cans_plasticbox_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab2_ex3.log

# ----- [175/415] place_cans_plasticbox_ab3_ex1 -----
place_cans_plasticbox_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab3_ex1.log

# ----- [176/415] place_cans_plasticbox_ab3_ex3 -----
place_cans_plasticbox_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab3_ex3.log

# ----- [177/415] place_cans_plasticbox_ab3_ex4 -----
place_cans_plasticbox_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab3_ex4.log

# ----- [178/415] place_cans_plasticbox_ab3_ex5 -----
place_cans_plasticbox_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab3_ex5.log

# ----- [179/415] place_cans_plasticbox_ab3_ex6 -----
place_cans_plasticbox_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab3_ex6.log

# ----- [180/415] place_cans_plasticbox_ab4_ex2 -----
place_cans_plasticbox_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab4_ex2.log

# ----- [181/415] place_cans_plasticbox_ab4_ex3 -----
place_cans_plasticbox_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab4_ex3.log

# ----- [182/415] place_container_plate_main -----
place_container_plate_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_container_plate demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_container_plate_main.log

# ----- [183/415] place_container_plate_ab1_ex1 -----
place_container_plate_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_container_plate demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_container_plate_ab1_ex1.log

# ----- [184/415] place_container_plate_ab1_ex2 -----
place_container_plate_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_container_plate demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_container_plate_ab1_ex2.log

# ----- [185/415] place_container_plate_ab1_ex3 -----
place_container_plate_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_container_plate demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_container_plate_ab1_ex3.log

# ----- [186/415] place_container_plate_ab2_ex2 -----
place_container_plate_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_container_plate demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_container_plate_ab2_ex2.log

# ----- [187/415] place_container_plate_ab2_ex3 -----
place_container_plate_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_container_plate demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_container_plate_ab2_ex3.log

# ----- [188/415] place_container_plate_ab3_ex1 -----
place_container_plate_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_container_plate demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_container_plate_ab3_ex1.log

# ----- [189/415] place_container_plate_ab3_ex3 -----
place_container_plate_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_container_plate demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_container_plate_ab3_ex3.log

# ----- [190/415] place_container_plate_ab3_ex4 -----
place_container_plate_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_container_plate demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_container_plate_ab3_ex4.log

# ----- [191/415] place_container_plate_ab3_ex5 -----
place_container_plate_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_container_plate demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_container_plate_ab3_ex5.log

# ----- [192/415] place_container_plate_ab3_ex6 -----
place_container_plate_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_container_plate demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_container_plate_ab3_ex6.log

# ----- [193/415] place_container_plate_ab4_ex2 -----
place_container_plate_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_container_plate demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_container_plate_ab4_ex2.log

# ----- [194/415] place_container_plate_ab4_ex3 -----
place_container_plate_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_container_plate demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_container_plate_ab4_ex3.log

# ----- [195/415] place_dual_shoes_main -----
place_dual_shoes_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_dual_shoes demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_dual_shoes_main.log

# ----- [196/415] place_dual_shoes_ab1_ex1 -----
place_dual_shoes_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_dual_shoes demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_dual_shoes_ab1_ex1.log

# ----- [197/415] place_dual_shoes_ab1_ex2 -----
place_dual_shoes_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_dual_shoes demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_dual_shoes_ab1_ex2.log

# ----- [198/415] place_dual_shoes_ab1_ex3 -----
place_dual_shoes_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_dual_shoes demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_dual_shoes_ab1_ex3.log

# ----- [199/415] place_dual_shoes_ab2_ex2 -----
place_dual_shoes_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_dual_shoes demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_dual_shoes_ab2_ex2.log

# ----- [200/415] place_dual_shoes_ab2_ex3 -----
place_dual_shoes_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_dual_shoes demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_dual_shoes_ab2_ex3.log

# ----- [201/415] place_dual_shoes_ab3_ex1 -----
place_dual_shoes_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_dual_shoes demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_dual_shoes_ab3_ex1.log

# ----- [202/415] place_dual_shoes_ab3_ex3 -----
place_dual_shoes_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_dual_shoes demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_dual_shoes_ab3_ex3.log

# ----- [203/415] place_dual_shoes_ab3_ex4 -----
place_dual_shoes_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_dual_shoes demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_dual_shoes_ab3_ex4.log

# ----- [204/415] place_dual_shoes_ab3_ex5 -----
place_dual_shoes_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_dual_shoes demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_dual_shoes_ab3_ex5.log

# ----- [205/415] place_dual_shoes_ab3_ex6 -----
place_dual_shoes_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_dual_shoes demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_dual_shoes_ab3_ex6.log

# ----- [206/415] place_dual_shoes_ab4_ex2 -----
place_dual_shoes_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_dual_shoes demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_dual_shoes_ab4_ex2.log

# ----- [207/415] place_dual_shoes_ab4_ex3 -----
place_dual_shoes_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_dual_shoes demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_dual_shoes_ab4_ex3.log

# ----- [208/415] place_empty_cup_main -----
place_empty_cup_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_empty_cup demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_empty_cup_main.log

# ----- [209/415] place_empty_cup_ab1_ex1 -----
place_empty_cup_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_empty_cup demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_empty_cup_ab1_ex1.log

# ----- [210/415] place_empty_cup_ab1_ex2 -----
place_empty_cup_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_empty_cup demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_empty_cup_ab1_ex2.log

# ----- [211/415] place_empty_cup_ab1_ex3 -----
place_empty_cup_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_empty_cup demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_empty_cup_ab1_ex3.log

# ----- [212/415] place_empty_cup_ab2_ex2 -----
place_empty_cup_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_empty_cup demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_empty_cup_ab2_ex2.log

# ----- [213/415] place_empty_cup_ab2_ex3 -----
place_empty_cup_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_empty_cup demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_empty_cup_ab2_ex3.log

# ----- [214/415] place_empty_cup_ab3_ex1 -----
place_empty_cup_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_empty_cup demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_empty_cup_ab3_ex1.log

# ----- [215/415] place_empty_cup_ab3_ex3 -----
place_empty_cup_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_empty_cup demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_empty_cup_ab3_ex3.log

# ----- [216/415] place_empty_cup_ab3_ex4 -----
place_empty_cup_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_empty_cup demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_empty_cup_ab3_ex4.log

# ----- [217/415] place_empty_cup_ab3_ex5 -----
place_empty_cup_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_empty_cup demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_empty_cup_ab3_ex5.log

# ----- [218/415] place_empty_cup_ab3_ex6 -----
place_empty_cup_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_empty_cup demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_empty_cup_ab3_ex6.log

# ----- [219/415] place_empty_cup_ab4_ex2 -----
place_empty_cup_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_empty_cup demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_empty_cup_ab4_ex2.log

# ----- [220/415] place_empty_cup_ab4_ex3 -----
place_empty_cup_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_empty_cup demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_empty_cup_ab4_ex3.log

# ----- [221/415] place_fan_main -----
place_fan_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_fan_main.log

# ----- [222/415] place_fan_ab1_ex1 -----
place_fan_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_fan_ab1_ex1.log

# ----- [223/415] place_fan_ab1_ex2 -----
place_fan_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_fan_ab1_ex2.log

# ----- [224/415] place_fan_ab1_ex3 -----
place_fan_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_fan_ab1_ex3.log

# ----- [225/415] place_fan_ab2_ex2 -----
place_fan_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_fan_ab2_ex2.log

# ----- [226/415] place_fan_ab2_ex3 -----
place_fan_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_fan_ab2_ex3.log

# ----- [227/415] place_fan_ab3_ex1 -----
place_fan_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_fan_ab3_ex1.log

# ----- [228/415] place_fan_ab3_ex3 -----
place_fan_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_fan_ab3_ex3.log

# ----- [229/415] place_fan_ab3_ex4 -----
place_fan_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_fan_ab3_ex4.log

# ----- [230/415] place_fan_ab3_ex5 -----
place_fan_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_fan_ab3_ex5.log

# ----- [231/415] place_fan_ab3_ex6 -----
place_fan_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_fan_ab3_ex6.log

# ----- [232/415] place_fan_ab4_ex2 -----
place_fan_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_fan_ab4_ex2.log

# ----- [233/415] place_fan_ab4_ex3 -----
place_fan_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_fan demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_fan_ab4_ex3.log

# ----- [234/415] place_object_basket_main -----
place_object_basket_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_basket demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_object_basket_main.log

# ----- [235/415] place_object_basket_ab1_ex1 -----
place_object_basket_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_basket demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_object_basket_ab1_ex1.log

# ----- [236/415] place_object_basket_ab1_ex2 -----
place_object_basket_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_basket demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_object_basket_ab1_ex2.log

# ----- [237/415] place_object_basket_ab1_ex3 -----
place_object_basket_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_basket demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_object_basket_ab1_ex3.log

# ----- [238/415] place_object_basket_ab2_ex2 -----
place_object_basket_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_basket demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_object_basket_ab2_ex2.log

# ----- [239/415] place_object_basket_ab2_ex3 -----
place_object_basket_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_basket demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_object_basket_ab2_ex3.log

# ----- [240/415] place_object_basket_ab3_ex1 -----
place_object_basket_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_basket demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_object_basket_ab3_ex1.log

# ----- [241/415] place_object_basket_ab3_ex3 -----
place_object_basket_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_basket demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_object_basket_ab3_ex3.log

# ----- [242/415] place_object_basket_ab3_ex4 -----
place_object_basket_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_basket demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_object_basket_ab3_ex4.log

# ----- [243/415] place_object_basket_ab3_ex5 -----
place_object_basket_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_basket demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_object_basket_ab3_ex5.log

# ----- [244/415] place_object_basket_ab3_ex6 -----
place_object_basket_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_basket demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_object_basket_ab3_ex6.log

# ----- [245/415] place_object_basket_ab4_ex2 -----
place_object_basket_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_basket demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_object_basket_ab4_ex2.log

# ----- [246/415] place_object_basket_ab4_ex3 -----
place_object_basket_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_basket demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_object_basket_ab4_ex3.log

# ----- [247/415] place_object_stand_main -----
place_object_stand_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_stand demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_object_stand_main.log

# ----- [248/415] place_object_stand_ab1_ex1 -----
place_object_stand_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_stand demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_object_stand_ab1_ex1.log

# ----- [249/415] place_object_stand_ab1_ex2 -----
place_object_stand_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_stand demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_object_stand_ab1_ex2.log

# ----- [250/415] place_object_stand_ab1_ex3 -----
place_object_stand_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_stand demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_object_stand_ab1_ex3.log

# ----- [251/415] place_object_stand_ab2_ex2 -----
place_object_stand_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_stand demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_object_stand_ab2_ex2.log

# ----- [252/415] place_object_stand_ab2_ex3 -----
place_object_stand_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_stand demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_object_stand_ab2_ex3.log

# ----- [253/415] place_object_stand_ab3_ex1 -----
place_object_stand_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_stand demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_object_stand_ab3_ex1.log

# ----- [254/415] place_object_stand_ab3_ex3 -----
place_object_stand_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_stand demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_object_stand_ab3_ex3.log

# ----- [255/415] place_object_stand_ab3_ex4 -----
place_object_stand_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_stand demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_object_stand_ab3_ex4.log

# ----- [256/415] place_object_stand_ab3_ex5 -----
place_object_stand_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_stand demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_object_stand_ab3_ex5.log

# ----- [257/415] place_object_stand_ab3_ex6 -----
place_object_stand_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_stand demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_object_stand_ab3_ex6.log

# ----- [258/415] place_object_stand_ab4_ex2 -----
place_object_stand_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_stand demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_object_stand_ab4_ex2.log

# ----- [259/415] place_object_stand_ab4_ex3 -----
place_object_stand_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_object_stand demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_object_stand_ab4_ex3.log

# ----- [260/415] place_phone_stand_main -----
place_phone_stand_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_phone_stand demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_phone_stand_main.log

# ----- [261/415] place_phone_stand_ab1_ex1 -----
place_phone_stand_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_phone_stand demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_phone_stand_ab1_ex1.log

# ----- [262/415] place_phone_stand_ab1_ex2 -----
place_phone_stand_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_phone_stand demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_phone_stand_ab1_ex2.log

# ----- [263/415] place_phone_stand_ab1_ex3 -----
place_phone_stand_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_phone_stand demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_phone_stand_ab1_ex3.log

# ----- [264/415] place_phone_stand_ab2_ex2 -----
place_phone_stand_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_phone_stand demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_phone_stand_ab2_ex2.log

# ----- [265/415] place_phone_stand_ab2_ex3 -----
place_phone_stand_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_phone_stand demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_phone_stand_ab2_ex3.log

# ----- [266/415] place_phone_stand_ab3_ex1 -----
place_phone_stand_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_phone_stand demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_phone_stand_ab3_ex1.log

# ----- [267/415] place_phone_stand_ab3_ex3 -----
place_phone_stand_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_phone_stand demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_phone_stand_ab3_ex3.log

# ----- [268/415] place_phone_stand_ab3_ex4 -----
place_phone_stand_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_phone_stand demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_phone_stand_ab3_ex4.log

# ----- [269/415] place_phone_stand_ab3_ex5 -----
place_phone_stand_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_phone_stand demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_phone_stand_ab3_ex5.log

# ----- [270/415] place_phone_stand_ab3_ex6 -----
place_phone_stand_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_phone_stand demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_phone_stand_ab3_ex6.log

# ----- [271/415] place_phone_stand_ab4_ex2 -----
place_phone_stand_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_phone_stand demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_phone_stand_ab4_ex2.log

# ----- [272/415] place_phone_stand_ab4_ex3 -----
place_phone_stand_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_phone_stand demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_phone_stand_ab4_ex3.log

# ----- [273/415] place_shoe_main -----
place_shoe_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_shoe demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_shoe_main.log

# ----- [274/415] place_shoe_ab1_ex1 -----
place_shoe_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_shoe demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/place_shoe_ab1_ex1.log

# ----- [275/415] place_shoe_ab1_ex2 -----
place_shoe_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_shoe demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/place_shoe_ab1_ex2.log

# ----- [276/415] place_shoe_ab1_ex3 -----
place_shoe_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_shoe demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/place_shoe_ab1_ex3.log

# ----- [277/415] place_shoe_ab2_ex2 -----
place_shoe_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_shoe demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/place_shoe_ab2_ex2.log

# ----- [278/415] place_shoe_ab2_ex3 -----
place_shoe_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_shoe demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/place_shoe_ab2_ex3.log

# ----- [279/415] place_shoe_ab3_ex1 -----
place_shoe_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_shoe demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/place_shoe_ab3_ex1.log

# ----- [280/415] place_shoe_ab3_ex3 -----
place_shoe_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_shoe demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/place_shoe_ab3_ex3.log

# ----- [281/415] place_shoe_ab3_ex4 -----
place_shoe_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_shoe demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/place_shoe_ab3_ex4.log

# ----- [282/415] place_shoe_ab3_ex5 -----
place_shoe_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_shoe demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/place_shoe_ab3_ex5.log

# ----- [283/415] place_shoe_ab3_ex6 -----
place_shoe_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_shoe demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/place_shoe_ab3_ex6.log

# ----- [284/415] place_shoe_ab4_ex2 -----
place_shoe_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_shoe demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/place_shoe_ab4_ex2.log

# ----- [285/415] place_shoe_ab4_ex3 -----
place_shoe_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh place_shoe demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/place_shoe_ab4_ex3.log

# ----- [286/415] press_stapler_main -----
press_stapler_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh press_stapler demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/press_stapler_main.log

# ----- [287/415] press_stapler_ab1_ex1 -----
press_stapler_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh press_stapler demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/press_stapler_ab1_ex1.log

# ----- [288/415] press_stapler_ab1_ex2 -----
press_stapler_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh press_stapler demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/press_stapler_ab1_ex2.log

# ----- [289/415] press_stapler_ab1_ex3 -----
press_stapler_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh press_stapler demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/press_stapler_ab1_ex3.log

# ----- [290/415] press_stapler_ab2_ex2 -----
press_stapler_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh press_stapler demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/press_stapler_ab2_ex2.log

# ----- [291/415] press_stapler_ab2_ex3 -----
press_stapler_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh press_stapler demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/press_stapler_ab2_ex3.log

# ----- [292/415] press_stapler_ab3_ex1 -----
press_stapler_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh press_stapler demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/press_stapler_ab3_ex1.log

# ----- [293/415] press_stapler_ab3_ex3 -----
press_stapler_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh press_stapler demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/press_stapler_ab3_ex3.log

# ----- [294/415] press_stapler_ab3_ex4 -----
press_stapler_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh press_stapler demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/press_stapler_ab3_ex4.log

# ----- [295/415] press_stapler_ab3_ex5 -----
press_stapler_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh press_stapler demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/press_stapler_ab3_ex5.log

# ----- [296/415] press_stapler_ab3_ex6 -----
press_stapler_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh press_stapler demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/press_stapler_ab3_ex6.log

# ----- [297/415] press_stapler_ab4_ex2 -----
press_stapler_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh press_stapler demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/press_stapler_ab4_ex2.log

# ----- [298/415] press_stapler_ab4_ex3 -----
press_stapler_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh press_stapler demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/press_stapler_ab4_ex3.log

# ----- [299/415] put_bottles_dustbin_main -----
put_bottles_dustbin_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh put_bottles_dustbin demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/put_bottles_dustbin_main.log

# ----- [300/415] put_bottles_dustbin_ab1_ex1 -----
put_bottles_dustbin_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab1_ex1.log

# ----- [301/415] put_bottles_dustbin_ab1_ex2 -----
put_bottles_dustbin_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab1_ex2.log

# ----- [302/415] put_bottles_dustbin_ab1_ex3 -----
put_bottles_dustbin_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab1_ex3.log

# ----- [303/415] put_bottles_dustbin_ab2_ex2 -----
put_bottles_dustbin_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab2_ex2.log

# ----- [304/415] put_bottles_dustbin_ab2_ex3 -----
put_bottles_dustbin_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab2_ex3.log

# ----- [305/415] put_bottles_dustbin_ab3_ex1 -----
put_bottles_dustbin_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab3_ex1.log

# ----- [306/415] put_bottles_dustbin_ab3_ex3 -----
put_bottles_dustbin_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab3_ex3.log

# ----- [307/415] put_bottles_dustbin_ab3_ex4 -----
put_bottles_dustbin_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab3_ex4.log

# ----- [308/415] put_bottles_dustbin_ab3_ex5 -----
put_bottles_dustbin_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab3_ex5.log

# ----- [309/415] put_bottles_dustbin_ab3_ex6 -----
put_bottles_dustbin_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab3_ex6.log

# ----- [310/415] put_bottles_dustbin_ab4_ex2 -----
put_bottles_dustbin_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab4_ex2.log

# ----- [311/415] put_bottles_dustbin_ab4_ex3 -----
put_bottles_dustbin_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh put_bottles_dustbin demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/put_bottles_dustbin_ab4_ex3.log

# ----- [312/415] rotate_qrcode_main -----
rotate_qrcode_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh rotate_qrcode demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/rotate_qrcode_main.log

# ----- [313/415] rotate_qrcode_ab1_ex1 -----
rotate_qrcode_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh rotate_qrcode demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/rotate_qrcode_ab1_ex1.log

# ----- [314/415] rotate_qrcode_ab1_ex2 -----
rotate_qrcode_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh rotate_qrcode demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/rotate_qrcode_ab1_ex2.log

# ----- [315/415] rotate_qrcode_ab1_ex3 -----
rotate_qrcode_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh rotate_qrcode demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/rotate_qrcode_ab1_ex3.log

# ----- [316/415] rotate_qrcode_ab2_ex2 -----
rotate_qrcode_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh rotate_qrcode demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/rotate_qrcode_ab2_ex2.log

# ----- [317/415] rotate_qrcode_ab2_ex3 -----
rotate_qrcode_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh rotate_qrcode demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/rotate_qrcode_ab2_ex3.log

# ----- [318/415] rotate_qrcode_ab3_ex1 -----
rotate_qrcode_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh rotate_qrcode demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/rotate_qrcode_ab3_ex1.log

# ----- [319/415] rotate_qrcode_ab3_ex3 -----
rotate_qrcode_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh rotate_qrcode demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/rotate_qrcode_ab3_ex3.log

# ----- [320/415] rotate_qrcode_ab3_ex4 -----
rotate_qrcode_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh rotate_qrcode demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/rotate_qrcode_ab3_ex4.log

# ----- [321/415] rotate_qrcode_ab3_ex5 -----
rotate_qrcode_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh rotate_qrcode demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/rotate_qrcode_ab3_ex5.log

# ----- [322/415] rotate_qrcode_ab3_ex6 -----
rotate_qrcode_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh rotate_qrcode demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/rotate_qrcode_ab3_ex6.log

# ----- [323/415] rotate_qrcode_ab4_ex2 -----
rotate_qrcode_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh rotate_qrcode demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/rotate_qrcode_ab4_ex2.log

# ----- [324/415] rotate_qrcode_ab4_ex3 -----
rotate_qrcode_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh rotate_qrcode demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/rotate_qrcode_ab4_ex3.log

# ----- [325/415] scan_object_main -----
scan_object_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/scan_object_main.log

# ----- [326/415] scan_object_ab1_ex1 -----
scan_object_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/scan_object_ab1_ex1.log

# ----- [327/415] scan_object_ab1_ex2 -----
scan_object_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/scan_object_ab1_ex2.log

# ----- [328/415] scan_object_ab1_ex3 -----
scan_object_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/scan_object_ab1_ex3.log

# ----- [329/415] scan_object_ab2_ex2 -----
scan_object_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/scan_object_ab2_ex2.log

# ----- [330/415] scan_object_ab2_ex3 -----
scan_object_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/scan_object_ab2_ex3.log

# ----- [331/415] scan_object_ab3_ex1 -----
scan_object_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/scan_object_ab3_ex1.log

# ----- [332/415] scan_object_ab3_ex3 -----
scan_object_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/scan_object_ab3_ex3.log

# ----- [333/415] scan_object_ab3_ex4 -----
scan_object_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/scan_object_ab3_ex4.log

# ----- [334/415] scan_object_ab3_ex5 -----
scan_object_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/scan_object_ab3_ex5.log

# ----- [335/415] scan_object_ab3_ex6 -----
scan_object_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/scan_object_ab3_ex6.log

# ----- [336/415] scan_object_ab4_ex2 -----
scan_object_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/scan_object_ab4_ex2.log

# ----- [337/415] scan_object_ab4_ex3 -----
scan_object_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh scan_object demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/scan_object_ab4_ex3.log

# ----- [338/415] shake_bottle_main -----
shake_bottle_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh shake_bottle demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/shake_bottle_main.log

# ----- [339/415] shake_bottle_ab1_ex1 -----
shake_bottle_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh shake_bottle demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/shake_bottle_ab1_ex1.log

# ----- [340/415] shake_bottle_ab1_ex2 -----
shake_bottle_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh shake_bottle demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/shake_bottle_ab1_ex2.log

# ----- [341/415] shake_bottle_ab1_ex3 -----
shake_bottle_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh shake_bottle demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/shake_bottle_ab1_ex3.log

# ----- [342/415] shake_bottle_ab2_ex2 -----
shake_bottle_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh shake_bottle demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/shake_bottle_ab2_ex2.log

# ----- [343/415] shake_bottle_ab2_ex3 -----
shake_bottle_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh shake_bottle demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/shake_bottle_ab2_ex3.log

# ----- [344/415] shake_bottle_ab3_ex1 -----
shake_bottle_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh shake_bottle demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/shake_bottle_ab3_ex1.log

# ----- [345/415] shake_bottle_ab3_ex3 -----
shake_bottle_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh shake_bottle demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/shake_bottle_ab3_ex3.log

# ----- [346/415] shake_bottle_ab3_ex4 -----
shake_bottle_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh shake_bottle demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/shake_bottle_ab3_ex4.log

# ----- [347/415] shake_bottle_ab3_ex5 -----
shake_bottle_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh shake_bottle demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/shake_bottle_ab3_ex5.log

# ----- [348/415] shake_bottle_ab3_ex6 -----
shake_bottle_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh shake_bottle demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/shake_bottle_ab3_ex6.log

# ----- [349/415] shake_bottle_ab4_ex2 -----
shake_bottle_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh shake_bottle demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/shake_bottle_ab4_ex2.log

# ----- [350/415] shake_bottle_ab4_ex3 -----
shake_bottle_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh shake_bottle demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/shake_bottle_ab4_ex3.log

# ----- [351/415] stack_blocks_three_main -----
stack_blocks_three_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_three demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/stack_blocks_three_main.log

# ----- [352/415] stack_blocks_three_ab1_ex1 -----
stack_blocks_three_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_three demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/stack_blocks_three_ab1_ex1.log

# ----- [353/415] stack_blocks_three_ab1_ex2 -----
stack_blocks_three_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_three demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/stack_blocks_three_ab1_ex2.log

# ----- [354/415] stack_blocks_three_ab1_ex3 -----
stack_blocks_three_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_three demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/stack_blocks_three_ab1_ex3.log

# ----- [355/415] stack_blocks_three_ab2_ex2 -----
stack_blocks_three_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_three demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/stack_blocks_three_ab2_ex2.log

# ----- [356/415] stack_blocks_three_ab2_ex3 -----
stack_blocks_three_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_three demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/stack_blocks_three_ab2_ex3.log

# ----- [357/415] stack_blocks_three_ab3_ex1 -----
stack_blocks_three_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_three demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/stack_blocks_three_ab3_ex1.log

# ----- [358/415] stack_blocks_three_ab3_ex3 -----
stack_blocks_three_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_three demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/stack_blocks_three_ab3_ex3.log

# ----- [359/415] stack_blocks_three_ab3_ex4 -----
stack_blocks_three_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_three demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/stack_blocks_three_ab3_ex4.log

# ----- [360/415] stack_blocks_three_ab3_ex5 -----
stack_blocks_three_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_three demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/stack_blocks_three_ab3_ex5.log

# ----- [361/415] stack_blocks_three_ab3_ex6 -----
stack_blocks_three_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_three demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/stack_blocks_three_ab3_ex6.log

# ----- [362/415] stack_blocks_three_ab4_ex2 -----
stack_blocks_three_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_three demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/stack_blocks_three_ab4_ex2.log

# ----- [363/415] stack_blocks_three_ab4_ex3 -----
stack_blocks_three_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_three demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/stack_blocks_three_ab4_ex3.log

# ----- [364/415] stack_blocks_two_main -----
stack_blocks_two_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_two demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/stack_blocks_two_main.log

# ----- [365/415] stack_blocks_two_ab1_ex1 -----
stack_blocks_two_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_two demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/stack_blocks_two_ab1_ex1.log

# ----- [366/415] stack_blocks_two_ab1_ex2 -----
stack_blocks_two_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_two demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/stack_blocks_two_ab1_ex2.log

# ----- [367/415] stack_blocks_two_ab1_ex3 -----
stack_blocks_two_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_two demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/stack_blocks_two_ab1_ex3.log

# ----- [368/415] stack_blocks_two_ab2_ex2 -----
stack_blocks_two_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_two demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/stack_blocks_two_ab2_ex2.log

# ----- [369/415] stack_blocks_two_ab2_ex3 -----
stack_blocks_two_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_two demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/stack_blocks_two_ab2_ex3.log

# ----- [370/415] stack_blocks_two_ab3_ex1 -----
stack_blocks_two_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_two demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/stack_blocks_two_ab3_ex1.log

# ----- [371/415] stack_blocks_two_ab3_ex3 -----
stack_blocks_two_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_two demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/stack_blocks_two_ab3_ex3.log

# ----- [372/415] stack_blocks_two_ab3_ex4 -----
stack_blocks_two_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_two demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/stack_blocks_two_ab3_ex4.log

# ----- [373/415] stack_blocks_two_ab3_ex5 -----
stack_blocks_two_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_two demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/stack_blocks_two_ab3_ex5.log

# ----- [374/415] stack_blocks_two_ab3_ex6 -----
stack_blocks_two_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_two demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/stack_blocks_two_ab3_ex6.log

# ----- [375/415] stack_blocks_two_ab4_ex2 -----
stack_blocks_two_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_two demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/stack_blocks_two_ab4_ex2.log

# ----- [376/415] stack_blocks_two_ab4_ex3 -----
stack_blocks_two_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_blocks_two demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/stack_blocks_two_ab4_ex3.log

# ----- [377/415] stack_bowls_three_main -----
stack_bowls_three_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_three demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/stack_bowls_three_main.log

# ----- [378/415] stack_bowls_three_ab1_ex1 -----
stack_bowls_three_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_three demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/stack_bowls_three_ab1_ex1.log

# ----- [379/415] stack_bowls_three_ab1_ex2 -----
stack_bowls_three_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_three demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/stack_bowls_three_ab1_ex2.log

# ----- [380/415] stack_bowls_three_ab1_ex3 -----
stack_bowls_three_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_three demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/stack_bowls_three_ab1_ex3.log

# ----- [381/415] stack_bowls_three_ab2_ex2 -----
stack_bowls_three_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_three demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/stack_bowls_three_ab2_ex2.log

# ----- [382/415] stack_bowls_three_ab2_ex3 -----
stack_bowls_three_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_three demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/stack_bowls_three_ab2_ex3.log

# ----- [383/415] stack_bowls_three_ab3_ex1 -----
stack_bowls_three_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_three demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/stack_bowls_three_ab3_ex1.log

# ----- [384/415] stack_bowls_three_ab3_ex3 -----
stack_bowls_three_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_three demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/stack_bowls_three_ab3_ex3.log

# ----- [385/415] stack_bowls_three_ab3_ex4 -----
stack_bowls_three_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_three demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/stack_bowls_three_ab3_ex4.log

# ----- [386/415] stack_bowls_three_ab3_ex5 -----
stack_bowls_three_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_three demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/stack_bowls_three_ab3_ex5.log

# ----- [387/415] stack_bowls_three_ab3_ex6 -----
stack_bowls_three_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_three demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/stack_bowls_three_ab3_ex6.log

# ----- [388/415] stack_bowls_three_ab4_ex2 -----
stack_bowls_three_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_three demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/stack_bowls_three_ab4_ex2.log

# ----- [389/415] stack_bowls_three_ab4_ex3 -----
stack_bowls_three_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_three demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/stack_bowls_three_ab4_ex3.log

# ----- [390/415] stack_bowls_two_main -----
stack_bowls_two_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_two demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/stack_bowls_two_main.log

# ----- [391/415] stack_bowls_two_ab1_ex1 -----
stack_bowls_two_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_two demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/stack_bowls_two_ab1_ex1.log

# ----- [392/415] stack_bowls_two_ab1_ex2 -----
stack_bowls_two_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_two demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/stack_bowls_two_ab1_ex2.log

# ----- [393/415] stack_bowls_two_ab1_ex3 -----
stack_bowls_two_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_two demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/stack_bowls_two_ab1_ex3.log

# ----- [394/415] stack_bowls_two_ab2_ex2 -----
stack_bowls_two_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_two demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/stack_bowls_two_ab2_ex2.log

# ----- [395/415] stack_bowls_two_ab2_ex3 -----
stack_bowls_two_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_two demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/stack_bowls_two_ab2_ex3.log

# ----- [396/415] stack_bowls_two_ab3_ex1 -----
stack_bowls_two_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_two demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/stack_bowls_two_ab3_ex1.log

# ----- [397/415] stack_bowls_two_ab3_ex3 -----
stack_bowls_two_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_two demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/stack_bowls_two_ab3_ex3.log

# ----- [398/415] stack_bowls_two_ab3_ex4 -----
stack_bowls_two_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_two demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/stack_bowls_two_ab3_ex4.log

# ----- [399/415] stack_bowls_two_ab3_ex5 -----
stack_bowls_two_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_two demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/stack_bowls_two_ab3_ex5.log

# ----- [400/415] stack_bowls_two_ab3_ex6 -----
stack_bowls_two_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_two demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/stack_bowls_two_ab3_ex6.log

# ----- [401/415] stack_bowls_two_ab4_ex2 -----
stack_bowls_two_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_two demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/stack_bowls_two_ab4_ex2.log

# ----- [402/415] stack_bowls_two_ab4_ex3 -----
stack_bowls_two_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh stack_bowls_two demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/stack_bowls_two_ab4_ex3.log

# ----- [403/415] turn_switch_main -----
turn_switch_main
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh turn_switch demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/turn_switch_main.log

# ----- [404/415] turn_switch_ab1_ex1 -----
turn_switch_ab1_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh turn_switch demo_clean 100 0 0 ab1_ex1 2>&1 | tee ../../Logs/train/turn_switch_ab1_ex1.log

# ----- [405/415] turn_switch_ab1_ex2 -----
turn_switch_ab1_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh turn_switch demo_clean 100 0 0 ab1_ex2 2>&1 | tee ../../Logs/train/turn_switch_ab1_ex2.log

# ----- [406/415] turn_switch_ab1_ex3 -----
turn_switch_ab1_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh turn_switch demo_clean 100 0 0 ab1_ex3 2>&1 | tee ../../Logs/train/turn_switch_ab1_ex3.log

# ----- [407/415] turn_switch_ab2_ex2 -----
turn_switch_ab2_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh turn_switch demo_clean 100 0 0 ab2_ex2 2>&1 | tee ../../Logs/train/turn_switch_ab2_ex2.log

# ----- [408/415] turn_switch_ab2_ex3 -----
turn_switch_ab2_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh turn_switch demo_clean 100 0 0 ab2_ex3 2>&1 | tee ../../Logs/train/turn_switch_ab2_ex3.log

# ----- [409/415] turn_switch_ab3_ex1 -----
turn_switch_ab3_ex1
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh turn_switch demo_clean 100 0 0 ab3_ex1 2>&1 | tee ../../Logs/train/turn_switch_ab3_ex1.log

# ----- [410/415] turn_switch_ab3_ex3 -----
turn_switch_ab3_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh turn_switch demo_clean 100 0 0 ab3_ex3 2>&1 | tee ../../Logs/train/turn_switch_ab3_ex3.log

# ----- [411/415] turn_switch_ab3_ex4 -----
turn_switch_ab3_ex4
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh turn_switch demo_clean 100 0 0 ab3_ex4 2>&1 | tee ../../Logs/train/turn_switch_ab3_ex4.log

# ----- [412/415] turn_switch_ab3_ex5 -----
turn_switch_ab3_ex5
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh turn_switch demo_clean 100 0 0 ab3_ex5 2>&1 | tee ../../Logs/train/turn_switch_ab3_ex5.log

# ----- [413/415] turn_switch_ab3_ex6 -----
turn_switch_ab3_ex6
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh turn_switch demo_clean 100 0 0 ab3_ex6 2>&1 | tee ../../Logs/train/turn_switch_ab3_ex6.log

# ----- [414/415] turn_switch_ab4_ex2 -----
turn_switch_ab4_ex2
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh turn_switch demo_clean 100 0 0 ab4_ex2 2>&1 | tee ../../Logs/train/turn_switch_ab4_ex2.log

# ----- [415/415] turn_switch_ab4_ex3 -----
turn_switch_ab4_ex3
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
bash train.sh turn_switch demo_clean 100 0 0 ab4_ex3 2>&1 | tee ../../Logs/train/turn_switch_ab4_ex3.log
