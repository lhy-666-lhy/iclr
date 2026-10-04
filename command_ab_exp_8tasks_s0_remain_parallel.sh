# Ablation-exp REMAIN parallel | seed=0 | demo_clean | n=100
# Remain: 37 | NO serial | 1 train / GPU
# Layout: 5 platform jobs = 4×8GPU(full) + 1×8GPU(5 trains)
# Alt 4GPU split also listed at bottom as comment reference
# Submit: name = job line; paste #!/bin/sh body; pick 8-GPU node
# ckpt: policy/SPACE/checkpoints/<task>-train_<enc>-100_0/
# log:  Logs/train/<task>_<enc>_s0.log

# ----- [0/4] abexp_par_n0_s0  [8GPU]  8 parallel -----
abexp_par_n0_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par_n0_s0 launch 8 parallel"
(
  echo "[$(date)] start handover_block ab_exp1 seed=0 gpu=0"
  bash train.sh handover_block demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/handover_block_ab_exp1_s0.log
  echo "[$(date)] done  handover_block ab_exp1 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp1 seed=0 gpu=1"
  bash train.sh hanging_mug demo_clean 100 0 1 ab_exp1 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp1_s0.log
  echo "[$(date)] done  hanging_mug ab_exp1 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp1 seed=0 gpu=2"
  bash train.sh place_object_basket demo_clean 100 0 2 ab_exp1 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp1_s0.log
  echo "[$(date)] done  place_object_basket ab_exp1 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp1 seed=0 gpu=3"
  bash train.sh scan_object demo_clean 100 0 3 ab_exp1 2>&1 | tee ../../Logs/train/scan_object_ab_exp1_s0.log
  echo "[$(date)] done  scan_object ab_exp1 seed=0 gpu=3 exit=$?"
) &
(
  echo "[$(date)] start pick_dual_bottles ab_exp1 seed=0 gpu=4"
  bash train.sh pick_dual_bottles demo_clean 100 0 4 ab_exp1 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab_exp1_s0.log
  echo "[$(date)] done  pick_dual_bottles ab_exp1 seed=0 gpu=4 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp1 seed=0 gpu=5"
  bash train.sh place_bread_skillet demo_clean 100 0 5 ab_exp1 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp1_s0.log
  echo "[$(date)] done  place_bread_skillet ab_exp1 seed=0 gpu=5 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp1 seed=0 gpu=6"
  bash train.sh place_bread_basket demo_clean 100 0 6 ab_exp1 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp1_s0.log
  echo "[$(date)] done  place_bread_basket ab_exp1 seed=0 gpu=6 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp1 seed=0 gpu=7"
  bash train.sh place_cans_plasticbox demo_clean 100 0 7 ab_exp1 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp1_s0.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp1 seed=0 gpu=7 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par_n0_s0 ALL_DONE"

# map: gpu0=handover_block:ab_exp1, gpu1=hanging_mug:ab_exp1, gpu2=place_object_basket:ab_exp1, gpu3=scan_object:ab_exp1, gpu4=pick_dual_bottles:ab_exp1, gpu5=place_bread_skillet:ab_exp1, gpu6=place_bread_basket:ab_exp1, gpu7=place_cans_plasticbox:ab_exp1

# ----- [1/4] abexp_par_n1_s0  [8GPU]  8 parallel -----
abexp_par_n1_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par_n1_s0 launch 8 parallel"
(
  echo "[$(date)] start hanging_mug ab_exp2 seed=0 gpu=0"
  bash train.sh hanging_mug demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp2_s0.log
  echo "[$(date)] done  hanging_mug ab_exp2 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp2 seed=0 gpu=1"
  bash train.sh place_object_basket demo_clean 100 0 1 ab_exp2 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp2_s0.log
  echo "[$(date)] done  place_object_basket ab_exp2 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp2 seed=0 gpu=2"
  bash train.sh scan_object demo_clean 100 0 2 ab_exp2 2>&1 | tee ../../Logs/train/scan_object_ab_exp2_s0.log
  echo "[$(date)] done  scan_object ab_exp2 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp2 seed=0 gpu=3"
  bash train.sh place_bread_basket demo_clean 100 0 3 ab_exp2 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp2_s0.log
  echo "[$(date)] done  place_bread_basket ab_exp2 seed=0 gpu=3 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp2 seed=0 gpu=4"
  bash train.sh place_cans_plasticbox demo_clean 100 0 4 ab_exp2 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp2_s0.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp2 seed=0 gpu=4 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp3 seed=0 gpu=5"
  bash train.sh hanging_mug demo_clean 100 0 5 ab_exp3 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp3_s0.log
  echo "[$(date)] done  hanging_mug ab_exp3 seed=0 gpu=5 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp3 seed=0 gpu=6"
  bash train.sh place_object_basket demo_clean 100 0 6 ab_exp3 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp3_s0.log
  echo "[$(date)] done  place_object_basket ab_exp3 seed=0 gpu=6 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp3 seed=0 gpu=7"
  bash train.sh scan_object demo_clean 100 0 7 ab_exp3 2>&1 | tee ../../Logs/train/scan_object_ab_exp3_s0.log
  echo "[$(date)] done  scan_object ab_exp3 seed=0 gpu=7 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par_n1_s0 ALL_DONE"

# map: gpu0=hanging_mug:ab_exp2, gpu1=place_object_basket:ab_exp2, gpu2=scan_object:ab_exp2, gpu3=place_bread_basket:ab_exp2, gpu4=place_cans_plasticbox:ab_exp2, gpu5=hanging_mug:ab_exp3, gpu6=place_object_basket:ab_exp3, gpu7=scan_object:ab_exp3

# ----- [2/4] abexp_par_n2_s0  [8GPU]  8 parallel -----
abexp_par_n2_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par_n2_s0 launch 8 parallel"
(
  echo "[$(date)] start place_bread_basket ab_exp3 seed=0 gpu=0"
  bash train.sh place_bread_basket demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp3_s0.log
  echo "[$(date)] done  place_bread_basket ab_exp3 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp3 seed=0 gpu=1"
  bash train.sh place_cans_plasticbox demo_clean 100 0 1 ab_exp3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp3_s0.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp3 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp4 seed=0 gpu=2"
  bash train.sh hanging_mug demo_clean 100 0 2 ab_exp4 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp4_s0.log
  echo "[$(date)] done  hanging_mug ab_exp4 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp4 seed=0 gpu=3"
  bash train.sh place_object_basket demo_clean 100 0 3 ab_exp4 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp4_s0.log
  echo "[$(date)] done  place_object_basket ab_exp4 seed=0 gpu=3 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp4 seed=0 gpu=4"
  bash train.sh scan_object demo_clean 100 0 4 ab_exp4 2>&1 | tee ../../Logs/train/scan_object_ab_exp4_s0.log
  echo "[$(date)] done  scan_object ab_exp4 seed=0 gpu=4 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp4 seed=0 gpu=5"
  bash train.sh place_bread_skillet demo_clean 100 0 5 ab_exp4 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp4_s0.log
  echo "[$(date)] done  place_bread_skillet ab_exp4 seed=0 gpu=5 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp4 seed=0 gpu=6"
  bash train.sh place_bread_basket demo_clean 100 0 6 ab_exp4 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp4_s0.log
  echo "[$(date)] done  place_bread_basket ab_exp4 seed=0 gpu=6 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp4 seed=0 gpu=7"
  bash train.sh place_cans_plasticbox demo_clean 100 0 7 ab_exp4 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp4_s0.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp4 seed=0 gpu=7 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par_n2_s0 ALL_DONE"

# map: gpu0=place_bread_basket:ab_exp3, gpu1=place_cans_plasticbox:ab_exp3, gpu2=hanging_mug:ab_exp4, gpu3=place_object_basket:ab_exp4, gpu4=scan_object:ab_exp4, gpu5=place_bread_skillet:ab_exp4, gpu6=place_bread_basket:ab_exp4, gpu7=place_cans_plasticbox:ab_exp4

# ----- [3/4] abexp_par_n3_s0  [8GPU]  8 parallel -----
abexp_par_n3_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par_n3_s0 launch 8 parallel"
(
  echo "[$(date)] start handover_block ab_exp5 seed=0 gpu=0"
  bash train.sh handover_block demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/handover_block_ab_exp5_s0.log
  echo "[$(date)] done  handover_block ab_exp5 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp5 seed=0 gpu=1"
  bash train.sh hanging_mug demo_clean 100 0 1 ab_exp5 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp5_s0.log
  echo "[$(date)] done  hanging_mug ab_exp5 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp5 seed=0 gpu=2"
  bash train.sh place_object_basket demo_clean 100 0 2 ab_exp5 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp5_s0.log
  echo "[$(date)] done  place_object_basket ab_exp5 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp5 seed=0 gpu=3"
  bash train.sh scan_object demo_clean 100 0 3 ab_exp5 2>&1 | tee ../../Logs/train/scan_object_ab_exp5_s0.log
  echo "[$(date)] done  scan_object ab_exp5 seed=0 gpu=3 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp5 seed=0 gpu=4"
  bash train.sh place_bread_basket demo_clean 100 0 4 ab_exp5 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp5_s0.log
  echo "[$(date)] done  place_bread_basket ab_exp5 seed=0 gpu=4 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp5 seed=0 gpu=5"
  bash train.sh place_cans_plasticbox demo_clean 100 0 5 ab_exp5 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp5_s0.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp5 seed=0 gpu=5 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp6 seed=0 gpu=6"
  bash train.sh hanging_mug demo_clean 100 0 6 ab_exp6 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp6_s0.log
  echo "[$(date)] done  hanging_mug ab_exp6 seed=0 gpu=6 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp6 seed=0 gpu=7"
  bash train.sh place_object_basket demo_clean 100 0 7 ab_exp6 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp6_s0.log
  echo "[$(date)] done  place_object_basket ab_exp6 seed=0 gpu=7 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par_n3_s0 ALL_DONE"

# map: gpu0=handover_block:ab_exp5, gpu1=hanging_mug:ab_exp5, gpu2=place_object_basket:ab_exp5, gpu3=scan_object:ab_exp5, gpu4=place_bread_basket:ab_exp5, gpu5=place_cans_plasticbox:ab_exp5, gpu6=hanging_mug:ab_exp6, gpu7=place_object_basket:ab_exp6

# ----- [4/4] abexp_par_n4_s0  [8GPU]  5 parallel -----
abexp_par_n4_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par_n4_s0 launch 5 parallel"
(
  echo "[$(date)] start scan_object ab_exp6 seed=0 gpu=0"
  bash train.sh scan_object demo_clean 100 0 0 ab_exp6 2>&1 | tee ../../Logs/train/scan_object_ab_exp6_s0.log
  echo "[$(date)] done  scan_object ab_exp6 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start pick_dual_bottles ab_exp6 seed=0 gpu=1"
  bash train.sh pick_dual_bottles demo_clean 100 0 1 ab_exp6 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab_exp6_s0.log
  echo "[$(date)] done  pick_dual_bottles ab_exp6 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp6 seed=0 gpu=2"
  bash train.sh place_bread_skillet demo_clean 100 0 2 ab_exp6 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp6_s0.log
  echo "[$(date)] done  place_bread_skillet ab_exp6 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp6 seed=0 gpu=3"
  bash train.sh place_bread_basket demo_clean 100 0 3 ab_exp6 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp6_s0.log
  echo "[$(date)] done  place_bread_basket ab_exp6 seed=0 gpu=3 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp6 seed=0 gpu=4"
  bash train.sh place_cans_plasticbox demo_clean 100 0 4 ab_exp6 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp6_s0.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp6 seed=0 gpu=4 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par_n4_s0 ALL_DONE"

# map: gpu0=scan_object:ab_exp6, gpu1=pick_dual_bottles:ab_exp6, gpu2=place_bread_skillet:ab_exp6, gpu3=place_bread_basket:ab_exp6, gpu4=place_cans_plasticbox:ab_exp6

# ========== ALTERNATIVE: pure 4-GPU nodes (10 jobs) ==========
# 9×4GPU full(36) + 1×4GPU with 1 train = 37

# ----- ALT [0/9] abexp_par4_n0_s0  [4GPU]  4 parallel -----
abexp_par4_n0_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par4_n0_s0 launch 4 parallel"
(
  echo "[$(date)] start handover_block ab_exp1 seed=0 gpu=0"
  bash train.sh handover_block demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/handover_block_ab_exp1_s0.log
  echo "[$(date)] done  handover_block ab_exp1 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp1 seed=0 gpu=1"
  bash train.sh hanging_mug demo_clean 100 0 1 ab_exp1 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp1_s0.log
  echo "[$(date)] done  hanging_mug ab_exp1 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp1 seed=0 gpu=2"
  bash train.sh place_object_basket demo_clean 100 0 2 ab_exp1 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp1_s0.log
  echo "[$(date)] done  place_object_basket ab_exp1 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp1 seed=0 gpu=3"
  bash train.sh scan_object demo_clean 100 0 3 ab_exp1 2>&1 | tee ../../Logs/train/scan_object_ab_exp1_s0.log
  echo "[$(date)] done  scan_object ab_exp1 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par4_n0_s0 ALL_DONE"

# map: gpu0=handover_block:ab_exp1, gpu1=hanging_mug:ab_exp1, gpu2=place_object_basket:ab_exp1, gpu3=scan_object:ab_exp1

# ----- ALT [1/9] abexp_par4_n1_s0  [4GPU]  4 parallel -----
abexp_par4_n1_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par4_n1_s0 launch 4 parallel"
(
  echo "[$(date)] start pick_dual_bottles ab_exp1 seed=0 gpu=0"
  bash train.sh pick_dual_bottles demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab_exp1_s0.log
  echo "[$(date)] done  pick_dual_bottles ab_exp1 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp1 seed=0 gpu=1"
  bash train.sh place_bread_skillet demo_clean 100 0 1 ab_exp1 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp1_s0.log
  echo "[$(date)] done  place_bread_skillet ab_exp1 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp1 seed=0 gpu=2"
  bash train.sh place_bread_basket demo_clean 100 0 2 ab_exp1 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp1_s0.log
  echo "[$(date)] done  place_bread_basket ab_exp1 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp1 seed=0 gpu=3"
  bash train.sh place_cans_plasticbox demo_clean 100 0 3 ab_exp1 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp1_s0.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp1 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par4_n1_s0 ALL_DONE"

# map: gpu0=pick_dual_bottles:ab_exp1, gpu1=place_bread_skillet:ab_exp1, gpu2=place_bread_basket:ab_exp1, gpu3=place_cans_plasticbox:ab_exp1

# ----- ALT [2/9] abexp_par4_n2_s0  [4GPU]  4 parallel -----
abexp_par4_n2_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par4_n2_s0 launch 4 parallel"
(
  echo "[$(date)] start hanging_mug ab_exp2 seed=0 gpu=0"
  bash train.sh hanging_mug demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp2_s0.log
  echo "[$(date)] done  hanging_mug ab_exp2 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp2 seed=0 gpu=1"
  bash train.sh place_object_basket demo_clean 100 0 1 ab_exp2 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp2_s0.log
  echo "[$(date)] done  place_object_basket ab_exp2 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp2 seed=0 gpu=2"
  bash train.sh scan_object demo_clean 100 0 2 ab_exp2 2>&1 | tee ../../Logs/train/scan_object_ab_exp2_s0.log
  echo "[$(date)] done  scan_object ab_exp2 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp2 seed=0 gpu=3"
  bash train.sh place_bread_basket demo_clean 100 0 3 ab_exp2 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp2_s0.log
  echo "[$(date)] done  place_bread_basket ab_exp2 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par4_n2_s0 ALL_DONE"

# map: gpu0=hanging_mug:ab_exp2, gpu1=place_object_basket:ab_exp2, gpu2=scan_object:ab_exp2, gpu3=place_bread_basket:ab_exp2

# ----- ALT [3/9] abexp_par4_n3_s0  [4GPU]  4 parallel -----
abexp_par4_n3_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par4_n3_s0 launch 4 parallel"
(
  echo "[$(date)] start place_cans_plasticbox ab_exp2 seed=0 gpu=0"
  bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp2_s0.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp2 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp3 seed=0 gpu=1"
  bash train.sh hanging_mug demo_clean 100 0 1 ab_exp3 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp3_s0.log
  echo "[$(date)] done  hanging_mug ab_exp3 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp3 seed=0 gpu=2"
  bash train.sh place_object_basket demo_clean 100 0 2 ab_exp3 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp3_s0.log
  echo "[$(date)] done  place_object_basket ab_exp3 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp3 seed=0 gpu=3"
  bash train.sh scan_object demo_clean 100 0 3 ab_exp3 2>&1 | tee ../../Logs/train/scan_object_ab_exp3_s0.log
  echo "[$(date)] done  scan_object ab_exp3 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par4_n3_s0 ALL_DONE"

# map: gpu0=place_cans_plasticbox:ab_exp2, gpu1=hanging_mug:ab_exp3, gpu2=place_object_basket:ab_exp3, gpu3=scan_object:ab_exp3

# ----- ALT [4/9] abexp_par4_n4_s0  [4GPU]  4 parallel -----
abexp_par4_n4_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par4_n4_s0 launch 4 parallel"
(
  echo "[$(date)] start place_bread_basket ab_exp3 seed=0 gpu=0"
  bash train.sh place_bread_basket demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp3_s0.log
  echo "[$(date)] done  place_bread_basket ab_exp3 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp3 seed=0 gpu=1"
  bash train.sh place_cans_plasticbox demo_clean 100 0 1 ab_exp3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp3_s0.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp3 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp4 seed=0 gpu=2"
  bash train.sh hanging_mug demo_clean 100 0 2 ab_exp4 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp4_s0.log
  echo "[$(date)] done  hanging_mug ab_exp4 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp4 seed=0 gpu=3"
  bash train.sh place_object_basket demo_clean 100 0 3 ab_exp4 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp4_s0.log
  echo "[$(date)] done  place_object_basket ab_exp4 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par4_n4_s0 ALL_DONE"

# map: gpu0=place_bread_basket:ab_exp3, gpu1=place_cans_plasticbox:ab_exp3, gpu2=hanging_mug:ab_exp4, gpu3=place_object_basket:ab_exp4

# ----- ALT [5/9] abexp_par4_n5_s0  [4GPU]  4 parallel -----
abexp_par4_n5_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par4_n5_s0 launch 4 parallel"
(
  echo "[$(date)] start scan_object ab_exp4 seed=0 gpu=0"
  bash train.sh scan_object demo_clean 100 0 0 ab_exp4 2>&1 | tee ../../Logs/train/scan_object_ab_exp4_s0.log
  echo "[$(date)] done  scan_object ab_exp4 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp4 seed=0 gpu=1"
  bash train.sh place_bread_skillet demo_clean 100 0 1 ab_exp4 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp4_s0.log
  echo "[$(date)] done  place_bread_skillet ab_exp4 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp4 seed=0 gpu=2"
  bash train.sh place_bread_basket demo_clean 100 0 2 ab_exp4 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp4_s0.log
  echo "[$(date)] done  place_bread_basket ab_exp4 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp4 seed=0 gpu=3"
  bash train.sh place_cans_plasticbox demo_clean 100 0 3 ab_exp4 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp4_s0.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp4 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par4_n5_s0 ALL_DONE"

# map: gpu0=scan_object:ab_exp4, gpu1=place_bread_skillet:ab_exp4, gpu2=place_bread_basket:ab_exp4, gpu3=place_cans_plasticbox:ab_exp4

# ----- ALT [6/9] abexp_par4_n6_s0  [4GPU]  4 parallel -----
abexp_par4_n6_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par4_n6_s0 launch 4 parallel"
(
  echo "[$(date)] start handover_block ab_exp5 seed=0 gpu=0"
  bash train.sh handover_block demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/handover_block_ab_exp5_s0.log
  echo "[$(date)] done  handover_block ab_exp5 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp5 seed=0 gpu=1"
  bash train.sh hanging_mug demo_clean 100 0 1 ab_exp5 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp5_s0.log
  echo "[$(date)] done  hanging_mug ab_exp5 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp5 seed=0 gpu=2"
  bash train.sh place_object_basket demo_clean 100 0 2 ab_exp5 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp5_s0.log
  echo "[$(date)] done  place_object_basket ab_exp5 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp5 seed=0 gpu=3"
  bash train.sh scan_object demo_clean 100 0 3 ab_exp5 2>&1 | tee ../../Logs/train/scan_object_ab_exp5_s0.log
  echo "[$(date)] done  scan_object ab_exp5 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par4_n6_s0 ALL_DONE"

# map: gpu0=handover_block:ab_exp5, gpu1=hanging_mug:ab_exp5, gpu2=place_object_basket:ab_exp5, gpu3=scan_object:ab_exp5

# ----- ALT [7/9] abexp_par4_n7_s0  [4GPU]  4 parallel -----
abexp_par4_n7_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par4_n7_s0 launch 4 parallel"
(
  echo "[$(date)] start place_bread_basket ab_exp5 seed=0 gpu=0"
  bash train.sh place_bread_basket demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp5_s0.log
  echo "[$(date)] done  place_bread_basket ab_exp5 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp5 seed=0 gpu=1"
  bash train.sh place_cans_plasticbox demo_clean 100 0 1 ab_exp5 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp5_s0.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp5 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp6 seed=0 gpu=2"
  bash train.sh hanging_mug demo_clean 100 0 2 ab_exp6 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp6_s0.log
  echo "[$(date)] done  hanging_mug ab_exp6 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp6 seed=0 gpu=3"
  bash train.sh place_object_basket demo_clean 100 0 3 ab_exp6 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp6_s0.log
  echo "[$(date)] done  place_object_basket ab_exp6 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par4_n7_s0 ALL_DONE"

# map: gpu0=place_bread_basket:ab_exp5, gpu1=place_cans_plasticbox:ab_exp5, gpu2=hanging_mug:ab_exp6, gpu3=place_object_basket:ab_exp6

# ----- ALT [8/9] abexp_par4_n8_s0  [4GPU]  4 parallel -----
abexp_par4_n8_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par4_n8_s0 launch 4 parallel"
(
  echo "[$(date)] start scan_object ab_exp6 seed=0 gpu=0"
  bash train.sh scan_object demo_clean 100 0 0 ab_exp6 2>&1 | tee ../../Logs/train/scan_object_ab_exp6_s0.log
  echo "[$(date)] done  scan_object ab_exp6 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start pick_dual_bottles ab_exp6 seed=0 gpu=1"
  bash train.sh pick_dual_bottles demo_clean 100 0 1 ab_exp6 2>&1 | tee ../../Logs/train/pick_dual_bottles_ab_exp6_s0.log
  echo "[$(date)] done  pick_dual_bottles ab_exp6 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp6 seed=0 gpu=2"
  bash train.sh place_bread_skillet demo_clean 100 0 2 ab_exp6 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp6_s0.log
  echo "[$(date)] done  place_bread_skillet ab_exp6 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp6 seed=0 gpu=3"
  bash train.sh place_bread_basket demo_clean 100 0 3 ab_exp6 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp6_s0.log
  echo "[$(date)] done  place_bread_basket ab_exp6 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par4_n8_s0 ALL_DONE"

# map: gpu0=scan_object:ab_exp6, gpu1=pick_dual_bottles:ab_exp6, gpu2=place_bread_skillet:ab_exp6, gpu3=place_bread_basket:ab_exp6

# ----- ALT [9/9] abexp_par4_n9_s0  [4GPU]  1 parallel -----
abexp_par4_n9_s0
#!/bin/sh
export WANDB_MODE=offline
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp_par4_n9_s0 launch 1 parallel"
(
  echo "[$(date)] start place_cans_plasticbox ab_exp6 seed=0 gpu=0"
  bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab_exp6 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp6_s0.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp6 seed=0 gpu=0 exit=$?"
) &
wait
echo "[$(date)] node=abexp_par4_n9_s0 ALL_DONE"

# map: gpu0=place_cans_plasticbox:ab_exp6

