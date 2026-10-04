# Ablation-exp 6tasks × ab_exp1..6 | seed=0 | demo_clean | n=100 | epochs=500
# SKIP: handover_block, pick_dual_bottles
# Tasks: hanging_mug place_object_basket scan_object place_bread_skillet place_bread_basket place_cans_plasticbox
# Trainings: 6×6 = 36 | NO serial | 1 train / GPU
# Layout (recommended): 4×8GPU + 1×4GPU
# Saves: checkpoints/<task>-train_<enc>-100_0/500.ckpt
# Log:   Logs/train/<task>_<enc>_s0_e500.log
# Requires train_policy.sh env: NUM_EPOCHS / CHECKPOINT_EVERY / RESUME
# Submit: name=job line; paste #!/bin/sh body; pick 8GPU or 4GPU as labeled

# ----- [0/4] abexp6t_e500_n0_s0  [8GPU]  8 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_n0_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_n0_s0 launch 8 parallel epochs=500"
(
  echo "[$(date)] start hanging_mug ab_exp1 seed=0 gpu=0 epochs=500"
  bash train.sh hanging_mug demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp1_s0_e500.log
  echo "[$(date)] done  hanging_mug ab_exp1 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp1 seed=0 gpu=1 epochs=500"
  bash train.sh place_object_basket demo_clean 100 0 1 ab_exp1 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp1_s0_e500.log
  echo "[$(date)] done  place_object_basket ab_exp1 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp1 seed=0 gpu=2 epochs=500"
  bash train.sh scan_object demo_clean 100 0 2 ab_exp1 2>&1 | tee ../../Logs/train/scan_object_ab_exp1_s0_e500.log
  echo "[$(date)] done  scan_object ab_exp1 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp1 seed=0 gpu=3 epochs=500"
  bash train.sh place_bread_skillet demo_clean 100 0 3 ab_exp1 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp1_s0_e500.log
  echo "[$(date)] done  place_bread_skillet ab_exp1 seed=0 gpu=3 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp1 seed=0 gpu=4 epochs=500"
  bash train.sh place_bread_basket demo_clean 100 0 4 ab_exp1 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp1_s0_e500.log
  echo "[$(date)] done  place_bread_basket ab_exp1 seed=0 gpu=4 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp1 seed=0 gpu=5 epochs=500"
  bash train.sh place_cans_plasticbox demo_clean 100 0 5 ab_exp1 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp1_s0_e500.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp1 seed=0 gpu=5 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp2 seed=0 gpu=6 epochs=500"
  bash train.sh hanging_mug demo_clean 100 0 6 ab_exp2 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp2_s0_e500.log
  echo "[$(date)] done  hanging_mug ab_exp2 seed=0 gpu=6 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp2 seed=0 gpu=7 epochs=500"
  bash train.sh place_object_basket demo_clean 100 0 7 ab_exp2 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp2_s0_e500.log
  echo "[$(date)] done  place_object_basket ab_exp2 seed=0 gpu=7 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_n0_s0 ALL_DONE"

# map: gpu0=hanging_mug:ab_exp1, gpu1=place_object_basket:ab_exp1, gpu2=scan_object:ab_exp1, gpu3=place_bread_skillet:ab_exp1, gpu4=place_bread_basket:ab_exp1, gpu5=place_cans_plasticbox:ab_exp1, gpu6=hanging_mug:ab_exp2, gpu7=place_object_basket:ab_exp2

# ----- [1/4] abexp6t_e500_n1_s0  [8GPU]  8 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_n1_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_n1_s0 launch 8 parallel epochs=500"
(
  echo "[$(date)] start scan_object ab_exp2 seed=0 gpu=0 epochs=500"
  bash train.sh scan_object demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/scan_object_ab_exp2_s0_e500.log
  echo "[$(date)] done  scan_object ab_exp2 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp2 seed=0 gpu=1 epochs=500"
  bash train.sh place_bread_skillet demo_clean 100 0 1 ab_exp2 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp2_s0_e500.log
  echo "[$(date)] done  place_bread_skillet ab_exp2 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp2 seed=0 gpu=2 epochs=500"
  bash train.sh place_bread_basket demo_clean 100 0 2 ab_exp2 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp2_s0_e500.log
  echo "[$(date)] done  place_bread_basket ab_exp2 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp2 seed=0 gpu=3 epochs=500"
  bash train.sh place_cans_plasticbox demo_clean 100 0 3 ab_exp2 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp2_s0_e500.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp2 seed=0 gpu=3 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp3 seed=0 gpu=4 epochs=500"
  bash train.sh hanging_mug demo_clean 100 0 4 ab_exp3 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp3_s0_e500.log
  echo "[$(date)] done  hanging_mug ab_exp3 seed=0 gpu=4 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp3 seed=0 gpu=5 epochs=500"
  bash train.sh place_object_basket demo_clean 100 0 5 ab_exp3 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp3_s0_e500.log
  echo "[$(date)] done  place_object_basket ab_exp3 seed=0 gpu=5 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp3 seed=0 gpu=6 epochs=500"
  bash train.sh scan_object demo_clean 100 0 6 ab_exp3 2>&1 | tee ../../Logs/train/scan_object_ab_exp3_s0_e500.log
  echo "[$(date)] done  scan_object ab_exp3 seed=0 gpu=6 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp3 seed=0 gpu=7 epochs=500"
  bash train.sh place_bread_skillet demo_clean 100 0 7 ab_exp3 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp3_s0_e500.log
  echo "[$(date)] done  place_bread_skillet ab_exp3 seed=0 gpu=7 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_n1_s0 ALL_DONE"

# map: gpu0=scan_object:ab_exp2, gpu1=place_bread_skillet:ab_exp2, gpu2=place_bread_basket:ab_exp2, gpu3=place_cans_plasticbox:ab_exp2, gpu4=hanging_mug:ab_exp3, gpu5=place_object_basket:ab_exp3, gpu6=scan_object:ab_exp3, gpu7=place_bread_skillet:ab_exp3

# ----- [2/4] abexp6t_e500_n2_s0  [8GPU]  8 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_n2_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_n2_s0 launch 8 parallel epochs=500"
(
  echo "[$(date)] start place_bread_basket ab_exp3 seed=0 gpu=0 epochs=500"
  bash train.sh place_bread_basket demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp3_s0_e500.log
  echo "[$(date)] done  place_bread_basket ab_exp3 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp3 seed=0 gpu=1 epochs=500"
  bash train.sh place_cans_plasticbox demo_clean 100 0 1 ab_exp3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp3_s0_e500.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp3 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp4 seed=0 gpu=2 epochs=500"
  bash train.sh hanging_mug demo_clean 100 0 2 ab_exp4 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp4_s0_e500.log
  echo "[$(date)] done  hanging_mug ab_exp4 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp4 seed=0 gpu=3 epochs=500"
  bash train.sh place_object_basket demo_clean 100 0 3 ab_exp4 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp4_s0_e500.log
  echo "[$(date)] done  place_object_basket ab_exp4 seed=0 gpu=3 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp4 seed=0 gpu=4 epochs=500"
  bash train.sh scan_object demo_clean 100 0 4 ab_exp4 2>&1 | tee ../../Logs/train/scan_object_ab_exp4_s0_e500.log
  echo "[$(date)] done  scan_object ab_exp4 seed=0 gpu=4 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp4 seed=0 gpu=5 epochs=500"
  bash train.sh place_bread_skillet demo_clean 100 0 5 ab_exp4 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp4_s0_e500.log
  echo "[$(date)] done  place_bread_skillet ab_exp4 seed=0 gpu=5 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp4 seed=0 gpu=6 epochs=500"
  bash train.sh place_bread_basket demo_clean 100 0 6 ab_exp4 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp4_s0_e500.log
  echo "[$(date)] done  place_bread_basket ab_exp4 seed=0 gpu=6 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp4 seed=0 gpu=7 epochs=500"
  bash train.sh place_cans_plasticbox demo_clean 100 0 7 ab_exp4 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp4_s0_e500.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp4 seed=0 gpu=7 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_n2_s0 ALL_DONE"

# map: gpu0=place_bread_basket:ab_exp3, gpu1=place_cans_plasticbox:ab_exp3, gpu2=hanging_mug:ab_exp4, gpu3=place_object_basket:ab_exp4, gpu4=scan_object:ab_exp4, gpu5=place_bread_skillet:ab_exp4, gpu6=place_bread_basket:ab_exp4, gpu7=place_cans_plasticbox:ab_exp4

# ----- [3/4] abexp6t_e500_n3_s0  [8GPU]  8 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_n3_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_n3_s0 launch 8 parallel epochs=500"
(
  echo "[$(date)] start hanging_mug ab_exp5 seed=0 gpu=0 epochs=500"
  bash train.sh hanging_mug demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp5_s0_e500.log
  echo "[$(date)] done  hanging_mug ab_exp5 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp5 seed=0 gpu=1 epochs=500"
  bash train.sh place_object_basket demo_clean 100 0 1 ab_exp5 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp5_s0_e500.log
  echo "[$(date)] done  place_object_basket ab_exp5 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp5 seed=0 gpu=2 epochs=500"
  bash train.sh scan_object demo_clean 100 0 2 ab_exp5 2>&1 | tee ../../Logs/train/scan_object_ab_exp5_s0_e500.log
  echo "[$(date)] done  scan_object ab_exp5 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp5 seed=0 gpu=3 epochs=500"
  bash train.sh place_bread_skillet demo_clean 100 0 3 ab_exp5 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp5_s0_e500.log
  echo "[$(date)] done  place_bread_skillet ab_exp5 seed=0 gpu=3 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp5 seed=0 gpu=4 epochs=500"
  bash train.sh place_bread_basket demo_clean 100 0 4 ab_exp5 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp5_s0_e500.log
  echo "[$(date)] done  place_bread_basket ab_exp5 seed=0 gpu=4 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp5 seed=0 gpu=5 epochs=500"
  bash train.sh place_cans_plasticbox demo_clean 100 0 5 ab_exp5 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp5_s0_e500.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp5 seed=0 gpu=5 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp6 seed=0 gpu=6 epochs=500"
  bash train.sh hanging_mug demo_clean 100 0 6 ab_exp6 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp6_s0_e500.log
  echo "[$(date)] done  hanging_mug ab_exp6 seed=0 gpu=6 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp6 seed=0 gpu=7 epochs=500"
  bash train.sh place_object_basket demo_clean 100 0 7 ab_exp6 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp6_s0_e500.log
  echo "[$(date)] done  place_object_basket ab_exp6 seed=0 gpu=7 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_n3_s0 ALL_DONE"

# map: gpu0=hanging_mug:ab_exp5, gpu1=place_object_basket:ab_exp5, gpu2=scan_object:ab_exp5, gpu3=place_bread_skillet:ab_exp5, gpu4=place_bread_basket:ab_exp5, gpu5=place_cans_plasticbox:ab_exp5, gpu6=hanging_mug:ab_exp6, gpu7=place_object_basket:ab_exp6

# ----- [4/4] abexp6t_e500_n4_s0  [4GPU]  4 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_n4_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_n4_s0 launch 4 parallel epochs=500"
(
  echo "[$(date)] start scan_object ab_exp6 seed=0 gpu=0 epochs=500"
  bash train.sh scan_object demo_clean 100 0 0 ab_exp6 2>&1 | tee ../../Logs/train/scan_object_ab_exp6_s0_e500.log
  echo "[$(date)] done  scan_object ab_exp6 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp6 seed=0 gpu=1 epochs=500"
  bash train.sh place_bread_skillet demo_clean 100 0 1 ab_exp6 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp6_s0_e500.log
  echo "[$(date)] done  place_bread_skillet ab_exp6 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp6 seed=0 gpu=2 epochs=500"
  bash train.sh place_bread_basket demo_clean 100 0 2 ab_exp6 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp6_s0_e500.log
  echo "[$(date)] done  place_bread_basket ab_exp6 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp6 seed=0 gpu=3 epochs=500"
  bash train.sh place_cans_plasticbox demo_clean 100 0 3 ab_exp6 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp6_s0_e500.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp6 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_n4_s0 ALL_DONE"

# map: gpu0=scan_object:ab_exp6, gpu1=place_bread_skillet:ab_exp6, gpu2=place_bread_basket:ab_exp6, gpu3=place_cans_plasticbox:ab_exp6

# ========== ALTERNATIVE: pure 4-GPU nodes (9 jobs × 4) ==========

# ----- [0/8] abexp6t_e500_4g_n0_s0  [4GPU]  4 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_4g_n0_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_4g_n0_s0 launch 4 parallel epochs=500"
(
  echo "[$(date)] start hanging_mug ab_exp1 seed=0 gpu=0 epochs=500"
  bash train.sh hanging_mug demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp1_s0_e500.log
  echo "[$(date)] done  hanging_mug ab_exp1 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp1 seed=0 gpu=1 epochs=500"
  bash train.sh place_object_basket demo_clean 100 0 1 ab_exp1 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp1_s0_e500.log
  echo "[$(date)] done  place_object_basket ab_exp1 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp1 seed=0 gpu=2 epochs=500"
  bash train.sh scan_object demo_clean 100 0 2 ab_exp1 2>&1 | tee ../../Logs/train/scan_object_ab_exp1_s0_e500.log
  echo "[$(date)] done  scan_object ab_exp1 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp1 seed=0 gpu=3 epochs=500"
  bash train.sh place_bread_skillet demo_clean 100 0 3 ab_exp1 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp1_s0_e500.log
  echo "[$(date)] done  place_bread_skillet ab_exp1 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_4g_n0_s0 ALL_DONE"

# map: gpu0=hanging_mug:ab_exp1, gpu1=place_object_basket:ab_exp1, gpu2=scan_object:ab_exp1, gpu3=place_bread_skillet:ab_exp1

# ----- [1/8] abexp6t_e500_4g_n1_s0  [4GPU]  4 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_4g_n1_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_4g_n1_s0 launch 4 parallel epochs=500"
(
  echo "[$(date)] start place_bread_basket ab_exp1 seed=0 gpu=0 epochs=500"
  bash train.sh place_bread_basket demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp1_s0_e500.log
  echo "[$(date)] done  place_bread_basket ab_exp1 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp1 seed=0 gpu=1 epochs=500"
  bash train.sh place_cans_plasticbox demo_clean 100 0 1 ab_exp1 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp1_s0_e500.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp1 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp2 seed=0 gpu=2 epochs=500"
  bash train.sh hanging_mug demo_clean 100 0 2 ab_exp2 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp2_s0_e500.log
  echo "[$(date)] done  hanging_mug ab_exp2 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp2 seed=0 gpu=3 epochs=500"
  bash train.sh place_object_basket demo_clean 100 0 3 ab_exp2 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp2_s0_e500.log
  echo "[$(date)] done  place_object_basket ab_exp2 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_4g_n1_s0 ALL_DONE"

# map: gpu0=place_bread_basket:ab_exp1, gpu1=place_cans_plasticbox:ab_exp1, gpu2=hanging_mug:ab_exp2, gpu3=place_object_basket:ab_exp2

# ----- [2/8] abexp6t_e500_4g_n2_s0  [4GPU]  4 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_4g_n2_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_4g_n2_s0 launch 4 parallel epochs=500"
(
  echo "[$(date)] start scan_object ab_exp2 seed=0 gpu=0 epochs=500"
  bash train.sh scan_object demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/scan_object_ab_exp2_s0_e500.log
  echo "[$(date)] done  scan_object ab_exp2 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp2 seed=0 gpu=1 epochs=500"
  bash train.sh place_bread_skillet demo_clean 100 0 1 ab_exp2 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp2_s0_e500.log
  echo "[$(date)] done  place_bread_skillet ab_exp2 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp2 seed=0 gpu=2 epochs=500"
  bash train.sh place_bread_basket demo_clean 100 0 2 ab_exp2 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp2_s0_e500.log
  echo "[$(date)] done  place_bread_basket ab_exp2 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp2 seed=0 gpu=3 epochs=500"
  bash train.sh place_cans_plasticbox demo_clean 100 0 3 ab_exp2 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp2_s0_e500.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp2 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_4g_n2_s0 ALL_DONE"

# map: gpu0=scan_object:ab_exp2, gpu1=place_bread_skillet:ab_exp2, gpu2=place_bread_basket:ab_exp2, gpu3=place_cans_plasticbox:ab_exp2

# ----- [3/8] abexp6t_e500_4g_n3_s0  [4GPU]  4 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_4g_n3_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_4g_n3_s0 launch 4 parallel epochs=500"
(
  echo "[$(date)] start hanging_mug ab_exp3 seed=0 gpu=0 epochs=500"
  bash train.sh hanging_mug demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp3_s0_e500.log
  echo "[$(date)] done  hanging_mug ab_exp3 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp3 seed=0 gpu=1 epochs=500"
  bash train.sh place_object_basket demo_clean 100 0 1 ab_exp3 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp3_s0_e500.log
  echo "[$(date)] done  place_object_basket ab_exp3 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp3 seed=0 gpu=2 epochs=500"
  bash train.sh scan_object demo_clean 100 0 2 ab_exp3 2>&1 | tee ../../Logs/train/scan_object_ab_exp3_s0_e500.log
  echo "[$(date)] done  scan_object ab_exp3 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp3 seed=0 gpu=3 epochs=500"
  bash train.sh place_bread_skillet demo_clean 100 0 3 ab_exp3 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp3_s0_e500.log
  echo "[$(date)] done  place_bread_skillet ab_exp3 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_4g_n3_s0 ALL_DONE"

# map: gpu0=hanging_mug:ab_exp3, gpu1=place_object_basket:ab_exp3, gpu2=scan_object:ab_exp3, gpu3=place_bread_skillet:ab_exp3

# ----- [4/8] abexp6t_e500_4g_n4_s0  [4GPU]  4 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_4g_n4_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_4g_n4_s0 launch 4 parallel epochs=500"
(
  echo "[$(date)] start place_bread_basket ab_exp3 seed=0 gpu=0 epochs=500"
  bash train.sh place_bread_basket demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp3_s0_e500.log
  echo "[$(date)] done  place_bread_basket ab_exp3 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp3 seed=0 gpu=1 epochs=500"
  bash train.sh place_cans_plasticbox demo_clean 100 0 1 ab_exp3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp3_s0_e500.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp3 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp4 seed=0 gpu=2 epochs=500"
  bash train.sh hanging_mug demo_clean 100 0 2 ab_exp4 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp4_s0_e500.log
  echo "[$(date)] done  hanging_mug ab_exp4 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp4 seed=0 gpu=3 epochs=500"
  bash train.sh place_object_basket demo_clean 100 0 3 ab_exp4 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp4_s0_e500.log
  echo "[$(date)] done  place_object_basket ab_exp4 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_4g_n4_s0 ALL_DONE"

# map: gpu0=place_bread_basket:ab_exp3, gpu1=place_cans_plasticbox:ab_exp3, gpu2=hanging_mug:ab_exp4, gpu3=place_object_basket:ab_exp4

# ----- [5/8] abexp6t_e500_4g_n5_s0  [4GPU]  4 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_4g_n5_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_4g_n5_s0 launch 4 parallel epochs=500"
(
  echo "[$(date)] start scan_object ab_exp4 seed=0 gpu=0 epochs=500"
  bash train.sh scan_object demo_clean 100 0 0 ab_exp4 2>&1 | tee ../../Logs/train/scan_object_ab_exp4_s0_e500.log
  echo "[$(date)] done  scan_object ab_exp4 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp4 seed=0 gpu=1 epochs=500"
  bash train.sh place_bread_skillet demo_clean 100 0 1 ab_exp4 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp4_s0_e500.log
  echo "[$(date)] done  place_bread_skillet ab_exp4 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp4 seed=0 gpu=2 epochs=500"
  bash train.sh place_bread_basket demo_clean 100 0 2 ab_exp4 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp4_s0_e500.log
  echo "[$(date)] done  place_bread_basket ab_exp4 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp4 seed=0 gpu=3 epochs=500"
  bash train.sh place_cans_plasticbox demo_clean 100 0 3 ab_exp4 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp4_s0_e500.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp4 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_4g_n5_s0 ALL_DONE"

# map: gpu0=scan_object:ab_exp4, gpu1=place_bread_skillet:ab_exp4, gpu2=place_bread_basket:ab_exp4, gpu3=place_cans_plasticbox:ab_exp4

# ----- [6/8] abexp6t_e500_4g_n6_s0  [4GPU]  4 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_4g_n6_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_4g_n6_s0 launch 4 parallel epochs=500"
(
  echo "[$(date)] start hanging_mug ab_exp5 seed=0 gpu=0 epochs=500"
  bash train.sh hanging_mug demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp5_s0_e500.log
  echo "[$(date)] done  hanging_mug ab_exp5 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp5 seed=0 gpu=1 epochs=500"
  bash train.sh place_object_basket demo_clean 100 0 1 ab_exp5 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp5_s0_e500.log
  echo "[$(date)] done  place_object_basket ab_exp5 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start scan_object ab_exp5 seed=0 gpu=2 epochs=500"
  bash train.sh scan_object demo_clean 100 0 2 ab_exp5 2>&1 | tee ../../Logs/train/scan_object_ab_exp5_s0_e500.log
  echo "[$(date)] done  scan_object ab_exp5 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp5 seed=0 gpu=3 epochs=500"
  bash train.sh place_bread_skillet demo_clean 100 0 3 ab_exp5 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp5_s0_e500.log
  echo "[$(date)] done  place_bread_skillet ab_exp5 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_4g_n6_s0 ALL_DONE"

# map: gpu0=hanging_mug:ab_exp5, gpu1=place_object_basket:ab_exp5, gpu2=scan_object:ab_exp5, gpu3=place_bread_skillet:ab_exp5

# ----- [7/8] abexp6t_e500_4g_n7_s0  [4GPU]  4 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_4g_n7_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_4g_n7_s0 launch 4 parallel epochs=500"
(
  echo "[$(date)] start place_bread_basket ab_exp5 seed=0 gpu=0 epochs=500"
  bash train.sh place_bread_basket demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp5_s0_e500.log
  echo "[$(date)] done  place_bread_basket ab_exp5 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp5 seed=0 gpu=1 epochs=500"
  bash train.sh place_cans_plasticbox demo_clean 100 0 1 ab_exp5 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp5_s0_e500.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp5 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start hanging_mug ab_exp6 seed=0 gpu=2 epochs=500"
  bash train.sh hanging_mug demo_clean 100 0 2 ab_exp6 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp6_s0_e500.log
  echo "[$(date)] done  hanging_mug ab_exp6 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket ab_exp6 seed=0 gpu=3 epochs=500"
  bash train.sh place_object_basket demo_clean 100 0 3 ab_exp6 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp6_s0_e500.log
  echo "[$(date)] done  place_object_basket ab_exp6 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_4g_n7_s0 ALL_DONE"

# map: gpu0=place_bread_basket:ab_exp5, gpu1=place_cans_plasticbox:ab_exp5, gpu2=hanging_mug:ab_exp6, gpu3=place_object_basket:ab_exp6

# ----- [8/8] abexp6t_e500_4g_n8_s0  [4GPU]  4 parallel | NUM_EPOCHS=500 -----
abexp6t_e500_4g_n8_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=abexp6t_e500_4g_n8_s0 launch 4 parallel epochs=500"
(
  echo "[$(date)] start scan_object ab_exp6 seed=0 gpu=0 epochs=500"
  bash train.sh scan_object demo_clean 100 0 0 ab_exp6 2>&1 | tee ../../Logs/train/scan_object_ab_exp6_s0_e500.log
  echo "[$(date)] done  scan_object ab_exp6 seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet ab_exp6 seed=0 gpu=1 epochs=500"
  bash train.sh place_bread_skillet demo_clean 100 0 1 ab_exp6 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp6_s0_e500.log
  echo "[$(date)] done  place_bread_skillet ab_exp6 seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket ab_exp6 seed=0 gpu=2 epochs=500"
  bash train.sh place_bread_basket demo_clean 100 0 2 ab_exp6 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp6_s0_e500.log
  echo "[$(date)] done  place_bread_basket ab_exp6 seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox ab_exp6 seed=0 gpu=3 epochs=500"
  bash train.sh place_cans_plasticbox demo_clean 100 0 3 ab_exp6 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp6_s0_e500.log
  echo "[$(date)] done  place_cans_plasticbox ab_exp6 seed=0 gpu=3 exit=$?"
) &
wait
echo "[$(date)] node=abexp6t_e500_4g_n8_s0 ALL_DONE"

# map: gpu0=scan_object:ab_exp6, gpu1=place_bread_skillet:ab_exp6, gpu2=place_bread_basket:ab_exp6, gpu3=place_cans_plasticbox:ab_exp6

