# e500 remain | 1 job = 1 GPU | NUM_EPOCHS=500
# Count: 33 | skip existing 500.ckpt
# Submit: name=<job> ; paste #!/bin/sh body ; pick 1GPU
# ckpt: policy/SPACE/checkpoints/<task>-train_<enc>-100_0/500.ckpt
# log:  Logs/train/<task>_<enc>_s0_e500.log
# exit code = train exit (no false success)

# ----- [0/33] e500_1g_hanging_mug_ab_exp2_s0  [1GPU] -----
e500_1g_hanging_mug_ab_exp2_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start hanging_mug ab_exp2 seed=0 gpu=0 epochs=500"
bash train.sh hanging_mug demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp2_s0_e500.log
ec=$?
echo "[$(date)] done hanging_mug ab_exp2 exit=$ec"
exit $ec

# ----- [1/33] e500_1g_hanging_mug_ab_exp3_s0  [1GPU] -----
e500_1g_hanging_mug_ab_exp3_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start hanging_mug ab_exp3 seed=0 gpu=0 epochs=500"
bash train.sh hanging_mug demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp3_s0_e500.log
ec=$?
echo "[$(date)] done hanging_mug ab_exp3 exit=$ec"
exit $ec

# ----- [2/33] e500_1g_hanging_mug_ab_exp4_s0  [1GPU] -----
e500_1g_hanging_mug_ab_exp4_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start hanging_mug ab_exp4 seed=0 gpu=0 epochs=500"
bash train.sh hanging_mug demo_clean 100 0 0 ab_exp4 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp4_s0_e500.log
ec=$?
echo "[$(date)] done hanging_mug ab_exp4 exit=$ec"
exit $ec

# ----- [3/33] e500_1g_hanging_mug_ab_exp6_s0  [1GPU] -----
e500_1g_hanging_mug_ab_exp6_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start hanging_mug ab_exp6 seed=0 gpu=0 epochs=500"
bash train.sh hanging_mug demo_clean 100 0 0 ab_exp6 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp6_s0_e500.log
ec=$?
echo "[$(date)] done hanging_mug ab_exp6 exit=$ec"
exit $ec

# ----- [4/33] e500_1g_place_object_basket_main_s0  [1GPU] -----
e500_1g_place_object_basket_main_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_object_basket main seed=0 gpu=0 epochs=500"
bash train.sh place_object_basket demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_object_basket_main_s0_e500.log
ec=$?
echo "[$(date)] done place_object_basket main exit=$ec"
exit $ec

# ----- [5/33] e500_1g_place_object_basket_ab_exp1_s0  [1GPU] -----
e500_1g_place_object_basket_ab_exp1_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_object_basket ab_exp1 seed=0 gpu=0 epochs=500"
bash train.sh place_object_basket demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp1_s0_e500.log
ec=$?
echo "[$(date)] done place_object_basket ab_exp1 exit=$ec"
exit $ec

# ----- [6/33] e500_1g_place_object_basket_ab_exp2_s0  [1GPU] -----
e500_1g_place_object_basket_ab_exp2_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_object_basket ab_exp2 seed=0 gpu=0 epochs=500"
bash train.sh place_object_basket demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp2_s0_e500.log
ec=$?
echo "[$(date)] done place_object_basket ab_exp2 exit=$ec"
exit $ec

# ----- [7/33] e500_1g_place_object_basket_ab_exp3_s0  [1GPU] -----
e500_1g_place_object_basket_ab_exp3_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_object_basket ab_exp3 seed=0 gpu=0 epochs=500"
bash train.sh place_object_basket demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp3_s0_e500.log
ec=$?
echo "[$(date)] done place_object_basket ab_exp3 exit=$ec"
exit $ec

# ----- [8/33] e500_1g_place_object_basket_ab_exp4_s0  [1GPU] -----
e500_1g_place_object_basket_ab_exp4_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_object_basket ab_exp4 seed=0 gpu=0 epochs=500"
bash train.sh place_object_basket demo_clean 100 0 0 ab_exp4 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp4_s0_e500.log
ec=$?
echo "[$(date)] done place_object_basket ab_exp4 exit=$ec"
exit $ec

# ----- [9/33] e500_1g_place_object_basket_ab_exp5_s0  [1GPU] -----
e500_1g_place_object_basket_ab_exp5_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_object_basket ab_exp5 seed=0 gpu=0 epochs=500"
bash train.sh place_object_basket demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp5_s0_e500.log
ec=$?
echo "[$(date)] done place_object_basket ab_exp5 exit=$ec"
exit $ec

# ----- [10/33] e500_1g_place_object_basket_ab_exp6_s0  [1GPU] -----
e500_1g_place_object_basket_ab_exp6_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_object_basket ab_exp6 seed=0 gpu=0 epochs=500"
bash train.sh place_object_basket demo_clean 100 0 0 ab_exp6 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp6_s0_e500.log
ec=$?
echo "[$(date)] done place_object_basket ab_exp6 exit=$ec"
exit $ec

# ----- [11/33] e500_1g_scan_object_main_s0  [1GPU] -----
e500_1g_scan_object_main_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start scan_object main seed=0 gpu=0 epochs=500"
bash train.sh scan_object demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/scan_object_main_s0_e500.log
ec=$?
echo "[$(date)] done scan_object main exit=$ec"
exit $ec

# ----- [12/33] e500_1g_scan_object_ab_exp1_s0  [1GPU] -----
e500_1g_scan_object_ab_exp1_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start scan_object ab_exp1 seed=0 gpu=0 epochs=500"
bash train.sh scan_object demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/scan_object_ab_exp1_s0_e500.log
ec=$?
echo "[$(date)] done scan_object ab_exp1 exit=$ec"
exit $ec

# ----- [13/33] e500_1g_scan_object_ab_exp3_s0  [1GPU] -----
e500_1g_scan_object_ab_exp3_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start scan_object ab_exp3 seed=0 gpu=0 epochs=500"
bash train.sh scan_object demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/scan_object_ab_exp3_s0_e500.log
ec=$?
echo "[$(date)] done scan_object ab_exp3 exit=$ec"
exit $ec

# ----- [14/33] e500_1g_scan_object_ab_exp4_s0  [1GPU] -----
e500_1g_scan_object_ab_exp4_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start scan_object ab_exp4 seed=0 gpu=0 epochs=500"
bash train.sh scan_object demo_clean 100 0 0 ab_exp4 2>&1 | tee ../../Logs/train/scan_object_ab_exp4_s0_e500.log
ec=$?
echo "[$(date)] done scan_object ab_exp4 exit=$ec"
exit $ec

# ----- [15/33] e500_1g_scan_object_ab_exp5_s0  [1GPU] -----
e500_1g_scan_object_ab_exp5_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start scan_object ab_exp5 seed=0 gpu=0 epochs=500"
bash train.sh scan_object demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/scan_object_ab_exp5_s0_e500.log
ec=$?
echo "[$(date)] done scan_object ab_exp5 exit=$ec"
exit $ec

# ----- [16/33] e500_1g_place_bread_skillet_main_s0  [1GPU] -----
e500_1g_place_bread_skillet_main_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_skillet main seed=0 gpu=0 epochs=500"
bash train.sh place_bread_skillet demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_bread_skillet_main_s0_e500.log
ec=$?
echo "[$(date)] done place_bread_skillet main exit=$ec"
exit $ec

# ----- [17/33] e500_1g_place_bread_skillet_ab_exp1_s0  [1GPU] -----
e500_1g_place_bread_skillet_ab_exp1_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_skillet ab_exp1 seed=0 gpu=0 epochs=500"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp1_s0_e500.log
ec=$?
echo "[$(date)] done place_bread_skillet ab_exp1 exit=$ec"
exit $ec

# ----- [18/33] e500_1g_place_bread_skillet_ab_exp2_s0  [1GPU] -----
e500_1g_place_bread_skillet_ab_exp2_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_skillet ab_exp2 seed=0 gpu=0 epochs=500"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp2_s0_e500.log
ec=$?
echo "[$(date)] done place_bread_skillet ab_exp2 exit=$ec"
exit $ec

# ----- [19/33] e500_1g_place_bread_skillet_ab_exp3_s0  [1GPU] -----
e500_1g_place_bread_skillet_ab_exp3_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_skillet ab_exp3 seed=0 gpu=0 epochs=500"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp3_s0_e500.log
ec=$?
echo "[$(date)] done place_bread_skillet ab_exp3 exit=$ec"
exit $ec

# ----- [20/33] e500_1g_place_bread_skillet_ab_exp4_s0  [1GPU] -----
e500_1g_place_bread_skillet_ab_exp4_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_skillet ab_exp4 seed=0 gpu=0 epochs=500"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab_exp4 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp4_s0_e500.log
ec=$?
echo "[$(date)] done place_bread_skillet ab_exp4 exit=$ec"
exit $ec

# ----- [21/33] e500_1g_place_bread_skillet_ab_exp5_s0  [1GPU] -----
e500_1g_place_bread_skillet_ab_exp5_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_skillet ab_exp5 seed=0 gpu=0 epochs=500"
bash train.sh place_bread_skillet demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp5_s0_e500.log
ec=$?
echo "[$(date)] done place_bread_skillet ab_exp5 exit=$ec"
exit $ec

# ----- [22/33] e500_1g_place_bread_basket_main_s0  [1GPU] -----
e500_1g_place_bread_basket_main_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_basket main seed=0 gpu=0 epochs=500"
bash train.sh place_bread_basket demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_bread_basket_main_s0_e500.log
ec=$?
echo "[$(date)] done place_bread_basket main exit=$ec"
exit $ec

# ----- [23/33] e500_1g_place_bread_basket_ab_exp1_s0  [1GPU] -----
e500_1g_place_bread_basket_ab_exp1_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_basket ab_exp1 seed=0 gpu=0 epochs=500"
bash train.sh place_bread_basket demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp1_s0_e500.log
ec=$?
echo "[$(date)] done place_bread_basket ab_exp1 exit=$ec"
exit $ec

# ----- [24/33] e500_1g_place_bread_basket_ab_exp2_s0  [1GPU] -----
e500_1g_place_bread_basket_ab_exp2_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_basket ab_exp2 seed=0 gpu=0 epochs=500"
bash train.sh place_bread_basket demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp2_s0_e500.log
ec=$?
echo "[$(date)] done place_bread_basket ab_exp2 exit=$ec"
exit $ec

# ----- [25/33] e500_1g_place_bread_basket_ab_exp4_s0  [1GPU] -----
e500_1g_place_bread_basket_ab_exp4_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_basket ab_exp4 seed=0 gpu=0 epochs=500"
bash train.sh place_bread_basket demo_clean 100 0 0 ab_exp4 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp4_s0_e500.log
ec=$?
echo "[$(date)] done place_bread_basket ab_exp4 exit=$ec"
exit $ec

# ----- [26/33] e500_1g_place_bread_basket_ab_exp5_s0  [1GPU] -----
e500_1g_place_bread_basket_ab_exp5_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_bread_basket ab_exp5 seed=0 gpu=0 epochs=500"
bash train.sh place_bread_basket demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp5_s0_e500.log
ec=$?
echo "[$(date)] done place_bread_basket ab_exp5 exit=$ec"
exit $ec

# ----- [27/33] e500_1g_place_cans_plasticbox_main_s0  [1GPU] -----
e500_1g_place_cans_plasticbox_main_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_cans_plasticbox main seed=0 gpu=0 epochs=500"
bash train.sh place_cans_plasticbox demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_cans_plasticbox_main_s0_e500.log
ec=$?
echo "[$(date)] done place_cans_plasticbox main exit=$ec"
exit $ec

# ----- [28/33] e500_1g_place_cans_plasticbox_ab_exp1_s0  [1GPU] -----
e500_1g_place_cans_plasticbox_ab_exp1_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_cans_plasticbox ab_exp1 seed=0 gpu=0 epochs=500"
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab_exp1 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp1_s0_e500.log
ec=$?
echo "[$(date)] done place_cans_plasticbox ab_exp1 exit=$ec"
exit $ec

# ----- [29/33] e500_1g_place_cans_plasticbox_ab_exp2_s0  [1GPU] -----
e500_1g_place_cans_plasticbox_ab_exp2_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_cans_plasticbox ab_exp2 seed=0 gpu=0 epochs=500"
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp2_s0_e500.log
ec=$?
echo "[$(date)] done place_cans_plasticbox ab_exp2 exit=$ec"
exit $ec

# ----- [30/33] e500_1g_place_cans_plasticbox_ab_exp3_s0  [1GPU] -----
e500_1g_place_cans_plasticbox_ab_exp3_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_cans_plasticbox ab_exp3 seed=0 gpu=0 epochs=500"
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab_exp3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp3_s0_e500.log
ec=$?
echo "[$(date)] done place_cans_plasticbox ab_exp3 exit=$ec"
exit $ec

# ----- [31/33] e500_1g_place_cans_plasticbox_ab_exp4_s0  [1GPU] -----
e500_1g_place_cans_plasticbox_ab_exp4_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_cans_plasticbox ab_exp4 seed=0 gpu=0 epochs=500"
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab_exp4 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp4_s0_e500.log
ec=$?
echo "[$(date)] done place_cans_plasticbox ab_exp4 exit=$ec"
exit $ec

# ----- [32/33] e500_1g_place_cans_plasticbox_ab_exp5_s0  [1GPU] -----
e500_1g_place_cans_plasticbox_ab_exp5_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] start place_cans_plasticbox ab_exp5 seed=0 gpu=0 epochs=500"
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp5_s0_e500.log
ec=$?
echo "[$(date)] done place_cans_plasticbox ab_exp5 exit=$ec"
exit $ec
