# Remain e500 | 32 on 2×8GPU (2/GPU) + 1 alone
# EXCLUDE alone: place_cans_plasticbox:ab_exp5
# ckpt: policy/SPACE/checkpoints/<task>-train_<enc>-100_0/500.ckpt

# ----- e500_remain_n0_s0  [8GPU]  2 concurrent / GPU | NUM_EPOCHS=500 -----
e500_remain_n0_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=e500_remain_n0_s0 launch 16 jobs @ 2 concurrent/GPU, epochs=500"
rm -f /tmp/e500_fail_* 2>/dev/null || true
(
  # GPU 0: ['hanging_mug:ab_exp2', 'place_bread_skillet:main']
  (
    echo "[$(date)] start hanging_mug ab_exp2 seed=0 gpu=0 epochs=500"
    bash train.sh hanging_mug demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp2_s0_e500.log
    ec=$?
    echo "[$(date)] done  hanging_mug ab_exp2 seed=0 gpu=0 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_hanging_mug_ab_exp2; fi
  ) &
  (
    echo "[$(date)] start place_bread_skillet main seed=0 gpu=0 epochs=500"
    bash train.sh place_bread_skillet demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/place_bread_skillet_main_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_bread_skillet main seed=0 gpu=0 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_bread_skillet_main; fi
  ) &
  wait
) &
(
  # GPU 1: ['hanging_mug:ab_exp3', 'place_bread_skillet:ab_exp1']
  (
    echo "[$(date)] start hanging_mug ab_exp3 seed=0 gpu=1 epochs=500"
    bash train.sh hanging_mug demo_clean 100 0 1 ab_exp3 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp3_s0_e500.log
    ec=$?
    echo "[$(date)] done  hanging_mug ab_exp3 seed=0 gpu=1 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_hanging_mug_ab_exp3; fi
  ) &
  (
    echo "[$(date)] start place_bread_skillet ab_exp1 seed=0 gpu=1 epochs=500"
    bash train.sh place_bread_skillet demo_clean 100 0 1 ab_exp1 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp1_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_bread_skillet ab_exp1 seed=0 gpu=1 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_bread_skillet_ab_exp1; fi
  ) &
  wait
) &
(
  # GPU 2: ['hanging_mug:ab_exp4', 'place_bread_skillet:ab_exp2']
  (
    echo "[$(date)] start hanging_mug ab_exp4 seed=0 gpu=2 epochs=500"
    bash train.sh hanging_mug demo_clean 100 0 2 ab_exp4 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp4_s0_e500.log
    ec=$?
    echo "[$(date)] done  hanging_mug ab_exp4 seed=0 gpu=2 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_hanging_mug_ab_exp4; fi
  ) &
  (
    echo "[$(date)] start place_bread_skillet ab_exp2 seed=0 gpu=2 epochs=500"
    bash train.sh place_bread_skillet demo_clean 100 0 2 ab_exp2 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp2_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_bread_skillet ab_exp2 seed=0 gpu=2 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_bread_skillet_ab_exp2; fi
  ) &
  wait
) &
(
  # GPU 3: ['hanging_mug:ab_exp6', 'place_bread_skillet:ab_exp3']
  (
    echo "[$(date)] start hanging_mug ab_exp6 seed=0 gpu=3 epochs=500"
    bash train.sh hanging_mug demo_clean 100 0 3 ab_exp6 2>&1 | tee ../../Logs/train/hanging_mug_ab_exp6_s0_e500.log
    ec=$?
    echo "[$(date)] done  hanging_mug ab_exp6 seed=0 gpu=3 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_hanging_mug_ab_exp6; fi
  ) &
  (
    echo "[$(date)] start place_bread_skillet ab_exp3 seed=0 gpu=3 epochs=500"
    bash train.sh place_bread_skillet demo_clean 100 0 3 ab_exp3 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp3_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_bread_skillet ab_exp3 seed=0 gpu=3 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_bread_skillet_ab_exp3; fi
  ) &
  wait
) &
(
  # GPU 4: ['place_object_basket:main', 'place_bread_skillet:ab_exp4']
  (
    echo "[$(date)] start place_object_basket main seed=0 gpu=4 epochs=500"
    bash train.sh place_object_basket demo_clean 100 0 4 main 2>&1 | tee ../../Logs/train/place_object_basket_main_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_object_basket main seed=0 gpu=4 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_object_basket_main; fi
  ) &
  (
    echo "[$(date)] start place_bread_skillet ab_exp4 seed=0 gpu=4 epochs=500"
    bash train.sh place_bread_skillet demo_clean 100 0 4 ab_exp4 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp4_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_bread_skillet ab_exp4 seed=0 gpu=4 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_bread_skillet_ab_exp4; fi
  ) &
  wait
) &
(
  # GPU 5: ['place_object_basket:ab_exp1', 'place_bread_skillet:ab_exp5']
  (
    echo "[$(date)] start place_object_basket ab_exp1 seed=0 gpu=5 epochs=500"
    bash train.sh place_object_basket demo_clean 100 0 5 ab_exp1 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp1_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_object_basket ab_exp1 seed=0 gpu=5 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_object_basket_ab_exp1; fi
  ) &
  (
    echo "[$(date)] start place_bread_skillet ab_exp5 seed=0 gpu=5 epochs=500"
    bash train.sh place_bread_skillet demo_clean 100 0 5 ab_exp5 2>&1 | tee ../../Logs/train/place_bread_skillet_ab_exp5_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_bread_skillet ab_exp5 seed=0 gpu=5 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_bread_skillet_ab_exp5; fi
  ) &
  wait
) &
(
  # GPU 6: ['place_object_basket:ab_exp2', 'place_bread_basket:main']
  (
    echo "[$(date)] start place_object_basket ab_exp2 seed=0 gpu=6 epochs=500"
    bash train.sh place_object_basket demo_clean 100 0 6 ab_exp2 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp2_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_object_basket ab_exp2 seed=0 gpu=6 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_object_basket_ab_exp2; fi
  ) &
  (
    echo "[$(date)] start place_bread_basket main seed=0 gpu=6 epochs=500"
    bash train.sh place_bread_basket demo_clean 100 0 6 main 2>&1 | tee ../../Logs/train/place_bread_basket_main_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_bread_basket main seed=0 gpu=6 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_bread_basket_main; fi
  ) &
  wait
) &
(
  # GPU 7: ['place_object_basket:ab_exp3', 'place_bread_basket:ab_exp1']
  (
    echo "[$(date)] start place_object_basket ab_exp3 seed=0 gpu=7 epochs=500"
    bash train.sh place_object_basket demo_clean 100 0 7 ab_exp3 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp3_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_object_basket ab_exp3 seed=0 gpu=7 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_object_basket_ab_exp3; fi
  ) &
  (
    echo "[$(date)] start place_bread_basket ab_exp1 seed=0 gpu=7 epochs=500"
    bash train.sh place_bread_basket demo_clean 100 0 7 ab_exp1 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp1_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_bread_basket ab_exp1 seed=0 gpu=7 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_bread_basket_ab_exp1; fi
  ) &
  wait
) &
wait
FAIL=0
ls /tmp/e500_fail_* >/dev/null 2>&1 && FAIL=1
echo "[$(date)] node=e500_remain_n0_s0 ALL_DONE fail=$FAIL"
if [ "$FAIL" -ne 0 ]; then ls /tmp/e500_fail_*; exit 1; fi
exit 0

# map: gpu0=['hanging_mug:ab_exp2', 'place_bread_skillet:main'] | gpu1=['hanging_mug:ab_exp3', 'place_bread_skillet:ab_exp1'] | gpu2=['hanging_mug:ab_exp4', 'place_bread_skillet:ab_exp2'] | gpu3=['hanging_mug:ab_exp6', 'place_bread_skillet:ab_exp3'] | gpu4=['place_object_basket:main', 'place_bread_skillet:ab_exp4'] | gpu5=['place_object_basket:ab_exp1', 'place_bread_skillet:ab_exp5'] | gpu6=['place_object_basket:ab_exp2', 'place_bread_basket:main'] | gpu7=['place_object_basket:ab_exp3', 'place_bread_basket:ab_exp1']

# ----- e500_remain_n1_s0  [8GPU]  2 concurrent / GPU | NUM_EPOCHS=500 -----
e500_remain_n1_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=e500_remain_n1_s0 launch 16 jobs @ 2 concurrent/GPU, epochs=500"
rm -f /tmp/e500_fail_* 2>/dev/null || true
(
  # GPU 0: ['place_object_basket:ab_exp4', 'place_bread_basket:ab_exp2']
  (
    echo "[$(date)] start place_object_basket ab_exp4 seed=0 gpu=0 epochs=500"
    bash train.sh place_object_basket demo_clean 100 0 0 ab_exp4 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp4_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_object_basket ab_exp4 seed=0 gpu=0 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_object_basket_ab_exp4; fi
  ) &
  (
    echo "[$(date)] start place_bread_basket ab_exp2 seed=0 gpu=0 epochs=500"
    bash train.sh place_bread_basket demo_clean 100 0 0 ab_exp2 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp2_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_bread_basket ab_exp2 seed=0 gpu=0 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_bread_basket_ab_exp2; fi
  ) &
  wait
) &
(
  # GPU 1: ['place_object_basket:ab_exp5', 'place_bread_basket:ab_exp4']
  (
    echo "[$(date)] start place_object_basket ab_exp5 seed=0 gpu=1 epochs=500"
    bash train.sh place_object_basket demo_clean 100 0 1 ab_exp5 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp5_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_object_basket ab_exp5 seed=0 gpu=1 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_object_basket_ab_exp5; fi
  ) &
  (
    echo "[$(date)] start place_bread_basket ab_exp4 seed=0 gpu=1 epochs=500"
    bash train.sh place_bread_basket demo_clean 100 0 1 ab_exp4 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp4_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_bread_basket ab_exp4 seed=0 gpu=1 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_bread_basket_ab_exp4; fi
  ) &
  wait
) &
(
  # GPU 2: ['place_object_basket:ab_exp6', 'place_bread_basket:ab_exp5']
  (
    echo "[$(date)] start place_object_basket ab_exp6 seed=0 gpu=2 epochs=500"
    bash train.sh place_object_basket demo_clean 100 0 2 ab_exp6 2>&1 | tee ../../Logs/train/place_object_basket_ab_exp6_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_object_basket ab_exp6 seed=0 gpu=2 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_object_basket_ab_exp6; fi
  ) &
  (
    echo "[$(date)] start place_bread_basket ab_exp5 seed=0 gpu=2 epochs=500"
    bash train.sh place_bread_basket demo_clean 100 0 2 ab_exp5 2>&1 | tee ../../Logs/train/place_bread_basket_ab_exp5_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_bread_basket ab_exp5 seed=0 gpu=2 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_bread_basket_ab_exp5; fi
  ) &
  wait
) &
(
  # GPU 3: ['scan_object:main', 'place_cans_plasticbox:main']
  (
    echo "[$(date)] start scan_object main seed=0 gpu=3 epochs=500"
    bash train.sh scan_object demo_clean 100 0 3 main 2>&1 | tee ../../Logs/train/scan_object_main_s0_e500.log
    ec=$?
    echo "[$(date)] done  scan_object main seed=0 gpu=3 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_scan_object_main; fi
  ) &
  (
    echo "[$(date)] start place_cans_plasticbox main seed=0 gpu=3 epochs=500"
    bash train.sh place_cans_plasticbox demo_clean 100 0 3 main 2>&1 | tee ../../Logs/train/place_cans_plasticbox_main_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_cans_plasticbox main seed=0 gpu=3 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_cans_plasticbox_main; fi
  ) &
  wait
) &
(
  # GPU 4: ['scan_object:ab_exp1', 'place_cans_plasticbox:ab_exp1']
  (
    echo "[$(date)] start scan_object ab_exp1 seed=0 gpu=4 epochs=500"
    bash train.sh scan_object demo_clean 100 0 4 ab_exp1 2>&1 | tee ../../Logs/train/scan_object_ab_exp1_s0_e500.log
    ec=$?
    echo "[$(date)] done  scan_object ab_exp1 seed=0 gpu=4 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_scan_object_ab_exp1; fi
  ) &
  (
    echo "[$(date)] start place_cans_plasticbox ab_exp1 seed=0 gpu=4 epochs=500"
    bash train.sh place_cans_plasticbox demo_clean 100 0 4 ab_exp1 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp1_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_cans_plasticbox ab_exp1 seed=0 gpu=4 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_cans_plasticbox_ab_exp1; fi
  ) &
  wait
) &
(
  # GPU 5: ['scan_object:ab_exp3', 'place_cans_plasticbox:ab_exp2']
  (
    echo "[$(date)] start scan_object ab_exp3 seed=0 gpu=5 epochs=500"
    bash train.sh scan_object demo_clean 100 0 5 ab_exp3 2>&1 | tee ../../Logs/train/scan_object_ab_exp3_s0_e500.log
    ec=$?
    echo "[$(date)] done  scan_object ab_exp3 seed=0 gpu=5 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_scan_object_ab_exp3; fi
  ) &
  (
    echo "[$(date)] start place_cans_plasticbox ab_exp2 seed=0 gpu=5 epochs=500"
    bash train.sh place_cans_plasticbox demo_clean 100 0 5 ab_exp2 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp2_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_cans_plasticbox ab_exp2 seed=0 gpu=5 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_cans_plasticbox_ab_exp2; fi
  ) &
  wait
) &
(
  # GPU 6: ['scan_object:ab_exp4', 'place_cans_plasticbox:ab_exp3']
  (
    echo "[$(date)] start scan_object ab_exp4 seed=0 gpu=6 epochs=500"
    bash train.sh scan_object demo_clean 100 0 6 ab_exp4 2>&1 | tee ../../Logs/train/scan_object_ab_exp4_s0_e500.log
    ec=$?
    echo "[$(date)] done  scan_object ab_exp4 seed=0 gpu=6 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_scan_object_ab_exp4; fi
  ) &
  (
    echo "[$(date)] start place_cans_plasticbox ab_exp3 seed=0 gpu=6 epochs=500"
    bash train.sh place_cans_plasticbox demo_clean 100 0 6 ab_exp3 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp3_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_cans_plasticbox ab_exp3 seed=0 gpu=6 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_cans_plasticbox_ab_exp3; fi
  ) &
  wait
) &
(
  # GPU 7: ['scan_object:ab_exp5', 'place_cans_plasticbox:ab_exp4']
  (
    echo "[$(date)] start scan_object ab_exp5 seed=0 gpu=7 epochs=500"
    bash train.sh scan_object demo_clean 100 0 7 ab_exp5 2>&1 | tee ../../Logs/train/scan_object_ab_exp5_s0_e500.log
    ec=$?
    echo "[$(date)] done  scan_object ab_exp5 seed=0 gpu=7 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_scan_object_ab_exp5; fi
  ) &
  (
    echo "[$(date)] start place_cans_plasticbox ab_exp4 seed=0 gpu=7 epochs=500"
    bash train.sh place_cans_plasticbox demo_clean 100 0 7 ab_exp4 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp4_s0_e500.log
    ec=$?
    echo "[$(date)] done  place_cans_plasticbox ab_exp4 seed=0 gpu=7 exit=$ec"
    if [ "$ec" -ne 0 ]; then touch /tmp/e500_fail_place_cans_plasticbox_ab_exp4; fi
  ) &
  wait
) &
wait
FAIL=0
ls /tmp/e500_fail_* >/dev/null 2>&1 && FAIL=1
echo "[$(date)] node=e500_remain_n1_s0 ALL_DONE fail=$FAIL"
if [ "$FAIL" -ne 0 ]; then ls /tmp/e500_fail_*; exit 1; fi
exit 0

# map: gpu0=['place_object_basket:ab_exp4', 'place_bread_basket:ab_exp2'] | gpu1=['place_object_basket:ab_exp5', 'place_bread_basket:ab_exp4'] | gpu2=['place_object_basket:ab_exp6', 'place_bread_basket:ab_exp5'] | gpu3=['scan_object:main', 'place_cans_plasticbox:main'] | gpu4=['scan_object:ab_exp1', 'place_cans_plasticbox:ab_exp1'] | gpu5=['scan_object:ab_exp3', 'place_cans_plasticbox:ab_exp2'] | gpu6=['scan_object:ab_exp4', 'place_cans_plasticbox:ab_exp3'] | gpu7=['scan_object:ab_exp5', 'place_cans_plasticbox:ab_exp4']

# ----- e500_remain_alone_cans_abexp5  [1GPU] -----
e500_remain_alone_cans_abexp5
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] alone place_cans_plasticbox ab_exp5 gpu=0 epochs=500"
bash train.sh place_cans_plasticbox demo_clean 100 0 0 ab_exp5 2>&1 | tee ../../Logs/train/place_cans_plasticbox_ab_exp5_s0_e500.log
ec=$?
echo "[$(date)] done place_cans_plasticbox ab_exp5 exit=$ec"
exit $ec
