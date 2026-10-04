# main × 6 tasks | seed=0 | demo_clean | n=100 | epochs=500
# SKIP: handover_block, pick_dual_bottles
# Tasks: hanging_mug place_object_basket scan_object place_bread_skillet place_bread_basket place_cans_plasticbox
# Layout: 1×8GPU node, 6 parallel (gpu0-5; gpu6-7 idle)
# Saves: checkpoints/<task>-train_main-100_0/500.ckpt
# Log:   Logs/train/<task>_main_s0_e500.log
# Submit: name=job line; paste #!/bin/sh body; pick 8-GPU node

# ----- [0/0] main6t_e500_n0_s0  [8GPU]  6 parallel | NUM_EPOCHS=500 -----
main6t_e500_n0_s0
#!/bin/sh
export WANDB_MODE=offline
export NUM_EPOCHS=500
export CHECKPOINT_EVERY=500
export RESUME=false
cd policy/SPACE
mkdir -p ../../Logs/train
echo "[$(date)] node=main6t_e500_n0_s0 launch 6 parallel epochs=500 encoder=main"
(
  echo "[$(date)] start hanging_mug main seed=0 gpu=0 epochs=500"
  bash train.sh hanging_mug demo_clean 100 0 0 main 2>&1 | tee ../../Logs/train/hanging_mug_main_s0_e500.log
  echo "[$(date)] done  hanging_mug main seed=0 gpu=0 exit=$?"
) &
(
  echo "[$(date)] start place_object_basket main seed=0 gpu=1 epochs=500"
  bash train.sh place_object_basket demo_clean 100 0 1 main 2>&1 | tee ../../Logs/train/place_object_basket_main_s0_e500.log
  echo "[$(date)] done  place_object_basket main seed=0 gpu=1 exit=$?"
) &
(
  echo "[$(date)] start scan_object main seed=0 gpu=2 epochs=500"
  bash train.sh scan_object demo_clean 100 0 2 main 2>&1 | tee ../../Logs/train/scan_object_main_s0_e500.log
  echo "[$(date)] done  scan_object main seed=0 gpu=2 exit=$?"
) &
(
  echo "[$(date)] start place_bread_skillet main seed=0 gpu=3 epochs=500"
  bash train.sh place_bread_skillet demo_clean 100 0 3 main 2>&1 | tee ../../Logs/train/place_bread_skillet_main_s0_e500.log
  echo "[$(date)] done  place_bread_skillet main seed=0 gpu=3 exit=$?"
) &
(
  echo "[$(date)] start place_bread_basket main seed=0 gpu=4 epochs=500"
  bash train.sh place_bread_basket demo_clean 100 0 4 main 2>&1 | tee ../../Logs/train/place_bread_basket_main_s0_e500.log
  echo "[$(date)] done  place_bread_basket main seed=0 gpu=4 exit=$?"
) &
(
  echo "[$(date)] start place_cans_plasticbox main seed=0 gpu=5 epochs=500"
  bash train.sh place_cans_plasticbox demo_clean 100 0 5 main 2>&1 | tee ../../Logs/train/place_cans_plasticbox_main_s0_e500.log
  echo "[$(date)] done  place_cans_plasticbox main seed=0 gpu=5 exit=$?"
) &
wait
echo "[$(date)] node=main6t_e500_n0_s0 ALL_DONE"

# map: gpu0=hanging_mug:main, gpu1=place_object_basket:main, gpu2=scan_object:main, gpu3=place_bread_skillet:main, gpu4=place_bread_basket:main, gpu5=place_cans_plasticbox:main
