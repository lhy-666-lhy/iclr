#!/bin/sh
# Platform jobs: Robot-DP3 main H16, 6 RLBench2 tasks, seed0 then seed1 then seed2, num_epochs=3000.
# Copy the job-name line into the platform name field.
# Paste the #!/bin/sh block as the launch command.
# HORIZON=16  N_ACTION_STEPS=8  N_OBS_STEPS=3  NUM_EPOCHS=3000
# RESUME=true
# Checkpoint:
#   RLBench2/SPACE/exp_logs/ckpt/<TASK>/train_robot_dp3_main_<TASK>_robot_dp3_main_main_h16_seed<SEED>/
# Tasks:
#   bimanual_handover_item_easy
#   bimanual_pick_laptop
#   bimanual_lift_ball
#   bimanual_push_box
#   bimanual_lift_tray
#   bimanual_sweep_to_dustpan

# ----- [0/6] rlb2_handover_h16_s012 -----
rlb2_handover_h16_s012
#!/bin/sh
export WANDB_MODE=offline
cd RLBench2/SPACE
TASK=bimanual_handover_item_easy
for SEED in 0 1 2; do
  echo "[$(date)] start ${TASK} main H=16 Ta=8 epochs=3000 seed=${SEED}"
  HORIZON=16 N_ACTION_STEPS=8 N_OBS_STEPS=3 NUM_EPOCHS=3000 RESUME=true \
  ADDITION_INFO=main_h16 \
  LOG=./${TASK}_main_h16_s${SEED}.log \
  GPU=0 \
  bash scripts/training/train_robot_dp3.sh "${TASK}" "${SEED}"
  echo "[$(date)] done  ${TASK} main H=16 Ta=8 seed=${SEED}"
done

# ----- [1/6] rlb2_laptop_h16_s012 -----
rlb2_laptop_h16_s012
#!/bin/sh
export WANDB_MODE=offline
cd RLBench2/SPACE
TASK=bimanual_pick_laptop
for SEED in 0 1 2; do
  echo "[$(date)] start ${TASK} main H=16 Ta=8 epochs=3000 seed=${SEED}"
  HORIZON=16 N_ACTION_STEPS=8 N_OBS_STEPS=3 NUM_EPOCHS=3000 RESUME=true \
  ADDITION_INFO=main_h16 \
  LOG=./${TASK}_main_h16_s${SEED}.log \
  GPU=0 \
  bash scripts/training/train_robot_dp3.sh "${TASK}" "${SEED}"
  echo "[$(date)] done  ${TASK} main H=16 Ta=8 seed=${SEED}"
done

# ----- [2/6] rlb2_ball_h16_s012 -----
rlb2_ball_h16_s012
#!/bin/sh
export WANDB_MODE=offline
cd RLBench2/SPACE
TASK=bimanual_lift_ball
for SEED in 0 1 2; do
  echo "[$(date)] start ${TASK} main H=16 Ta=8 epochs=3000 seed=${SEED}"
  HORIZON=16 N_ACTION_STEPS=8 N_OBS_STEPS=3 NUM_EPOCHS=3000 RESUME=true \
  ADDITION_INFO=main_h16 \
  LOG=./${TASK}_main_h16_s${SEED}.log \
  GPU=0 \
  bash scripts/training/train_robot_dp3.sh "${TASK}" "${SEED}"
  echo "[$(date)] done  ${TASK} main H=16 Ta=8 seed=${SEED}"
done

# ----- [3/6] rlb2_box_h16_s012 -----
rlb2_box_h16_s012
#!/bin/sh
export WANDB_MODE=offline
cd RLBench2/SPACE
TASK=bimanual_push_box
for SEED in 0 1 2; do
  echo "[$(date)] start ${TASK} main H=16 Ta=8 epochs=3000 seed=${SEED}"
  HORIZON=16 N_ACTION_STEPS=8 N_OBS_STEPS=3 NUM_EPOCHS=3000 RESUME=true \
  ADDITION_INFO=main_h16 \
  LOG=./${TASK}_main_h16_s${SEED}.log \
  GPU=0 \
  bash scripts/training/train_robot_dp3.sh "${TASK}" "${SEED}"
  echo "[$(date)] done  ${TASK} main H=16 Ta=8 seed=${SEED}"
done

# ----- [4/6] rlb2_tray_h16_s012 -----
rlb2_tray_h16_s012
#!/bin/sh
export WANDB_MODE=offline
cd RLBench2/SPACE
TASK=bimanual_lift_tray
for SEED in 0 1 2; do
  echo "[$(date)] start ${TASK} main H=16 Ta=8 epochs=3000 seed=${SEED}"
  HORIZON=16 N_ACTION_STEPS=8 N_OBS_STEPS=3 NUM_EPOCHS=3000 RESUME=true \
  ADDITION_INFO=main_h16 \
  LOG=./${TASK}_main_h16_s${SEED}.log \
  GPU=0 \
  bash scripts/training/train_robot_dp3.sh "${TASK}" "${SEED}"
  echo "[$(date)] done  ${TASK} main H=16 Ta=8 seed=${SEED}"
done

# ----- [5/6] rlb2_dustpan_h16_s012 -----
rlb2_dustpan_h16_s012
#!/bin/sh
export WANDB_MODE=offline
cd RLBench2/SPACE
TASK=bimanual_sweep_to_dustpan
for SEED in 0 1 2; do
  echo "[$(date)] start ${TASK} main H=16 Ta=8 epochs=3000 seed=${SEED}"
  HORIZON=16 N_ACTION_STEPS=8 N_OBS_STEPS=3 NUM_EPOCHS=3000 RESUME=true \
  ADDITION_INFO=main_h16 \
  LOG=./${TASK}_main_h16_s${SEED}.log \
  GPU=0 \
  bash scripts/training/train_robot_dp3.sh "${TASK}" "${SEED}"
  echo "[$(date)] done  ${TASK} main H=16 Ta=8 seed=${SEED}"
done
