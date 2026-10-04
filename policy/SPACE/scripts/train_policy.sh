#!/usr/bin/env bash
DEBUG=False
save_ckpt=True

alg_name=${1}
# task choices: See TASK.md
task_name=${2}
setting=${3}
expert_data_num=${4}
config_name=${alg_name}
addition_info=${5}
seed=${6}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${7}
# robot_aware_encoder_type: main | ex1.. | geo|topo | surf* | inner* | pn|pn2|pt
encoder_type=${8:-main}
# Data zarr uses task_config (e.g. demo_clean); setting may be train_<encoder> for ckpt isolation.
task_config=${9:-${setting}}
zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}.zarr"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mrobot_aware_encoder_type: ${encoder_type}\033[0m"
echo -e "\033[33msetting (ckpt): ${setting}\033[0m"
echo -e "\033[33mzarr: ${zarr_path}\033[0m"
echo -e "\033[33mrun_dir: ${run_dir}\033[0m"

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd spacepolicy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Optional MLP width overrides (Cross-Arm / Injection):
#   HAND_H=512 INJ_H=512 bash train.sh ... ab1_ex1
EXTRA_OVERRIDES=()
if [[ -n "${HAND_H:-}" ]]; then
  EXTRA_OVERRIDES+=( "policy.robot_aware_hand_hidden_dim=${HAND_H}" )
  echo -e "\033[33mhand_hidden_dim=${HAND_H}\033[0m"
fi
if [[ -n "${INJ_H:-}" ]]; then
  EXTRA_OVERRIDES+=( "policy.robot_aware_injection_hidden_dim=${INJ_H}" )
  echo -e "\033[33minjection_hidden_dim=${INJ_H}\033[0m"
fi
# Optional short-run overrides, e.g. NUM_EPOCHS=500 RESUME=false bash train.sh ...
if [[ -n "${NUM_EPOCHS:-}" ]]; then
  EXTRA_OVERRIDES+=( "training.num_epochs=${NUM_EPOCHS}" )
  EXTRA_OVERRIDES+=( "training.checkpoint_every=${CHECKPOINT_EVERY:-${NUM_EPOCHS}}" )
  echo -e "\033[33mnum_epochs=${NUM_EPOCHS}  checkpoint_every=${CHECKPOINT_EVERY:-${NUM_EPOCHS}}\033[0m"
fi
if [[ -n "${RESUME:-}" ]]; then
  EXTRA_OVERRIDES+=( "training.resume=${RESUME}" )
  echo -e "\033[33mresume=${RESUME}\033[0m"
fi

python train_space.py --config-name=${config_name}.yaml \
                            task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            expert_data_num=${expert_data_num} \
                            setting=${setting} \
                            policy.use_robot_aware=true \
                            policy.robot_aware_encoder_type=${encoder_type} \
                            task.dataset.robot_aware_encoder_type=${encoder_type} \
                            task.dataset.zarr_path=${zarr_path} \
                            task.dataset.use_robot_pc_cache=true \
                            "${EXTRA_OVERRIDES[@]}"
