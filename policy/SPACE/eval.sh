#!/usr/bin/env bash
# Eval — encoder_type and ckpt_setting use the same abN_exM id as training.
#
# Usage:
#   bash eval.sh <task> <config> <ckpt_setting> <n> <seed> <gpu> <encoder_type>
#
# Rule:
#   train:  bash train.sh TASK CFG N SEED GPU ab1_ex2
#           → run_tag / ckpt_setting = train_ab1_ex2
#   eval:   bash eval.sh  TASK CFG train_ab1_ex2 N SEED GPU ab1_ex2
#
# Examples:
#   bash eval.sh handover_block demo_clean train_main    100 0 0 main
#   bash eval.sh handover_block demo_clean train_ab1_ex2 100 0 0 ab1_ex2
#   bash eval.sh handover_block demo_clean train_ab3_ex5 100 0 0 ab3_ex5
#   bash eval.sh handover_block demo_clean train_ab4_ex2 100 0 0 ab4_ex2

set -euo pipefail

policy_name=SPACE
task_name=${1:?"usage: bash eval.sh <task> <config> train_<abN_exM> <n> <seed> <gpu> <abN_exM>"}
task_config=${2:?"missing task_config"}
ckpt_setting=${3:?"missing ckpt_setting e.g. train_ab1_ex2"}
expert_data_num=${4:?"missing expert_data_num"}
seed=${5:?"missing seed"}
gpu_id=${6:?"missing gpu_id"}
encoder_type=${7:-main}

VALID_TYPES="main ab1_ex1 ab1_ex2 ab1_ex3 ab1_ex4 ab1ex2_hp1 ab1ex2_hp2 ab1ex2_hp3 ab1ex2_hp4 ab2_ex2 ab2_ex3 ab2_ex4 ab3_ex1 ab3_ex2 ab3_ex3 ab3_ex4 ab3_ex5 ab3_ex6 ab4_ex1 ab4_ex2 ab4_ex3 ab_exp1 ab_exp2 ab_exp3 ab_exp4 ab_exp5 ab_exp6 ex1 ex2 ex3 ex4 geo topo surf8 surf16 surf32 inner8 inner16 inner32 pn pn2 pt"
if [[ ! " ${VALID_TYPES} " =~ " ${encoder_type} " ]]; then
  echo -e "\033[31m[eval.sh] unknown encoder_type='${encoder_type}'\033[0m"
  echo "Canonical: main ab1_ex{1..4} ab2_ex{2,3,4} ab3_ex{1..6} ab4_ex{1,2,3}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu: ${gpu_id} | ckpt_setting: ${ckpt_setting} | type: ${encoder_type}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --robot_aware_encoder_type ${encoder_type} \
    --use_robot_aware true
