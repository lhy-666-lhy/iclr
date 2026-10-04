#!/usr/bin/env bash
# RGB train — same abN_exM type ids as train.sh
set -euo pipefail

task_name=${1:?"usage: bash train_rgb.sh <task> <config> <n> <seed> <gpu> [abN_exM] [run_tag]"}
task_config=${2:?"missing task_config"}
expert_data_num=${3:?"missing expert_data_num"}
seed=${4:?"missing seed"}
gpu_id=${5:?"missing gpu_id"}
encoder_type=${6:-main}
run_tag=${7:-train_${encoder_type}_rgb}

VALID_TYPES="main ab1_ex1 ab1_ex2 ab1_ex3 ab1_ex4 ab1ex2_hp1 ab1ex2_hp2 ab1ex2_hp3 ab1ex2_hp4 ab2_ex2 ab2_ex3 ab2_ex4 ab3_ex1 ab3_ex2 ab3_ex3 ab3_ex4 ab3_ex5 ab3_ex6 ab4_ex1 ab4_ex2 ab4_ex3 ab_exp1 ab_exp2 ab_exp3 ab_exp4 ab_exp5 ab_exp6 ex1 ex2 ex3 ex4 geo topo surf8 surf16 surf32 inner8 inner16 inner32 pn pn2 pt"
if [[ ! " ${VALID_TYPES} " =~ " ${encoder_type} " ]]; then
  echo -e "\033[31m[train_rgb.sh] unknown encoder_type='${encoder_type}'\033[0m"
  exit 1
fi

echo -e "\033[36m[train_rgb] type=${encoder_type} run_tag=${run_tag}\033[0m"

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh "${task_name}" "${task_config}" "${expert_data_num}"
fi

bash scripts/train_policy_rgb.sh \
  robot_space \
  "${task_name}" \
  "${task_config}" \
  "${expert_data_num}" \
  "${run_tag}" \
  "${seed}" \
  "${gpu_id}" \
  "${encoder_type}"
