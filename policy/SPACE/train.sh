#!/usr/bin/env bash
# Train robot-aware SPACE — use exact paper ids: main | ab{N}_ex{M}
#
# Usage:
#   bash train.sh <task> <task_config> <n> <seed> <gpu> [encoder_type] [run_tag]
#
# Canonical encoder_type (recommended):
#   main
#   ab1_ex1 | ab1_ex2 | ab1_ex3 | ab1_ex4          # Ablation-1 fusion
#   ab1ex2_hp1 | ab1ex2_hp2 | ab1ex2_hp3 | ab1ex2_hp4  # ab1_ex2 hand-MLP variants
#   ab2_ex2 | ab2_ex3 | ab2_ex4                    # Ablation-2 (no ab2_ex1)
#   ab3_ex1 | ab3_ex2 | ab3_ex3 | ab3_ex4 | ab3_ex5 | ab3_ex6
#   ab4_ex1 | ab4_ex2 | ab4_ex3                    # robot geo/topo backbone
#   ab_exp1 | ab_exp2 | ab_exp3 | ab_exp4 | ab_exp5 | ab_exp6  # topo-first late fusion
#
# Defaults: encoder_type=main , run_tag=train_<encoder_type>
#
# Optional MLP widths (env):
#   HAND_H  → policy.robot_aware_hand_hidden_dim      (Cross-Arm)
#   INJ_H   → policy.robot_aware_injection_hidden_dim (Injection)
# Examples:
#   HAND_H=512 INJ_H=512 bash train.sh handover_block demo_clean 100 0 0 ab1_ex1 train_ab1_ex1_h512_i512
#   HAND_H=1024 bash train.sh handover_block demo_clean 100 0 0 ab1_ex2 train_ab1_ex2_h1024
#
# Examples:
#   bash train.sh handover_block demo_clean 100 0 0 main
#   bash train.sh handover_block demo_clean 100 0 0 ab1_ex2
#   bash train.sh handover_block demo_clean 100 0 0 ab1ex2_hp1
#   bash train.sh handover_block demo_clean 100 0 0 ab3_ex5
# Eval pair:
#   bash eval.sh handover_block demo_clean train_ab1_ex2 100 0 0 ab1_ex2

set -euo pipefail

task_name=${1:?"usage: bash train.sh <task> <config> <n> <seed> <gpu> [abN_exM|main] [run_tag]"}
task_config=${2:?"missing task_config"}
expert_data_num=${3:?"missing expert_data_num"}
seed=${4:?"missing seed"}
gpu_id=${5:?"missing gpu_id"}
encoder_type=${6:-main}
run_tag=${7:-train_${encoder_type}}

# Canonical + short aliases
VALID_TYPES="main ab1_ex1 ab1_ex2 ab1_ex3 ab1_ex4 ab1ex2_hp1 ab1ex2_hp2 ab1ex2_hp3 ab1ex2_hp4 ab2_ex2 ab2_ex3 ab2_ex4 ab3_ex1 ab3_ex2 ab3_ex3 ab3_ex4 ab3_ex5 ab3_ex6 ab4_ex1 ab4_ex2 ab4_ex3 ab_exp1 ab_exp2 ab_exp3 ab_exp4 ab_exp5 ab_exp6 ex1 ex2 ex3 ex4 geo topo surf8 surf16 surf32 inner8 inner16 inner32 pn pn2 pt"
if [[ ! " ${VALID_TYPES} " =~ " ${encoder_type} " ]]; then
  echo -e "\033[31m[train.sh] unknown encoder_type='${encoder_type}'\033[0m"
  echo "Canonical: main ab1_ex{1..4} ab2_ex{2,3,4} ab3_ex{1..6} ab4_ex{1,2,3} ab_exp{1..6}"
  exit 1
fi

echo -e "\033[36m[train.sh] task=${task_name} config=${task_config} n=${expert_data_num}\033[0m"
echo -e "\033[36m[train.sh] seed=${seed} gpu=${gpu_id}\033[0m"
echo -e "\033[36m[train.sh] encoder_type=${encoder_type}  run_tag=${run_tag}\033[0m"
echo -e "\033[36m[train.sh] ckpt dir: checkpoints/${task_name}-${run_tag}-${expert_data_num}_${seed}/\033[0m"

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    bash process_data.sh "${task_name}" "${task_config}" "${expert_data_num}"
fi

# setting=run_tag so ckpt path matches eval.sh (train_<encoder>).
# zarr still comes from task_config (demo_clean), overridden in train_policy.sh.
bash scripts/train_policy.sh \
  robot_space \
  "${task_name}" \
  "${run_tag}" \
  "${expert_data_num}" \
  "${run_tag}" \
  "${seed}" \
  "${gpu_id}" \
  "${encoder_type}" \
  "${task_config}"
