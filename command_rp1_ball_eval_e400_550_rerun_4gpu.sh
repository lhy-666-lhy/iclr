# PPI_rp1 eval RERUN: ball epochs 400 450 500 550 | 4GPU
# ckpt: train_ppi_rp1_8gpus_ball_rp1_ppi_rp1_20260902-ball-fullmesh_seed0
#   exp_logs/eval_runs/ball_epochs_seq_${RUN_TAG}/
#   exp_logs/eval_runs/ball_4gpu_epoch{400,450,500,550}_model_peract2seeds_infer1000_${RUN_TAG}/
# Submit: name=job + paste #!/bin/sh body + pick 4GPU

# ----- ppi_rp1_ball_eval_e400_450_500_550_rerun_4gpu  [4GPU] -----
ppi_rp1_ball_eval_e400_450_500_550_rerun_4gpu
#!/bin/sh
cd RLBench2/SPACE
TASK=ball \
CKPT_EPOCHS="400 450 500 550" \
NUM_INFER=1000 \
TOTAL_EPS=100 \
NGPU=4 \
RUN_TAG="ball_e400_550_rerun_$(date +%Y%m%d_%H%M%S)_$(hostname -s)" \
  bash scripts/inference/evaluate_ppi_rp1_epochs_4gpu.sh
