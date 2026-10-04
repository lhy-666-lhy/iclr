# PPI_rp1 eval: box epochs 450 500 550 600 | 4GPU
# Submit: name=job + #!/bin/sh body + 4GPU 

# ckpt: train_ppi_rp1_8gpus_box_rp1_ppi_rp1_20260902-box-fullmesh_seed0

# ----- ppi_rp1_box_eval_e450_500_550_600_4gpu  [4GPU] -----
ppi_rp1_box_eval_e450_500_550_600_4gpu
#!/bin/sh
cd RLBench2/SPACE
TASK=box CKPT_EPOCHS="450 500 550 600" NUM_INFER=1000 TOTAL_EPS=100 NGPU=4 \
  bash scripts/inference/evaluate_ppi_rp1_epochs_4gpu.sh
