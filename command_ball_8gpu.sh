# PPI lift_ball | 8GPU | paper script settings
# Submit: name=job line; paste #!/bin/sh body; pick 8GPU
# Wrapper -> run_ppi_train_ball_8gpu.sh (stats path is one line, pipefail-safe)

# ----- ppi_ball_e1000_8gpu_s0  [8GPU] -----
ppi_ball_e1000_8gpu_s0
#!/bin/sh
bash RLBench2/SPACE/run_ppi_train_ball_8gpu.sh
