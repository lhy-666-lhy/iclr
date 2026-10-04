# PPI_rp1 x 4 tasks (ball / dustpan=sweep / tray / box=pushbox) | 8GPU | e1000
# Submit: name=job line + paste #!/bin/sh body + pick 8GPU
# Requires: robot_pc / norm_stats / pcd / dino / point_flow
# Log: RLBench2/SPACE/Logs/train_ppi_rp1_{ball,dustpan,tray,box}_8gpu.log
# PPI_ROOT=RLBench2/SPACE

# ----- ppi_rp1_ball_e1000_8gpu_s0  [8GPU] -----
ppi_rp1_ball_e1000_8gpu_s0
#!/bin/sh
bash RLBench2/SPACE/run_ppi_rp1_ball_8gpu.sh

# ----- ppi_rp1_dustpan_e1000_8gpu_s0  [8GPU]  (=sweep) -----
ppi_rp1_dustpan_e1000_8gpu_s0
#!/bin/sh
bash RLBench2/SPACE/run_ppi_rp1_dustpan_8gpu.sh

# ----- ppi_rp1_tray_e1000_8gpu_s0  [8GPU]  (=lifttray) -----
ppi_rp1_tray_e1000_8gpu_s0
#!/bin/sh
bash RLBench2/SPACE/run_ppi_rp1_tray_8gpu.sh

# ----- ppi_rp1_box_e1000_8gpu_s0  [8GPU]  (=pushbox) -----
ppi_rp1_box_e1000_8gpu_s0
#!/bin/sh
bash RLBench2/SPACE/run_ppi_rp1_box_8gpu.sh
