# PPI x 3 tasks (drawer / laptop / handover_easy) | original + PPI_rp1 | 8GPU
# Submit: name=job line + paste #!/bin/sh body + pick 8GPU
# Log: RLBench2/SPACE/Logs/train_ppi_*_8gpu.log  /  train_ppi_rp1_*_8gpu.log
# PPI_ROOT=RLBench2/SPACE

# Original PPI (EE-proxy)

# ----- ppi_drawer_e1000_8gpu_s0  [8GPU] -----
ppi_drawer_e1000_8gpu_s0
#!/bin/sh
bash RLBench2/SPACE/run_ppi_train_drawer_8gpu.sh

# ----- ppi_laptop_e1000_8gpu_s0  [8GPU] -----
ppi_laptop_e1000_8gpu_s0
#!/bin/sh
bash RLBench2/SPACE/run_ppi_train_laptop_8gpu.sh

# ----- ppi_handover_easy_e1000_8gpu_s0  [8GPU] -----
ppi_handover_easy_e1000_8gpu_s0
#!/bin/sh
bash RLBench2/SPACE/run_ppi_train_handover_easy_8gpu.sh

# =============================================================================
# PPI_rp1 link meshrobot_pc 

# =============================================================================

# ----- ppi_rp1_drawer_e1000_8gpu_s0  [8GPU] -----
ppi_rp1_drawer_e1000_8gpu_s0
#!/bin/sh
bash RLBench2/SPACE/run_ppi_rp1_drawer_8gpu.sh

# ----- ppi_rp1_laptop_e1000_8gpu_s0  [8GPU] -----
ppi_rp1_laptop_e1000_8gpu_s0
#!/bin/sh
bash RLBench2/SPACE/run_ppi_rp1_laptop_8gpu.sh

# ----- ppi_rp1_handover_easy_e1000_8gpu_s0  [8GPU] -----
ppi_rp1_handover_easy_e1000_8gpu_s0
#!/bin/sh
bash RLBench2/SPACE/run_ppi_rp1_handover_easy_8gpu.sh
