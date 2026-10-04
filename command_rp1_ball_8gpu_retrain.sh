# PPI_rp1 lift_ball RETRAIN | 8GPU | full DualPanda link mesh (no EE-proxy)
# Fresh run: addition_info=20260906-ball-fullmesh-retrain (does NOT overwrite 20260902-ball-fullmesh)
# Prerequisite: precompute robot_pc first (see precompute_dual_panda_robot_pc.sh)
# Submit: name=job line; paste #!/bin/sh body; pick 8GPU

# ----- ppi_rp1_ball_e1000_8gpu_s0_retrain  [8GPU] -----
ppi_rp1_ball_e1000_8gpu_s0_retrain
#!/bin/sh
bash RLBench2/SPACE/run_ppi_rp1_ball_8gpu.sh
