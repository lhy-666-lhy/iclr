#!/bin/sh
# Precompute robot point clouds for SPACE zarr datasets.
#
# Usage:
#   bash scripts/precompute_robot_pc.sh handover_block demo_clean 100
#   bash scripts/precompute_robot_pc.sh --all demo_clean 100
#   bash scripts/precompute_robot_pc.sh --all demo_clean 100 --overwrite

cd "$(dirname "$0")/.." || exit 1

python scripts/precompute_robot_pc.py "$@"
