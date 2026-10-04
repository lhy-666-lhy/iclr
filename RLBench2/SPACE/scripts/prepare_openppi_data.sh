#!/usr/bin/env bash
# Prepare Open-PPI data for paper PPI training (7 RLBench2 tasks).
# Uses HF mirror. Prefer official training_processed over local regeneration.
#
# Usage:
#   bash scripts/prepare_openppi_data.sh
#   bash scripts/prepare_openppi_data.sh --download-only
#   bash scripts/prepare_openppi_data.sh --extract-only
#   bash scripts/prepare_openppi_data.sh --fix-push-box
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

PPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$PPI_ROOT/data"
TEMP_RAW="RLBench2/temp/Open-PPI/training_raw_tar"
LOG="$PPI_ROOT/Logs"
mkdir -p "$LOG" "$DATA/training_raw" "$DATA/training_processed" "$PPI_ROOT/pretrained_models"

TASKS=(
  bimanual_handover_item_easy
  bimanual_lift_ball
  bimanual_lift_tray
  bimanual_pick_laptop
  bimanual_push_box
  bimanual_put_item_in_drawer
  bimanual_sweep_to_dustpan
)

link_raw() {
  echo "[link] training_raw -> local extracted demos"
  for t in "${TASKS[@]}"; do
    src="$TEMP_RAW/$t"
    dst="$DATA/training_raw/$t"
    if [[ -d "$src" ]]; then
      ln -sfn "$src" "$dst"
      n=$(ls -d "$src/all_variations/episodes"/episode* 2>/dev/null | wc -l)
      echo "  $t episodes≈$n"
    else
      echo "  MISSING $src"
    fi
  done
}

download_processed() {
  echo "[download] training_processed via $HF_ENDPOINT"
  PPI_DATA="$DATA" python3 - <<'PY'
import os
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="yuyinyang3y/Open-PPI",
    repo_type="dataset",
    allow_patterns=["training_processed/**"],
    local_dir=os.environ["PPI_DATA"],
    max_workers=8,
)
print("downloaded to", path)
PY
}

extract_processed() {
  echo "[extract] unpack *.tar under training_processed"
  find "$DATA/training_processed" -type f -name '*.tar' | sort | while read -r tarf; do
    dest="$(dirname "$tarf")"
    echo "  extracting $(basename "$tarf") -> $dest"
    tar -xf "$tarf" -C "$dest"
  done
  echo "[check] instruction_embeddings.pkl"
  ls -lah "$DATA/training_processed/instruction_embeddings.pkl"
  echo "[check] norm_stats count=$(ls "$DATA/training_processed/norm_stats" | wc -l)"
  for t in "${TASKS[@]}"; do
    for m in point_cloud dino_feature point_flow; do
      ep="$DATA/training_processed/$m/$t/all_variations/episodes"
      n=$(ls -d "$ep"/episode* 2>/dev/null | wc -l)
      echo "  $m/$t episodes=$n"
    done
  done
}

fix_push_box() {
  echo "[fix] re-download bimanual_push_box episodes_0_19.tar"
  OUT="$TEMP_RAW/bimanual_push_box/all_variations/episodes"
  mkdir -p "$OUT" "$(dirname "$TEMP_RAW")/_hf_dl"
  python3 - <<'PY'
import os, shutil, subprocess
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="yuyinyang3y/Open-PPI",
    repo_type="dataset",
    filename="training_raw_tar/bimanual_push_box/all_variations/episodes/episodes_0_19.tar",
    local_dir="RLBench2/temp/Open-PPI/_hf_dl",
)
dst = "RLBench2/temp/Open-PPI/training_raw_tar/bimanual_push_box/all_variations/episodes/episodes_0_19.tar"
shutil.copy2(p, dst)
print("size_GB", round(os.path.getsize(dst)/1e9, 3))
eps = subprocess.check_output(
    ["bash", "-lc", f"tar -tf {dst} | awk -F/ '{{print \$1}}' | sort -u"]
).decode().strip().splitlines()
print("n_eps", len(eps), "sample", eps[:5])
if len(eps) < 20:
    raise SystemExit(f"still incomplete tar: only {len(eps)} episodes — upstream may be broken")
subprocess.check_call(["tar", "-xf", dst, "-C", os.path.dirname(dst)])
print("extracted ok")
PY
}

mode="${1:-all}"
case "$mode" in
  --download-only) link_raw; download_processed ;;
  --extract-only) extract_processed ;;
  --fix-push-box) fix_push_box ;;
  all|--all|"")
    link_raw
    download_processed
    extract_processed
    echo "[note] if push_box missing ep1-19: bash scripts/prepare_openppi_data.sh --fix-push-box"
    ;;
  *)
    echo "unknown arg: $mode"; exit 1 ;;
esac

echo "[DONE] data under $DATA"
