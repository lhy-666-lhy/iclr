#!/usr/bin/env bash
# Recover weights left in *-demo_clean-* dirs (last overwrite) into the correct
# train_<encoder_type> directory, based on cfg inside the ckpt.
#
# Usage:
#   bash policy/SPACE/scripts/recover_demo_clean_ckpts.sh
#   bash policy/SPACE/scripts/recover_demo_clean_ckpts.sh --dry-run
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_ROOT="$ROOT/policy/SPACE/checkpoints"
PY="python"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# Remove broken train_* → demo_clean symlinks created by old run_eval_8x4090.sh
echo "[INFO] removing symlink leftovers under train_* dirs"
find "$CKPT_ROOT" -maxdepth 2 -type l -name '3000.ckpt' | while read -r link; do
  target="$(readlink -f "$link" || true)"
  if [[ "$target" == *"-demo_clean-"* ]]; then
    echo "  rm $link -> $target"
    if (( DRY_RUN == 0 )); then
      rm -f "$link"
    fi
  fi
done

echo "[INFO] mapping demo_clean ckpts by embedded encoder_type"
for src_dir in "$CKPT_ROOT"/*-demo_clean-*; do
  [[ -d "$src_dir" ]] || continue
  src="$src_dir/3000.ckpt"
  [[ -f "$src" && ! -L "$src" ]] || continue

  info="$("$PY" - <<PY
import torch, dill
from pathlib import Path
p = Path(r"$src")
payload = torch.load(p.open("rb"), pickle_module=dill, map_location="cpu")
enc = str(payload["cfg"].policy.robot_aware_encoder_type)
# parent dir: <task>-demo_clean-<n>_<seed>
name = Path(r"$src_dir").name
# split from the right: task may contain dashes? tasks here are underscore words
# pattern: {task}-demo_clean-{n}_{seed}
import re
m = re.match(r"^(?P<task>.+)-demo_clean-(?P<n>\d+)_(?P<seed>\d+)$", name)
if not m:
    raise SystemExit(f"bad dir name: {name}")
print(m.group("task"))
print(m.group("n"))
print(m.group("seed"))
print(enc)
PY
)"
  task="$(echo "$info" | sed -n '1p')"
  n="$(echo "$info" | sed -n '2p')"
  seed="$(echo "$info" | sed -n '3p')"
  enc="$(echo "$info" | sed -n '4p')"

  dst_dir="$CKPT_ROOT/${task}-train_${enc}-${n}_${seed}"
  dst="$dst_dir/3000.ckpt"
  echo "[MAP] $(basename "$src_dir")  encoder=${enc}"
  echo "      -> $dst"
  if [[ -e "$dst" && ! -L "$dst" ]]; then
    echo "      keep existing real file"
    continue
  fi
  if (( DRY_RUN )); then
    echo "      (dry-run) would copy"
    continue
  fi
  mkdir -p "$dst_dir"
  rm -f "$dst"
  cp -n "$src" "$dst" || cp "$src" "$dst"
  echo "      copied"
done

echo "[DONE]"
