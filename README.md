# SPACE

Code for **SPACE**: a robot-aware 3D diffusion policy that encodes robot geometry and topology as extra tokens.

This snapshot covers two simulators:

| Benchmark | Path | Entry |
|-----------|------|--------|
| RobotWin / Sapien | `policy/SPACE` | `train.sh`, `eval.sh` |
| RLBench2 (bimanual) | `RLBench2/SPACE` | `ddp_train.py`, `run_8gpu.sh` |

## Layout

```
policy/SPACE/          # SPACE on RobotWin
  train.sh eval.sh
  spacepolicy/         # training code (package: backbone)
  block/               # robot PC, fusion
RLBench2/SPACE/        # SPACE on RLBench2
  space/               # policy package
  ddp_train.py
script/                # env install
```

## RobotWin

Install:

```bash
bash script/_install.sh
cd policy/SPACE
```

Train / eval (from `policy/SPACE`):

```bash
bash train.sh <task> demo_clean 100 0 0 main
bash eval.sh  <task> demo_clean train_main 100 0 0 main
```

`encoder_type` uses paper ids: `main`, `ab1_ex1`–`ab1_ex4`, `ab2_ex2`–`ab2_ex4`, `ab3_ex1`–`ab3_ex6`, `ab4_ex1`–`ab4_ex3`.

## RLBench2

```bash
cd RLBench2/SPACE
pip install -e . -r requirements.txt
# set COPPELIASIM_ROOT, then:
bash run_8gpu.sh
# or:
torchrun --nproc_per_node 8 ddp_train.py --config-name ppi_rp1 task=ball_rp1
```

Set `WANDB_MODE=offline` unless you need logging.

## Notes

- Checkpoints go under `policy/SPACE/checkpoints` or `RLBench2/SPACE/exp_logs`.
- Third-party weights (CLIP, DINOv2) are not in this repo; pass local paths or let the official download run.
- Paper details (tasks, ablations, hyperparameters) are in the PDF.
