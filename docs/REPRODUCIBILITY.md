# Reproduction guide

## Tested environment

| Component | Version / commit |
|---|---|
| OS | Ubuntu 22.04 |
| Python | 3.11.15 |
| LeIsaac | `24d3bcd3f1e4585740fc79921782c41617237812` plus the experiment layer in this repo |
| Isaac Lab submodule | `37ddf626871758333d6ed89cf64ad702aef127d0` |
| Isaac Lab Python package | 0.54.2 |
| LeRobot | 0.4.4 |
| PyTorch | 2.7.0+cu128 |
| NumPy | 1.26.0 |
| PyArrow | 24.0.0 |
| CUDA / cuDNN | 12.8 / 9.7.1 |
| GPU | NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB |
| Driver | 580.159.03 |

Isaac Lab/LeIsaac installations are environment-specific. Follow the upstream
[LeIsaac setup guide](https://lightwheelai.github.io/leisaac/docs/getting-started/)
first. This repository is the experiment layer, not a vendored simulator.

## Installation layout

Copy or symlink `experiments/` and `configs/` into a compatible LeIsaac checkout
so that `experiments/` is directly below the project root. The training scripts
expect an `activate.sh` supplied by the local LeIsaac environment.

For analysis-only tests:

```bash
python -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r requirements-analysis.txt
pytest experiments/tests -q
python tools/validate_public_repo.py
```

## Dataset protocol

The dataset itself is not published here. To reproduce the reported study, use:

- 30 full expert episodes at 30 FPS;
- front and wrist RGB plus joint state/action;
- a successful final three-orange episode condition;
- stable placement for 10 consecutive frames;
- Gate 2 event slicing;
- Gate 3 target success plus strict preservation of all prefix oranges.

Run the effective-data and B3 audits before training:

```bash
python experiments/audit_pick_orange_effective_data.py --help
python experiments/audit_pick_orange_b3_slices.py --help
python experiments/prepare_strict_prefix_gate3.py --help
```

Do not infer a B3 success label from a 340-frame slice. Valid releases in this
study occurred at actions 350–358.

## Training

The final scripts keep batch size 64 and use two training lanes by default:

```bash
tmux new-session -d -s pickorange_train \
  'TRAIN_MAX_PARALLEL=2 bash experiments/run_pick_orange_double_steps_train_parallel.sh 30'
```

Expected final checkpoints:

```text
A0: 030000, 036000, 042000
A1/B1: 010000, 012000, 014000
A1/B2: 010000, 012000, 014000
A1/B3: 010000, 012000, 014000
```

Review dataset names and output roots in the script before execution. It can
resume a run and prune transient recovery checkpoints within that run.

## Horizon protocols

The default is unchanged:

```text
native_horizon:
  A0 = 1020 policy actions = 2040 simulation steps ≈ 34 s
  A1 = 420 × 3 actions = 2520 simulation steps ≈ 42 s
```

Matched horizon is opt-in:

```text
matched_horizon:
  A0 = 1260 policy actions = 2520 simulation steps ≈ 42 s
  A1 = 420 × 3 actions = 2520 simulation steps ≈ 42 s
```

Explicit protocol runs require the output directory to contain the protocol
name. Replace checkpoint placeholders with local paths:

```bash
python experiments/eval_pick_orange_joint6_ablation.py \
  --group a0 \
  --checkpoint /path/to/a0/pretrained_model \
  --episodes 20 --seed 2026 \
  --horizon_protocol native_horizon \
  --output_dir outputs/eval/pickorange/a0_native_horizon \
  --device cuda --headless

python experiments/eval_pick_orange_joint6_ablation.py \
  --group a0 \
  --checkpoint /path/to/a0/pretrained_model \
  --episodes 20 --seed 2026 \
  --horizon_protocol matched_horizon \
  --output_dir outputs/eval/pickorange/a0_matched_horizon \
  --device cuda --headless

python experiments/eval_pick_orange_joint6_ablation.py \
  --group a1 \
  --checkpoint_001 /path/to/b1/pretrained_model \
  --checkpoint_002 /path/to/b2/pretrained_model \
  --checkpoint_003 /path/to/b3/pretrained_model \
  --policy_steps_per_orange 420 \
  --episodes 20 --seed 2026 \
  --horizon_protocol native_horizon \
  --output_dir outputs/eval/pickorange/a1_native_horizon \
  --device cuda --headless
```

## Smoke test

Run dry validation first. Actual simulator execution is intentionally gated:

```bash
python experiments/smoke_pick_orange_eval_protocol.py \
  --episodes 1 \
  --a0-checkpoint /path/to/a0/pretrained_model \
  --b1-checkpoint /path/to/b1/pretrained_model \
  --b2-checkpoint /path/to/b2/pretrained_model \
  --b3-checkpoint /path/to/b3/pretrained_model \
  --b3-dataset /path/to/b3/file-000.parquet

# Only after checking GPU memory and active training processes:
python experiments/smoke_pick_orange_eval_protocol.py <same-arguments> --execute
```

Smoke status uses `PASS`, `WARN`, `FAIL`, and `SKIPPED`, split into code/input,
simulation, and resource-safety sections. Smoke output is excluded from formal
success-rate aggregation.

## Rebuild public charts

```bash
python tools/render_result_charts.py
```

The chart generator uses only the Python standard library and reads
`results/summary.json`.

The three-way temporal-aggregation analyzer operates on full evaluator output
directories and verifies 20 episodes, policy-call boundaries, checkpoint
immutability and physical-state pairing before writing statistics:

```bash
python experiments/analyze_temporal_three_way.py \
  --h100-off /path/to/h100_no_aggregation_paired \
  --h1-off /path/to/h1_no_aggregation \
  --h1-aggregation /path/to/h1_temporal_aggregation_001 \
  --output /path/to/analysis_three_way
```

The repository vendors the resulting sanitized summary and 20-row paired
episode table, but not checkpoints or full simulator logs.

## Reproducibility boundary

This repository publishes code, configuration, sanitized evaluation records,
figures and representative media. It intentionally omits simulator assets,
expert datasets, checkpoints and large runtime outputs. As a result, readers can
fully verify the reported tables and analysis code, but cannot rerun policies
without obtaining compatible LeIsaac assets and local model/data artifacts.
