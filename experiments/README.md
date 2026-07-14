# Experiment code map

These files are a curated experiment layer from the completed 30-demo study.
They expect to run from a compatible LeIsaac project root.

| File | Role |
|---|---|
| `train_pick_orange_double_steps.sh` | Final batch-64 A0/A1 training, resume and checkpoint-set validation |
| `run_pick_orange_double_steps_train_parallel.sh` | Two-lane training orchestration |
| `run_pick_orange_double_steps_eval.py` | Up-to-three-lane formal evaluation orchestration |
| `eval_pick_orange_joint6_ablation.py` | Full-task A0/A1 evaluator, horizon protocols and overrun traces |
| `eval_pick_orange_a1_isolated.py` | B1/B2/B3 isolated evaluator with oracle provenance |
| `audit_pick_orange_effective_data.py` | Frame/window/exposure accounting |
| `audit_pick_orange_b3_slices.py` | Video, timestamp, finite-value and strict-prefix audit |
| `prepare_strict_prefix_gate3.py` | Strict-prefix Gate 3 preparation |
| `pick_orange_analysis.py` | Dependency-light statistics and pairing helpers |
| `analyze_pick_orange_results.py` | Paired and two-level result analysis |
| `smoke_pick_orange_eval_protocol.py` | Dry-by-default protocol and resource-safety smoke test |
| `run_pick_orange_30_only_pipeline.py` | Persistent retrying eval-to-report state machine |
| `status_pick_orange_30_only.py` | Atomic live status writer |
| `report_pick_orange_30_only.py` | Final 30-demo report aggregation |

## Safety notes

- Simulator execution in the smoke test requires `--execute`.
- Formal outputs and smoke outputs use separate directory trees.
- Horizon protocol names are required in explicitly selected output paths.
- The training shell can move a failed pre-checkpoint run aside and prune
  transient checkpoints inside its target output directory. Review paths before
  use.
- The code intentionally does not contain datasets, checkpoints, simulator
  assets or an environment activation script.

