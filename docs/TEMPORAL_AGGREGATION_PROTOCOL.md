# Paired ACT temporal aggregation protocol

This inference-only experiment compares the G4 A1-B1 14k checkpoint at fixed
`K=100`, `H=1`, 420 policy actions, two simulation steps per action, seed 2026,
and 20 paired initial states.

- `h1_no_aggregation`: predict a fresh 100-action chunk every environment step
  and execute only the newest chunk's first action.
- `h1_temporal_aggregation_001`: predict a fresh 100-action chunk every step and
  pass the raw chunk through LeRobot `ACTTemporalEnsembler(coeff=0.01)` before
  postprocessing and execution.

No training, checkpoint mutation, controller change, moving average, or other
action smoothing is permitted.

## Safety-gated execution

Pause any server deletion workflow first. Do not run while another protected GPU
task is active.

```bash
python experiments/run_pick_orange_temporal_aggregation.py \
  --checkpoint /path/to/G4_A1_B1_014000/pretrained_model \
  --formal-root outputs/eval/pick_orange_temporal_aggregation \
  --smoke-root outputs/smoke/pick_orange_temporal_aggregation \
  --device cuda:0 \
  --execute \
  --deletion-pause-confirmed
```

The runner performs, in order:

1. resource/path/checkpoint preflight;
2. generation or validation of a fixed 20-state manifest;
3. paired one-episode smoke;
4. paired three-episode smoke;
5. paired 20-episode formal comparison;
6. paired analysis and a separate incremental `tar.gz` archive.

Any smoke failure, active GPU compute process, refreshed-state mismatch, abnormal
Isaac exit, missing output, or checkpoint SHA256 change stops the pipeline.
Formal condition directories are never overwritten.

## Pairing protocol

Each manifest row stores robot joint position/velocity and Orange001/002/003 plus
Plate root pose/velocity. For every episode, the evaluator:

1. resets the environment;
2. restores the manifest state;
3. performs `sim.step(render=False)`, `scene.update`, and one no-op environment
   action to refresh observations and cameras;
4. runs the no-aggregation rollout;
5. repeats steps 1–3 from the same manifest row;
6. compares the two refreshed physical-state vectors with `atol=1e-6`;
7. runs temporal aggregation only if the pair matches.

A mismatch writes `PAIRING_FAILED.json`, raises immediately, and does not create
`COMPLETED`.

## Claim boundary

The result may answer only whether temporal aggregation improved the observed B1
primitive behavior for this checkpoint and paired sample at `K=100`, `H=1`, and
`coeff=0.01`. Twenty episodes are not a precise estimate of the true success
rate. The result does not generalize to all ACT policies, the complete A0/A1
task, hardware, or another temporal coefficient.
