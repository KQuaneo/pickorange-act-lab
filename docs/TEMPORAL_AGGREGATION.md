# Strictly paired ACT temporal-aggregation ablation

This inference-only experiment isolates two ACT deployment choices for the G4
A1-B1 14k checkpoint:

1. how many actions are executed before replanning (`H`); and
2. whether overlapping `K=100` chunks are temporally aggregated.

No policy was retrained. All three groups used the same 20 initialization IDs,
serialized robot/object/plate state, restoration procedure, seed, 420 executed
actions and two simulation steps per action.

| Controller | Success | Wilson 95% | Mean/median max lift | Contact-or-better | Calls/episode | Mean wall time |
|---|---:|---:|---:|---:|---:|---:|
| `H=100`, aggregation off | 5/20 | 11.2–46.9% | 0.0427/0.0057 m | 13/20 | 5 | 35.23 s |
| `H=1`, aggregation off | 0/20 | 0.0–16.1% | 0.0121/0.0000 m | 7/20 | 420 | 47.92 s |
| `H=1`, aggregation 0.01 | 5/20 | 11.2–46.9% | 0.0418/0.0045 m | 7/20 | 420 | 48.03 s |

![Strictly paired temporal-aggregation result](../assets/temporal-aggregation-three-way.svg)

## Episode-level transitions

The triple code is `(H=100 off, H=1 off, H=1 aggregation)`:

| Transition | Episodes | Meaning |
|---|---:|---|
| `000` | 13 | all three failed |
| `101` | 3 | H=100 and aggregation succeeded |
| `100` | 2 | only H=100 succeeded |
| `001` | 2 | only temporal aggregation succeeded |

Pairwise exact McNemar results:

- H=100 off versus H=1 off: 5 versus 0 discordant successes, `p=0.0625`;
- H=1 off versus aggregation: 0 versus 5, `p=0.0625`;
- H=100 off versus aggregation: two exclusive successes in each direction,
  `p=1.0`.

The sample supports the narrow observation that temporal aggregation recovered
H=1 performance to the observed H=100 success level. It does **not** support a
claim that aggregation outperformed H=100, nor a general optimal-controller
claim.

## What changed in the control signal

Mean executed-action delta L2 was 0.03051 for H=100, 0.01321 for plain H=1 and
0.02035 for aggregated H=1. Thus temporal aggregation did not act as a simple
sliding-average smoother: it increased motion variation relative to plain H=1
while recovering successful placements. H=100 produced more contact-or-better
episodes (13/20) than either H=1 arm (7/20), despite tying the aggregated arm
on final success.

The paired max-lift difference was +0.02976 m for aggregation relative to plain
H=1 and -0.00089 m relative to H=100. These are descriptive statistics over 20
paired episodes.

## Pairing and integrity

- Physical-state pairing passed for 20/20 episodes.
- H=100 executed `100+100+100+100+20` actions with policy calls at
  `[0, 100, 200, 300, 400]`.
- Checkpoint SHA256, size and modification time were unchanged.
- Separate Isaac processes were not pixel-identical. Camera signatures remain
  a documented rendering warning; initialization IDs, simulator state and the
  observation-refresh protocol were matched.
- The aggregation arm used LeRobot's ACT temporal ensembler over predictions
  for the same absolute time step. It was not a moving average over executed
  actions.

Machine-readable evidence:

- [`temporal_aggregation_three_way.json`](../results/raw/temporal_aggregation_three_way.json)
- [`temporal_aggregation_paired_episodes.jsonl`](../results/raw/temporal_aggregation_paired_episodes.jsonl)

This result applies only to `K=100`, the tested `H` values, coefficient 0.01,
the G4 B1 checkpoint and these paired simulator starts. It does not generalize
to full A0/A1, other ACT checkpoints, other coefficients or hardware.
