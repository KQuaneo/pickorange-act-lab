# Complete experiment index and comparison policy

This index separates every completed training generation by dataset semantics,
batch size, horizon and scientific role. “Earlier” does not automatically mean
“baseline”: only protocol-compatible cells are placed in the primary chart.

## Training generations

| Generation | Trained policies | Data and batch | Evaluation evidence | Decision |
|---|---|---|---|---|
| G1: joint6 ablation | A0, A2, B1/B2/B3; A3 reused B policies | 24 train / 6 val, batch 128 | A0–A3 × four checkpoints; B1–B3 isolated; 560 rollouts | Include as diagnostic ablation |
| G2: SingleOrange | Orange001 horizon 25/50/100 | 24 train / 6 val, batch 128 | 60 rollouts | Include as primitive diagnostic |
| G3: Gate30 baseline | A0 to 21k; B1/B2/B3 to 7k | 30 train, batch 64; B3 target-success 29 episodes | 120 legacy-protocol + 40 corrected full-task + 60 isolated rollouts | Include separately as protocol history |
| G4: doubled strict-prefix | A0 to 42k; B1/B2/B3 to 14k | batch 64; strict B3 28 episodes | 120 full-task + 60 isolated rollouts | Primary final benchmark |

Across training generations and inference-only ablations, the repository
contains summaries for **1,160 evaluated rollout episodes**. This is an
inventory count, not a pooled sample:
the protocols differ and their success rates are never averaged together.

## Why G3 is worth adding

The previously omitted G3 intermediate checkpoints contain a useful result:

| Method | Checkpoint | Legacy full success | Stage horizon |
|---|---:|---:|---:|
| A0 | 15k | 0/20 | 1,020 total actions |
| A0 | 18k | 0/20 | 1,020 total actions |
| A0 | 21k | 0/20 | 1,020 total actions |
| A1 | 5k per primitive | **1/20** | 340×3 |
| A1 | 6k per primitive | **1/20** | 340×3 |
| A1 | 7k per primitive | 0/20 | 340×3 |

This strengthens two conclusions:

1. observed A1 success was not unique to the final 14k checkpoint; and
2. rollout performance was non-monotonic with training steps.

Each 1/20 cell has a Wilson 95% interval of approximately 0.9–23.6%, so these
single successes are historical observations, not evidence of a statistically
resolved improvement.

It does **not** establish a clean step-scaling curve. G3 used the old 340-action
boundary, while G4 uses 420 actions and a stricter B3 dataset. The G3 final 7k
policies were re-evaluated under the corrected 420-action protocol for the final
historical archive and again achieved 0/20. Those corrected re-evaluations are
not included in the primary chart or table because the initialization sequences
are not fully paired with G4.

Machine-readable evidence: [`historical-gate30-340.json`](../results/historical-gate30-340.json).

## Inference-only studies

| Study | Episodes | Pairing | Role |
|---|---:|---|---|
| Fixed-K RHC, H=100/25/10/1 | 80 | only episode 0 aligned across H | descriptive horizon sweep |
| H=100 off / H=1 off / H=1 aggregation | 60 | strict physical pairing 20/20 | paired temporal-aggregation ablation |

The paired study observed 5/20, 0/20 and 5/20 respectively. Aggregation
recovered the plain H=1 success level to the H=100 baseline but did not exceed
it. See [`TEMPORAL_AGGREGATION.md`](TEMPORAL_AGGREGATION.md).

## What should stay out of the main comparison

- Checkpoints that were trained but never given a formal rollout evaluation.
- Training loss alone, unless used to illustrate the offline/closed-loop gap.
- G1 batch-128 values as if they were direct controls for G4 batch-64 runs.
- G2 SingleOrange success as if it were a three-orange success rate.
- G3 340-action outcomes pooled with G4 420-action outcomes.
- Any cancelled 50-demo preparation or training marker; no 50-demo result was
  completed or used.

This tiered presentation preserves the useful history without inflating the
headline result or hiding protocol changes.
