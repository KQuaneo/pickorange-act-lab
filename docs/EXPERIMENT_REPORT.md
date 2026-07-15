# PickOrange-ACT unified experiment report

**Study scope:** 30 expert demonstrations, simulation only  
**Task:** `LeIsaac-SO101-PickOrange-v0`  
**Final evaluation seed:** 2026  
**Report date:** 2026-07-14

## Executive summary

This study asks a practical embodied-AI question: when an ACT policy fails on a
long, contact-rich manipulation task, is the main limitation long-horizon task
structure, primitive control, data quality, or stage transition behavior?

The investigation progressed from a monolithic ACT baseline to explicit stage
conditioning, multiple primitive policies, isolated oracle-initialized tests,
single-orange horizon sweeps, strict-prefix data audits, doubled-step training,
and fixed-time scheduler diagnostics.

The strongest observed full-task result was **3/20 (15%)** from the final A1
three-policy system. No full success was observed for A0 at any final
checkpoint. These native-horizon runs used 1,020 A0 actions versus 1,260 A1
actions; matched-horizon capability exists but has no formal 20-episode result.
The G4 isolated primitives achieved only 30%/45%/30%, with oracle
initialization for B2/B3. The underlying primitives therefore remain unreliable
independently of the sequential composition problem. A data audit rejected 2/30
B3 slices and identified releases after the former 340-action cutoff. These
findings led to a 420-action stage horizon and explicit scheduler-overrun
measurement.

The outcome is a technically meaningful partial success, not a solved task.
The main contribution is the evidence chain connecting data semantics,
low-level contact behavior, scheduling, horizon fairness and end-to-end outcome.
The study does not establish that multi-policy decomposition is statistically
superior to the monolithic policy.

## 1. Task and model

The SO-101 arm must sequentially pick three oranges from a tabletop and place
all three in a plate. The final success condition is evaluated from simulator
state: every orange must be inside the plate tolerance at the final step and
must satisfy a 10-frame stable placement condition. Robot rest is not required
by this evaluator.

| Item | Setting |
|---|---|
| Observations | front RGB, wrist RGB, robot joint state |
| Images | 480×640, 30 FPS expert data |
| Action | six direct joint-position targets; no IK |
| Policy | ACT with ResNet-18 visual encoder |
| Chunk / action horizon | 100 / 100 |
| Final batch size | 64 |
| Final A0 training | 42,000 steps; retained 30k/36k/42k |
| Final A1 training | 14,000 steps per primitive; retained 10k/12k/14k |
| Formal rollout | 20 episodes/configuration, seed 2026 |

## 2. Experimental questions and groups

| Group | Training | Inference | Question |
|---|---|---|---|
| A0 | one ACT on the full episode | one policy, start to finish | Can vanilla ACT solve the long task? |
| A1 | separate B1/B2/B3 ACT policies | fixed-time scheduler | Does temporal decomposition help? |
| A2 | one ACT plus stage/target features | oracle stage feature | Is stage ambiguity the main problem? |
| A3 | reuse A1 primitive policies | privileged success detector | Does event-triggered switching help? |
| B1/B2/B3 | isolated A1 primitives | B2/B3 oracle initialized | What is the primitive capability ceiling? |
| C0/C1 | single-orange ACT | chunk/horizon sweep | Is local execution horizon the bottleneck? |

The word “oracle” is intentionally narrow. A1 full-task inference uses fixed
time boundaries and is not a success oracle. A2 receives a privileged stage
identifier. A3 uses simulator state for switching. Isolated B2/B3 synthesize a
successful prefix and therefore cannot be interpreted as sequential rollouts.

### 2.1 Training lineage and comparison policy

The completed work contains four training generations rather than one run:

| Generation | Training | Protocol role |
|---|---|---|
| G1 | batch-128 A0/A2 to 21k and B1/B2/B3 to 7k; 24 train / 6 validation | A0–A3 methodological ablation |
| G2 | three batch-128 SingleOrange policies to 7k | action-horizon diagnostic |
| G3 | batch-64 Gate30 A0 to 21k and B1/B2/B3 to 7k | legacy 340-action history; final checkpoints also re-evaluated at 420 |
| G4 | batch-64 A0 to 42k and strict-prefix B1/B2/B3 to 14k | primary final benchmark |

The summaries cover 1,160 rollout episodes across historical, diagnostic and
final protocols, including the 80-episode fixed-K RHC sweep and 60-episode
strictly paired temporal-aggregation comparison. This
demonstrates evaluation scope but is not one statistical sample; rates are
never pooled across incompatible cells.

G3 A1 produced 1/20 full success at 5k and 6k per primitive under the legacy
340×3 protocol, followed by 0/20 at 7k. This evidence is worth preserving
because it predates the final doubled-step run and reinforces non-monotonic
checkpoint behavior. It remains outside the main bar chart because G4 changed
the stage horizon to 420 and tightened B3 from 29 target-success slices to 28
strict-prefix slices. The G3 final 7k policies were separately re-evaluated at
420×3 and again achieved 0/20.

See the [complete experiment index](EXPERIMENT_INDEX.md) for the inclusion
decision and the exact historical checkpoint table.

## 3. Experiment sequence

### 3.1 Data/schema and replay gates

Before training, the workflow validated the task schema, HDF5-to-LeRobot
conversion, scene alignment and successful replay semantics. Later preparation
used event-based stable placement rather than an assumed fixed number of frames.

The final ACT sampling accounting treats every frame as a sample anchor; future
actions at episode ends are padded. For that reason, “30 episodes” alone is not
a sufficient exposure measure.

| Dataset | Strict episodes | Anchors | Full unpadded windows | Approx. anchor exposure |
|---|---:|---:|---:|---:|
| A0 full | 30 | 35,040 | 32,070 | 76.71× |
| B1 / Orange001 | 30 | 9,431 | 6,461 | 95.01× |
| B2 / Orange002 | 30 | 11,068 | 8,098 | 80.95× |
| B3 / Orange003 | 28 | 11,014 | 8,242 | 81.35× |

### 3.2 Legacy long-task ablation: A0–A3

The first formal study trained A0/A2 to 21k steps and each A1 primitive to 7k
steps at batch 128. Four aligned checkpoint levels were evaluated with 20
episodes each. No group produced an observed full three-orange success.

| Group | Best Orange001 | Best Orange002 | Best Orange003 | Full success observed |
|---|---:|---:|---:|---:|
| A0 | 50% | 10% | 0% | 0/20 per checkpoint |
| A1 | 20% | 15% | 0% | 0/20 per checkpoint |
| A2 | 50% | 5% | 0% | 0/20 per checkpoint |
| A3 | 10% | 0% | 0% | 0/20 per checkpoint |

![Legacy stage success](../assets/pick_orange_stage_success_by_checkpoint.png)

Training and validation loss decreased, but rollout success did not track it.
This is evidence of an offline-to-closed-loop gap under the tested setup; it
does not isolate whether the cause is vision, action averaging, dynamics,
contact timing or data coverage.

![Validation loss versus rollout](../assets/pick_orange_validation_vs_eval.png)

A2 did not outperform A0 despite explicit stage information. A3 usually could
not trigger a reliable first-stage success event. Together these observations
weakened—but did not eliminate—the hypothesis that stage identity alone was the
dominant limitation.

### 3.3 Isolated primitive evaluation

The A1 policies were then evaluated separately. B1 starts from the normal task
initialization. B2/B3 use oracle initialization with already-completed oranges
placed in the plate, robot joints set to the corresponding expert subtask start,
and camera observations rendered after the synthetic state is applied.

The G3 archive records 45%/50%/15% for its final 7k B1/B2/B3 checkpoints.
Earlier checkpoint sweeps recorded legacy best values of 45%/40%/25%. These
measurements are retained only as historical protocol evidence and are excluded
from the G4 primary table and charts.

This distinction matters: isolated success quantifies primitive capability in
a controlled start-state distribution, not the probability that three
primitives will succeed in sequence.

### 3.4 SingleOrange horizon sweep

Orange001 was trained/evaluated as a standalone primitive with chunk/action
horizons 25, 50 and 100.

| Horizon | Success |
|---:|---:|
| 25 | 4/20 (20%) |
| 50 | 4/20 (20%) |
| 100 | 6/20 (30%) |

![Single-orange horizon comparison](../assets/single_orange_chunk_comparison.png)

The comparison is diagnostic rather than an equal-compute architecture
ablation: changing action horizon also changes control refresh, smoothing and
contact response. Under the tested settings, simply shortening the horizon did
not improve success.

### 3.5 Failure taxonomy and gripper timing

Across 320 Orange001 outcomes, 195 (60.9%) were classified as `no_effect`, 30
as contact/tiny lift, seven as partial lift, two as high lift without placement,
one as placed then lost and 85 as final success.

![Orange001 failure taxonomy](../assets/orange001_failure_types_by_group.png)

Expert sequences shared a highly regular gripper schedule: first close around
frame 178, a longest closed segment around 120 frames, and reopening around
frame 298. The combination of frequent no-effect failures and fixed timing is
consistent with sensitivity to spatial trajectory error. This is an engineering
hypothesis supported by the observed pattern, not a controlled causal result.

![Expert gripper timing](../assets/expert_gripper_timing.png)

### 3.6 Strict-prefix B3 audit

The weak legacy B3 outcome motivated a complete audit of the third-orange
training slices. The audit checked decoded video/frame counts, finite arrays,
timestamp continuity, target stable placement and preservation of every prior
orange.

| Check | Result |
|---|---:|
| Integrity | 30/30 |
| Orange003 target success | 29/30 |
| Strict-prefix success | 28/30 |

The two rejections were `target_not_stably_placed` and
`prefix_not_intact:Orange002`. More importantly, valid B3 release actions were
observed at 350–358, after the legacy 340-action boundary. The final experiment
therefore uses 420 actions per A1 stage; historical 340-action results remain
separately labeled rather than silently rewritten.

### 3.7 Doubled-step final experiment

The final study kept batch size 64 and the 30-demo dataset fixed. A0 was trained
to 42k; each A1 primitive to 14k. Only three new checkpoints were retained, with
all primary comparisons restricted to the protocol-consistent G4 run.

| Method | Checkpoint | Full success | Wilson 95% | Mean final oranges |
|---|---:|---:|---:|---:|
| A0 | 30k | 0/20 | 0.0–16.1% | 0.70 |
| A0 | 36k | 0/20 | 0.0–16.1% | 0.55 |
| A0 | 42k | 0/20 | 0.0–16.1% | 0.35 |
| A1 | 10k | 2/20 | 2.8–30.1% | 0.65 |
| A1 | 12k | 0/20 | 0.0–16.1% | 0.70 |
| A1 | 14k | **3/20** | **5.2–36.0%** | **0.75** |

The non-monotonic 10k/12k/14k result is another warning that training loss or
step count alone is not a reliable model-selection signal for contact-rich
rollouts. With 20 episodes, apparent differences have substantial uncertainty.

The final G4 isolated evaluation was:

| Primitive | G4 14k |
|---|---:|
| B1 | 6/20 (30%) |
| B2, oracle initialized | 9/20 (45%) |
| B3, oracle initialized | 6/20 (30%) |

These values are below 50% even in the isolated setting, and B2/B3 benefit from
oracle initialization. Primitive control is therefore an independent
reliability bottleneck; the full-task failures should not be attributed solely
to long-horizon composition.

The independently trained G3 checkpoints are preserved in the
[experiment index](EXPERIMENT_INDEX.md) as historical evidence. They are not a
direct baseline because training lineage, B3 dataset semantics and evaluation
initialization sequences are not fully matched.

### 3.8 Native versus matched horizon

The historical/final default remains `native_horizon`:

| Protocol | A0 | A1 | Same total horizon? |
|---|---:|---:|---|
| `native_horizon` | 1,020 actions / 2,040 sim steps / 34s | 420×3 / 2,520 sim steps / 42s | No |
| `matched_horizon` | 1,260 actions / 2,520 sim steps / 42s | 420×3 / 2,520 sim steps / 42s | Yes |

`matched_horizon` is an opt-in evaluator capability. It was not substituted for
the reported historical benchmark and has no formal 20-episode result in this
study. Output paths and protocol-specific summary names prevent accidental
overwrites.

### 3.9 Fixed-time scheduler overrun

For a stage that becomes stably successful before its fixed boundary:

```text
post_success_overrun = stage_switch_step - target_first_stably_satisfied_step
```

If stable success is never reached, the satisfaction and overrun fields are
JSON `null`, not zero.

| Stage | n | Mean | Median | Q25–Q75 | Prefix destroyed during overrun |
|---|---:|---:|---:|---:|---:|
| Orange001 | 3 | 63.67 | 14.0 | 13.0–89.5 | 0 |
| Orange002 | 4 | 89.75 | 87.5 | 81.5–95.75 | 0 |
| Orange003 | 7 | 97.71 | 103.0 | 27.5–120.0 | 0 |

Seven overrun/next-stage-deviation pairs yielded Pearson r≈0.632. The sample is
small and one normalized stage-deviation family is numerically sensitive to
near-zero expert variance. The result is therefore retained only as a
descriptive diagnostic; no causal or robust inferential claim is made.

### 3.10 Fixed-K receding-horizon inference ablation

The final inference-only ablation held the trained G4 B1 14k policy and its
prediction chunk size fixed at `K=100`, disabled temporal ensembling, and varied
only the executed prefix `H in {100, 25, 10, 1}`. Every cell used 420 policy
actions, two simulation steps per action, seed 2026 and 20 episodes.

| H | Success | Wilson 95% | Contact-or-better | Calls | Discarded predictions | Mean wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | **5/20** | 11.2–46.9% | 12/20 | 100 | 1,600 | 33.84s |
| 25 | 3/20 | 5.2–36.0% | 15/20 | 340 | 25,600 | 33.45s |
| 10 | 2/20 | 2.8–30.1% | 14/20 | 840 | 75,600 | 30.54s |
| 1 | 1/20 | 0.9–23.6% | 7/20 | 8,400 | 831,600 | 49.43s |

The pre-registered selection order—success count, contact-or-better count,
median maximum lift, then larger H—selected `H*=100`. Thus no shorter execution
horizon improved observed B1 success, while aggressive replanning greatly
increased inference and discarded-action cost. For H=1, the one evaluator
success was inside the plate at the final step but had not remained there for
the separate ten-step stable-placement diagnostic; the raw taxonomy therefore
labels it `high_lift_without_placement` rather than `final_success`.

This comparison is not paired. Only episode 0 shared the same initialization
ID and raw start state across every H; subsequent reset states diverged.
Paired McNemar/bootstrap tests were consequently prohibited, and the observed
rates are reported descriptively with Wilson intervals.

The planned full A0/A1 follow-up was not executed. Because `H*=100`, the
pre-planned comparison of the existing `H=100` controller against `H*` would
duplicate the baseline exactly rather than test a new controller. This avoids
redundant simulator runs but does not resolve the distinct A0/A1
native-versus-matched total-horizon limitation.

### 3.11 Strictly paired temporal aggregation

The final inference ablation reused one validated 20-state manifest for three
controllers: H=100 without aggregation, H=1 without aggregation, and H=1 with
LeRobot ACT temporal aggregation at coefficient 0.01. All groups used the G4
B1 14k checkpoint, K=100, 420 executed actions and seed 2026.

| Controller | Success | Wilson 95% | Mean/median max lift | Contact-or-better | Calls/episode |
|---|---:|---:|---:|---:|---:|
| H=100 off | 5/20 | 11.2–46.9% | 0.0427/0.0057 m | 13/20 | 5 |
| H=1 off | 0/20 | 0.0–16.1% | 0.0121/0.0000 m | 7/20 | 420 |
| H=1 aggregation 0.01 | 5/20 | 11.2–46.9% | 0.0418/0.0045 m | 7/20 | 420 |

![Strictly paired temporal aggregation](../assets/temporal-aggregation-three-way.svg)

The episode transition `(H100 off, H1 off, H1 aggregation)` was `000` for 13
episodes, `101` for three, `100` for two and `001` for two. Aggregation versus
plain H=1 had five favorable and zero unfavorable success transitions (exact
McNemar `p=0.0625`). Aggregation versus H=100 had two exclusive successes in
each direction (`p=1.0`). Mean paired max lift changed by +0.02976 m relative
to plain H=1 and -0.00089 m relative to H=100.

Thus aggregation recovered the observed H=1 success level but did not exceed
the paired H=100 baseline. H=100 also reached contact-or-better more often,
while plain H=1 had the lowest action-delta metric and no final successes. The
controllers therefore differ in behavior even where aggregate success ties.
See the [dedicated report](TEMPORAL_AGGREGATION.md).

Physical-state pairing passed 20/20, including initialization ID, serialized
robot/object/plate state and refresh protocol. Separate Isaac launches were not
pixel-identical; this is retained as a rendering caveat rather than silently
presented as exact image pairing.

## 4. Automation and operational reliability

Long-running work was executed under tmux with a supervisor and live status
writer. Training used two stable GPU lanes: one A0 lane and one sequential A1
lane. Evaluation used up to three parallel jobs. The final 30-demo pipeline:

- persists a state JSON and completion markers;
- reuses completed summaries on restart;
- lowers evaluation parallelism after failure;
- uses bounded exponential retry delays;
- waits on low disk capacity rather than failing mid-run;
- separates smoke, formal evaluation and public report outputs;
- validates the exact retained-checkpoint set.

The cancelled 50-demo extension was removed from the active pipeline. No
50-demo result is used anywhere in this report.

## 5. Interpretation

The evidence supports six measured conclusions:

1. In the current native-horizon samples, A1 produced 3/20 end-to-end successes
   whereas A0 produced 0/20. Because the horizons are 1,260 versus 1,020 actions
   and no formal matched-horizon 20-episode result exists, this is not evidence
   of statistically conclusive superiority.
2. Primitive policies remain unreliable even under controlled initialization;
   this is an independent bottleneck, not merely a consequence of long-horizon
   composition.
3. Stage transitions are not merely a software detail: success can occur long
   before the fixed boundary, while valid B3 releases can occur after the old
   boundary.
4. Dataset semantics materially affect the experiment. A small nominal dataset
   required strict-prefix filtering and exposure accounting at the frame/window
   level.
5. In the fixed-K B1 inference ablation, shorter replanning horizons did not
   improve observed success; `H*=100` was selected, and cross-H results remain
   descriptive because initialization pairing failed after episode 0.
6. Under the later 20-state paired protocol, temporal aggregation recovered
   H=1 from 0/20 to 5/20, matching but not outperforming H=100. This narrows the
   control diagnosis without establishing a generally optimal controller.

The experiment does not prove that fixed-time overrun causes the next stage to
fail, that A1 is statistically superior in general, or that the policies will
transfer to hardware.

## 6. Recommended next experiments

1. Freeze the current protocol and repeat A0/A1 with additional independent
   seeds before selecting a winner.
2. Run the opt-in A0 matched-horizon diagnostic to isolate the eight-second
   native-horizon difference.
3. Replace fixed-time switching with a non-privileged visual success detector
   only after primitive reliability improves.
4. Increase demonstration diversity around failed contact modes rather than
   only increasing identical trajectory count.
5. Replicate the paired H=100/H=1/aggregation comparison with independent seeds
   before drawing a general execution-horizon or ensembling conclusion.
6. Add closed-loop gripper/contact logic or a residual controller and compare
   against the current position-only execution.
7. Evaluate on physical SO-101 hardware before making any sim-to-real claim.

## 7. Evidence index

- Public compact result: [`results/summary.json`](../results/summary.json)
- Sanitized raw results: [`results/raw/`](../results/raw/)
- Evaluators: [`experiments/eval_pick_orange_joint6_ablation.py`](../experiments/eval_pick_orange_joint6_ablation.py)
- Strict B3 audit: [`experiments/audit_pick_orange_b3_slices.py`](../experiments/audit_pick_orange_b3_slices.py)
- Statistical helpers: [`experiments/pick_orange_analysis.py`](../experiments/pick_orange_analysis.py)
- Resilient pipeline: [`experiments/run_pick_orange_30_only_pipeline.py`](../experiments/run_pick_orange_30_only_pipeline.py)
- Paired temporal aggregation: [`TEMPORAL_AGGREGATION.md`](TEMPORAL_AGGREGATION.md)
