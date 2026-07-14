# Public asset manifest

| Asset | Meaning | Included? |
|---|---|---|
| `media/b3-strict-prefix-expert-demo.mp4` | Expert episode 1, B3 slice, `STRICT PASS` | Yes |
| `media/a1-failure-rollout.mp4` | Representative A1 episode 0 failure, native 42-second horizon | Yes, explicitly labeled failure |
| `assets/b3-audit-overview.jpg` | Contact-sheet overview from the 30-episode B3 audit | Yes |
| `assets/final-full-task-results.svg` | Final A0/A1 20-episode comparison | Generated from public summary |
| `assets/isolated-primitive-results.svg` | Final isolated B1/B2/B3 comparison | Generated from public summary |
| Historical PNG figures | Legacy A0–A3, validation, W&B and failure diagnostics | Yes |
| Expert datasets / HDF5 / parquet | Large private runtime artifacts | No |
| Model checkpoints | Large runtime artifacts | No |
| Simulator assets | Governed by upstream projects | No |
| W&B credentials / local `.env` | Secrets | No |

The public videos are qualitative evidence only. The expert clip is not a
policy rollout, and the policy clip is not a successful cherry-picked result.
Formal success rates come from the machine-readable evaluation summaries.

