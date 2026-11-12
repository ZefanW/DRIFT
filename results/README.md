# Per-run results for Olmo3

Accuracy in percent, one row per continued-SFT run. Every condition has three runs.

| File | Runs | Conditions |
|---|---:|---|
| [`olmo3_per_run.csv`](olmo3_per_run.csv) | 30 | The eight selection methods of the main table, No Debiasing, and BM25 with signed rollouts |
| [`olmo3_ablation_per_run.csv`](olmo3_ablation_per_run.csv) | 27 | Rollout weighting and budget, cosine normalization, and the two self-distillation controls |

## Columns

| Column | Meaning |
|---|---|
| `method` | Condition label |
| `run_id` | Name of the original experiment |
| `run_number` | Index of the run within its condition |
| `BBH` … `GPQA_D` | Accuracy on each benchmark. `LCB_v5` keeps the paper's label for [LiveCodeBench](../evaluation/README.md#livecodebench-version) |
| `Validation_source` | Mean of BBH, MBPP+, MATH and MMLU-Pro within the run |
| `Additional` | Mean of ZebraLogic, LiveCodeBench, OlympiadBench and GPQA Diamond within the run |
| `All` | Mean of the eight benchmarks within the run |

## Summary

Mean ± sample standard deviation over the three runs of each condition, computed from the two files.

```python
import pandas as pd

runs = pd.concat([pd.read_csv("results/olmo3_per_run.csv"),
                  pd.read_csv("results/olmo3_ablation_per_run.csv")])
cols = ["Validation_source", "Additional", "All"]
print(runs.groupby("method", sort=False)[cols].agg(["mean", "std"]).round(2))
```

### Main table

| Method | Validation source | Additional | All |
|---|---:|---:|---:|
| Random | 52.86 ± 0.14 | 25.21 ± 0.16 | 39.04 ± 0.15 |
| DSIR | 45.98 ± 0.09 | 22.76 ± 0.04 | 34.37 ± 0.02 |
| QuRating | 52.81 ± 0.17 | 25.71 ± 0.31 | 39.26 ± 0.23 |
| BM25 | 53.17 ± 0.12 | 25.92 ± 0.42 | 39.54 ± 0.19 |
| LESS | 51.29 ± 0.20 | 23.96 ± 0.40 | 37.63 ± 0.29 |
| RDS | 52.66 ± 0.21 | 25.50 ± 0.29 | 39.08 ± 0.24 |
| IF (GraSS) | 52.51 ± 0.21 | 24.81 ± 0.17 | 38.66 ± 0.08 |
| **DRIFT** | **53.76 ± 0.13** | **26.51 ± 0.24** | **40.14 ± 0.12** |

### Ablations and controls

| Label in the files | Condition | Validation source | Additional | All |
|---|---|---:|---:|---:|
| `1P` | One correct rollout per query | 53.32 ± 0.18 | 25.75 ± 0.18 | 39.53 ± 0.16 |
| `1P + 1N (-1)` | One correct and one incorrect rollout per query, signed | 53.69 ± 0.27 | 26.25 ± 0.26 | 39.97 ± 0.26 |
| `All P + All N, 4 rollouts` | Signed rollouts, 4 per query | 53.74 ± 0.14 | 26.01 ± 0.13 | 39.87 ± 0.13 |
| `All P + All N, 8 rollouts` | Signed rollouts, 8 per query | 53.51 ± 0.14 | 26.39 ± 0.09 | 39.95 ± 0.05 |
| `All P + All N, 16 rollouts` | Signed rollouts, 16 per query | 53.54 ± 0.01 | 26.35 ± 0.25 | 39.95 ± 0.12 |
| `All P + All N (+1)` | All 32 rollouts, every one weighted +1 | 52.87 ± 0.21 | 25.19 ± 0.16 | 39.03 ± 0.19 |
| `No Debiasing` | Raw influence scores | 52.42 ± 0.23 | 26.44 ± 0.08 | 39.43 ± 0.15 |
| `Cosine` | Cosine normalization in place of the per-task norm correction | 53.31 ± 0.19 | 26.53 ± 0.13 | 39.92 ± 0.06 |
| `BM25 (signed rollouts)` | BM25 retrieval with DRIFT's rollouts and correctness weights | 53.00 ± 0.10 | 25.60 ± 0.39 | 39.30 ± 0.15 |
| `Self-Distillation` | Continued SFT on the verified correct rollouts | 51.78 ± 0.17 | 23.49 ± 0.11 | 37.64 ± 0.13 |
| `Self-Distillation + Random` | The same rollouts mixed 1:1 with random corpus examples | 52.56 ± 0.27 | 24.18 ± 0.24 | 38.37 ± 0.22 |

DRIFT in the main table is the signed 32-rollout condition with per-task norm correction.

## Agreement with the paper

The files keep the precision of the original metric records, and no value was adjusted to match a table.

- **Main table.** `All` agrees with the paper to two decimals for every method, in mean and standard deviation. A few per-benchmark entries differ in the last digit.
- **Ablations.** Summaries agree with the paper up to the last digit, except for the 8-rollout condition, where the recovered runs average 39.95 on `All` and the paper reports 39.99.

## Not included

- The starting checkpoint, which is evaluated once and has no training runs. Its scores are in the paper's Table 1.
- The off-policy reference, All P, Packing Correction and Unified Orthogonalization conditions. Their runs have not been matched to the original experiment records yet.
- OpenR1-Distill-7B.

## Provenance

[`olmo3_per_run_sources.json`](olmo3_per_run_sources.json) and [`olmo3_ablation_per_run_sources.json`](olmo3_ablation_per_run_sources.json) list the source metric file and its SHA-256 for every run, together with the full comparison against the paper's tables.
