# OLMo3 per-run evaluation results

Accuracy is reported in percent, with one row per recorded continual-SFT run.

| File | Coverage |
|---|---|
| [olmo3_per_run.csv](olmo3_per_run.csv) | 24 main-table runs; three No Debiasing runs; three signed-BM25 runs |
| [olmo3_ablation_per_run.csv](olmo3_ablation_per_run.csv) | 27 runs covering 1P, 1P+1N, 4/8/16 rollouts, all-positive weighting, Cosine, and both self-distillation controls |

`run_id` identifies the original experiment; `run_number` is its ordinal within the condition. `Validation_source`, `Additional`, and `All` are calculated within each run. Use sample standard deviation (`ddof=1`) to summarize repeated runs.

Values retain the precision of their source records. The main-table `All` summaries match the paper to two decimal places, while some individual benchmark summaries differ. The recovered 8-rollout ablation also differs from the paper. Exact source references, checksums, and summary comparisons are in [main-result provenance](olmo3_per_run_sources.json) and [ablation provenance](olmo3_ablation_per_run_sources.json); the exported values have not been adjusted to match a table.

This export covers OLMo3. Per-run records for Off-policy reference, All P, Packing Correction, and Unified Orthogonalization are not included because their original experiment mappings are not yet confirmed. The starting Base checkpoint is not a three-run continual-training condition.
