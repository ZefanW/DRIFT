# Run the DRIFT selection

This guide computes attribution scores and selects data yourself, for Olmo3 or for another checkpoint. To retrain on the subset selected in the paper, use the [reproduction guide](reproduce_olmo3.md) instead.

## Inputs

DRIFT needs three things: a checkpoint that has finished SFT, the corpus it was trained on, and validation queries with a way to verify responses.

For Olmo3 these are `allenai/Olmo-3-7B-Instruct-SFT`, the Dolci-Instruct-SFT corpus without tool-use rows, and the paper's 8,220 validation queries (7,500 MATH, 596 MBPP, 70 MMLU-Pro and 54 BBH). The MBPP queries are disjoint from the MBPP+ evaluation set. The queries, the 32 rollouts per query and their correctness labels are on [Hugging Face](https://huggingface.co/datasets/ZefanW/DRIFT-OLMo3-Validation-Rollouts) in their original schema.

Corpus rows keep every conversation turn:

```json
{"id":"unique-row-id","messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Response"}]}
```

Validation rows hold the prompt and a verifier. The reference answer goes in the verifier, not in the prompt:

```json
{"id":"query-0","task":"math","messages":[{"role":"user","content":"Question"}],"verifier":{"type":"math","answer":"Reference answer"}}
```

| Verifier | Use |
|---|---|
| `math` | Mathematical equivalence with the reference answer |
| `choice` | Multiple-choice label |
| `exact` | Exact match after whitespace and Unicode normalization |
| `external` | Graded outside the pipeline, for code execution or benchmark-specific scoring |

Corpus IDs must be unique. If your source has duplicate IDs, assign row-based IDs when converting it.

## Generate, attribute and select

Install `.[models,data,math]` for the Transformers backend. `.[rollout]` installs the vLLM backend with its own PyTorch pin, so keep it in a separate environment from training if the versions differ.

When every query uses a built-in verifier, one command runs all stages:

```bash
python scripts/run_method.py --config configs/olmo3.json \
  --corpus data/corpus.jsonl --queries data/validation.jsonl \
  --output runs/olmo3_selection --gpus 0,1,2,3,4,5,6,7 --backend transformers
```

With `external` verifiers, stop after the rollouts, grade them, and pass the rewards back:

```bash
run() {
  python scripts/run_method.py --config configs/olmo3.json \
    --corpus data/corpus.jsonl --queries data/validation.jsonl \
    --output runs/olmo3_selection --gpus 0,1,2,3,4,5,6,7 "$@"
}

run --backend transformers --stage prepare
run --backend transformers --stage rollout

# grade the generated responses, then:
run --stage label --external-labels data/rewards.jsonl
run --stage extract
run --stage score
run --stage select
```

`data/rewards.jsonl` has one line per externally graded response, `{"id":"query-0:0","reward":1}` for a correct response and `-1` for an incorrect one.

The selected conversations are written to `runs/olmo3_selection/selection/subset.jsonl`. `--model` overrides the checkpoint in the configuration, and `--dry-run` prints the commands of each stage without running them.

## Configuration

[`configs/olmo3.json`](../configs/olmo3.json) holds the paper's Olmo3 settings.

| Setting | Value |
|---|---|
| Rollouts | 32 per query, temperature 0.6, top-p 0.95 |
| Gradients | All model parameters, projected to 16,384 dimensions |
| Scoring | Per-task norm correction, clipping at the 1st and 99th percentiles, z-scoring |
| Selection | Top 10% |

Both extraction passes, over the corpus and over the rollouts, must use the same checkpoint, tokenizer, parameter subset and projection. For a rollout, only the sampled response is supervised, and it has to fit in the extraction context in full.

For another model, copy the file and set `model` and `revision`. The paper's OpenR1-Distill-7B run used 8 rollouts per query and excluded the embedding layer (`exclude_parameters`).

## Disk space

Feature extraction writes one shard per GPU, and the merge step checks that every row is covered before scoring. For about 2M examples at dimension 16,384, one float32 feature table takes roughly 130 GB. Keeping the shards and the merged table doubles that, and the rollout features come on top.

## Differences from the paper's run

Recomputing the Olmo3 selection with this code does not return exactly the released subset.

- **Projection.** The paper's run drew its random projection with dattri. This implementation draws it with NumPy's PCG64, so the same seed gives a different matrix.
- **Regression.** `fit_mode: legacy-float32` keeps the float32 fitting convention of the original run. `stable` uses float64 OLS and can change the ranking.
- **Task grouping.** The original scoring merged its 40 subtask columns into 7 MATH, 18 BBH, 1 MBPP and 14 MMLU-Pro columns before the norm correction.

The [released subset](../artifacts/) is the one to use when reproducing the paper's downstream numbers. Each run saves its projection settings and fit diagnostics next to the scores.
