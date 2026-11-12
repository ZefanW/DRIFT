# Recompute DRIFT selection

Use this route to compute attribution and select data yourself. To reproduce training on the **paper’s existing OLMo3 selection**, use [the frozen-subset route](reproduce_olmo3.md).

## Inputs

Use the completed SFT checkpoint `allenai/Olmo-3-7B-Instruct-SFT`, its original non-tool corpus, and validation queries. The paper uses 8,220 queries: 7,500 MATH, 54 BBH, 596 MBPP, and 70 MMLU-Pro. MBPP validation problems are disjoint from the MBPP+ evaluation set.

The [original validation queries, rollouts, and correctness labels](https://huggingface.co/datasets/ZefanW/DRIFT-OLMo3-Validation-Rollouts) are available as compressed JSONL files. The dataset card explains their original schema, including `is_original` to distinguish reference answers from generated responses.

Corpus rows preserve all conversation turns:

```json
{"id":"unique-row-id","messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Response"}]}
```

Validation rows contain the prompt and a verifier. Do not append the reference answer to the generation prompt:

```json
{"id":"query-0","task":"math","messages":[{"role":"user","content":"Question"}],"verifier":{"type":"math","answer":"Reference answer"}}
```

Supported verifiers are `math`, `choice`, `exact`, and `external`. Use external labels for code execution or benchmark-specific grading, as `{ "id": "query-0:0", "reward": 1 }` (or `-1`) for every generated response requiring external verification. Corpus IDs must be unique; assign row-based IDs when converting a corpus with duplicate source IDs.

## Generate, attribute, and select

Install `.[models,data,math]` for the Transformers route. `.[rollout]` installs the optional vLLM backend and its compatible PyTorch requirements; use a separate environment from continuation training when versions differ.

```bash
MODEL=allenai/Olmo-3-7B-Instruct-SFT
CORPUS=data/corpus.jsonl
QUERIES=data/validation.jsonl
RUN=runs/olmo3_selection

python scripts/run_method.py --config configs/olmo3.json \
  --model "$MODEL" --corpus "$CORPUS" --queries "$QUERIES" \
  --output "$RUN" --gpus 0,1,2,3,4,5,6,7 --backend transformers \
  --stage prepare
python scripts/run_method.py --config configs/olmo3.json \
  --model "$MODEL" --corpus "$CORPUS" --queries "$QUERIES" \
  --output "$RUN" --gpus 0,1,2,3,4,5,6,7 --backend transformers \
  --stage rollout
```

Evaluate externally graded responses, then continue:

```bash
python scripts/run_method.py --config configs/olmo3.json \
  --model "$MODEL" --corpus "$CORPUS" --queries "$QUERIES" \
  --output "$RUN" --gpus 0,1,2,3,4,5,6,7 \
  --stage label --external-labels data/rewards.jsonl
for STAGE in extract score select; do
  python scripts/run_method.py --config configs/olmo3.json \
    --model "$MODEL" --corpus "$CORPUS" --queries "$QUERIES" \
    --output "$RUN" --gpus 0,1,2,3,4,5,6,7 --stage "$STAGE"
done
```

Omit `--external-labels` when all queries use built-in verifiers. `--stage all` is available for that case. The selected original conversations are written to `$RUN/selection/subset.jsonl`.

The OLMo3 configuration uses 32 responses per query, temperature 0.6, top-p 0.95, all model parameters, and 16,384-dimensional projected gradients. It selects 10% after task-wise norm correction, 1st/99th percentile clipping, and z-scoring. Both extraction passes must use the same checkpoint, tokenizer, parameter subset, and projection. Anchors supervise only the final sampled response and must fit the extraction context in full.

## Implementation notes

The portable implementation saves its projection and fit diagnostics. Its NumPy PCG64 projection is not the historical dattri random matrix; re-running it therefore need not recover the original selected IDs. `legacy-float32` preserves the earlier float32 fitting convention, while `stable` uses float64 OLS and can change the ranking. The paper’s original scoring additionally grouped 40 subtask columns into 7 MATH, 18 BBH, 1 MBPP, and 14 MMLU-Pro columns before correction. The frozen selection is the authoritative input for reproducing the reported downstream experiment.

Feature extraction writes independent GPU shards; merge verifies complete coverage before scoring. For 1.92M examples at dimension 16,384, one float32 feature table is about 126 GB. Retaining shards and the merged table requires roughly twice that, plus validation features. Choose disk space and contexts accordingly; no full-corpus feature computation is needed for the frozen-subset route.
