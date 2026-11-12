# Step-by-step DRIFT pipeline

Run the commands below from the repository root. See the [project homepage](../README.md) for the quick start.

DRIFT uses on-policy responses to select examples from an existing supervised fine-tuning corpus. Begin with a checkpoint that has completed its SFT schedule, sample and verify validation responses, score the original corpus using projected gradients, and continue SFT on the selected subset.

This repository implements the method and its data pipeline. It does not include the historical checkpoints, frozen rollouts, or private experiment artifacts needed to reproduce every reported paper number. The included examples are small fixtures, not benchmark results.

## Install and check the CPU implementation

Run these commands from the repository root. The core package requires Python 3.10+ and NumPy; it does not install or import PyTorch for CPU scoring.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

drift demo --output runs/cpu-demo
python -m pytest
```

The demo uses synthetic features and exercises task weighting, norm correction, and selection. Install optional components separately as needed:

```bash
python -m pip install -e '.[models]'   # Transformers gradient extraction and generation
python -m pip install -e '.[data]'     # Parquet input
python -m pip install -e '.[rollout]'  # vLLM generation backend
python -m pip install -e '.[math]'     # math-verify answer verification
```

Use a GPU environment compatible with the chosen PyTorch and vLLM packages. `--backend transformers` needs the `models` extra but not the `rollout` extra. The pinned vLLM dependency is a separate installation choice; the NumPy core works without it.

The saved example responses can also be checked without loading a model:

```bash
mkdir -p runs/fixture-check
drift prepare --input examples/corpus.jsonl \
  --output runs/fixture-check/corpus.jsonl
drift label --queries examples/queries.jsonl \
  --rollouts examples/rollouts.jsonl \
  --output runs/fixture-check/anchors.jsonl
```

## Run the selection pipeline

Provide your original SFT corpus and validation queries in the formats below. Use the same frozen SFT checkpoint and tokenizer for generation and both extraction passes. A local model snapshot is convenient; a Hugging Face model ID also works, with `--revision` available to pin a commit.

The configured runner launches every method stage and keeps one log per child process. It supports any positive number of independent GPU replicas, provided the model fits on each GPU. Preview the commands without creating output files or loading a model:

```bash
python scripts/run_method.py --config configs/olmo3.json \
  --corpus ./data/corpus.jsonl --queries ./data/queries.jsonl \
  --output ./runs/my-drift --gpus 0,1 --backend transformers \
  --stage all --dry-run
```

Remove `--dry-run` to execute. Use `--model ./models/my-sft-checkpoint` to override the configured public model, or `--config configs/olmo2_1b.json` for the OLMo2 1B configuration. The `rollout` extra enables `--backend vllm`; Transformers needs only the `models` extra. Both configurations use 4,096-token extraction and a 16,384-dimensional projection; OLMo3 samples 32 responses per query, while OLMo2 samples 16. Edit a copy of the configuration to freeze your chosen settings and an optional model `revision`.

`--stage all` runs through subset export; continuation training is separate. It requires built-in verifiers or `--external-labels` for these exact responses. For external evaluation, run `--stage prepare`, then `--stage rollout`, grade the generated responses, and continue with `--stage label --external-labels ./data/rewards.jsonl`, followed by `extract`, `score`, and `select`. Use the same output root and GPU count across these runner invocations. Each stage requires new outputs and can consume already completed upstream stages; it does not resume partial outputs. A failed rank stops its sibling processes owned by this runner and prevents later stages from starting. Logs and status remain under `runner_logs/`.

For a small real-model check, run the fixture corpus with `configs/smoke.json` (13 conversations, 2 queries, 2 responses per query, 128-dimensional features). This validates plumbing and is not a quality evaluation:

```bash
python scripts/run_method.py --config configs/smoke.json \
  --model ./models/my-sft-checkpoint --corpus examples/corpus.jsonl \
  --queries examples/queries.jsonl --output runs/model-smoke --gpus 0
```

The equivalent individual CLI commands are below. This single-GPU example uses built-in verifiers and assumes that each validation prompt plus its generation budget fits the selected context. Choose a new output directory for each run; do not execute this block into a completed runner output.

```bash
set -euo pipefail

MODEL=./models/my-sft-checkpoint
CORPUS=./data/corpus.jsonl
QUERIES=./data/queries.jsonl
RUN=./runs/my-drift
mkdir -p "$RUN"

# 1. Preserve every turn of each original corpus conversation.
drift prepare --input "$CORPUS" --output "$RUN/corpus.jsonl"

# 2. Generate responses to validation prompts, without injecting references.
CUDA_VISIBLE_DEVICES=0 drift rollout \
  --queries "$QUERIES" --model "$MODEL" \
  --output "$RUN/rollout-rank-0" --backend transformers \
  --samples 32 --temperature 0.6 --top-p 0.95 --top-k -1 \
  --max-new-tokens 2048 --max-model-len 4096 --seed 42

# 3. Assign each generated response a reward of +1 or -1.
drift label --queries "$QUERIES" \
  --rollouts "$RUN/rollout-rank-0/rollouts.jsonl" \
  --output "$RUN/anchors.jsonl"

# 4. Extract the original corpus and validation anchors with identical settings.
CUDA_VISIBLE_DEVICES=0 drift extract \
  --input "$RUN/corpus.jsonl" --output "$RUN/corpus-rank-0" \
  --model "$MODEL" --dimension 16384 --projection-seed 42 \
  --chunk-elements 536870911 --max-length 4096 \
  --dtype bfloat16 --attention sdpa --device cuda:0

CUDA_VISIBLE_DEVICES=0 drift extract \
  --input "$RUN/anchors.jsonl" --output "$RUN/anchors-rank-0" \
  --model "$MODEL" --dimension 16384 --projection-seed 42 \
  --chunk-elements 536870911 --max-length 4096 \
  --dtype bfloat16 --attention sdpa --device cuda:0

# 5. Validate coverage and produce ordered feature tables, even for one rank.
drift merge --shards "$RUN/corpus-rank-0" --output "$RUN/corpus-features"
drift merge --shards "$RUN/anchors-rank-0" --output "$RUN/anchor-features"

# 6. Compute task centroids, raw scores, and gradient-norm correction.
drift score --corpus-features "$RUN/corpus-features" \
  --anchor-features "$RUN/anchor-features" \
  --fit-mode stable --output "$RUN/scores"

# 7. Export the top floor(0.10 * N) complete conversations.
drift select --scores-dir "$RUN/scores" --fraction 0.1 \
  --corpus "$RUN/corpus.jsonl" --output "$RUN/selection"
```

To use externally evaluated queries in the individual commands, add `--external-labels ./data/rewards.jsonl` to the labeling step after evaluating these newly generated responses.

The final training data is `runs/my-drift/selection/subset.jsonl`. `selected_ids.jsonl` records rank and score; `subset.jsonl` preserves the original corpus order and complete messages.

All output stage directories must be new. Failed or incomplete shards are rejected instead of being silently reused or filled with zero gradients.

## Continue SFT on the selected subset

Use the **same starting SFT checkpoint**. The supplied wrapper invokes the public ms-swift trainer; install it in a separate training environment using the [pinned setup](reproducibility.md#continual-sft-and-outcome-evaluation). Its default recipe packs eight 32,768-token sequences per optimizer update, with full-parameter BF16 training and 500 updates:

```bash
python scripts/continue_sft.py --model ./models/my-sft-checkpoint \
  --dataset ./runs/my-drift/selection/subset.jsonl \
  --output ./runs/my-drift/continued-sft --gpus 0,1,2,3 \
  --max-steps 500 --max-length 32768 --global-sequences 8 --dry-run
```

Remove `--dry-run` to train. Packing requires compatible FlashAttention, and multi-GPU training requires DeepSpeed. Gradient accumulation adjusts to the GPU count, which must divide `--global-sequences`. For a small setup check, `--no-packing --attn-impl sdpa` avoids FlashAttention and changes the token budget. The bundled `drift_chat` plugin reuses the attribution mask and the checkpoint’s chat template for training. The wrapper retains `train.log`, `training_manifest.json`, and model checkpoints in a new output directory. Held-out benchmark evaluation remains a separately recorded stage.

## Input formats

The original corpus is JSONL, Parquet, or a directory of Parquet shards. Each record needs `messages`, containing the original `system`, `user`, and `assistant` turns as string content. At least one assistant response is required. Existing IDs must be unique; `prepare` assigns row-based IDs if none are supplied. An optional `source` field is retained.

```json
{"id":"train-0","messages":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}],"source":"my-corpus"}
```

Validation queries are JSONL with unique string IDs, a task name, the full prompt conversation ending in a user message, and a verifier:

```json
{"id":"query-0","task":"arithmetic","messages":[{"role":"user","content":"What is 2 + 2? Reply with the answer only."}],"verifier":{"type":"exact","answer":"4"}}
```

Available verifier types:

| Type | Required configuration | Behavior |
|---|---|---|
| `choice` | `answer`, optionally `choices` | Extracts an unambiguous answer label, such as `Answer: B`. |
| `exact` | String `answer` | Compares the whole completion after declared normalization; optionally uses a one-capture `extract_regex`. |
| `math` | `answer` and the `math` extra | Parses and verifies mathematical equivalence with `math-verify`. |
| `external` | An external labels file at labeling time | Imports rewards from your benchmark evaluator, including code execution or other task-specific grading. |

External labels use the generated rollout ID (`query_id:sample_index`) and an exact reward of `+1` or `-1`:

```json
{"id":"code-query-0:0","reward":1}
{"id":"code-query-0:1","reward":-1}
```

The file must cover **exactly** the rollouts whose queries use `external`: missing, duplicate, or extra IDs fail validation. Optional `query_id` and `task` fields are checked when present. The package does not execute generated code; use the benchmark's evaluation harness to produce these labels. See [reproducibility.md](reproducibility.md) for imported rollout records and verifier details.

## What is computed

Each projected feature is a per-example gradient of causal **token-mean CE over assistant content and its termination token**. Corpus gradients supervise all assistant turns; anchor gradients supervise only the final generated answer, with earlier turns retained as context. Prompt tokens, assistant headers, and following turn separators are masked. Anchors that exceed the extraction context are rejected rather than shortened. The signed GraSS sketch samples coordinates and hashes them with random signs; correctness rewards are applied once, when constructing task centroids.

For task `k`, query `q`, and its `N_q` rollouts, each anchor receives weight `reward / (Q_k * N_q)`, where `Q_k` is the number of queries in that task. All-correct and all-incorrect queries both contribute. A corpus feature is scored by its dot product with each task centroid. DRIFT then fits a separate relation between log score magnitude and log projected-gradient norm, divides by `norm ** beta`, preserves the score's sign, and sums the corrected task scores. Selection defaults to the highest-scoring 10%, with earlier corpus rows breaking ties.

`--fit-mode stable` uses centered float64 OLS with an intercept. `--fit-mode legacy-float32` reproduces the historical float32-input `np.polyfit` fitting convention and exposes its numerical rank, singular values, cutoff, and coefficients. These modes can yield different slopes and subsets. Clipping and z-scoring are disabled by default; opt in with `--clip 1 99 --zscore` when that is the intended score-processing protocol.

## Models, GPUs, and reproducibility

The implementation supports Transformers causal language models with a fast tokenizer and an explicit chat template that preserves assistant content and supplies a special-token terminator. Anchors additionally require their rendered prompt prefix to match generation. No particular model family or hard-coded special-token ID is required. The public configurations use `allenai/Olmo-3-7B-Instruct-SFT` and `allenai/OLMo-2-0425-1B-SFT`; these specify method settings, not a claim of reproduced benchmark scores.

Generation and feature extraction use independently launched model replicas with `--rank` and `--world-size`; there is no fixed GPU count. All ranks of a dataset must be supplied to `merge`. Corpus and anchor extraction may use different rank counts, but their checkpoint, tokenizer, masking, parameter subset, and projection fingerprints must match. See the [multi-GPU commands and artifact contract](reproducibility.md).

The portable projector uses NumPy PCG64, fixed chunks in global parameter-coordinate order, and FP32 accumulation. It follows the sparse construction but does **not** recreate the historical dattri random matrix or floating-point bitstream. An identical seed therefore does not promise the same historical top-10% subset. Save the generated projection specification and manifests alongside any reported result.

See [release validation](validation.md) for executed checks and their limits.

License: [MIT](../LICENSE).
