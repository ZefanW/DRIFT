# Reproducibility and artifact contracts

This document describes the executable method in this release. It complements Section 3 and Appendix A of the [paper](https://arxiv.org/abs/2606.18307). The [step-by-step guide](pipeline.md) provides the full `prepare → rollout → label → extract → merge → score → select` sequence, with separate extraction and merge passes for corpus examples and validation anchors. The paper's checkpoints, rollouts and result artifacts are not bundled. Throughout, "historical" refers to the original code and artifacts behind the paper's reported numbers.

## Freeze the experimental inputs

Start from a model that has completed its original SFT schedule, and retain that same checkpoint for generation, attribution, and subsequent continual SFT. Use a local snapshot or pass an immutable Hub `--revision` to both `rollout` and `extract`. Preserve the original corpus, the validation query file, verifier configuration, tokenizer, generation settings, selected IDs, and continual-training configuration.

The rollout manifest records the resolved Hub revision where available, tokenizer files, template, EOS configuration, prompt hashes, decoding settings, and software versions. For a local model it records weight filenames and sizes, not full weight-content hashes. Extraction fingerprints local weight contents. Keeping the local snapshot immutable links these stages; the score command does not independently prove that an arbitrary imported rollout came from the declared model.

All examples use user-selected files and output roots. Generated directories are intentionally not overwriteable. Preserve incomplete artifacts for diagnosis and use a new directory for an explicitly chosen rerun.

## Corpus and validation semantics

`prepare` keeps complete `messages` rather than constructing a first-question/last-answer pair. It accepts JSONL and Parquet, keeps an optional `source`, assigns missing corpus IDs, and rejects duplicate IDs or conversations without assistant responses. Parquet shards are traversed by sorted filename and then original row order. The canonical JSONL is the input to extraction and subset export; its hash binds those stages.

Validation queries have unique string IDs and fields `task`, `messages`, and `verifier`. Their final message must be `user`. A task is the balancing unit: use a consistent task definition rather than accidentally replacing it with a source filename or generation run ID. References in `verifier.answer` are used for grading, not added to the generation prompt or substituted for a generated response.

Generation preserves short answers, empty outputs, and length-limited outputs in its records. It honors the tokenizer/model EOS configuration, uses no positive minimum generation length, and refuses prompts whose length plus requested generation budget exceeds the selected context length. It never silently shortens a validation prompt to make it fit.

### Importing rollouts

`label` can also consume JSONL produced outside this package. A minimal record is:

```json
{"id":"query-0:0","query_id":"query-0","task":"arithmetic","sample_index":0,"messages":[{"role":"user","content":"What is 2 + 2? Reply with the answer only."},{"role":"assistant","content":"4"}],"completion":"4","finish_reason":"stop","truncated":false}
```

The prefix messages must exactly equal the corresponding query's messages; the final assistant message must exactly equal `completion`. IDs must be `query_id:sample_index`, with canonical integer sample indices starting at zero and no gaps. All queries must have equal sample counts before exclusions. Duplicate IDs, unknown queries, task mismatches, missing query coverage, and incompatible completed rollout manifests are rejected. Supported finish reasons are `stop`, `length`, `eos`, `stop_sequence`, `error`, and `abort`.

The built-in generator additionally records per-request seeds, token IDs, token counts, and prompt hashes. Imported records should retain comparable provenance, but structural validation alone cannot establish how they were generated.

### Verifiers and external rewards

`choice` uses explicit answer markers, a bare label, or a complete single-line choice such as `B. ...`; it does not scan arbitrary reasoning for the reference letter. Contradictory markers are ambiguous. `exact` defaults to NFKC Unicode normalization, stripping, whitespace collapse, and case-sensitive comparison. Its optional `normalization` object can override `unicode`, `strip`, `collapse_whitespace`, and `casefold`; an optional `extract_regex` must contain exactly one capture group and produce exactly one match. `math` uses the installed `math-verify` version, recorded in the audit.

Use `external` for a benchmark-specific evaluator, including code tasks. An external labels JSONL contains `{ "id": "query:0", "reward": 1 }` or reward `-1`. IDs must cover exactly all externally verified rollouts, including any that will subsequently be excluded. Optional `task` and `query_id` fields must match. Booleans, zero, fractional rewards, duplicate IDs, and extra labels are rejected. The package imports outcomes; it does not run generated programs or infer benchmark correctness from response length.

Empty, truncated, interrupted, or technically unverifiable responses fail labeling by default. An ordinary nonempty answer that fails the verifier's declared extraction rule is graded incorrect and logged as a parse failure. A verifier exception is not converted into a negative reward. `--allow-exclusions` explicitly accepts a changed anchor population and records excluded IDs and queries left without anchors in `<anchors.jsonl>.audit.json`.

## Assistant-token gradients and projection

Extraction computes per-example causal language-model CE averaged over supervised assistant tokens. Corpus examples supervise all assistant responses; reward-bearing anchors supervise only the final generated answer, keeping historical assistant responses as prompt context. The mask includes assistant content and its actual special-token terminator, and excludes assistant headers and following turn separators. Content spans are located in the fully rendered conversation, allowing historical and final responses to use different termination tokens. A fast tokenizer and an explicit chat template that preserves assistant content are required. Rendered-text tokenization must agree with the template's tokenization, and anchors must preserve the prompt prefix used for generation. Arbitrary tool/multimodal templates are not supported by this text-only schema.

`--max-length` limits the entire tokenized conversation. Corpus conversations may be right-truncated for gradient computation; subset export still preserves the original complete messages. Anchors must fit in full, including the generated answer and its terminator: an overlength anchor is rejected, not shortened. Inspect prompt and answer lengths before extraction. Rows with no supervised targets are rejected. `--skip-no-supervision` can exclude such corpus rows and records them in `rejected.jsonl`; reward-bearing anchors are never skipped by that option. The selection fraction then applies to surviving corpus rows, and the excluded count is carried into score and selection manifests.

All model parameters participate by default. `--exclude-parameters REGEX` explicitly changes that subset and must match between corpus and anchors. The model performs forward/backward without an optimizer update. Ordinary extraction uses evaluation mode. Optional gradient checkpointing enables training mode, disables dropout modules, and rejects model configurations with nonzero dropout to avoid functional dropout changing the gradients. This restriction does not apply to ordinary evaluation-mode extraction. Each rank holds a complete model replica.

The projector orders parameters lexicographically and treats them as one coordinate sequence. In each fixed-size global chunk it samples up to `4 * dimension` coordinates without replacement, assigns each a random output coordinate and a ±1 sign, and adds their contributions. It applies no extra rescaling or cosine normalization. Projection signs and correctness rewards are separate: stored features contain CE gradient sketches; reward signs enter the centroid only once.

`projection.json` records the parameter layout, PCG64 RNG, seed, chunk size, dimension, accumulation precision, and coordinate checksum. `projection.npz` stores the actual sampled coordinates, destinations, and signs. This release uses fixed global chunks and FP32 accumulation; the historical dattri implementation used a different RNG/chunk implementation and floating-point path. The sparse construction is portable, but equal seeds do not imply historical matrix, bitstream, or subset identity. GPU summation and model kernels can also differ across environments.

## Configured runner and independent GPU ranks

`scripts/run_method.py` reads a JSON configuration and calls `python -m drift` for each stage. It uses the same Python environment as the runner. The configuration determines the checkpoint, decoding, extraction, fit mode, and fraction; `--model` and `--backend` can override those two settings. Add an immutable `revision` to a copied configuration to pin both model-loading stages.

| Configuration | Public SFT checkpoint | Rollouts per query | Temperature / top-p / top-k |
|---|---|---:|---|
| `configs/olmo3.json` | `allenai/Olmo-3-7B-Instruct-SFT` | 32 | 0.6 / 0.95 / -1 |
| `configs/olmo2_1b.json` | `allenai/OLMo-2-0425-1B-SFT` | 16 | 1.0 / 1.0 / 40 |

Both use a 2,048-token generation budget, 4,096-token context, all model parameters, 16,384 projection dimensions, stable fitting, and a 10% selection fraction. `olmo3.json` uses the rollout count, sampling temperature, projection dimension and selection fraction of the paper's Olmo3 run (Table 7 and Appendix A.1). `olmo2_1b.json` targets a smaller model that the paper does not evaluate.

For example, after `prepare`, `rollout`, and `label` have created `corpus.jsonl` and `anchors.jsonl` under the run root, execute both extraction passes and their merges:

```bash
python scripts/run_method.py --config configs/olmo3.json \
  --model ./models/my-sft-checkpoint \
  --corpus ./data/corpus.jsonl --queries ./data/queries.jsonl \
  --output ./runs/my-drift --gpus 0,1 --stage extract
```

Choose `--stage score` and then `--stage select` with the same arguments to finish. Stage names and prerequisites are explicit:

| Runner stage | Required upstream artifacts | Output under the run root |
|---|---|---|
| `prepare` | User corpus | `corpus.jsonl` |
| `rollout` | User validation queries | `rollout-rank-R/rollouts.jsonl` |
| `label` | Every rollout rank and optional external labels | `anchors.jsonl` |
| `extract` | Canonical corpus and anchors | `corpus-rank-R/`, `anchors-rank-R/`, `corpus-features/`, `anchor-features/` |
| `score` | Both merged feature tables | `scores/` |
| `select` | Scores and canonical corpus | `selection/` |
| `all` | User corpus and queries; any needed external labels | All of the above, in dependency order |

`all` excludes continuation training. Before a real all-stage launch, the runner rejects external verifiers without a supplied labels file. External labels must grade the actual generated responses; usually run generation first, evaluate those responses, and invoke `label --external-labels` as a separate runner stage. Built-in verifiers allow an uninterrupted all-stage run. `--dry-run` prints the commands without creating output directories, reading model weights, or starting child processes; it does not validate the contents of input datasets.

Every stage output must be new, including all extraction shards and merge destinations. A stage can consume completed inputs from previous invocations, but partial stage outputs are not resumed or overwritten. Keep the GPU count consistent across runner stages so its expected rollout-rank paths match. Use the individual CLI commands when combining artifact layouts or rank counts from different runs.

Each invocation records its configuration, runner source hash, exact commands, exit codes, and per-child logs in `runner_logs/<invocation>/`. If one rank fails, the runner terminates only process groups it started, escalating after a bounded grace period. It preserves failure logs and never launches the subsequent merge or stage. It does not retry automatically or signal unrelated GPU processes.

Both generation and extraction assign input rows by `index % world_size == rank`. The runner launches one independent process per chosen GPU, with its own output directory and a full model replica. There is no fixed GPU count and no tensor-parallel model partition. Request seeds derive from the configured seed, query ID, and sample index, independently of rank count. This controls request seeding, not cross-hardware bitwise generation.

Corpus and anchor extraction can have different world sizes. Within each dataset, merge requires exactly ranks `0..W-1`, matching input hashes and feature settings, and complete row coverage. Across datasets, score requires identical feature fingerprints: model identity, tokenizer vocabulary/template, projection, masking, context length, dtype, attention implementation, selected parameters, and recorded library versions. Rank count is not part of that mathematical feature identity.

## Task balancing, fitting, and selection

For query `q` in task `k`, let `N_q` be its number of retained rollouts and `Q_k` the number of retained queries. The centroid is

```text
c_k = sum_q sum_i reward[q,i] * feature[q,i] / (Q_k * N_q)
raw[n,k] = dot(corpus_feature[n], c_k)
norm[n] = L2(corpus_feature[n])
```

Tasks are ordered lexically. Repeating every rollout of one query leaves its task weight unchanged. All-correct and all-incorrect queries remain valid anchors; zero centroids are retained and reported. Norms refer to projected gradients, not unprojected full-parameter gradients.

Each task fits `log(abs(raw)) = intercept + beta * log(norm)`. The corrected score divides by `norm ** beta`, retains the raw sign, and **does not subtract the fitted intercept**. Corrected task scores are summed.

| Fit mode | Numerical definition | Diagnostic behavior |
|---|---|---|
| `stable` | Float64 OLS with centered/scaled log norms and an intercept | Requires rank two; reports coefficients and design diagnostics. |
| `legacy-float32` | Float32 logarithms passed to `np.polyfit(..., 1)` with its default cutoff; float32 power/division correction | Reports actual rank, singular values, cutoff, coefficients, and rank-deficient tasks. |

The legacy mode exposes the earlier numerical convention rather than describing a rank-truncated fit as ordinary full-rank OLS. It is not an automatic reproduction of historical task grouping, filtering, generation, or features. Compare each run's fit report instead of assuming the two modes give the same beta.

Zero raw scores remain zero and are excluded from logarithmic fitting. Zero norms are allowed only with all-zero raw scores for that row. An all-zero task stays zero with an explicit no-fit status. A nonzero task with fewer than two usable observations or constant log norms fails rather than receiving an invented slope. Non-finite values, unrepresentable corrections, and other invalid numeric inputs are rejected.

By default, aggregate scores are neither clipped nor standardized. `--clip LOW HIGH` specifies percentile bounds on the task-summed score; `--zscore` then applies population standardization. These options never modify the individual corrected task scores. Standardizing a constant aggregate fails.

Selection returns `floor(fraction * N)` candidates, ordered by decreasing score and then increasing original feature-row position. The CLI rejects an empty selection. `selected_ids.jsonl` has ranking order; exported `subset.jsonl` has original corpus order. Supply the exact canonical corpus used for extraction: select checks its hash before exporting complete messages.

## Outputs and resource planning

| Stage | Main artifacts |
|---|---|
| Prepare | Canonical corpus JSONL and `<corpus>.manifest.json` |
| Rollout | `rollouts.jsonl`, manifest, tokenizer snapshot, model configuration |
| Label | Anchor JSONL and `<anchors>.audit.json` |
| Extract | Rank manifest, `projection.json`, `projection.npz`, feature/metadata chunks, rejection ledger |
| Merge | `features.npy`, ordered records, rejected records, complete manifest |
| Score | Raw scores, projected norms, centroids, corrected scores, aggregate scores, records, fit manifest |
| Select | Ranked selected IDs, optional complete-message subset, selection manifest |

Extraction saves explicit original row indices and IDs. Merge checks chunk checksums, rank ownership, dimensions, unique row/ID coverage, and rejection records; it refuses missing features rather than filling them with zeros. Feature tables and score products are checked against their manifests before downstream use.

Saved features use float32: budget `N * dimension * 4` bytes per feature table, plus validation features and metadata. Merge writes directly to the final disk-backed table while retaining source chunks, so allow approximately two copies of each feature dataset plus metadata and score outputs; there is no third full-size scratch feature matrix. Loading and scoring use disk-backed arrays and bounded feature blocks; metadata and regression working arrays still consume RAM. Profile the intended model/context on a small declared input before scheduling a full-corpus extraction; there is no fixed runtime estimate or required GPU count.

## Continual SFT and outcome evaluation

Train on the exported original-corpus subset starting from the same completed SFT checkpoint. Keep the tokenizer/chat template, assistant loss mask, sequence length, packing, optimizer, learning rate, and training budget explicit. Applying a subset selected at one checkpoint to a different checkpoint tests subset transfer and changes the experiment.

`scripts/continue_sft.py` calls the public ms-swift SFT command. Install it in a separate environment to keep training dependencies independent from vLLM:

```bash
python3.12 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -r requirements-train.txt
python -m pip install -e .
```

Use Python 3.12 for this training environment. The requirements pin the reviewed ms-swift commit (`3.12.0.dev0`), PyTorch 2.9.1, Transformers 4.57.3, and datasets 3.6.0; datasets 4.x is incompatible with this trainer’s non-packed length statistics. Default packed training additionally needs FlashAttention compatible with the chosen PyTorch/CUDA environment; install a matching build before launching the packed recipe. Preserve the resolved environment versions with the experiment.

```bash
python scripts/continue_sft.py --model ./models/my-sft-checkpoint \
  --dataset ./runs/my-drift/selection/subset.jsonl \
  --output ./runs/my-drift/continued-sft --gpus 0,1,2,3 \
  --max-steps 500 --max-length 32768 --global-sequences 8 \
  --learning-rate 1e-5 --min-learning-rate 1e-6 \
  --seed 42 --data-seed 42 --dry-run
```

Remove `--dry-run` to execute. The wrapper uses full-parameter BF16 training, per-device batch size one, gradient checkpointing, packed sequences, AdamW, a 10% warmup, and a cosine schedule with the specified minimum learning rate. Multi-GPU runs use DeepSpeed ZeRO-2. Gradient accumulation is `global_sequences / GPU_count`, so eight packed 32,768-token sequences yield 262,144 nominal tokens per update; actual supervised token counts depend on messages and packing. GPU count must divide the global sequence count. Choose enough memory for this recipe and a model supporting its context length.

For a limited setup check without FlashAttention, explicitly change the recipe:

```bash
python scripts/continue_sft.py --model ./models/my-sft-checkpoint \
  --dataset ./runs/my-drift/selection/subset.jsonl \
  --output ./runs/my-drift/sft-smoke --gpus 0 \
  --max-steps 2 --max-length 512 --global-sequences 1 \
  --no-packing --attn-impl sdpa --dry-run
```

This checks a different training budget and is not the default continuation experiment. Real training requires a new output directory and records the dataset hash, command, settings, trainer version, return code, and nominal token budget in `training_manifest.json`, with stdout/stderr in `train.log` and weights under `checkpoints/`. The wrapper loads `scripts/swift_template.py` as an external plugin. Its `drift_chat` template calls the same `assistant_labels` function as attribution, preserving the checkpoint’s full chat template and supervising all assistant responses plus their actual end markers. Right truncation matches corpus attribution; samples with no surviving assistant targets fail. The plugin supports text-only causal SFT, including native packing and dynamic padding. A successful wrapper exit establishes process completion, not held-out model quality.

The CPU demo, unit tests and artifact checks establish code behavior. Evaluating a trained checkpoint on held-out benchmarks is a separate stage with its own dataset versions, prompts, sampling repeats and evaluators. Appendix A.1 of the paper lists the evaluation configurations behind its reported numbers.
