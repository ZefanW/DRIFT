# Olmo3 evaluation

`scripts/evaluate_olmo3.py` evaluates a local Olmo3 checkpoint on the paper's eight benchmarks. It runs OLMES and LightEval in separate environments, checks the question and response counts against [`protocol.json`](protocol.json), and writes `summary.json`.

## Setup

Linux, Python 3.11 and CUDA GPUs. The two harnesses need separate virtual environments, because OLMES requires NumPy 1 and Datasets 3 while LightEval requires NumPy 2 and Datasets 4. From the DRIFT repository root:

```bash
git clone https://github.com/allenai/olmes.git vendor/olmes
git -C vendor/olmes checkout f122a7bad99551ffb3f5feec14169af794926f6b
git -C vendor/olmes apply ../../evaluation/patches/olmes-olmo3-main.patch
git clone https://github.com/huggingface/lighteval.git vendor/lighteval
git -C vendor/lighteval checkout 7c1cd62716b0a198a630c26d781430c54726cd02
git -C vendor/lighteval apply ../../evaluation/patches/lighteval-0.13.0-drift.patch

python3.11 -m venv .venv-olmes
.venv-olmes/bin/pip install -c evaluation/constraints-olmes.txt -e 'vendor/olmes[gpu]'
python3.11 -m venv .venv-lighteval
.venv-lighteval/bin/pip install -c evaluation/constraints-lighteval.txt -e 'vendor/lighteval[vllm]'
```

Two datasets are gated. Accept the access conditions of `Idavidrein/gpqa` and `allenai/ZebraLogicBench-private` on Hugging Face and log in from both environments. The public `allenai/ZebraLogicBench` has placeholder solutions and cannot be used for scoring.

MBPP+ and LiveCodeBench execute model-generated code, so run the evaluation in an isolated environment.

## Run

```bash
.venv-lighteval/bin/python scripts/evaluate_olmo3.py \
  --checkpoint /path/to/checkpoint-500 --output runs/evaluation-500 \
  --olmes-python "$PWD/.venv-olmes/bin/python" \
  --lighteval-python "$PWD/.venv-lighteval/bin/python" \
  --olmes-source "$PWD/vendor/olmes" \
  --lighteval-source "$PWD/vendor/lighteval" \
  --gpus 0,1,2,3,4,5,6,7 --preflight-only
```

`--preflight-only` checks the patched sources and both environments without evaluating. Remove it to run the three evaluation commands in sequence. `--dry-run` only prints those commands.

The launcher verifies the hashes of the patched sources, records package versions and checkpoint metadata, and checks them again after each stage. It refuses to overwrite an existing output directory and stops at the first failure.

## Protocol

| Benchmark | Harness | Questions | Responses per question | Temperature | Metric |
|---|---|---:|---:|---:|---|
| BBH, 18 multiple-choice subtasks | OLMES | 4,365 | 1 | 0 | Micro average |
| MBPP+ | OLMES | 378 | 8 | 0.6 | Pass@1 |
| MATH | OLMES | 5,000 | 1 | 0 | Macro average over 7 subjects |
| MMLU-Pro | LightEval | 12,032 | 1 | 0 | Extractive match |
| ZebraLogic | OLMES | 1,000 | 4 | 0.6 | Whole-grid accuracy, averaged over responses |
| LiveCodeBench | OLMES | 1,055 | 4 | 0.6 | Pass@1 |
| OlympiadBench, English math | LightEval | 674 | 4 | 0.6 | Accuracy, averaged over responses |
| GPQA Diamond | LightEval | 198 | 16 | 0.6 | Pass@1 |

`All` is the unrounded mean of the eight percentages. The first four benchmarks form the validation-source average and the last four the additional average.

[`protocol.json`](protocol.json) holds these counts and the aggregation rules. [`sources.json`](sources.json) holds the upstream commits and the hashes of the patched files.

### LiveCodeBench version

The paper labels this benchmark LiveCodeBench v5. The loader used for the paper's runs ignored the `release_v5` setting and read six files with 1,055 problems in total, which is the public `release_v6` collection. `release_v5` has 880 problems. The harness here pins the 1,055-problem collection, so it evaluates the same problems as the paper. [`olmes_patch_provenance.json`](olmes_patch_provenance.json) records the dataset commit and the six files.

### Generation settings

| Harness | Context | Sampling |
|---|---|---|
| OLMES | 4,096 tokens for prompt plus completion, batch size 64, completion budget set per batch by prompt length | Top-p 0.95 |
| LightEval | 32,768 tokens, up to 4,096 new tokens | Top-p 0.95, sampling seed 42, model seed 1234 |

The per-request sampling seeds of the original OLMES runs were not saved, so sampled benchmarks will not match the paper run for run.

## Status

The patches apply cleanly to the pinned upstream commits and pass CPU checks. The eight benchmarks have not been re-run end to end on GPU from a fresh install. The package lock of the original environment was not kept, so `constraints-olmes.txt` and `constraints-lighteval.txt` list the versions found in the evaluation environments.

## Sources and licenses

| Harness | Upstream commit | License |
|---|---|---|
| OLMES | [`f122a7b`](https://github.com/allenai/olmes/commit/f122a7bad99551ffb3f5feec14169af794926f6b) | Apache 2.0, in [`licenses/olmes-Apache-2.0.txt`](licenses/olmes-Apache-2.0.txt) |
| LightEval | [`7c1cd62`](https://github.com/huggingface/lighteval/commit/7c1cd62716b0a198a630c26d781430c54726cd02) | MIT, in [`licenses/lighteval-MIT.txt`](licenses/lighteval-MIT.txt) |

The patches change task definitions, scoring and the evaluator compatibility needed to run them. Benchmark datasets keep the licenses and access conditions of their providers.
