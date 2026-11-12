# OLMo3 evaluation

This entry point evaluates a local OLMo3 checkpoint on the eight tasks used in the paper. It runs three commands sequentially, validates the expected questions and response counts, and writes `summary.json`. Existing output directories are rejected; failures stop without retries or terminating unrelated processes.

The patches preserve the audited task and scoring settings. They passed patch-application and CPU checks, **not a fresh eight-task GPU run**. The original installed-package lock was not archived; the constraints below describe the inspected evaluation environments, not a claim of bitwise historical reproducibility.

## Setup

Use Linux, Python 3.11, CUDA-compatible GPUs, and two separate virtual environments: OLMES requires NumPy 1 / Datasets 3, while LightEval requires NumPy 2 / Datasets 4. From the DRIFT repository root:

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

Accept the Hugging Face access conditions for `Idavidrein/gpqa` and `allenai/ZebraLogicBench-private`, and authenticate each environment using your own account. The latter is the answer-bearing dataset; `allenai/ZebraLogicBench` has placeholder solutions and is not a substitute. Code benchmarks execute generated programs through the evaluators; run them in the isolated evaluation environment.

```bash
.venv-lighteval/bin/python scripts/evaluate_olmo3.py \
  --checkpoint /path/to/checkpoint-500 --output runs/evaluation-500 \
  --olmes-python "$PWD/.venv-olmes/bin/python" \
  --lighteval-python "$PWD/.venv-lighteval/bin/python" \
  --olmes-source "$PWD/vendor/olmes" \
  --lighteval-source "$PWD/vendor/lighteval" \
  --gpus 0,1,2,3,4,5,6,7 --preflight-only
```

Remove `--preflight-only` to evaluate. `--dry-run` only prints the three commands and does not check installations. The launcher verifies patched source hashes, records installed package versions and checkpoint metadata, and checks the same fingerprints before and after each stage. Weight provenance remains the responsibility of the checkpoint's training manifest.

## Protocol

| Task | Questions | Responses per question | Temperature | Reported metric |
|---|---:|---:|---:|---|
| BBH, 18 multiple-choice tasks | 4365 | 1 | 0 | Micro average |
| MBPP+ | 378 | 8 | 0.6 | Pass@1 |
| MATH | 5000 | 1 | 0 | Macro average across 7 subjects |
| MMLU-Pro | 12032 | 1 | 0 | Extractive match |
| ZebraLogic | 1000 | 4 | 0.6 | Complete-grid accuracy averaged over responses |
| LiveCodeBench historical snapshot | 1055 | 4 | 0.6 | Pass@1 |
| OlympiadBench English math | 674 | 4 | 0.6 | Average response accuracy |
| GPQA Diamond | 198 | 16 | 0.6 | Pass@1 |

**LiveCodeBench version correction:** the paper used the label LCBv5, but its local JSON loader ignored `release_v5` and loaded six files totaling 1055 problems. The corresponding public collection is `release_v6`; `release_v5` has 880 problems. This release fixes the public data reconstruction to the actual 1055-problem collection, rather than silently changing the evaluated set. Its dataset commit and six-file evidence are recorded in `olmes_patch_provenance.json`.

OLMES uses a **4096-token prompt-plus-completion window**, with batch size 64; its shared per-batch output budget depends on prompt length. LightEval uses a 32768-token window and up to 4096 new tokens, top-p 0.95, sampling seed 42, and model seed 1234. The default uses eight visible GPUs. OLMES's historical per-request sampling seeds were not archived, so repeated sampled results need not be identical.

`protocol.json` specifies the counts and aggregation. `sources.json` specifies upstream commits and checked source hashes. The final score is the unrounded arithmetic mean of the eight percentages; the first four tasks form the target average.

## Sources and licenses

- [OLMES upstream commit](https://github.com/allenai/olmes/commit/f122a7bad99551ffb3f5feec14169af794926f6b), Apache 2.0; license retained in `licenses/olmes-Apache-2.0.txt`.
- [LightEval upstream commit](https://github.com/huggingface/lighteval/commit/7c1cd62716b0a198a630c26d781430c54726cd02), MIT; license retained in `licenses/lighteval-MIT.txt`.

Only task, scoring, and required evaluator compatibility changes are included. Dataset licenses and access conditions remain those of their respective providers. A fresh dependency installation and full GPU run remain unverified.
