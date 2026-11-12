# Reproduce the OLMo3-7B experiment

This route uses the paper’s frozen selection: **192,453 conversations from 1,924,533 non-tool candidates**. Rebuilding that subset avoids differences caused by regenerating rollouts or random projections.

## Environment

Use Linux, Python 3.12, and an NVIDIA GPU environment. The recorded training configuration used 8 × 80 GB GPUs. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
pip install -e '.[data]'
pip install 'flash-attn==2.8.3' --no-build-isolation
```

FlashAttention must match the installed PyTorch/CUDA build. `requirements-train.txt` pins the available training stack; it is not a claim that every dependency version from the original run was archived.

## Data and checkpoint

The [selection manifest](../artifacts/olmo3_selection_manifest.json) fixes:

- Model: `allenai/Olmo-3-7B-Instruct-SFT`, revision `d9687e20d263cd0876dea143723b3c4974d19935`.
- Source data: `allenai/Dolci-Instruct-SFT`, revision `b779ae7ae5c757ccea6abb04a05204470bf1e602`, train split. This is the original downloaded parent corpus. Records with null `source` are excluded before matching.
- Selection: the original selected parquet’s **row order and multiplicity**, recorded by message-content SHA256. IDs alone are insufficient because the corpus contains duplicate IDs.

```bash
python scripts/prepare_olmo3_subset.py --output data/olmo3_drift.jsonl
```

The script streams the public source and writes the selected conversations in their original training order. Content hashes include every role/content turn; empty optional tool fields are omitted. Missing or mismatching conversations stop reconstruction. Temporary text is cached on disk, not accumulated in RAM. Allow several GB of free disk space for dataset reads, the temporary database, and the output.

If the pinned corpus is already downloaded as Parquet shards:

```bash
python scripts/prepare_olmo3_subset.py \
  --corpus /path/to/Dolci-Instruct-SFT \
  --output data/olmo3_drift.jsonl
```

The released index contains hashes and source IDs, not the original conversation text. Dataset terms are provided on the [Dolci dataset card](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT).

## Training

```bash
python scripts/continue_sft.py \
  --model allenai/Olmo-3-7B-Instruct-SFT \
  --revision d9687e20d263cd0876dea143723b3c4974d19935 \
  --dataset data/olmo3_drift.jsonl --output runs/olmo3_drift \
  --gpus 0,1,2,3,4,5,6,7
```

| Setting | Recorded OLMo3 recipe |
|---|---|
| Updates / packed context | 500 / 32,768 tokens |
| Global batch | 8 sequences, nominally 262,144 tokens |
| Learning rate | 1e-5 peak, cosine decay to 1e-6, 10% warmup |
| Optimizer | Fused AdamW, betas (0.9, 0.95), weight decay 0, clipping 1 |
| Precision / memory | BF16, gradient checkpointing, FlashAttention 2, Liger, ZeRO-2 |
| Template / truncation | Swift `dummy` + `jinja`; delete overlength examples |
| Randomization | Training seed 42; data seed 42; dataset and dataloader shuffling |

`--template-mode paper` is the default. It preserves the original Jinja training path, whose loss covers the rendered sequence. The optional `assistant` mode uses assistant-only loss and changes this training recipe. Attribution’s assistant-token gradients are a separate computation.

The original arguments include `optim_args=eps=1e-6` and `adam_epsilon=1e-8`; the pinned Transformers fused-AdamW path uses `adam_epsilon`. Both values are retained. Four GPUs can use `--gpus 0,1,2,3`; accumulation doubles to preserve the global batch, although hardware and kernels can still affect results.

Outputs include `training_manifest.json`, `train.log`, and `checkpoints/checkpoint-500`. Use a fresh output directory. Add `--dry-run` to inspect the actual command.

## Evaluation

The paper evaluates BBH, MBPP+, MATH, MMLU-Pro, ZebraLogic, LiveCodeBench v5, OlympiadBench, and GPQA Diamond. See [evaluation setup](../evaluation/README.md) for the released harness configuration and commands.

| Benchmark | Decoding / aggregation |
|---|---|
| BBH | Greedy; micro-average over the selected 18 subtasks |
| MATH | Greedy; macro-average over seven subjects |
| MMLU-Pro | Greedy |
| MBPP+ | Temperature 0.6, top-p 0.95, avg@8 |
| ZebraLogic, LCB v5, OlympiadBench | Temperature 0.6, top-p 0.95, avg@4 |
| GPQA Diamond | Temperature 0.6, top-p 0.95, avg@16 |

`All` is the arithmetic mean of the eight benchmark accuracies. `Additional` averages ZebraLogic, LCB v5, OlympiadBench, and GPQA Diamond. The paper reports 39.17 → 40.14 for `All`; these are the paper’s reported values, not measurements from rerunning the release code.
