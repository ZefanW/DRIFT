# Reproduce the Olmo3 experiment

This guide retrains Olmo3-7B-Instruct-SFT on the subset DRIFT selected in the [paper](https://arxiv.org/abs/2606.18307) and evaluates the result. It uses the released selection of **192,453 conversations out of 1,924,533 candidates**, so no rollouts or gradient features are recomputed. To run the selection yourself, see the [selection guide](selection.md).

## Environment

Linux, Python 3.12 and NVIDIA GPUs. The paper's training run used eight 80 GB GPUs. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
pip install -e '.[data]'
pip install 'flash-attn==2.8.3' --no-build-isolation
```

FlashAttention has to match your PyTorch and CUDA build. `requirements-train.txt` pins the main training dependencies. The full lock file of the original run was not kept.

## Data and checkpoint

The [selection manifest](../artifacts/olmo3_selection_manifest.json) pins everything the subset depends on.

| Item | Pinned value |
|---|---|
| Model | `allenai/Olmo-3-7B-Instruct-SFT` at revision `d9687e20d263cd0876dea143723b3c4974d19935` |
| Corpus | `allenai/Dolci-Instruct-SFT` at revision `b779ae7ae5c757ccea6abb04a05204470bf1e602`, train split, rows with a null `source` removed |
| Selection | 192,453 rows in their original training order, each identified by a SHA-256 of its message content |

Rows are matched by content hash because the corpus contains duplicate IDs. The [index format](../artifacts/README.md) is described next to the file.

```bash
python scripts/prepare_olmo3_subset.py --output data/olmo3_drift.jsonl
```

The script streams the corpus from the Hub and writes the selected conversations in training order, duplicates included. It stops if a conversation is missing or does not match its hash. Leave several GB of free disk for the dataset cache, a temporary database and the output.

If you already have the pinned corpus as Parquet shards:

```bash
python scripts/prepare_olmo3_subset.py \
  --corpus /path/to/Dolci-Instruct-SFT \
  --output data/olmo3_drift.jsonl
```

The index stores hashes and source IDs only. The conversations come from the Dolci dataset under the terms on its [dataset card](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT).

## Training

```bash
python scripts/continue_sft.py \
  --model allenai/Olmo-3-7B-Instruct-SFT \
  --revision d9687e20d263cd0876dea143723b3c4974d19935 \
  --dataset data/olmo3_drift.jsonl --output runs/olmo3_drift \
  --gpus 0,1,2,3,4,5,6,7
```

The defaults are the paper's Olmo3 recipe.

| Setting | Value |
|---|---|
| Updates and context | 500 updates on packed 32,768-token sequences |
| Global batch | 8 sequences, 262,144 tokens |
| Learning rate | 1e-5 peak, cosine decay to 1e-6, 10% warmup |
| Optimizer | Fused AdamW, betas (0.9, 0.95), weight decay 0, gradient clipping 1 |
| Precision and memory | BF16, gradient checkpointing, FlashAttention 2, Liger kernels, DeepSpeed ZeRO-2 |
| Template and truncation | ms-swift `dummy` template with the `jinja` backend, overlength examples dropped |
| Seeds | Training seed 42, data seed 42, with dataset and dataloader shuffling |

`--template-mode paper` is the default and reproduces the original training path, in which the loss covers the whole rendered sequence. `--template-mode assistant` restricts the loss to assistant tokens and is a different recipe. Attribution always uses assistant-token gradients, whatever this flag is set to.

The original arguments set both `optim_args=eps=1e-6` and `adam_epsilon=1e-8`. With the pinned Transformers version the fused AdamW path reads `adam_epsilon`. Both are passed as they were.

With four GPUs, pass `--gpus 0,1,2,3`. Gradient accumulation doubles to keep the global batch.

The run writes `training_manifest.json`, `train.log` and `checkpoints/checkpoint-500`. Each run needs a new output directory, and `--dry-run` prints the training command without starting it.

## Evaluation

Evaluate `checkpoints/checkpoint-500` on the eight benchmarks with the [evaluation harness](../evaluation/README.md). It lists the decoding settings, the number of sampled responses and the metric of each benchmark.

The paper's numbers for this experiment:

| Model | All | Additional |
|---|---:|---:|
| Starting checkpoint | 39.17 | 25.49 |
| After continued SFT on the DRIFT subset, three runs | 40.14 ± 0.12 | 26.51 ± 0.24 |

`All` is the mean of the eight benchmark accuracies and `Additional` the mean of ZebraLogic, LiveCodeBench, OlympiadBench and GPQA Diamond. The three runs behind the second row are in [`results/`](../results/).
