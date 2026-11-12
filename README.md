# DRIFT

**Data Selection for LLM Instruction Tuning via On-Policy Attribution**

Zefan Wang · Lincheng Li · Tianyu Yu · Yuan Yao

**[Paper](https://arxiv.org/abs/2606.18307) · [PDF](https://arxiv.org/pdf/2606.18307) · [OLMo3 reproduction](docs/reproduce_olmo3.md)**

DRIFT improves an already SFT-trained model by selecting better examples from its original training corpus. It uses the model’s own validation responses, correctness-weighted gradient attribution, and gradient-norm correction to select the top 10% for continued SFT.

## OLMo3-7B results

Accuracy (%) from the paper's main table, with standard deviations from the appendix. Continued-SFT results are mean ± standard deviation across three runs, including training and evaluation randomness.

| Method | BBH | MBPP+ | MATH | MMLU-Pro | ZebraLogic | LCB v5 | OlympiadBench | GPQA-D | All | Additional |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 44.63 | 56.42 | 66.16 | 44.22 | 17.23 | 15.43 | 34.27 | 35.01 | 39.17 | 25.49 |
| Random | 44.45 ± 0.29 | 56.34 ± 0.17 | 66.40 ± 0.28 | 44.25 ± 0.15 | 17.67 ± 0.96 | 14.38 ± 0.63 | 35.04 ± 0.47 | 33.77 ± 0.68 | 39.04 ± 0.15 | 25.21 ± 0.16 |
| DSIR | 39.97 ± 0.23 | 52.63 ± 0.88 | 55.11 ± 0.42 | 36.22 ± 0.82 | 17.68 ± 0.34 | 12.73 ± 0.68 | 27.83 ± 1.07 | 32.82 ± 0.70 | 34.37 ± 0.02 | 22.76 ± 0.04 |
| Qurating | 44.27 ± 0.21 | 55.96 ± 0.05 | 66.89 ± 0.37 | 44.14 ± 0.16 | 17.77 ± 0.58 | 15.49 ± 0.94 | 34.98 ± 0.26 | 34.61 ± 0.55 | 39.26 ± 0.23 | 25.71 ± 0.31 |
| BM25 | 44.44 ± 0.40 | 55.74 ± 0.99 | 67.82 ± 0.27 | 44.66 ± 0.14 | 17.48 ± 0.51 | 15.13 ± 1.04 | **36.18** ± 0.64 | **34.90** ± 1.36 | 39.54 ± 0.19 | 25.92 ± 0.42 |
| LESS | 43.38 ± 0.21 | 53.80 ± 0.20 | 64.60 ± 0.40 | 43.38 ± 0.14 | 16.14 ± 0.90 | 13.45 ± 0.70 | 33.06 ± 0.63 | 33.21 ± 0.51 | 37.63 ± 0.29 | 23.96 ± 0.40 |
| RDS | 43.68 ± 0.13 | 55.59 ± 0.24 | 66.56 ± 0.39 | 44.80 ± 0.18 | 17.76 ± 1.09 | 14.59 ± 0.84 | 35.38 ± 0.55 | 34.27 ± 0.67 | 39.08 ± 0.24 | 25.50 ± 0.29 |
| IF (GraSS) | 44.58 ± 0.35 | 54.97 ± 0.28 | 65.69 ± 0.26 | 44.81 ± 0.21 | 16.98 ± 0.35 | 13.67 ± 0.78 | 34.21 ± 0.17 | 34.35 ± 0.41 | 38.66 ± 0.08 | 24.80 ± 0.17 |
| **DRIFT** | **45.86** ± 0.30 | **56.40** ± 0.22 | **67.84** ± 0.43 | **44.94** ± 0.19 | **19.95** ± 0.94 | **16.28** ± 0.58 | 35.25 ± 0.56 | 34.57 ± 0.92 | **40.14** ± 0.12 | **26.51** ± 0.24 |

All averages the eight benchmarks; Additional averages the last four. Bold marks the best continued-SFT mean. No standard deviation is reported for Base.

Download the [per-run evaluation results](results/README.md) for the main comparison and available ablations.

“Base” is the checkpoint **after its original SFT schedule**. Continued training uses 192,453 selected examples, 500 updates, a 262,144-token global batch, and learning rate 1e-5.

## Reproduce OLMo3-7B

- **Model:** [allenai/Olmo-3-7B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT).
- **Data:** [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT); 1,924,533 non-tool candidate examples.
- **Paper’s selected subset:** [ordered selection index](artifacts/olmo3_selected.jsonl.gz), with fixed [model/data versions](artifacts/olmo3_selection_manifest.json). The reconstruction script preserves the selected messages, duplicates, and training order.
- **Validation rollouts and labels:** [Hugging Face dataset](https://huggingface.co/datasets/ZefanW/DRIFT-OLMo3-Validation-Rollouts), containing 8,220 validation queries and 263,040 generated responses with correctness labels.

### 1. Install

Use Linux, Python 3.12, and NVIDIA GPUs. The paper training setup uses 8 × 80 GB GPUs.

```bash
git clone https://github.com/ZefanW/DRIFT.git
cd DRIFT
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
pip install -e '.[data]'
```

Install FlashAttention 2 compatible with your PyTorch/CUDA build; see the [environment setup](docs/reproduce_olmo3.md#environment).

### 2. Rebuild the selected training data

```bash
python scripts/prepare_olmo3_subset.py --output data/olmo3_drift.jsonl
```

This streams the pinned public Dolci dataset and reconstructs the exact selected conversations using content hashes. It avoids recomputing full-corpus influence scores.

### 3. Continue SFT

```bash
python scripts/continue_sft.py \
  --model allenai/Olmo-3-7B-Instruct-SFT \
  --revision d9687e20d263cd0876dea143723b3c4974d19935 \
  --dataset data/olmo3_drift.jsonl --output runs/olmo3_drift \
  --gpus 0,1,2,3,4,5,6,7
```

The defaults follow the recorded OLMo3 training configuration. See [reproduction and evaluation](docs/reproduce_olmo3.md) for the eight benchmarks and [DRIFT selection](docs/selection.md) to recompute attribution on your own data.

## Citation

```bibtex
@misc{wang2026driftdataselectionllm,
  title={DRIFT: Data Selection for LLM Instruction Tuning via On-Policy Attribution},
  author={Zefan Wang and Lincheng Li and Tianyu Yu and Yuan Yao},
  year={2026},
  eprint={2606.18307},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2606.18307}
}
```

Code is released under the [MIT License](LICENSE). The linked model and datasets retain their original licenses.
