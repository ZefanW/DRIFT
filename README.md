# DRIFT

**Data Selection for LLM Instruction Tuning via On-Policy Attribution**

Zefan Wang · Lincheng Li · Tianyu Yu · Yuan Yao

**[Paper](https://arxiv.org/abs/2606.18307) · [PDF](https://arxiv.org/pdf/2606.18307) · [OLMo3 reproduction](docs/reproduce_olmo3.md)**

DRIFT improves an already SFT-trained model by selecting better examples from its original training corpus. It uses the model’s own validation responses, correctness-weighted gradient attribution, and gradient-norm correction to select the top 10% for continued SFT.

## OLMo3-7B results

Accuracy reported in the paper (Table 1):

| Model | Eight-benchmark average | Additional benchmarks |
|---|---:|---:|
| OLMo3-7B-Instruct-SFT | 39.17 | 25.49 |
| + DRIFT | **40.14** | **26.51** |

“Base” is the checkpoint **after its original SFT schedule**. Continued training uses 192,453 selected examples, 500 updates, a 262,144-token global batch, and learning rate 1e-5.

## Reproduce OLMo3-7B

- **Model:** [allenai/Olmo-3-7B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT).
- **Data:** [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT); 1,924,533 non-tool candidate examples.
- **Paper’s selected subset:** [ordered selection index](artifacts/olmo3_selected.jsonl.gz), with fixed [model/data versions](artifacts/olmo3_selection_manifest.json). The reconstruction script preserves the selected messages, duplicates, and training order.

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
