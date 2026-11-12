# Release validation

Checks performed on 2026-11-12 for version 0.1.0:

| Check | Observed result |
|---|---|
| CPU suite | 111 tests and 31 subtests passed on Python 3.14 / NumPy 2.5.4. |
| CPU demo and saved-response verification | Synthetic scoring/selection and signed-reward fixture labeling completed. |
| Real OLMo2 1B selection | 13 corpus conversations, 2 validation queries, 4 generated responses, 128-dimensional features, and export of 2 selected conversations completed. |
| Real OLMo3 7B selection | The same small pipeline completed, including the corpus's multi-turn example and its distinct intermediate/final assistant terminators. |
| vLLM generation | OLMo2 1B generated 4 responses; all stopped normally, with no empty or length-limited outputs. |
| Continued SFT | The supplied Swift wrapper and `drift_chat` plugin completed 2 optimizer steps on the small OLMo2 subset and saved checkpoint weights and trainer state. Chat headers were present in inputs and masked from loss. |
| Orchestration | A deliberately failing rank stopped its owned sibling process, prevented merge, and left an unrelated process running. Dry runs created no output. |

GPU checks used an A800 80 GB. Transformers checks used Python 3.12, PyTorch 2.9.1, Transformers 4.57.3, and NumPy 1.26.4. The vLLM check used vLLM 0.11.0, PyTorch 2.8.0, and Transformers 4.57.6. Continued SFT used the commit pinned in `requirements-train.txt` with datasets 3.6.0; its check used SDPA, no packing, a 256-token limit, and one sequence per update.

These checks establish execution and artifact consistency on small inputs. They do not measure selection quality or reproduce paper benchmark scores. The full corpus, default 500-update packed training recipe, and multi-GPU DeepSpeed training were not rerun for this release. Backend/kernel changes can alter floating-point results. See [reproducibility.md](reproducibility.md) for the numerical modes and projection differences.

To repeat the small real-model selection check with the shipped fixtures:

```bash
python scripts/run_method.py --config configs/smoke.json \
  --model ./models/my-sft-checkpoint --corpus examples/corpus.jsonl \
  --queries examples/queries.jsonl --output runs/model-smoke --gpus 0
```

Use a fresh output directory for each model. The default fixture smoke backend is Transformers; `--backend vllm` selects the alternative generator. CPU tests require no GPU libraries or model downloads.
