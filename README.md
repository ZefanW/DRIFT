<div align="center">

# DRIFT

**Data Selection for LLM Instruction Tuning via On-Policy Attribution**

[![arXiv](https://img.shields.io/badge/arXiv-2606.18307-b31b1b.svg)](https://arxiv.org/abs/2606.18307)
[![Project page](https://img.shields.io/badge/project-page-c2603c.svg)](https://zefanw.github.io/DRIFT/)
[![Rollouts on Hugging Face](https://img.shields.io/badge/Hugging%20Face-validation%20rollouts-ffb000.svg?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/ZefanW/DRIFT-OLMo3-Validation-Rollouts)
[![Tests](https://github.com/ZefanW/DRIFT/actions/workflows/tests.yml/badge.svg)](https://github.com/ZefanW/DRIFT/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-2ea44f.svg)](LICENSE)

[Paper](https://arxiv.org/abs/2606.18307) · [Project page](https://zefanw.github.io/DRIFT/) · [Results](#results) · [Artifacts](#released-artifacts) · [Reproduce](#reproduce-the-olmo3-experiment) · [Your data](#run-drift-on-your-own-data) · [Citation](#citation)

<sub>Zefan Wang, Lincheng Li, Tianyu Yu, Yuan Yao · Tsinghua University</sub>

</div>

Data selection for instruction tuning is usually studied for efficiency. DRIFT studies whether selection can also raise the ceiling and produce a better model than training on the full corpus.

DRIFT starts from a model that has already finished SFT on all of its data. It samples the model's own responses to a set of validation queries, labels each response as correct or incorrect, and ranks the original SFT corpus by its influence on those signed responses. Continuing SFT on the top 10% improves Olmo3-7B-Instruct-SFT and OpenR1-Distill-7B over their full-data checkpoints and outperforms every selection baseline we evaluated.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/overview-dark.svg">
    <img src="assets/overview-light.svg" width="100%" alt="DRIFT overview. Step 1 samples and verifies on-policy rollouts. Step 2 builds signed validation centroids. Step 3 scores the SFT corpus and removes the gradient-norm trend. Step 4 continues SFT on the top 10%.">
  </picture>
</p>

## Method

Influence functions are a local approximation, so they are most reliable when the validation target lies close to the current parameters. A model's own responses are much closer targets than external reference answers. DRIFT attributes on those responses and signs them by correctness.

1. **On-policy rollouts.** Sample several responses to each validation query from the checkpoint and verify them. Reference answers are used only for grading.
2. **Signed validation centroids.** Take the token-mean CE gradient of each response over its assistant tokens and compress it with a [GraSS](https://arxiv.org/abs/2505.18976) sparse projection. Average the gradients within each validation task, weighting correct responses by `+1` and incorrect ones by `−1`.
3. **Debiased influence.** Score every corpus example by its dot product with each task centroid. Within each task, fit a power law between score magnitude and gradient norm, remove the fitted norm component, and keep the sign. Sum the corrected scores over tasks.
4. **Continued SFT.** Keep the top 10% of the corpus and continue training from the same checkpoint.

## Results

Accuracy (%) after continued SFT on 10% of each model's original SFT corpus (paper, Table 1).

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Method</th>
      <th colspan="2" align="center">Olmo3-7B-Instruct-SFT</th>
      <th colspan="2" align="center">OpenR1-Distill-7B</th>
    </tr>
    <tr>
      <th align="right">All</th><th align="right">Add.</th>
      <th align="right">All</th><th align="right">Add.</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><i>Full data</i></td><td align="right"><i>39.17</i></td><td align="right"><i>25.49</i></td><td align="right"><i>57.23</i></td><td align="right"><i>46.24</i></td></tr>
    <tr><td>Random</td><td align="right">39.04</td><td align="right">25.21</td><td align="right">57.40</td><td align="right">46.65</td></tr>
    <tr><td>DSIR</td><td align="right">34.37</td><td align="right">22.76</td><td align="right">56.24</td><td align="right">45.43</td></tr>
    <tr><td>QuRating</td><td align="right">39.26</td><td align="right">25.71</td><td align="right">57.78</td><td align="right">47.05</td></tr>
    <tr><td>BM25</td><td align="right">39.54</td><td align="right">25.92</td><td align="right">57.32</td><td align="right">46.54</td></tr>
    <tr><td>LESS</td><td align="right">37.63</td><td align="right">23.96</td><td align="right">57.28</td><td align="right">46.44</td></tr>
    <tr><td>RDS</td><td align="right">39.08</td><td align="right">25.50</td><td align="right">57.63</td><td align="right">46.91</td></tr>
    <tr><td>IF (GraSS)</td><td align="right">38.66</td><td align="right">24.80</td><td align="right">57.85</td><td align="right">47.69</td></tr>
    <tr><td><b>DRIFT</b></td><td align="right"><b>40.14</b></td><td align="right"><b>26.51</b></td><td align="right"><b>58.66</b></td><td align="right"><b>48.73</b></td></tr>
  </tbody>
</table>

*Full data* is the checkpoint after its original SFT on the whole corpus. Every other row continues SFT from it, and bold marks the best of them in each column. **All** averages eight benchmarks. **Add.** averages the four that supply no validation queries. Olmo3 numbers are means over three runs.

<details>
<summary>Per-benchmark results on Olmo3-7B-Instruct-SFT</summary>
<br>
<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Method</th>
      <th colspan="4" align="center">Validation-source benchmarks</th>
      <th colspan="4" align="center">Additional benchmarks</th>
    </tr>
    <tr>
      <th align="right">BBH</th><th align="right">MBPP+</th><th align="right">MATH</th><th align="right">MMLU<br>Pro</th>
      <th align="right">Zebra<br>Logic</th><th align="right">LCB</th><th align="right">Olympiad<br>Bench</th><th align="right">GPQA<br>Diamond</th>
    </tr>
  </thead>
  <tbody>
      <tr><td><i>Full data</i></td><td align="right"><i>44.63</i></td><td align="right"><i>56.42</i></td><td align="right"><i>66.16</i></td><td align="right"><i>44.22</i></td><td align="right"><i>17.23</i></td><td align="right"><i>15.43</i></td><td align="right"><i>34.27</i></td><td align="right"><i>35.01</i></td></tr>
      <tr><td>Random</td><td align="right">44.45</td><td align="right">56.34</td><td align="right">66.40</td><td align="right">44.25</td><td align="right">17.67</td><td align="right">14.38</td><td align="right">35.04</td><td align="right">33.77</td></tr>
      <tr><td>DSIR</td><td align="right">39.97</td><td align="right">52.63</td><td align="right">55.11</td><td align="right">36.22</td><td align="right">17.68</td><td align="right">12.73</td><td align="right">27.83</td><td align="right">32.82</td></tr>
      <tr><td>QuRating</td><td align="right">44.27</td><td align="right">55.96</td><td align="right">66.89</td><td align="right">44.14</td><td align="right">17.77</td><td align="right">15.49</td><td align="right">34.98</td><td align="right">34.61</td></tr>
      <tr><td>BM25</td><td align="right">44.44</td><td align="right">55.74</td><td align="right">67.82</td><td align="right">44.66</td><td align="right">17.48</td><td align="right">15.13</td><td align="right"><b>36.18</b></td><td align="right"><b>34.90</b></td></tr>
      <tr><td>LESS</td><td align="right">43.38</td><td align="right">53.80</td><td align="right">64.60</td><td align="right">43.38</td><td align="right">16.14</td><td align="right">13.45</td><td align="right">33.06</td><td align="right">33.21</td></tr>
      <tr><td>RDS</td><td align="right">43.68</td><td align="right">55.59</td><td align="right">66.56</td><td align="right">44.80</td><td align="right">17.76</td><td align="right">14.59</td><td align="right">35.38</td><td align="right">34.27</td></tr>
      <tr><td>IF (GraSS)</td><td align="right">44.58</td><td align="right">54.97</td><td align="right">65.69</td><td align="right">44.81</td><td align="right">16.98</td><td align="right">13.67</td><td align="right">34.21</td><td align="right">34.35</td></tr>
      <tr><td><b>DRIFT</b></td><td align="right"><b>45.86</b></td><td align="right"><b>56.40</b></td><td align="right"><b>67.84</b></td><td align="right"><b>44.94</b></td><td align="right"><b>19.95</b></td><td align="right"><b>16.28</b></td><td align="right">35.25</td><td align="right">34.57</td></tr>
  </tbody>
</table>

LCB is LiveCodeBench.

</details>

Standard deviations and per-run records for Olmo3 are in [`results/`](results/).

## Released artifacts

| Artifact | Location | Contents |
|---|---|---|
| Selected subset | [`artifacts/`](artifacts/) | The 192,453 conversations DRIFT selected from about 2M candidates, as an ordered index that rebuilds the training file from the public corpus |
| Validation rollouts | [Hugging Face](https://huggingface.co/datasets/ZefanW/DRIFT-OLMo3-Validation-Rollouts) | 8,220 validation queries and 263,040 sampled responses with correctness labels |
| Per-run results | [`results/`](results/) | 57 continued-SFT runs behind the main table and the ablations |
| Evaluation harness | [`evaluation/`](evaluation/) | Protocol, pinned OLMES and LightEval patches, and a launcher for the eight benchmarks |
| Selection code | [`src/drift/`](src/drift/) | Rollout generation, verification, gradient features, scoring and subset export |
| Training recipe | [`scripts/continue_sft.py`](scripts/continue_sft.py) | The paper's continued-SFT configuration on ms-swift |

All artifacts are for Olmo3-7B-Instruct-SFT.

## Reproduce the Olmo3 experiment

This route retrains on the subset selected in the paper, so no rollouts or gradient features are recomputed. It needs Linux, Python 3.12 and NVIDIA GPUs. The paper trained on eight 80 GB GPUs.

```bash
git clone https://github.com/ZefanW/DRIFT.git
cd DRIFT
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
pip install -e '.[data]'
pip install 'flash-attn==2.8.3' --no-build-isolation   # must match your PyTorch and CUDA build

# 1. Rebuild the selected training data from the public Dolci corpus
python scripts/prepare_olmo3_subset.py --output data/olmo3_drift.jsonl

# 2. Continue SFT from the full-data checkpoint
python scripts/continue_sft.py \
  --model allenai/Olmo-3-7B-Instruct-SFT \
  --revision d9687e20d263cd0876dea143723b3c4974d19935 \
  --dataset data/olmo3_drift.jsonl --output runs/olmo3_drift \
  --gpus 0,1,2,3,4,5,6,7
```

Step 3 evaluates `runs/olmo3_drift/checkpoints/checkpoint-500` on the eight benchmarks with the [evaluation harness](evaluation/README.md), which runs in its own environments. The [reproduction guide](docs/reproduce_olmo3.md) covers the pinned model and corpus revisions, the training recipe and the evaluation protocol.

## Run DRIFT on your own data

The selection pipeline works for any checkpoint that has finished SFT, given its training corpus and a set of validation queries with verifiers. Copy [`configs/olmo3.json`](configs/olmo3.json), set `model` and `revision` for your checkpoint, and run:

```bash
pip install -e '.[models,data,math]'

python scripts/run_method.py --config configs/my_model.json \
  --corpus data/corpus.jsonl --queries data/validation.jsonl \
  --output runs/my_selection --gpus 0,1,2,3 --backend transformers
```

The selected conversations are written to `runs/my_selection/selection/subset.jsonl`, ready for `continue_sft.py`. Add `--dry-run` to print the commands of each stage first. The [selection guide](docs/selection.md) describes the input formats, external graders for code tasks, and the disk space the gradient features need.

Recomputing the Olmo3 selection with this code gives a different subset from the released one, because the paper's runs used a different random projection. Use the released subset to reproduce the paper's numbers.

`drift demo --output runs/demo` runs the scoring stage on synthetic features on CPU, and `python -m pytest` runs the test suite. Both need only `pip install -e '.[dev]'`.

## Citation

```bibtex
@misc{wang2026drift,
  title         = {{DRIFT}: Data Selection for {LLM} Instruction Tuning via On-Policy Attribution},
  author        = {Wang, Zefan and Li, Lincheng and Yu, Tianyu and Yao, Yuan},
  year          = {2026},
  eprint        = {2606.18307},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2606.18307}
}
```

This repository follows arXiv v2 (September 2026). The first version (June 2026) was titled *DRIFT: Refining Instruction Data via On-Policy Data Attribution*.

## Acknowledgements

Gradient projection follows [GraSS](https://arxiv.org/abs/2505.18976). Training uses [ms-swift](https://github.com/modelscope/ms-swift), and evaluation uses [OLMES](https://github.com/allenai/olmes) and [LightEval](https://github.com/huggingface/lighteval).

## License

Code is released under the [MIT License](LICENSE). The linked model and datasets keep their own licenses.
