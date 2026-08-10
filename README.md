# AutoMI

**Hands-Free Motor Imagery EEG Classification via LLM Multi-Agents**

<p align="center">
  <a href="README_zh.md">简体中文</a> | <a href="https://doi.org/10.1016/j.jneumeth.2026.110871">Journal of Neuroscience Methods</a>
</p>

<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="version" src="https://img.shields.io/badge/version-v1.0.0-green">
  <img alt="license" src="https://img.shields.io/badge/license-GPL--3.0-orange">
  <img alt="DOI" src="https://img.shields.io/badge/DOI-10.1016%2Fj.jneumeth.2026.110871-blue">
  <img alt="docs" src="https://img.shields.io/badge/docs-English%20%7C%20%E4%B8%AD%E6%96%87-success">
</p>

![AutoMI framework](assets/automi_overview.png)

AutoMI is a fully automated multi-agent system for the iterative optimization of Motor Imagery (MI) EEG classification models. It couples a Q-learning policy with deterministic rules (hybrid decision-making), jointly evolves hyperparameters and model structures, and runs a rule-driven feedback pipeline with deterministic rollback and context compression for uninterrupted, long-horizon iterations.

## Highlights

- **Closed-loop multi-agent optimization** — planning, execution and output agents orchestrated by a LangGraph workflow with a rule-driven feedback node.
- **Hybrid decision-making** — Q-learning coupled with deterministic rules selects among `continue_current`, `parameter_evolution` and `structure_update`, avoiding unguided LLM trial-and-error.
- **Parameter–structure co-evolution** — synthesizes configuration overrides and executable structure modifications beyond predefined search spaces, with literature retrieval injecting structural insights.
- **Resilient long-horizon iterations** — deterministic rollback on execution failures, experience tracking with context compression, exception capturing and autonomous code repair.

## Results

Classification accuracy (%) on three MI-EEG datasets (73 subjects in total), as reported in the paper:

| Model | IV2a | OpenBMI | ECUST-MI |
|---|---|---|---|
| ShallowNet | 58.76 | 63.59 | 63.58 |
| EEGNet | 52.93 | 68.43 | 65.50 |
| IFNet | 69.98 | 71.94 | 69.31 |
| FBMSNet | 68.83 | 69.43 | 59.74 |
| EEGConformer | 54.44 | 70.26 | 70.03 |
| ADFCNN | 66.59 | 74.33 | 70.17 |
| CTNet | 53.16 | 54.73 | 61.82 |
| FBNAS | 59.20 | 68.81 | 63.77 |
| **AutoMI** | **77.62** | **78.08** | **83.02** |

Strictly without human intervention, AutoMI secures minimum mean-accuracy gains of 5.71% (IV2a), 2.92% (OpenBMI) and 3.20% (ECUST-MI) over the initial baselines, with maximum per-model improvements of 24.69%, 23.35% and 23.28% respectively.

## Paper

The manuscript is available at [`paper/AutoMI.pdf`](paper/AutoMI.pdf).

Ruiyu Zhao, Shurui Li, Xinjie He, Xingyu Wang, Andrzej Cichocki, Yunhe Lu, Jing Jin. *Hands-Free Motor Imagery EEG Classification via LLM Multi-Agents*.

If you use AutoMI in your research, please cite:

```bibtex
@article{automi2026zhao,
  title   = {Hands-free motor imagery EEG classification via LLM multi-agents},
  journal = {Journal of Neuroscience Methods},
  pages   = {110871},
  year    = {2026},
  issn    = {0165-0270},
  doi     = {https://doi.org/10.1016/j.jneumeth.2026.110871},
  author  = {Ruiyu Zhao and Shurui Li and Xinjie He and Xingyu Wang and Andrzej Cichocki and Yunhe Lu and Jing Jin},
}
```

## Installation

### Requirements

- **Python 3.10 or later** (minimum supported version).
- A CUDA-capable GPU is recommended for training.
- An LLM API key (Qwen / Claude) — see [LLM credentials](#llm-credentials).

### Setup

1. **Create and activate a virtual environment** (Python >= 3.10):

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

2. **Install PyTorch** matching your CUDA toolkit, then the remaining dependencies:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```

   > On a CPU-only machine you can skip the `--index-url` line and simply run `pip install -r requirements.txt` (it will pull the default CPU wheels).

### LLM credentials

AutoMI calls an LLM API for its planning / execution / output agents. Configure **one** provider via environment variables.

**Qwen (DashScope, OpenAI-compatible) — default:**

```bash
export QWEN_API_KEY="your-api-key"
# Optional: export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

**Or Claude (Anthropic):**

```bash
export AUTOMI_LLM_PROVIDER=claude
export ANTHROPIC_AUTH_TOKEN="your-token"
```

### Datasets

Datasets are **not** included in this repository. Download them and place them under a single root, then point `AUTOMI_DATA_ROOT` at it:

```
$AUTOMI_DATA_ROOT/
├── bcicIV2a/gdf/    # BCI Competition IV Dataset 2a (.gdf)
└── OpenBMI/mat/     # OpenBMI dataset (.mat)
```

```bash
export AUTOMI_DATA_ROOT=/path/to/datasets
```

## Quick Start

```bash
python main.py --model EEGNet --llm deepseek-v3.2 --datasets bcicIV2a --gpu 0
```

### Main options

- `--model / -m` — initial MI-EEG model(s), e.g. `ShallowConvNet EEGNet IFNet FBMSNet EEGConformer ADFCNN CTNet`. Default: `EEGNet`.
- `--llm` — LLM(s) used by the agents. Default: `deepseek-v3.2`.
- `--datasets / -d` — `bcicIV2a` and/or `OpenBMI`. Default: `bcicIV2a`.
- `--iterations / -i` — maximum iterations per subject. Default: `26`.
- `--max-workers / -w` — parallel subject processes. Default: `5`.
- `--gpu / -g` — GPU ids, comma separated (e.g. `0` or `0,1,2`). Default: `0`.
- `--max-param-failures` — consecutive non-improving `parameter_evolution` rounds before forcing `structure_update`. Default: `5`.
- `--conda-env` — conda environment name used by the runner. Default: `automi`.
- `--test` — smoke test with one subject per dataset.
- `--ablation` — `no-structure-update | random-action | no-experience | no-literature`.

Outputs are written to `output/` (or `output_ablation/<mode>/` when `--ablation` is set).

## Repository Structure

```
├── main.py                 # entry point
├── src/main/               # core system
│   ├── agents/             # planning / execution / output agents
│   ├── workflow/           # LangGraph workflow & feedback node
│   ├── rl/                 # Q-learning hybrid decision-making
│   ├── tools/              # literature retrieval, experience tracker, training tool
│   ├── prompts/            # agent prompts
│   ├── models/             # MI-EEG model zoo
│   ├── datasets/           # dataset loaders
│   ├── train/              # training pipeline
│   ├── configs/            # training & model configs
│   └── utils/              # LLM clients & shared config
├── paper/AutoMI.pdf        # manuscript
└── assets/                 # figures
```

## License

This project is licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE).
