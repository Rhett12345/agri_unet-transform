---
name: cloudmask-unet-runtime
description: Use when working in the ~/cloudmask/unet AGRI/MYD06 cloudmask U-Net project, including running, debugging, testing, training, evaluating, inference, GPU or CPU smoke tests, Conda environment cloudunet setup, DataLoader multiprocessing issues, checkpoints, fusion diagnostics, and UNet/cloudmask pipeline fixes.
---

# Cloudmask U-Net Runtime

## Scope

Use this skill for Codex tasks in `~/cloudmask/unet` that involve the AGRI/MYD06 cloud-property retrieval pipeline, U-Net training/evaluation/inference, fusion diagnostics, tests, checkpoints, runtime errors, Conda environment handling, GPU/CPU smoke tests, or DataLoader behavior.

Project facts:

- Project root: `~/cloudmask/unet`
- Default Conda environment: `cloudunet`
- The server has two GPUs.
- Most dependencies are already installed. Prefer the existing environment over reinstalling or upgrading packages.
- Current project entry points include `main.py`, `train.py`, `test.py`, `inference.py`, `tools/eval_checkpoints.py`, `tools/analyze_qc_failures.py`, and diagnostic scripts under `scripts/`.
- Configuration and many hyperparameters live in `config.py`; `UNET_WORKDIR` can override the default output root.

## Working Directory

- Before running any project command, first change to the project root:

```bash
cd ~/cloudmask/unet
```

- Do not assume the active terminal is already at the repository root.
- When reporting commands, include the project-root assumption or include `cd ~/cloudmask/unet && ...`.

## Environment

- Default to the existing Conda environment `cloudunet`.
- Preferred activation forms:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate cloudunet
```

or:

```bash
conda run -n cloudunet <command>
```

- Prefer using installed dependencies from `cloudunet`.
- Do not perform high-risk environment operations without explicit user confirmation, including:
  - `conda env remove`
  - overwriting environment specs from `pip freeze`
  - bulk upgrades or reinstalls of `torch`, CUDA packages, `numpy`, `scipy`, or the scientific stack
  - replacing the Python environment to solve a narrow code issue
- `requirements.txt` exists, but treat it as reference unless the user asks to install dependencies.

## GPU Rules

- The server has two GPUs.
- Before GPU work, use quick checks when appropriate:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

- If the current Codex sandbox or terminal cannot access GPUs, do not conclude that the project lacks GPU support or that CUDA is broken. Run CPU/import/smoke tests where possible and provide host/sandbox-external GPU commands for the user.
- GPU commands should explicitly set `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 python ...
CUDA_VISIBLE_DEVICES=0,1 python ...
```

- Run the smallest GPU smoke test before long training or full evaluation.
- This code generally chooses `cuda` when `torch.cuda.is_available()` is true; `train.py`, `test.py`, and `inference.py` do not currently expose a general `--device` CLI flag. Some diagnostic scripts under `scripts/` do expose `--device`.

## Testing Strategy

- Start with cheap checks:
  - import test
  - `--help` test
  - shape/model construction test
  - tiny batch or single-checkpoint smoke test
- Avoid full training unless the user explicitly asks for it or it is clearly required.
- For scripts, first inspect CLI options and config. Prefer existing controls such as:
  - `--workers` in `main.py` fusion
  - `--checkpoint` in `test.py` and `main.py`
  - `--batch_size` in `inference.py`
  - `--num-workers` in `tools/eval_checkpoints.py`
  - env overrides such as `UNET_WORKDIR`, `UNET_TRAIN_DATES`, `UNET_VAL_DATES`, `UNET_TEST_DATES`, and loss/checkpoint monitor env vars in `config.py`
- For training, `train.py` mainly uses `config.py` values such as `NUM_EPOCHS`, `BATCH_SIZE`, and `NUM_WORKERS`; do not invent CLI flags that are not present.
- If a Python multiprocessing/socket permission error appears, such as:

```text
PermissionError: [Errno 1] Operation not permitted
```

  first try a smoke test with DataLoader or multiprocessing workers set to `0` or `1` through existing knobs. Examples:
  - for checkpoint evaluation: `python tools/eval_checkpoints.py --num-workers 0 ...`
  - for fusion: `python main.py --stages fuse ... --workers 1`
  - for code-level tests: monkeypatch or temporarily set `cfg.NUM_WORKERS = 0` in a narrow test path

- Do not start by restructuring multiprocessing, datasets, or the pipeline for sandbox permission failures.
- Record the exact commands run, important outputs, and full error messages in the final response or working notes.

## Code Modification Rules

- Before editing, read the relevant files and nearby README/config documentation.
- Keep changes small and scoped to the requested behavior.
- Do not change model architecture, training hyperparameters, checkpoint selection logic, fusion thresholds, or QC thresholds unless the task explicitly asks for that.
- Do not hardcode absolute data paths unless the project already has an established convention for that path.
- Prefer env/config overrides for experiments where the project already supports them.
- Do not delete user data, fused HDF5 files, training outputs, checkpoints, logs, summaries, or diagnostic CSV/JSON files.
- Put new diagnostics or utility scripts under `tools/` or `scripts/` when practical.
- For diagnostics outputs, prefer CSV and/or JSON so runs are comparable.
- For any change that may affect training results, evaluation metrics, checkpoint compatibility, sample filtering, or fusion outputs, explicitly state the impact scope.

## Common Command Templates

Run from the project root unless the command includes `cd`.

Activate environment:

```bash
cd ~/cloudmask/unet
source ~/anaconda3/etc/profile.d/conda.sh && conda activate cloudunet
```

Check Python, Torch, and CUDA:

```bash
cd ~/cloudmask/unet
conda run -n cloudunet python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Check GPUs on the host:

```bash
nvidia-smi
```

Find likely entry points:

```bash
cd ~/cloudmask/unet
rg -n "argparse|if __name__ == .__main__.|def main\\(" . --glob "*.py"
```

Help/import smoke tests:

```bash
cd ~/cloudmask/unet
conda run -n cloudunet python main.py --help
conda run -n cloudunet python test.py --help
conda run -n cloudunet python inference.py --help
conda run -n cloudunet python -c "import config, dataset, model, train, test; print('imports ok')"
```

Model shape smoke test:

```bash
cd ~/cloudmask/unet
conda run -n cloudunet python - <<'PY'
import torch
import config as cfg
from model import build_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device).eval()
x = torch.randn(1, cfg.AGRI_CHANNELS, *cfg.PATCH_SIZE, device=device)
geo = torch.randn(1, 2, *cfg.PATCH_SIZE, device=device)
with torch.no_grad():
    clp, comp = model(x, geo=geo)
print(device, tuple(clp.shape), tuple(comp.shape))
PY
```

Run pytest smoke tests:

```bash
cd ~/cloudmask/unet
conda run -n cloudunet python -m pytest -q tests/test_test_dataloader.py tests/test_eval_checkpoints.py
```

Evaluate test split with an explicit checkpoint:

```bash
cd ~/cloudmask/unet
conda run -n cloudunet python test.py --checkpoint /path/to/checkpoint.pth
```

Evaluate multiple checkpoints with single-process DataLoader smoke mode:

```bash
cd ~/cloudmask/unet
conda run -n cloudunet python tools/eval_checkpoints.py --num-workers 0 --checkpoints HIR_COMP_UNet_AGRIonly_best_oa.pth
```

Single-GPU smoke command template:

```bash
cd ~/cloudmask/unet
CUDA_VISIBLE_DEVICES=0 conda run -n cloudunet python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Single-GPU checkpoint evaluation template:

```bash
cd ~/cloudmask/unet
CUDA_VISIBLE_DEVICES=0 conda run -n cloudunet python tools/eval_checkpoints.py --num-workers 0 --checkpoints HIR_COMP_UNet_AGRIonly_best_oa.pth
```

Dual-GPU visibility check template:

```bash
cd ~/cloudmask/unet
CUDA_VISIBLE_DEVICES=0,1 conda run -n cloudunet python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Fusion single-worker diagnostic template:

```bash
cd ~/cloudmask/unet
conda run -n cloudunet python main.py --stages fuse --split train --day YYYYMMDD --workers 1 --enable-qc-diagnostics
```

Analyze QC diagnostics:

```bash
cd ~/cloudmask/unet
conda run -n cloudunet python tools/analyze_qc_failures.py --input runs/qc_diagnostics/qc_gate_stats.csv
```

Inference template:

```bash
cd ~/cloudmask/unet
CUDA_VISIBLE_DEVICES=0 conda run -n cloudunet python inference.py --agri_file /path/to/FY4_AGRI_FILE.HDF --checkpoint /path/to/checkpoint.pth --batch_size 8
```

Full training template, only after smaller checks pass:

```bash
cd ~/cloudmask/unet
CUDA_VISIBLE_DEVICES=0 conda run -n cloudunet python main.py --stages train
```

## Reporting

When finishing a task, report:

- files changed
- commands actually run
- whether the command ran on CPU, GPU, or could not access GPU from the current sandbox
- key result or error
- any command the user should run on the host for GPU or long-running validation
