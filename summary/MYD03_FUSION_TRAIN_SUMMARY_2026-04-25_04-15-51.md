# MYD03 Fusion Fix + 4/2/2 Training Summary

Date: 2026-04-25 04:15:51  
Workspace: `/home/yuq/cloudmask/unet`  
Environment: `cloudunet`

## User Goal

Current phase focused on fixing the data-fusion bottleneck before further model tuning.

Requested experiment split:

- Train: 4 days
- Validation: 2 days
- Test: 2 days
- Run one complete training and test cycle before asking for further help

## Date Split Used

Updated in `config.py`:

```python
TRAIN_DATES = ["20190115", "20190415", "20190715", "20191015"]
VAL_DATES   = ["20190215", "20190815"]
TEST_DATES  = ["20190625", "20191225"]
```

## Core Fusion Fixes Implemented

Primary files changed:

- `fusion_io.py`
- `fusion_core.py`
- `fusion_config.py`
- `config.py`
- `tests/test_fusion_time.py`

Main changes:

1. MYD03 is now required by default for strict fusion.
2. MYD03 1 km geolocation is used as the primary geolocation source.
3. MYD03 `EV start time` is used as the primary scan-time source.
4. Missing MYD03 geolocation or scan time now rejects the MODIS granule by default instead of silently falling back to filename time.
5. MYD06 `Scan_Start_Time` remains a secondary scan-time source where appropriate.
6. File-level time fallback is disabled under strict defaults.
7. Fusion outputs now record:
   - `MATCH_DT_MEAN`
   - `MATCH_DT_MAX`
   - `MATCH_DIST_MEAN_KM`
   - `MATCH_DIST_P95_KM`
   - `scan_time_sources`
   - `geo_sources`
   - `fallback_granules`
8. Regression supervision is stricter than CLP supervision:
   - `REG_TIME_MAX_MIN = 2.0`
   - `REG_OVERLAP_FRAC_MIN = 0.5`
   - `REG_CLOUD_FRAC_MIN = 0.6`
   - `REG_PHASE_CONSISTENCY_MIN = 0.8`
9. `Samples/max_time_diff_min` now comes from `MATCH_DT_MAX`, not `MATCH_DT_MIN`.
10. Added `Samples/mean_time_diff_min` and `Samples/p95_match_dist_km`.

## MYD03 Verification

Real-file verification confirmed:

```text
myd03 (2030, 1354) EV start time (2030, 1354) (0.1986, 5.1716)
myd06 meta ('MYD03_1KM', 'EV start time', False)
```

This confirms MYD06 fusion metadata is using:

- geolocation source: `MYD03_1KM`
- scan-time source: `EV start time`
- fallback: `False`

## Fusion Commands Run

All fusion was regenerated with strict MYD03 settings:

```bash
conda run -n cloudunet python main.py --stages fuse --split train --overwrite --max_qc 1 --workers 8
conda run -n cloudunet python main.py --stages fuse --split val --overwrite --max_qc 1 --workers 8
conda run -n cloudunet python main.py --stages fuse --split test --overwrite --max_qc 1 --workers 8
```

Fusion logs showed `scantime=EV start time` throughout.

## Fused Dataset Summary

All current fused H5 outputs report:

- `scan_sources = EV start time`
- `geo_sources = MYD03_1KM`
- `fallback = 0`

### Train

- H5 files: 22
- patches: 19,255
- CLP clear/water/ice counts: `[6354932, 3641415, 4761260]`
- CLP fractions: `43.06% / 24.67% / 32.26%`
- valid CER pixels: 1,801,949
- valid COT pixels: 1,801,563
- valid CTH pixels: 2,225,608
- mean patch `dt_mean`: 3.2308 min
- mean patch `dt_max`: 3.4534 min
- mean overlap: 0.8521
- mean phase consistency: 0.9576
- mean cloud fraction: 0.5841
- mean p95 match distance: 2.9945 km

### Validation

- H5 files: 11
- patches: 2,995
- CLP clear/water/ice counts: `[899260, 560713, 765972]`
- CLP fractions: `40.40% / 25.19% / 34.41%`
- valid CER pixels: 396,380
- valid COT pixels: 396,429
- valid CTH pixels: 450,383
- mean patch `dt_mean`: 3.2849 min
- mean patch `dt_max`: 3.4944 min
- mean overlap: 0.8387
- mean phase consistency: 0.9574
- mean cloud fraction: 0.6087
- mean p95 match distance: 2.9944 km

### Test

- H5 files: 8
- patches: 1,723
- CLP clear/water/ice counts: `[745803, 341154, 211123]`
- CLP fractions: `57.45% / 26.28% / 16.26%`
- valid CER pixels: 106,108
- valid COT pixels: 106,060
- valid CTH pixels: 148,818
- mean patch `dt_mean`: 3.2300 min
- mean patch `dt_max`: 3.4520 min
- mean overlap: 0.8433
- mean phase consistency: 0.9624
- mean cloud fraction: 0.4678
- mean p95 match distance: 2.9944 km

## Stats

Command:

```bash
conda run -n cloudunet python main.py --stages stats
```

Result:

- stats file: `unet_workdir/stats/norm_stats.npz`
- computed from 1,800,260 valid pixels
- 3 train files were skipped by stats because they had no fully finite strict regression pixels

Important note:

`dataset.py::compute_and_save_stats()` still computes stats from pixels where all output channels are finite. This worked for this run, but for future stricter filtering it may be better to compute AGRI input stats from valid BT + valid CLP, and output stats per regression channel independently.

## GPU Note

Inside the default sandbox, CUDA was not visible:

```text
torch.cuda.is_available() = False
nvidia-smi failed inside sandbox
```

Outside the sandbox, the server correctly exposed two GPUs:

```text
NVIDIA GeForce RTX 4090, 24564 MiB
NVIDIA GeForce RTX 4090, 24564 MiB
```

PyTorch in `cloudunet` also saw CUDA outside the sandbox:

```text
cuda_available True
device_count 2
['NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090']
```

Training and test were therefore run outside the sandbox with escalated command execution so that CUDA was available.

## Training

Command:

```bash
conda run -n cloudunet python main.py --stages train
```

Training used CUDA:

```text
Training on cuda
```

Dataset size:

- train patches: 19,255
- val patches: 2,995
- train iters/epoch: 602
- val iters/epoch: 94

CLP train distribution:

```text
c0=43.1% | c1=24.7% | c2=32.3%
```

CLP loss weights:

```text
c0=0.88 | c1=1.16 | c2=1.02
```

Training stopped by early stopping at epoch 11.

Best checkpoint by current logic was epoch 1:

- best val loss: 1.791229
- best val OA: 37.47%
- checkpoint: `unet_workdir/model/HIR_COMP_UNet_AGRIonly_best.pth`

Final/last checkpoint:

- `unet_workdir/model/HIR_COMP_UNet_AGRIonly_last.pth`

Important checkpoint observation:

- Current code still saves best model by `val_loss`.
- Epoch 9 had higher `val_oa = 38.20%`, but was not selected because `val_loss` was worse.
- This confirms the earlier concern that checkpoint selection should add CLP-focused criteria such as macro accuracy or per-class balanced accuracy.

Best-val-loss row from `train_log.csv`:

```text
epoch=1
train_loss=2.1613
val_loss=1.7912
val_oa=37.47%
val_cer_rmse=5.45
val_cot_rmse=7.76
val_cth_rmse=2491.3 m
val_cls0_acc=64.24%
val_cls1_acc=8.64%
val_cls2_acc=22.16%
```

Highest validation OA row:

```text
epoch=9
val_loss=1.9952
val_oa=38.20%
val_cer_rmse=5.66
val_cot_rmse=7.74
val_cth_rmse=2606.0 m
val_cls0_acc=49.39%
val_cls1_acc=15.88%
val_cls2_acc=29.34%
```

## Test Evaluation

Command:

```bash
conda run -n cloudunet python main.py --stages test
```

Evaluation used CUDA:

```text
Evaluating on cuda
Loaded checkpoint /home/yuq/cloudmask/unet/unet_workdir/model/HIR_COMP_UNet_AGRIonly_best.pth
```

Outputs:

- `unet_workdir/eval/metrics_summary.csv`
- `unet_workdir/eval/confusion_matrix.png`
- `unet_workdir/eval/scatter_CER.png`
- `unet_workdir/eval/scatter_COT.png`
- `unet_workdir/eval/scatter_CTH.png`

Metrics from `metrics_summary.csv`:

```text
CLP_OA          = 30.9999 %
CLP_Clear_acc   = 54.5828 %
CLP_Water_acc   = 10.4094 %
CLP_Ice_acc     = 49.4290 %

CER_rmse = 10.1980 um
CER_mae  = 8.0272 um
CER_bias = 0.6387 um
CER_r    = 0.0635
CER_n    = 106108

COT_rmse = 20.8623
COT_mae  = 11.6823
COT_bias = -1.6484
COT_r    = -0.0618
COT_n    = 106060

CTH_rmse = 4412.4458 m
CTH_mae  = 3930.5332 m
CTH_bias = 1708.8954 m
CTH_r    = 0.3411
CTH_n    = 148818
```

## Verification

Commands completed:

```bash
conda run -n cloudunet pytest -q
```

Result:

```text
8 passed in 0.30s
```

Compile check:

```bash
conda run -n cloudunet python -m py_compile fusion_io.py fusion_core.py fusion_config.py data_fusion.py dataset.py train.py test.py
```

Result: passed with exit code 0.

## Current Interpretation

The major MYD03 failure mode has been fixed:

- no file-level scan-time fallback in the regenerated fused dataset
- all current fused outputs use MYD03 geolocation
- all current fused outputs use `EV start time`
- metadata confirms `fallback = 0`

However, model performance is still weak after strict fusion:

- CLP test OA is only about 31%
- Water accuracy is especially poor at about 10%
- CER and COT correlations are near zero
- CTH correlation is positive but still weak, with RMSE about 4.4 km

This suggests that simply fixing scan-time fallback is necessary but not sufficient. Remaining issues may include:

1. Residual spatial mismatch or patch-level misregistration.
2. Test split distribution mismatch, especially test being much more clear-dominant and less ice-heavy than val/train.
3. Current checkpoint selection by total validation loss not matching CLP or balanced-class goals.
4. Regression labels being sparse after strict gating, especially in test.
5. Model underfitting/overfitting pattern: train loss keeps decreasing while val loss worsens after epoch 1.

## Recommended Next Stage

Do not start by changing the model architecture. Start with diagnostics that tell whether the remaining error is data alignment or model learning.

Priority 1: Patch-level spatial visualization

- Save several test patches with:
  - AGRI BT channel images
  - CLP truth
  - CLP prediction
  - CTH truth
  - CTH prediction
  - optional CER/COT truth/pred
- Inspect whether truth/pred fields are globally shifted.
- If fields are shifted, continue debugging registration.
- If boundaries are merely blurred, then model/loss/architecture work is justified.

Priority 2: Fix checkpoint logic

- Save at least:
  - best by `val_loss`
  - best by `val_clp_macro_acc`
  - best by `val_oa`
- Report balanced CLP accuracy, not only OA.
- Current run proves that epoch 9 has better OA than epoch 1 but is not selected by the current checkpoint rule.

Priority 3: Improve validation/test comparability

- Current CLP fractions:
  - train: `43.06 / 24.67 / 32.26`
  - val: `40.40 / 25.19 / 34.41`
  - test: `57.45 / 26.28 / 16.26`
- Test is much more clear-heavy and much less ice-heavy.
- Next split should be seasonally stratified and distribution-checked before training.

Priority 4: Regression-specific quality analysis

- Bucket samples by:
  - `max_time_diff_min`
  - `mean_overlap_frac`
  - `mean_phase_consist`
  - `mean_cloud_frac`
  - `p95_match_dist_km`
- Report CTH/COT/CER RMSE per bucket.
- Confirm whether regression RMSE is dominated by low-quality or sparse-label patches.

Priority 5: Stats logic cleanup

- Current stats worked, but it is strict because it uses pixels where all 4 output channels are finite.
- For long-term robustness:
  - compute AGRI stats from valid input pixels
  - compute regression output stats per channel
  - do not require CER/COT/CTH all finite for input normalization

## Key Files For Next Session

Code:

- `config.py`
- `fusion_config.py`
- `fusion_io.py`
- `fusion_core.py`
- `data_fusion.py`
- `dataset.py`
- `train.py`
- `test.py`
- `tests/test_fusion_time.py`

Outputs:

- `unet_workdir/stats/norm_stats.npz`
- `unet_workdir/logs/train_log.csv`
- `unet_workdir/logs/pipeline.log`
- `unet_workdir/model/HIR_COMP_UNet_AGRIonly_best.pth`
- `unet_workdir/model/HIR_COMP_UNet_AGRIonly_last.pth`
- `unet_workdir/eval/metrics_summary.csv`
- `unet_workdir/eval/confusion_matrix.png`
- `unet_workdir/eval/scatter_CER.png`
- `unet_workdir/eval/scatter_COT.png`
- `unet_workdir/eval/scatter_CTH.png`

## Command Notes For Next Session

Use the `cloudunet` environment:

```bash
conda run -n cloudunet python ...
```

If running inside Codex sandbox, CUDA may not be visible. GPU commands may need escalated execution. Confirm with:

```bash
conda run -n cloudunet python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Expected outside sandbox:

```text
True
2
```

