# Small Training Run After Phase-Layered Fusion

Date: 2026-04-26  
Workspace: `/home/yuq/cloudmask/unet`

## Purpose

This run tested whether the modified MODIS-to-AGRI fusion logic improves a small end-to-end training workflow.

The fusion changes under test were:

- Phase-layered regression aggregation:
  - CLP is still decided by weighted majority vote.
  - CER/COT/CTH are aggregated only from cloudy candidates whose CLP equals the dominant phase.
- Spatial candidate cap:
  - `query_ball_point()` remains the coarse spatial prefilter.
  - Each AGRI pixel keeps only the nearest `EXPECTED_1KM_PER_AGRI` 1 km candidates, currently 16.
- CTH is treated as `CTH_1km` in the same collection path.
  - Historical `VALID_PIX_5KM` is still emitted for compatibility.
  - `VALID_PIX_CTH_1KM` is also emitted.

## Random Date Split

Common available days from FY4A, MYD06, and MYD03: 36.

Random seed used for selection:

```text
20260426
```

Selected dates:

```text
train = 20190725, 20190405, 20190305, 20190805
val   = 20190815, 20190125
test  = 20190925, 20191025
```

`config.py` was updated so `TRAIN_DATES`, `VAL_DATES`, and `TEST_DATES` match this split.

## Fusion Output

Fusion command pattern:

```bash
conda run -n cloudunet --no-capture-output python data_fusion.py \
  --split <split> --day <YYYYMMDD> --overwrite --workers 8 --max_qc 1
```

Actual paired output:

| split | H5 files | samples | by day |
|---|---:|---:|---|
| train | 22 | 18,316 | 20190305: 3,754; 20190405: 6,767; 20190725: 4,198; 20190805: 3,597 |
| val | 11 | 2,835 | 20190125: 1,001; 20190815: 1,834 |
| test | 12 | 3,719 | 20190925: 1,886; 20191025: 1,833 |

CLP finite-pixel class distribution:

| split | clear | water | ice |
|---|---:|---:|---:|
| train | 39.26% | 27.78% | 32.95% |
| val | 30.75% | 23.81% | 45.44% |
| test | 45.23% | 28.05% | 26.73% |

Regression finite pixels:

| split | CER | COT | CTH |
|---|---:|---:|---:|
| train | 2,088,251 | 2,090,458 | 2,543,669 |
| val | 420,231 | 420,281 | 478,482 |
| test | 306,646 | 306,801 | 391,309 |

Important observation:

- The random split is not class-balanced by date.
- Val is ice-heavy.
- Test is clear-heavy.
- This split is useful as a small smoke test, but not a clean old-vs-new fusion A/B.

## Stats

Stats command:

```bash
conda run -n cloudunet --no-capture-output python main.py --stages stats
```

Stats output:

```text
unet_workdir/stats/norm_stats.npz
```

Stats successfully used all 22 train H5 files after the config date update.

Final stats log summary:

```text
BT valid px = 18,111,488
output valid px = [13,679,768, 2,088,251, 2,090,458, 2,543,669]
```

Several small H5 files returned `None` in the stats worker because they did not contain enough valid pixels for independent channel statistics. The overall stats computation still succeeded.

## GPU

Default sandbox did not expose NVIDIA devices. `nvidia-smi` required escalated execution.

Visible GPUs with escalation:

```text
GPU0: NVIDIA GeForce RTX 4090, occupied by another python3 process
GPU1: NVIDIA GeForce RTX 4090, mostly free
```

Training and evaluation were run with:

```bash
CUDA_VISIBLE_DEVICES=1
```

PyTorch confirmation under escalated execution:

```text
cuda_available True
device0 NVIDIA GeForce RTX 4090
```

Training log confirmed:

```text
Training on cuda
```

## Training

Training command:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n cloudunet --no-capture-output \
  python main.py --stages train
```

Dataset sizes:

```text
train patches = 18,316
val patches   = 2,835
test patches  = 3,719
```

Training stopped early at epoch 11:

```text
Early stopping at epoch 11
Best val loss: 1.772935
Best val OA: 33.54%
```

Training speed on GPU1:

```text
~23.5 s / epoch after epoch 1
```

Key validation checkpoints:

| checkpoint / epoch | val loss | val OA | clear acc | water acc | ice acc | CER RMSE | COT RMSE | CTH RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best val loss, epoch 1 | 1.7729 | 33.54% | 32.33% | 38.95% | 30.31% | 5.62 | 7.49 | 2540.1 |
| best val OA, epoch 5 | 1.8261 | 40.41% | 37.15% | 17.19% | 43.73% | 5.76 | 7.27 | 2626.1 |
| last, epoch 11 | 2.1969 | 34.80% | 31.23% | 29.17% | 31.98% | 6.19 | 7.32 | 2983.5 |

Important training observation:

- The checkpoint saved by current code is based on lowest validation loss, not best CLP OA or balanced CLP accuracy.
- The best validation OA occurred at epoch 5, but that checkpoint was not saved.

Training log:

```text
unet_workdir/logs/train_log.csv
```

## Test Evaluation

Two checkpoints were evaluated:

1. Best validation loss checkpoint:

```text
unet_workdir/model/HIR_COMP_UNet_AGRIonly_best.pth
```

Saved metrics copy:

```text
unet_workdir/eval/metrics_summary_best_loss.csv
```

Results:

| metric | value |
|---|---:|
| CLP OA | 26.43% |
| Clear acc | 35.83% |
| Water acc | 53.37% |
| Ice acc | 20.53% |
| CER RMSE | 11.262 um |
| CER MAE | 8.894 um |
| CER Bias | -0.340 um |
| CER R | 0.2858 |
| COT RMSE | 16.643 |
| COT MAE | 8.738 |
| COT Bias | -3.385 |
| COT R | 0.0422 |
| CTH RMSE | 5167.3 m |
| CTH MAE | 4131.0 m |
| CTH Bias | 127.4 m |
| CTH R | 0.1529 |

2. Last checkpoint:

```text
unet_workdir/model/HIR_COMP_UNet_AGRIonly_last.pth
```

Saved metrics copy:

```text
unet_workdir/eval/metrics_summary_last.csv
```

Results:

| metric | value |
|---|---:|
| CLP OA | 25.95% |
| Clear acc | 33.37% |
| Water acc | 40.94% |
| Ice acc | 35.22% |
| CER RMSE | 13.798 um |
| CER MAE | 10.954 um |
| CER Bias | -0.337 um |
| CER R | -0.0861 |
| COT RMSE | 16.483 |
| COT MAE | 8.006 |
| COT Bias | -3.424 |
| COT R | -0.0100 |
| CTH RMSE | 6121.7 m |
| CTH MAE | 4882.9 m |
| CTH Bias | 680.8 m |
| CTH R | 0.0118 |

## Interpretation

This small run does not show improved CLP generalization.

Observed:

- Best-loss checkpoint test OA was only 26.43%.
- Last checkpoint test OA was 25.95%.
- These are below the previous reported old-run test OA around 31%, but the dates are different, so this is not a strict A/B comparison.
- CER correlation improved relative to previous near-zero behavior in the best-loss checkpoint (`R = 0.2858`).
- COT and CTH correlations remained weak.
- The validation and test date distributions are substantially different, which likely hurts generalization.

Likely causes:

1. The fusion change improves label physical consistency but does not by itself solve model generalization.
2. The random 4/2/2 split has strong date-level distribution shift:
   - Val: ice-heavy.
   - Test: clear-heavy.
3. Checkpoint selection is misaligned with the main target:
   - Current save criterion is validation total loss.
   - Best CLP OA was not saved.
4. The model still struggles with phase discrimination under small-data date splits.

## Recommended Next Steps

1. Add checkpoint saving for:

```text
best_val_oa
best_val_balanced_clp_acc
```

2. Report balanced CLP accuracy in train/val/test, not only OA.

3. Run a proper A/B experiment:

- Same dates.
- Same model seed.
- Same training settings.
- Old fusion vs new phase-layered fusion.

4. Avoid fully random date splits for serious conclusions.

Prefer a balanced split by season/cloud regime, or at minimum inspect class distribution before training.

5. Continue improving fusion diagnostics:

- Compare p95 match distance before/after.
- Compare CLP/CER/COT/CTH maps on the same patches.
- Track how many regression pixels are lost due to phase-layered filtering.

## Files Produced

Main outputs:

```text
unet_workdir/paired/train/
unet_workdir/paired/val/
unet_workdir/paired/test/
unet_workdir/stats/norm_stats.npz
unet_workdir/model/HIR_COMP_UNet_AGRIonly_best.pth
unet_workdir/model/HIR_COMP_UNet_AGRIonly_last.pth
unet_workdir/logs/train_log.csv
unet_workdir/eval/metrics_summary_best_loss.csv
unet_workdir/eval/metrics_summary_last.csv
```

Note:

- `unet_workdir/eval/metrics_summary.csv` was overwritten by the last checkpoint evaluation.
- Use `metrics_summary_best_loss.csv` and `metrics_summary_last.csv` for this experiment summary.
