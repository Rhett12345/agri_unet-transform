# AGRI U-Net Pipeline Session Summary - 2026-04-27

## Scope

This session focused on AGRI-only U-Net evaluation quality, test dataloader behavior, checkpoint comparison, and fusion QC diagnostics. The active project root was:

`/home/yuq/cloudmask/unet`

Standing guardrails followed:

- Do not change model structure, training hyperparameters, or fusion thresholds unless explicitly requested.
- Keep changes scoped to the requested step.
- Put new diagnostics/scripts under `tools/` or existing pipeline modules.
- Prefer CSV/JSON outputs for diagnostics.

## 1. Masked CLP Test Metrics

Problem:

- `test.py` was evaluating CLP labels after `labels[:,0].long()` without masking invalid CLP pixels.
- NaN / invalid CLP labels were counted as wrong test pixels.
- Test OA was therefore not comparable with train/val, which already use masks.

Implemented:

- CLP metrics now use a valid mask:
  - `torch.isfinite(labels[:,0])`
  - `label >= 0`
  - `label < CLP_CLASSES`
- OA, per-class recall/accuracy, macro metrics, and confusion matrix are computed only on valid CLP pixels.
- Test output now reports:
  - `valid_clp_pixels`
  - `total_pixels`
  - `valid_ratio`
- Added dynamic sanity coverage confirming NaN/invalid CLP does not enter the confusion matrix.
- Regression metrics were not changed for this step.

Relevant files:

- `test.py`

## 2. Test-Only Dataloader

Problem:

- `test.py` used `build_dataloaders()`, which constructs train/val/test datasets.
- Testing only needs the test split.
- Rebuilding train/val during test wastes time and can introduce unnecessary side effects.

Implemented:

- Added `build_test_dataloader(stats)` in `dataset.py`.
- `test.py` now uses only the test dataloader.
- `build_dataloaders()` behavior for training remains unchanged.
- Logs now make it clear only the test split is built during test.

Relevant files:

- `dataset.py`
- `test.py`
- `tests/test_test_dataloader.py`

## 3. `UNET_WORKDIR` Override

Problem:

- `tests/test_experiment_controls.py::test_config_allows_experiment_env_overrides` failed because `UNET_WORKDIR` did not override `ROOT`.

Implemented:

- Added an env-path helper in `config.py`.
- `ROOT` now defaults to `/data/Data_yuq/unet_workdir`, but can be overridden by `UNET_WORKDIR`.

Relevant files:

- `config.py`

## 4. Multi-Checkpoint Evaluation Script

Problem:

- Needed to evaluate several checkpoints on the same test split without retraining.

Implemented:

- Added `tools/eval_checkpoints.py`.
- Refactored `test.py` to expose reusable masked-evaluation collection logic.
- The script evaluates multiple checkpoints on one shared test split.
- Missing checkpoint paths are skipped with warnings.
- Outputs:
  - `runs/eval_checkpoints/summary.csv`
  - `runs/eval_checkpoints/summary.json`

Metrics emitted per checkpoint:

- Checkpoint name
- CLP valid pixels / total pixels / valid ratio
- Masked CLP OA
- Per-class precision / recall / F1
- Macro F1
- CER/COT/CTH valid pixels, RMSE, MAE, R

Regression metric detail:

- CER/COT/CTH are masked independently by channel finite masks.
- They do not require all regression channels to be valid simultaneously.

Observed checkpoint comparison:

- `HIR_COMP_UNet_AGRIonly_best_loss.pth`: OA about 43.99%, macro F1 about 41.59%.
- `HIR_COMP_UNet_AGRIonly_best_oa.pth`: OA about 48.15%, macro F1 about 45.99%.
- `HIR_COMP_UNet_AGRIonly_best_macro.pth`: OA about 47.81%, macro F1 about 45.50%.

Current recommendation for CLP diagnostics:

- `HIR_COMP_UNet_AGRIonly_best_oa.pth`

Relevant files:

- `tools/eval_checkpoints.py`
- `test.py`
- `tests/test_eval_checkpoints.py`

## 5. QC Gate Diagnostics In Fusion

Problem:

- Fusion success is low.
- Previous 36-day run from `out.log` had:
  - 957 AGRI scenes total
  - 202 successfully written
  - success rate about 21.1%
  - 743 post-qc all-zero scenes
  - all-zero ratio about 77.6%
- Existing logs only showed final post-qc counts, not which gate caused pixel loss.

Implemented:

- Added optional diagnostics to `fusion_io.apply_quality_filter()`.
- Default remains off.
- No QC thresholds were intentionally changed for this diagnostics step.
- No model/training/test/dataset logic was modified for this diagnostics step.
- Diagnostics are carried through the fusion call chain and written by `data_fusion.py`.

Outputs when enabled:

- `runs/qc_diagnostics/qc_gate_stats.csv`
- `runs/qc_diagnostics/qc_gate_stats.jsonl`

Enable controls:

- CLI: `--enable-qc-diagnostics`
- CLI output dir: `--qc-diagnostics-dir`
- Env: `ENABLE_QC_DIAGNOSTICS=true`
- Env: `FUSION_ENABLE_QC_DIAGNOSTICS=true`
- Env output dir: `FUSION_QC_DIAGNOSTICS_DIR=...`

Diagnostics fields include:

- `scene_id`
- `agri_file`
- `myd06_file`
- `myd03_file`
- raw valid counts for CLP/CER/COT/CTH
- independent gate counts:
  - `time_ok_px`
  - `overlap_ok_px`
  - `geo_ok_px`
  - `phase_ok_px`
  - `reg_time_ok_px`
  - `reg_overlap_ok_px`
  - `reg_cloud_ok_px`
  - `reg_phase_ok_px`
- cumulative counts:
  - `cumulative_base_px`
  - `cumulative_after_time_px`
  - `cumulative_after_overlap_px`
  - `cumulative_after_geo_px`
  - `cumulative_after_phase_px`
  - `cumulative_after_reg_time_px`
  - `cumulative_after_reg_overlap_px`
  - `cumulative_after_reg_cloud_px`
  - `cumulative_after_reg_phase_px`
- final valid counts:
  - `final_clp_px`
  - `final_cer_px`
  - `final_cot_px`
  - `final_cth_px`
- summary stats:
  - `time_delta_min_p50`
  - `time_delta_min_p90`
  - `time_delta_min_max`
  - `overlap_ratio`
  - `cloud_frac`
  - `phase_consistency`

Relevant files:

- `fusion_io.py`
- `data_fusion.py`
- `fusion_config.py`
- `main.py`
- `tests/test_fusion_time.py`

## 6. QC Failure Analyzer

Problem:

- Need a script to analyze `runs/qc_diagnostics/qc_gate_stats.csv` and infer why all-zero scenes fail.

Implemented:

- Added `tools/analyze_qc_failures.py`.
- Added `tests/test_analyze_qc_failures.py`.
- The script classifies:
  - success scene: final CLP + CER + COT + CTH > 0
  - zero scene: all four final counts are 0
- For each all-zero scene, it infers `inferred_failure_reason`:
  - first gate where cumulative count goes from >0 to 0
  - if no direct zero, the gate with the largest drop ratio

Outputs:

- `runs/qc_diagnostics/qc_failure_summary.csv`
- `runs/qc_diagnostics/sample_success_vs_zero.csv`
- `runs/qc_diagnostics/failure_reason_summary.csv`
- `runs/qc_diagnostics/qc_failure_report.md`

Console output:

- Overall success/zero scene counts and rates.
- Top failure gates.

Relevant files:

- `tools/analyze_qc_failures.py`
- `tests/test_analyze_qc_failures.py`

## 7. Current Diagnostics Run State

A full single-process day run for `20190105` was attempted with `n_workers=1`, but was too slow and was terminated after about 24 minutes before diagnostics were written.

Then a single-scene diagnostics probe was run successfully for:

- day: `20190105`
- scene: `20190105_054500`
- output h5: `/tmp/agri_qc_diag_probe_single/20190105/AGRI_MYD06_20190105_054500.h5`
- diagnostics:
  - `runs/qc_diagnostics/qc_gate_stats.csv`
  - `runs/qc_diagnostics/qc_gate_stats.jsonl`

Single-scene result:

- `ok=True`
- `msg=OK samples=493`
- final counts:
  - `final_clp_px=85790`
  - `final_cer_px=1958`
  - `final_cot_px=1850`
  - `final_cth_px=9017`

Important row values from `runs/qc_diagnostics/qc_gate_stats.csv`:

- `raw_clp_valid_px=129857`
- `raw_cer_valid_px=16390`
- `raw_cot_valid_px=16297`
- `raw_cth_valid_px=66989`
- `cumulative_base_px=129857`
- `cumulative_after_time_px=129857`
- `cumulative_after_overlap_px=129857`
- `cumulative_after_geo_px=85790`
- `cumulative_after_phase_px=85790`
- `cumulative_after_reg_time_px=10525`
- `cumulative_after_reg_overlap_px=10525`
- `cumulative_after_reg_cloud_px=10525`
- `cumulative_after_reg_phase_px=9652`
- `time_delta_min_p50=2.560936`
- `time_delta_min_p90=4.553824`
- `time_delta_min_max=7.410790`
- `overlap_ratio=0.022840`
- `cloud_frac=0.577443`
- `phase_consistency=0.728404`

Interpretation for this one successful scene:

- CLP did not lose pixels at time or overlap gates.
- CLP lost a large amount at geo gate: 129857 -> 85790.
- Regression supervision then dropped strongly at `reg_time`: 85790 cloudy-gated path to 10525.
- Final regression counts are lower because per-channel finite/range checks also apply.

This is only one successful scene. It does not answer the main all-zero-scene failure distribution. For that, run diagnostics over a broader set of scenes and then run `tools/analyze_qc_failures.py`.

## 8. Verification Already Run

Fusion diagnostics unit tests:

```bash
/home/yuq/anaconda3/envs/cloudunet/bin/python -m pytest -p no:cacheprovider tests/test_fusion_time.py
```

Result:

- `16 passed, 1 warning`

Full test suite:

```bash
/home/yuq/anaconda3/envs/cloudunet/bin/python -m pytest -p no:cacheprovider tests
```

Result:

- `26 passed, 1 warning`

Warning:

- Existing warning from `fusion_io.py`:
  - `RuntimeWarning: Mean of empty slice`
  - occurs in test coverage for empty `SAMPLE_WEIGHT`
  - not introduced as a failure

## 9. Useful Commands For Next Session

Run a small diagnostics fusion batch with multiprocessing off:

```bash
ENABLE_QC_DIAGNOSTICS=true /home/yuq/anaconda3/envs/cloudunet/bin/python data_fusion.py \
  --split train \
  --day 20190105 \
  --workers 1 \
  --max_qc 0 \
  --enable-qc-diagnostics \
  --qc-diagnostics-dir runs/qc_diagnostics
```

Note:

- This writes normal fusion outputs to the configured paired directory unless redirected through a direct `fuse_day(...)` call.
- For safe probing without touching official paired data, prefer a small custom call that writes `out_dir` under `/tmp`.

Analyze diagnostics:

```bash
/home/yuq/anaconda3/envs/cloudunet/bin/python tools/analyze_qc_failures.py \
  --input runs/qc_diagnostics/qc_gate_stats.csv \
  --out-dir runs/qc_diagnostics \
  --sample-n 5
```

Run checkpoint batch eval:

```bash
/home/yuq/anaconda3/envs/cloudunet/bin/python tools/eval_checkpoints.py \
  --checkpoints \
  /data/Data_yuq/unet_workdir/model/HIR_COMP_UNet_AGRIonly_best_loss.pth \
  /data/Data_yuq/unet_workdir/model/HIR_COMP_UNet_AGRIonly_best_oa.pth \
  /data/Data_yuq/unet_workdir/model/HIR_COMP_UNet_AGRIonly_best_macro.pth
```

Run test for one checkpoint:

```bash
/home/yuq/anaconda3/envs/cloudunet/bin/python test.py \
  --checkpoint /data/Data_yuq/unet_workdir/model/HIR_COMP_UNet_AGRIonly_best_oa.pth
```

## 10. Known Caveats / Next Work

- `runs/qc_diagnostics/qc_gate_stats.csv` currently contains only the one single-scene probe row unless a broader diagnostics run is executed.
- The all-zero-scene top failure gate distribution is therefore not known yet.
- A full day with `workers=1` was slow in the Codex sandbox. For production diagnostics, run by day in the user's normal shell, or add a targeted debug option later to limit number of scenes per day.
- Do not infer the global bottleneck from the single successful `20190105_054500` scene.
- Next practical step:
  1. Run diagnostics on a manageable subset with `workers=1`.
  2. Run `tools/analyze_qc_failures.py`.
  3. Inspect `failure_reason_summary.csv` to determine whether all-zero scenes mostly die at `time`, `overlap`, `geo`, `phase`, `reg_time`, `reg_cloud`, or `reg_phase`.

