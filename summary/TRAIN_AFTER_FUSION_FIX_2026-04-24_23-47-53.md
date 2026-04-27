# Training After Fusion Filter Fix

## Split

- train: `20190115`, `20191215`, `20190405`
- val: `20191225`
- test: `20190725`

## Data

- train: 24 H5, 22,265 patches
- val: 4 H5, 1,092 patches
- test: 6 H5, 1,310 patches
- train CLP distribution: clear `32.4%`, water `28.7%`, ice `38.9%`
- CLP loss weights: clear `1.01`, water `1.08`, ice `0.93`

## Training

- Device: CUDA
- Early stopped at epoch 15
- Best checkpoint by validation loss: epoch 5
- Best validation loss: `2.178658`
- Best validation OA at checkpoint: `44.64%`
- Validation class accuracy at checkpoint: clear `50.7%`, water `17.1%`, ice `28.3%`

## Test

- CLP OA: `28.57%`
- Clear accuracy: `37.63%`
- Water accuracy: `14.41%`
- Ice accuracy: `51.82%`
- CER RMSE: `12.793 um`
- COT RMSE: `22.430`
- CTH RMSE: `5920.886 m`

## Interpretation

The fusion/filter fix did recover clear supervision and prevented the previous water-only / no-clear sampling collapse. However, this random small split still generalizes poorly, especially for water. The likely next issues are distribution shift between selected days and checkpoint selection by multi-task loss rather than CLP macro accuracy.
