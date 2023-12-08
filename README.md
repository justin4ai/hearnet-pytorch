## HEAR-Net Pytorch Implementation

### Train

1. Put this directory under ghost directory.

2. 
```cd HEAR-Net```

```python3 train.py --source_images {SOURCE_PATH} --target_images {TARGET_PATH} --swapped_images {SWAPPED_PATH} heuristic_errors {ERROR_PATH} ```


Package & module dependencies will be updated.

First step : Implement end-to-end HEAR-Net pytorch code (ongoing)
- Model (done)
- Train (ongoing / almost done)
- Inference

Second step : Get the best pretrained model

Third step : Verify if the output is the same to what we wanted

Last step : Integrate AEI-Net and HEAR-Net into the full-pipeline model.