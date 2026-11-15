# Pipeline Entry Points

All scripts read base input, model, and output paths exclusively from `SETTINGS.json`.

## 1. Data Preprocessing
Decompresses lossless DICOM series using offline `pylibjpeg` decoders, applies anatomical joint center-cropping (`crop110`), resizes slices to `384 × 384`, normalizes per-series intensity distributions, and exports sequence metadata:
```bash
python src/prepare_data.py --config configs/caformer_s36_gru.yaml
