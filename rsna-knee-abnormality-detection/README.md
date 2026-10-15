# RSNA Knee Abnormality Detection: baseline fork with blend-weight experiments

Kaggle: RSNA Knee Abnormality Detection (12-finding knee MRI classification, macro ROC-AUC).

## What this is
An inference-only notebook that blends two independently trained model
branches (Raptor CoAtNet and Compact R7 ConvNeXt-small) by per-finding
percentile rank. No training happens in this notebook.

## My contribution
- Forked a public notebook, ran it on Kaggle (GPU T4 x2, internet off).
- Baseline blend (Raptor 0.5 / Compact 0.5): public score 0.956.
- Experiment: Raptor weight changed to 0.6 for the nine non-OA findings.
  Result: (fill in public score here once available).

## Credits (this is not my original work)
- Notebook forked from: Дворкин Евгений Владимирович (Kaggle)
- Source notebook: heliosli, "0.957 | Latest dataset | score in one click"
- Model weights come from public Kaggle datasets by their authors:
  RSNA Knee Compact R7 CNX FS SS, RSNA Knee selected SWA weights,
  RSNA OrthoFoundation Fold0 Epoch7. Linked, not re-uploaded.
All architectures, weights, preprocessing and ensemble methods belong to
their original authors.

## How to run
1. On Kaggle, attach the competition data and the three datasets above.
2. Set accelerator to GPU T4 x2 and turn internet off.
3. Run all, then submit `submission.csv`.

## Not included
Competition data and third-party model weights are not in this repo.