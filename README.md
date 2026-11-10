# NR-GLADNet

Official repository for **Noise-Resilient Global-Local Attentive Dual-Branch Network for Noncontact Acoustic Measurement and Bearing Fault Diagnosis**.

This release provides the model implementation, pretrained checkpoints, prepared STFT features, dataset splits, and an evaluation script for CUMTB and DCASE. It supports inference and testing with saved weights; training scripts are outside the scope of this release.

## Contents

| Path | Contents |
|---|---|
| `model/nr-gladnet.py` | NR-GLADNet model implementation |
| `test.py` | Evaluation using a checkpoint, split, and feature directory |
| `checkpoints/` | 22 checkpoints and SHA-256 checksums |
| `features/` | 22 prepared feature directories and SHA-256 checksums |
| `splits/` | CUMTB balanced random split and DCASE class random split |
| `ENVIRONMENT.md` | Tested software versions |

For each dataset, the feature directories cover one clean condition, five Gaussian noise conditions, and five fan interference conditions. CUMTB SNR levels are -8, -6, -4, -2, and 0 dB; DCASE SNR levels are -4, -2, 0, 2, and 4 dB.

## Evaluation

Set up the versions listed in [ENVIRONMENT.md](ENVIRONMENT.md), download the repository, and run:

```bash
python test.py
```

The default paths select the DCASE clean checkpoint, features, and split. To evaluate another condition, edit `SPLIT_PATH`, `WEIGHTS_PATH`, and `FEATURE_DIR` at the top of `test.py`, or pass the three paths directly:

```bash
python test.py --split "splits/<split>.json" --weights "checkpoints/<run>/last.pth" --features "features/<feature_directory>"
```

The script evaluates the split's test list, prints accuracy, macro precision, macro recall, macro F1, and the confusion matrix, and saves a JSON report under `results/`. Use `--device cpu` to select CPU evaluation.

The released CUMTB split is `balanced575_random_train_test`; the released DCASE split is `class_random_train_test`.

## Data availability

This repository distributes prepared features. Raw recordings are not included.

The DCASE 2023 Task 2 bearing subset used in the study is publicly available from [Zenodo](https://zenodo.org/records/7882613).

The CUMTB wind-turbine pitch-bearing dataset is subject to access and redistribution restrictions. Readers should contact the original data custodians for access to the raw recordings, subject to their permission.

## Citation

Citation information will be added when the associated article is published.

## License

A software license has not yet been specified.
