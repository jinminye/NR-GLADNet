# NR-GLADNet

Official repository for the manuscript:

**NR-GLADNet: Noise-Resilient Global-Local Attentive Dual-Branch Network for Acoustic Fault Diagnosis**

> Repository status: documentation scaffold only. The source code and reproducibility materials will be added in a subsequent release. No raw datasets, trained models, or experimental outputs are distributed in this initial version.

## Overview

NR-GLADNet is a noise-resilient dual-branch network for non-contact acoustic fault diagnosis. The method combines:

- adaptive soft-threshold denoising;
- a convolutional branch with global attention for local time-frequency features;
- a Transformer branch with local-global multi-head self-attention for global context;
- two-stage attentive fusion of local and global representations.

This repository will contain the model implementation, preprocessing and split-generation scripts, noise-generation utilities, configuration files, and instructions for reproducing the reported experiments.

## Data availability

The DCASE 2023 Task 2 bearing subset used in the study is publicly available from Zenodo:

<https://zenodo.org/records/7882613>

The CUMTB wind-turbine pitch-bearing dataset is subject to access and redistribution restrictions. The raw CUMTB recordings are therefore not included in this repository. Readers should contact the original data custodians for access, subject to their permission.

## Planned repository contents

- model implementations;
- training, evaluation, and ablation scripts;
- acoustic preprocessing and STFT utilities;
- controlled Gaussian-noise and recorded fan-interference generation scripts;
- configuration files and environment requirements;
- reproducibility instructions.

## Citation

Citation information will be added when the associated article is published.

## License

A software license will be specified when the source-code release is published.