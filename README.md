# Code and Experimental Results for Partial Parameter Sharing in Chinese Calligraphy Recognition

## 1. Overview

This repository contains the source code and experimental results for our research on multi-task Chinese calligraphy recognition.

The study investigates negative transfer in multi-task learning (MTL), particularly the performance degradation caused by fully shared feature representations.

We propose a **Partial Parameter Sharing** strategy based on ResNet18 to improve knowledge sharing between two recognition tasks while reducing task interference.

The main objective is to maintain or improve the performance of the primary task while preserving competitive recognition performance on the auxiliary task.

## 2. Research Tasks

The framework focuses on two tasks:

- **Character Recognition:** Identifying Chinese calligraphy characters.
- **Source Recognition:** Identifying the source associated with each calligraphy image.

## 3. Method

The proposed method adopts a partially shared ResNet18 architecture.

The experiments investigate different parameter-sharing strategies:

1. Single-task ResNet18 baseline.
2. Equal-weight full sharing.
3. Equal-weight partial sharing.
4. Fixed-weight full sharing.
5. Fixed-weight partial sharing (proposed method).

The proposed method is designed to reduce negative transfer through selective parameter sharing and task-specific feature learning.

## 4. Experimental Design

The experiments include:

- Three random seeds: 42, 123, and 2026.
- Single-task baseline comparisons.
- Full-sharing and partial-sharing ablation experiments.
- Different task-loss weighting strategies.
- Independent test-set evaluation.
- Source-only baseline evaluation.
- Model complexity and inference-speed analysis.
- Error analysis and class-frequency analysis.
- Paired prediction comparison using the exact McNemar test with Holm correction.

## 5. Repository Structure

```text
calligraphy_sci/
├── dual_splits/
├── paper_figs_dataset_examples/
├── results/
│   ├── ablation_2x2_statistics/
│   ├── ablation_2x2_test/
│   ├── final_test/
│   ├── final_error_analysis/
│   └── statistical_tests/
├── splits/
├── analyze_*.py
├── train_*.py
├── test_*.py
└── README.md
```

## 6. Dataset

The experiments use a Chinese calligraphy image dataset.

The original dataset images are not included in this repository. Users must obtain the dataset separately and configure the corresponding local data paths before running the experiments.

**Note:** Some experimental split files may contain machine-specific absolute paths. These paths need to be adjusted to match the local dataset directory.

## 7. Experimental Results

The `results/` directory contains experimental outputs, prediction files, statistical analysis, and evaluation summaries.

These materials support the comparisons between single-task learning, full parameter sharing, and partial parameter sharing.

## 8. Environment

The experiments were conducted using Python and PyTorch.

The exact software dependencies and execution configurations will be documented in the corresponding environment configuration files.

## 9. Reproducibility

Training scripts, evaluation scripts, dataset split files, and experimental results are provided to support experimental verification and reproducibility.

Random seeds are specified in the corresponding experiment scripts and output directories.

## 10. Citation

If you use this code in your research, please cite the associated paper:



## Trained Model Checkpoints (Zenodo)

The final trained models for our partial-sharing ResNet18 approach are publicly archived on Zenodo.

**Zenodo DOI (Version v1):** https://doi.org/10.5281/zenodo.24269713

**DOI for all versions:** https://doi.org/10.5281/zenodo.24269712

### Available Checkpoints

| File | Random Seed | Model |
|---|---|---|
| `best_model_seed42.pth` | 42 | Partial-Sharing ResNet18 |
| `best_model_seed123.pth` | 123 | Partial-Sharing ResNet18 |
| `best_model_seed2026.pth` | 2026 | Partial-Sharing ResNet18 |

All three checkpoints use the same architecture:

- Backbone: ResNet18 trained from scratch
- Shared layers: stem, layer1, layer2, and layer3
- Task-specific layers: independent layer4 blocks and classification heads
- Primary task: Character Recognition (6,249 classes)
- Auxiliary task: Calligraphic Source Recognition (19 classes)
- Loss function: L = L_character + 0.1 * L_source

### Held-Out Test Performance

Results are reported as mean +/- sample standard deviation over three random seeds.

| Metric | Partial Sharing (Ours) |
|---|---|
| Character Top-1 | 96.5662% +/- 0.1206% |
| Character Macro-F1 | 0.962637 +/- 0.001417 |
| Source Top-1 | 96.7256% +/- 0.0735% |
| Source Macro-F1 | 0.970277 +/- 0.000637 |

The final model improves Character Top-1 accuracy by 0.4250 percentage points compared with the standardized single-task ResNet18 baseline on the held-out test set.

### Dataset Availability

The experiments use a processed version of the Chinese Calligraphy Dataset.

Original dataset:
https://github.com/zhuojg/chinese-calligraphy-dataset

The original dataset images are not redistributed in the Zenodo model archive.

### Citation

If you use the trained checkpoints, please cite the Zenodo record:

Wen, H., & Liu, Z. (2026). *Trained Partial-Sharing ResNet18 Models for Dual-Task Chinese Calligraphy Recognition* (Version v1). Zenodo.

https://doi.org/10.5281/zenodo.24269713


*The bibliographic information will be added after publication.*

## 11. Contact

For questions regarding the implementation or experimental results, please contact the corresponding author.
