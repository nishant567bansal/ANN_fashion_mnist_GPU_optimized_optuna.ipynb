# 👗 GPU-Optimized Fashion MNIST Classification using ANN

A deep learning project implementing a fully configurable multi-layer Artificial Neural Network in PyTorch for image classification on the Fashion MNIST dataset. The pipeline features GPU-accelerated training, Bayesian hyperparameter optimization via Optuna, and rigorous regularization techniques — achieving **88.91% test accuracy**.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## Overview

This project demonstrates an end-to-end deep learning workflow for multi-class image classification. The focus is not just on achieving high accuracy but on building a **production-aware training pipeline** with:

- Dynamic, configurable ANN architecture with `BatchNorm`, `ReLU`, and `Dropout`
- GPU-accelerated training using CUDA (NVIDIA T4 on Google Colab)
- Custom `torch.utils.data.Dataset` and `DataLoader` with `pin_memory` for efficient host-to-device data transfer
- Bayesian hyperparameter search using **Optuna** across 10 trials
- Systematic evaluation via validation curves and regularization analysis

---

## Dataset

**Fashion MNIST** — a drop-in replacement for MNIST, consisting of 70,000 grayscale images of clothing articles across 10 categories.

| Split | Samples |
|-------|---------|
| Training (80%) | 48,000 |
| Test (20%) | 12,000 |

Each image is 28×28 pixels, flattened to a 784-dimensional feature vector. Pixel values are normalized to `[0, 1]` by dividing by 255.

**Class Labels:**

| Label | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Model Architecture

The model (`MyNN`) is a fully parameterizable feedforward neural network built using `nn.Sequential`. Each hidden layer block consists of:

```
Linear(input_dim → neurons_per_layer)
  → BatchNorm1d
  → ReLU
  → Dropout(p)
```

Followed by a final `Linear(neurons_per_layer → 10)` output layer (logits for CrossEntropyLoss).

**Key design choices:**
- **Batch Normalization** after each linear layer for training stability and faster convergence
- **Dropout** for regularization and preventing overfitting
- **ReLU** activations throughout to mitigate vanishing gradient issues
- Number of hidden layers, neurons per layer, and dropout rate are all tunable

---

## Training Pipeline

- **Loss Function:** `CrossEntropyLoss` (combines `LogSoftmax` + `NLLLoss` internally)
- **Device:** CUDA (`torch.device("cuda")`) with automatic CPU fallback
- **Data Loading:** Custom `CustomDataset` wrapping feature/label tensors, with `pin_memory=True` for faster GPU transfers and `shuffle=True` during training
- **Reproducibility:** Fixed seed via `torch.manual_seed(42)`

The training loop follows the standard PyTorch pattern:
1. Move batch to GPU
2. Forward pass
3. Compute loss
4. Zero gradients → Backprop → Optimizer step
5. Evaluate on test set using `torch.no_grad()`

---

## Hyperparameter Optimization

Bayesian optimization was performed using **Optuna** (`optuna.create_study(direction="maximize")`), maximizing validation accuracy over **10 trials**.

**Search Space:**

| Hyperparameter | Range / Choices |
|---|---|
| `num_hidden_layers` | 1 – 5 |
| `neurons_per_layer` | 8 – 128 (step 8) |
| `epochs` | 10 – 50 (step 10) |
| `learning_rate` | 1e-5 – 1e-1 (log scale) |
| `dropout_rate` | 0.1 – 0.5 (step 0.1) |
| `batch_size` | {16, 32, 64, 128} |
| `optimizer` | Adam, SGD, RMSprop |
| `weight_decay` | 1e-5 – 1e-3 (log scale) |

**Best Configuration Found (Trial 5):**

| Parameter | Value |
|---|---|
| `num_hidden_layers` | 3 |
| `neurons_per_layer` | 120 |
| `epochs` | 50 |
| `learning_rate` | 1.51e-4 |
| `dropout_rate` | 0.1 |
| `batch_size` | 128 |
| `optimizer` | RMSprop |
| `weight_decay` | 2.10e-5 |

---

## Results

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy (Optuna)** | **87.93%** |
| **Final Test Accuracy** | **88.91%** |
| **GPU Training Speedup** | ~3× faster than CPU baseline |
| **Hardware** | NVIDIA T4 (Google Colab) |

---

## Project Structure

```
├── ann_fashion_mnist_GPU_optimized_optuna.ipynb   # Main notebook
├── fashion-mnist_train.csv                         # Training data (Kaggle)
└── README.md
```

---

## Requirements

```
torch
torchvision
pandas
numpy
scikit-learn
matplotlib
optuna
```

Install with:

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib optuna
```

---

## How to Run

1. Clone the repository and place `fashion-mnist_train.csv` in the root directory (available on [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)).
2. Open the notebook in **Google Colab** (recommended — set runtime to GPU).
3. Run all cells sequentially. Optuna will conduct 10 trials and print the best hyperparameters and validation accuracy.

```python
# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")  # → Using device : cuda
```

4. The final model is trained using the best parameters identified by Optuna and evaluated on the held-out test set.

---

## Key Concepts Demonstrated

- Custom PyTorch `Dataset` and `DataLoader` with GPU memory pinning
- Dynamic ANN construction using `nn.Sequential` and programmatic layer stacking
- Batch Normalization and Dropout for regularization
- Bayesian hyperparameter optimization with Optuna
- GPU-accelerated deep learning on CUDA hardware
- Systematic model evaluation and generalization analysis

---

*Built with PyTorch · Optuna · Google Colab (T4 GPU)*
