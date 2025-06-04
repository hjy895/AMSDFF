# AMSDFF: Advanced Multi-Scale Dynamic Feature Fusion for Text Classification

AMSDFF is a state-of-the-art deep learning architecture that combines multi-scale attention mechanisms, dynamic feature fusion, and advanced pooling strategies to achieve superior performance on text classification tasks. The model outperforms across multiple benchmark datasets.

![image](https://github.com/user-attachments/assets/f0484832-b2ff-4316-a215-63597cb2dce8)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the Enhanced AMSDFF (Advanced Multi-Scale Dynamic Feature Fusion) model for text classification tasks.

### Key Features

- **Multi-Scale Attention:** Captures features at different granularities.
- **Dynamic Feature Fusion:** Adaptive combination of features using expert networks.
- **Advanced Pooling Strategy:** Combines attention, CLS token, and max pooling.
- **Optimized Architecture:** Efficient design for both accuracy and computational performance.

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU support)

## 📊 Supported Datasets

The model has been evaluated on four diverse text classification benchmarks spanning different domains, scales, and temporal ranges:

- News Categorization
AG News (2004-2005): Large-scale news classification with 127,600 samples across 4 categories:
BBC News (2004-2005): Balanced news categorization with 2,225 samples across 5 categories

- Financial & Topic Classification
Reuters (1987): Financial domain complexity with 10,788 samples across 46 topics:
20 Newsgroups (1993-1994): Forum discussion classification with 18,846 samples across 20 categories

## 🏗️ Architecture

### Model Components

1. **Multi-Scale Attention Module**
   - Captures features at different scales (1, 3, 5)
   - Depthwise separable convolutions for efficiency
   - Scale fusion with multi-head attention

2. **Dynamic Fusion Module**
   - Multiple expert networks
   - Learnable gating mechanism
   - Adaptive feature combination

3. **Advanced Pooling Strategy**
   - Attention-based pooling
   - CLS token pooling
   - Max pooling
   - Fusion of all pooling methods

### Technical Details

- **Base Model**: DistilBERT (can be configured)
- **Hidden Size**: 768
- **Attention Heads**: 8
- **Dropout**: 0.1-0.3 (layer-dependent)
- **Optimizer**: AdamW with cosine scheduling
- **Loss Functions**: Combined CE, Focal, and Label Smoothing

## 📁 Repository Structure

```
enhanced-amsdff/
├── data/          # Data loading and preprocessing
├── models/        # Model architectures
├── training/      # Training utilities
├── evaluation/    # Evaluation metrics
├── experiments/   # Experiment configurations
├── utils/         # Helper functions
└── results/       # Experiment results
```

## 🤝 Contributing

We welcome contributions!

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{enhanced_amsdff_2025,
  title={Enhanced AMSDFF: Advanced Multi-Scale Dynamic Feature Fusion for Text Classification},
  year={2025}
}
```


## 🙏 Acknowledgments

We thank the open-source community for providing the foundational models and datasets used in this research.
