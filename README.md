# AMSDFF: Advanced Multi-Scale Dynamic Feature Fusion for Text Classification

AMSDFF is a state-of-the-art deep learning architecture that combines multi-scale attention mechanisms, dynamic feature fusion, and advanced pooling strategies to achieve superior performance on text classification tasks. The model outperforms across multiple benchmark datasets.

![image](https://github.com/user-attachments/assets/53001830-f973-4418-b281-bd84bca5a215)

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

The model has been evaluated on multiple text classification benchmarks:
- News categorization datasets
- Sentiment analysis datasets
- Topic classification datasets
- Document classification datasets

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
