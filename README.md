# CS771 - Introduction to Machine Learning (Autumn 2024)  
## Mini-Project 2: Lifelong and Domain Adaptive Learning

**Team Members:**
- Abhinav Singh (220038)  
- Aditya Mathur (220068)  
- Patel Rhut Dipakbhai (220756)  
- Sahil Kumar Goyat (220937)  
- Darshan Dinesh Sethia (220326)

---

##  Overview

This repository contains our implementation for Mini-Project 2 of CS771. The project focuses on **Lifelong Learning** and **Domain Adaptation** using 20 datasets derived from CIFAR-10.

The work is divided into two main parts, each implemented in a separate Google Colab notebook:

- `task1_colab.ipynb`: Learning with Prototypes (LwP) on D1–D10.
- `task2_colab.ipynb`: Domain-adaptive learning on D11–D20.

The final report with full details of the experiments is available here:  
📄 [`CS771_report.pdf`](./CS771_report.pdf)

---

##  Problem Statement

- **Datasets**: D1–D20 from CIFAR-10.
  - D1 is labeled.
  - D2–D10 share the same distribution (in-distribution).
  - D11–D20 are from shifted but related distributions (out-of-distribution).
- **Goal**: Learn a model sequentially across datasets without revisiting old data, while maintaining high accuracy on all previous datasets.

---

##  Methods Summary

### Task 1: In-Distribution Learning
- **Feature Extraction**: ViT (Vision Transformer) pretrained on ImageNet.
- **Dimensionality Reduction**: Linear Discriminant Analysis (LDA).
- **Learning Algorithm**: Gaussian Mixture Models (GMM) + MAP estimation.
- **Continual Strategy**: Generative replay buffer using GMM-based sampling.

### Task 2: Domain Adaptive Learning
- **Augmentation**: RandMix (combines AdaIN, spatial/color perturbations, and noise).
- **Selective Sampling**: Top-2 Pseudo Labeling (T2PL) to improve centroid estimation.
- **Replay Strategy**: Combination of RandMix and GMM-based buffer samples.

---

##  Results Snapshot

- All models are evaluated on their own and previous datasets.
- Accuracy matrices (10×10 for Task 1 and 10×20 for Task 2) are reported in the PDF.
- Use of RandMix and T2PL significantly improves robustness in domain-shifted settings.

---

##  Repository Structure

```plaintext
.
├── task1_colab.ipynb       # Colab notebook for Task 1 (D1–D10)
├── task2_colab.ipynb       # Colab notebook for Task 2 (D11–D20)
├── CS771_report.pdf        # Final project report with methods and results
└── README.md               # This file
```
---

##  Useful Links
- [ViT Model on HuggingFace](https://huggingface.co/google/vit-base-patch16-224)
- [Presentation Video](https://www.youtube.com/watch?v=Wr5pD0Kw_zI)

---

##  References

1. Deja Vu: Continual Model Generalization for Unseen Domains (ICLR 2023)  
2. Lifelong Domain Adaptation via Consolidated Internal Distribution (NeurIPS 2021)
