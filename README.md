# Vision Transformer for Multi-Task Knee MRI Classification: An Attention-Based Approach

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

## Abstract

This repository presents **MRNetViT**, a deep learning framework for the automated diagnosis of knee injuries from Magnetic Resonance Imaging (MRI). Leveraging the Stanford MRNet dataset, we propose a custom architecture that combines a **Vision Transformer (ViT)** backbone with an **Attention-Based Aggregator** to handle variable-length slice sequences. Unlike traditional pooling methods, our attention mechanism dynamically learns the clinical significance of individual slices within an MRI volume. The model is trained using a multi-task learning objective to simultaneously detect **Abnormalities**, **ACL tears**, and **Meniscal tears**. Our approach achieves competitive performance, demonstrating the efficacy of transformers in volumetric medical imaging.

---

## 1. Introduction

Knee injuries are among the most common musculoskeletal conditions, often requiring MRI for definitive diagnosis. However, manual interpretation of MRI volumes—which consist of multiple 2D slices across different planes (sagittal, axial, coronal)—is time-consuming and prone to inter-observer variability.

This project implements a SOTA-inspired pipeline to automate this process. We focus on:
- **Volumetric Reasoning:** Integrating information across multiple slices.
- **Explainability:** Utilizing attention maps to identify key diagnostic features.
- **Robustness:** Evaluating performance on out-of-distribution (OOD) data.

---

## 2. Methodology

### 2.1 Dataset: MRNet
The model is trained and validated on the **MRNet dataset**, which contains 1,370 knee MRI exams. Each exam includes volumes in three orientations:
- **Sagittal:** 1,130 exams.
- **Coronal:** 1,130 exams.
- **Axial:** 1,130 exams.

Each volume is labeled for three clinical conditions: `Abnormal`, `ACL Tear`, and `Meniscal Tear`.

### 2.2 MRNetViT Architecture
The core architecture consists of three main components:

1.  **Feature Extractor:** A pre-trained `vit_small_patch16_224` (Vision Transformer) from the `timm` library. It processes each MRI slice independently to extract high-dimensional features ($\in \mathbb{R}^{384}$).
2.  **Attention-Based Aggregator:** Instead of standard max or average pooling, we employ a learnable attention mechanism. For a sequence of slices $S = \{s_1, s_2, ..., s_n\}$, the aggregator computes an importance weight $\alpha_i$ for each slice:
    $$\alpha_i = \text{softmax}(\text{MLP}(h_i))$$
    The global representation is then the weighted sum: $H = \sum \alpha_i h_i$.
3.  **Multi-Task Classifier:** A fully connected head that outputs three independent logits, one for each diagnostic condition.

### 2.3 Data Preprocessing & Augmentation
- **Normalization:** Standardized using ImageNet statistics.
- **Resizing:** All slices are resized to $224 \times 224$ pixels.
- **Sampling:** To handle variable depth, we uniformly sample 24 slices per volume.
- **Augmentation:** To improve generalization, we apply:
    - Random Horizontal Flips.
    - Random Affine Transformations (rotation, translation).
    - Gaussian Blur (to simulate varying MRI resolutions).

---

## 3. Training Strategy

We utilize a sophisticated training regimen to optimize the transformer backbone:

-   **Differential Learning Rates:** The ViT backbone is fine-tuned with a lower learning rate ($0.2 \times \eta$), while the attention head and classifier are trained with a higher base learning rate ($\eta = 2 \times 10^{-4}$).
-   **Loss Function:** Multi-label Binary Cross Entropy with Logits. We apply **Positional Weights** $[0.25, 4.0, 2.0]$ to account for class imbalance (specifically for ACL and Meniscal tears).
-   **Optimization:** `AdamW` optimizer with a weight decay of $10^{-2}$ and a `CosineAnnealingLR` scheduler.
-   **Mixed Precision:** Utilizes `torch.cuda.amp` for faster training and reduced memory footprint.

---

## 4. Evaluation & Results

### 4.1 Quantitative Performance
The model is evaluated using the **Area Under the ROC Curve (AUC)**, which is the standard metric for the MRNet challenge.

| Task | AUC (Sagittal) | Sensitivity | Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Abnormal** | 0.871 | 0.892 | 0.750 | 0.864 |
| **ACL Tear** | 0.834 | 0.785 | 0.821 | 0.742 |
| **Meniscus Tear** | 0.798 | 0.710 | 0.805 | 0.691 |
| **Mean AUC** | **0.834** | - | - | - |

*Note: Metrics may vary based on specific training runs and planes.*

### 4.2 Visualizations
The repository generates several diagnostic visualizations (see `eval_outputs/`):
- **ROC Curves:** Comparing performance across tasks.
- **Confusion Matrices:** Analyzing True/False positive rates.
- **Attention Maps:** Visualizing which slices and spatial regions the model "focused" on to make its prediction.

---

## 5. Robustness Analysis (OOD)

A critical requirement for medical AI is robustness to data from different scanners or institutions. We include a specialized script `run_ood_batch.py` to evaluate the model on **Out-of-Distribution (OOD)** images sourced from external clinical databases.

The OOD results are compared against the MRNet validation baseline in `ood_comparison.png`, providing insights into the model's generalization capabilities.

---

## 6. Installation & Usage

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support)
- `pip install -r requirements.txt` (including `timm`, `scikit-learn`, `matplotlib`, `seaborn`)

### Training
```bash
python train.py --data_root /path/to/mrnet --plane sagittal --epochs 50 --lr 2e-4
```

### Evaluation
```bash
python evaluate.py --data_root /path/to/mrnet --checkpoint best_model.pth --plane sagittal
```

### OOD Inference
```bash
python run_ood_batch.py --model_path best_model.pth --image_dir ood_images/ --auto_label
```

---

## 7. Project Structure

```text
├── dataset.py          # Data loading and augmentation pipeline
├── model.py            # MRNetViT architecture definition
├── train.py            # Multi-task training script
├── evaluate.py         # Comprehensive evaluation and metric generation
├── run_ood_batch.py    # Out-of-distribution robustness testing
├── gen_attention.py    # Attention map visualization
└── eval_outputs/       # Generated plots and metrics
```

---

## 9. Contributors & Collaborative Ownership

This project was developed as a cohesive group effort, with all members participating in the full lifecycle of the research, implementation, and documentation. While every team member maintains a comprehensive understanding of the entire MRNetViT pipeline, specific leads were assigned to ensure rigorous validation of key modules:

*   **Kishore Damodharan:** Led the architectural design of the **Data Preprocessing** pipeline, including volumetric sampling strategies and normalization protocols.
*   **Deepakraj Rajmohan:** Managed the **Model Building** and refinement of the Vision Transformer backbone and Attention-Based Aggregator.
*   **Poovarasan Pariselvam:** Orchestrated the **Training Pipeline**, optimization strategies, and differential learning rate configurations.
*   **Rohidh Govindan Choodamani:** Directed the **Evaluation Pipeline**, metric generation, and comprehensive validation against the MRNet baseline.
*   **Sriraaj Anandh:** Headed the **Visualization & OOD Robustness** testing, including attention map generation and external clinical data inference.

All contributors participated equally in the experimental design, ablation studies, and the final technical report.
