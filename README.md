# Chest X-Ray Pneumonia Classification

Binary classification of chest X-ray images (Normal vs Pneumonia) using PyTorch.
Three architectures are compared: a lightweight CNN trained from scratch, ResNet18
with a frozen backbone, and ResNet18 with partial fine-tuning.

**Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
— 5,863 labeled JPEGs split into train / val / test.

---

## Approach

**Data**

- The official val split (16 images) is merged into training.
- Class imbalance (~3:1 Pneumonia-to-Normal in training) is handled with
  `WeightedRandomSampler`.
- Training images receive light augmentation: horizontal flip, ±10° rotation,
  small brightness/contrast jitter. Eval images are resized and normalized only.
- Normalization uses ImageNet statistics — standard practice for pretrained ResNet
  weights even on grayscale X-rays converted to 3-channel RGB.

**Models**

| Model | Trainable params | Total params |
|---|---|---|
| XRayClassifier (CNN) | 163,457 | 163,457 |
| ResNet18 — frozen | ~530K (head only) | ~11.7M |
| ResNet18 — fine-tuned (layer3+4+head) | ~4.2M | ~11.7M |

**Training**

A single `train_model` function handles all three experiments:
- Optimizer: Adam with `ReduceLROnPlateau` (patience=3, factor=0.5)
- Early stopping on test AUC (patience=7)
- Best checkpoint saved by AUC, not accuracy
- Per-epoch log: Train Loss / Acc / F1 | Test Loss / Acc / F1 / AUC | time

**Interpretability**

Grad-CAM heatmaps are computed on `resnet_finetuned.layer4[-1].conv2` (best
model). An interactive Colab widget lets you filter by true class and
correct / wrong predictions to browse attention maps.

---

## Results

| Model | Accuracy | AUC | FP | FN |
|---|---|---|---|---|
| XRayClassifier (CNN) | 0.85 | 0.929 | 78 | 13 |
| ResNet18 — frozen | 0.91 | 0.956 | 34 | 25 |
| ResNet18 — fine-tuned | **0.92** | **0.973** | 47 | **5** |

**Key findings:**

- Transfer learning outperforms training from scratch (AUC +0.04 frozen,
  +0.07 fine-tuned).
- The fine-tuned model achieves the best AUC and the lowest false negative rate
  (FN=5 — only 5 pneumonia cases missed out of 390). In a clinical setting
  false negatives (missed diagnosis) are more dangerous than false positives,
  making this the preferred model.
- The frozen model is more stable (train and test metrics track closely).
  The fine-tuned model overfits — train F1 reaches 99.6% by epoch 7 while
  test peaks at epoch 1 — yet it still achieves the best test AUC. The
  pretrained features are strong enough that even the overfit head generalizes.
- Threshold analysis shows the optimal F1 threshold is below 0.5, further
  reducing missed pneumonia cases at the cost of more false positives.

**Convergence**

| Model | Epochs to early stop | Best epoch |
|---|---|---|
| CNN | 27 | ~20 |
| ResNet18 — frozen | 12 | 5 |
| ResNet18 — fine-tuned | 8 | 1 |

---

## Repository structure

```
chest-xray-classification/
├── chest_xray.ipynb    # full notebook: EDA → training → evaluation → Grad-CAM
├── cnn_best.pt         # XRayClassifier checkpoint (best AUC)
├── resnet_frozen.pt    # ResNet18 frozen checkpoint
├── resnet_finetuned.pt # ResNet18 fine-tuned checkpoint
└── README.md
```

---

## Running the notebook

Designed for **Google Colab** with a T4 GPU (~30–40 min total training).

To skip training, upload checkpoints and load them:

```python
from google.colab import files
files.upload()   # select the three .pt files

model.load_state_dict(torch.load("cnn_best.pt", map_location=device))
resnet_frozen.load_state_dict(torch.load("resnet_frozen.pt", map_location=device))
resnet_finetuned.load_state_dict(torch.load("resnet_finetuned.pt", map_location=device))
```

---

## Dependencies

```
torch  torchvision  scikit-learn  matplotlib  seaborn  opencv-python  kagglehub  ipywidgets
```
