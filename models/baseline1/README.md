# Baseline1 (B1) - Group Activity Recognition Model

## What is the Model?

**Baseline1 (B1)** is a simple CNN-based model for group activity recognition in volleyball videos.

### Architecture:
- **Backbone**: ResNet50 (pre-trained on ImageNet)
- **Frozen Layers**: Early layers (layer1) are frozen to preserve low-level features
- **Classification Head**: Custom fully-connected layer that maps ResNet50 features to 8 group activity classes
- **Input**: Single frame (224×224 RGB images)
- **Output**: Class logits for 8 volleyball group activities

### Key Features:
- Uses transfer learning from ImageNet
- Processes full frames (no person cropping)
- Single frame input (no temporal information)

---

## Accuracy Comparison

### Original Paper (CVPR 2016)
- **Dataset**: Collective Activity Dataset (CAD)
- **B1 Accuracy**: **63.0%**
- **Method**: Single frame CNN baseline on full dataset

### Current Implementation
- **Dataset**: Partial Volleyball dataset (only 2 videos, 36 samples)
- **Training Accuracy**: 94.44%
- **Validation Accuracy**: 77.78%
- **Note**: Results are not comparable due to limited data. Full evaluation requires complete dataset with proper annotations.

---

## Group Activity Classes (8 classes)
- `r_set`, `r_spike`, `r-pass`, `r_winpoint`
- `l_winpoint`, `l-pass`, `l-spike`, `l_set`

---

## Usage
```bash
cd /home/haythom/Group_Activity_Recognition
python models/baseline1/train.py
```

---

## References
Ibrahim, M. S., et al. "A Hierarchical Deep Temporal Model for Group Activity Recognition." CVPR 2016.

**B1 Accuracy in Original Paper: 63.0%**
