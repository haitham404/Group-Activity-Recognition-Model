# Group Activity Recognition (Volleyball)
PyTorch code for **group activity recognition** on the Volleyball dataset.

## ⚠️ Work-in-Progress
This repository contains **baseline implementations in progress**. Not all baselines are complete yet - this is intentional to show active development and room for contribution.

## Project Layout

- **`models/baseline1/`**: Baseline1 model (ResNet50) + training script
- **`data/`**: Dataset loading & annotation utilities (NEW)
- **`volleyball-datasets/`**: Dataset folder (videos / annotations) - *required*
- **`saved_models/`**: Saved model weights (created by training)
- **`runs/`**: TensorBoard logs (created by training)
- **`eval_utils/`**: Evaluation metrics (confusion matrix, F1-score, classification report)

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** You will also need the Volleyball dataset. See the Data section below.

## Data Preparation

The dataset should be structured as:

```
volleyball-datasets/
├── videos/
│   ├── 0/
│   │   ├── 24745/
│   │   │   ├── 0.jpg
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   ├── 24740/
│   │   │   ├── 0.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── 1/
│   │   └── ...
│   └── ...
├── annotations.txt  (per clip, see generate_sample_annotations.py)
└── annot_all.pkl    (full annotation pickle)
```

### If you don't have the dataset yet:

1. Place your `volleyball-datasets/` folder in the repo root or specify `--data-dir`
2. Generate sample `annotations.txt` files:
   ```bash
   python generate_sample_annotations.py --data-dir volleyball-datasets
   ```
3. Generate the annotation pickle:
   ```bash
   python -m data.volleyball_annot_loader
   ```

> **Tip**: The repo includes `generate_sample_annotations.py` to create basic annotations automatically, and `data/volleyball_annot_loader.py` (to be created) to convert them into the `.pkl` format.

## Training (Baseline1)

Run training with configurable data directory:

```bash
python models/baseline1/train.py --data-dir volleyball-datasets
```

**CLI arguments:**
- `--data-dir`: Path to dataset root (default: `volleyball-datasets` relative to CWD)
- `--epochs`: Number of epochs (default: 5)
- `--lr`: Learning rate (default: 1e-4)
- `--batch-size`: Batch size (default: 16)

Example with custom paths:
```bash
python models/baseline1/train.py --data-dir /path/to/volleyball-datasets --epochs 10 --lr 1e-3
```

The script will:
1. Load train/val splits automatically
2. Train a ResNet50 baseline (early layers frozen, new FC head for 8 classes)
3. Save best model to `saved_models/best_model_YYYYMMDD_HHMMSS.pth`
4. Log losses/accuracy to TensorBoard at `runs/baseline1_YYYYMMDD_HHMMSS`

## Evaluation

Evaluate a trained model:

```bash
python models/baseline1/eval.py --data-dir volleyball-datasets --model-path saved_models/best_model_*.pth
```

**CLI arguments:**
- `--data-dir`: Path to dataset root
- `--model-path`: Path to trained model weights (auto-detects latest if not specified)
- `--output-dir`: Directory to save confusion matrix (default: `./outputs`)

## Feature Extraction

Extract CNN features from the dataset:

```bash
python extract_features.py --data-dir volleyball-datasets --output-dir features
```

## Repository Status

This project is **actively developed** for the CV lab application. Current status:

| Component | Status |
|-----------|--------|
| Baseline1 (ResNet50) | ✅ Working |
| Data loader | ✅ Implemented |
| Training loop | ✅ Working |
| Evaluation metrics | ✅ Working |
| Baseline2 / Baseline3 | 🚧 In progress |
| Attention models | 📋 Planned |
| Dataset expansion | 📋 Planned |

## Results (Expected)

Baseline1 on Volleyball dataset typically achieves:
- **Validation Accuracy**: ~50-70% (depending on dataset size)
- **Test Accuracy**: ~45-65%

These results improve with more training epochs and data augmentation.

## Citation

If you use this code for your research, please cite accordingly and consider extending the baselines.
