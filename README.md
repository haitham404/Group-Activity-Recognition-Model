# Group Activity Recognition (Volleyball) — Simple Baselines

PyTorch code for **group activity recognition** on the Volleyball dataset, with a simple baseline model (`models/baseline1`).

## Project layout

- **`models/baseline1/`**: Baseline1 model + training script
- **`data/`**: dataset loading + annotation utilities
- **`volleyball-datasets/`**: dataset folder (videos / annotations) expected by the code
- **`saved_models/`**: saved model weights (created by training)
- **`runs/`**: TensorBoard logs (created by training)

## Setup

Create a virtual env and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: you will also need PyTorch / torchvision (not listed in `requirements.txt` in this repo).

## Data

The training script expects the dataset to exist under:

- `volleyball-datasets/videos/`
- `volleyball-datasets/annot_all.pkl`

If your dataset lives somewhere else, update `PROJECT_ROOT` in:

- `models/baseline1/train.py`

## Train (Baseline1)

- checks it's readme for details on the model and results

```bash
python models/baseline1/train.py
```


