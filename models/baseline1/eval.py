import os
import sys
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import B1_classifier
from data.data_loader import Group_Activity_DataSet
from eval_utils.eval_metrics import eval_model


# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate Baseline1 model")
parser.add_argument(
    "--data-dir",
    default="volleyball-datasets",
    help="Path to dataset root (default: volleyball-datasets under cwd)",
)
parser.add_argument(
    "--model-path",
    default=None,
    help="Path to trained model weights (default: saved_models/best_model_*.pth)",
)
parser.add_argument(
    "--output-dir",
    default="./outputs",
    help="Directory to save confusion matrix (default: ./outputs)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=5,
    help="Number of training epochs (default: 5)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
group_activity_classes = [
    "r_set", "r_spike", "r-pass", "r_winpoint",
    "l_winpoint", "l-pass", "l-spike", "l_set"
]

group_activity_labels = {name: i for i, name in enumerate(group_activity_classes)}


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------------------------
# Dataset path (configurable via --data-dir)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(args.data_dir)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
test_dataset = Group_Activity_DataSet(
    videos_path=os.path.join(PROJECT_ROOT, "volleyball_/videos"),
    annot_path=os.path.join(PROJECT_ROOT, "annot_all.pkl"),
    labels=group_activity_labels,
    split=[4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
    transform=test_transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# ---------------------------------------------------------------------------
# Load Model
# ---------------------------------------------------------------------------
# Determine model path: CLI arg → fallback to saved_models/best_model_*.pth
if args.model_path is not None:
    saved_model_path = args.model_path
else:
    saved_models_dir = "./saved_models"
    # Find the most recent checkpoint
    if os.path.isdir(saved_models_dir):
        ckpt_files = [
            f for f in os.listdir(saved_models_dir) if f.endswith(".pth")
        ]
        if ckpt_files:
            # Sort by timestamp in filename (YYYYMMDD_HHMMSS format)
            ckpt_files.sort(reverse=True)
            saved_model_path = os.path.join(saved_models_dir, ckpt_files[0])
        else:
            saved_model_path = None

if saved_model_path is None:
    print("WARNING: No model checkpoint found. Using randomly initialized weights.")
else:
    print(f"Loading model from: {saved_model_path}")
    model = B1_classifier(num_classes=8)
    model.load_state_dict(torch.load(saved_model_path, map_location=device))
    model.to(device)


# ---------------------------------------------------------------------------
# Evaluation output path (configurable via --output-dir)
# ---------------------------------------------------------------------------
os.makedirs(args.output_dir, exist_ok=True)
save_path = os.path.join(args.output_dir, "confusion_matrix.png")


# =========================
# Eval
# =========================
results = eval_model(
    model=model,
    test_loader=test_loader,
    device=device,
    class_names=group_activity_classes,
    save_path=save_path
)

print(results)