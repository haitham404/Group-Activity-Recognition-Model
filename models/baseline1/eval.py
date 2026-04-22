import sys
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import B1_classifier
from data.data_loader import Group_Activity_DataSet
from eval_utils.eval_metrics import eval_model


# =========================
# Paths
# =========================
PROJECT_ROOT = r"/mnt/New Volume/Deep Learning. DR Mostafa/05- Project 1 - CV/volleyball-datasets"

sys.path.append(PROJECT_ROOT)


# =========================
# Classes
# =========================
group_activity_classes = [
    "r_set", "r_spike", "r-pass", "r_winpoint",
    "l_winpoint", "l-pass", "l-spike", "l_set"
]

group_activity_labels = {name: i for i, name in enumerate(group_activity_classes)}


# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================
# Transform
# =========================
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# Dataset
# =========================
test_dataset = Group_Activity_DataSet(
    videos_path=f"{PROJECT_ROOT}/volleyball_/videos",
    annot_path=f"{PROJECT_ROOT}/annot_all.pkl",
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


# =========================
# Load Model
# =========================
saved_model_path = r"D:\Deep Learning. DR Mostafa\Group_Activity_Recognition\models\baseline1\outputs\trained_model\model_20250216_184456_4"

model = b1_classifier(num_classes=8)
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.to(device)


# =========================
# Eval
# =========================
save_path = r"/mnt/New Volume/Deep Learning. DR Mostafa/Group_activity_project/models/baseline1/outputs/confusion_matrix.png"

results = eval_model(
    model=model,
    test_loader=test_loader,
    device=device,
    class_names=group_activity_classes,
    save_path=save_path
)

print(results)