import os
import sys
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.data_loader import Group_Activity_Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.baseline1.model import B1Classifier


# =========================
# Device (CPU / GPU)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# Classes
# =========================
group_activity_classes = [
    "r_set", "r_spike", "r-pass", "r_winpoint",
    "l_winpoint", "l-pass", "l-spike", "l_set"
]

class_to_idx = {name: i for i, name in enumerate(group_activity_classes)}


# =========================
# Transforms
# =========================
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =========================
# Dataset Path
# =========================
PROJECT_ROOT = r"/home/haythom/Group_Activity_Recognition/volleyball-datasets"


# =========================
# Datasets
# =========================
train_dataset = Group_Activity_Dataset(
    videos_path=f"{PROJECT_ROOT}/videos",
    annot_path=f"{PROJECT_ROOT}/annot_all.pkl",
    labels=class_to_idx,
    split=[1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
    transform=train_transform
)

val_dataset = Group_Activity_Dataset(
    videos_path=f"{PROJECT_ROOT}/videos",
    annot_path=f"{PROJECT_ROOT}/annot_all.pkl",
    labels=class_to_idx,
    split=[0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
    transform=val_transform
)


# =========================
# DataLoaders (CPU optimized)
# =========================
# Check if datasets have samples
if len(train_dataset) == 0:
    print("\n" + "="*60)
    print("ERROR: Training dataset is empty!")
    print("="*60)
    print(f"Train dataset size: {len(train_dataset)}")
    print("\nTo fix this issue:")
    print("1. Ensure annotations.txt files exist in:")
    print("   /home/haythom/Group_Activity_Recognition/volleyball-datasets/videos/{video_id}/")
    print("\n2. Regenerate the pickle file:")
    print("   python -m data.volleyball_annot_loader")
    print("\n3. Then re-run this script")
    print("="*60)
    sys.exit(1)

if len(val_dataset) == 0:
    print("\nWarning: Validation dataset is empty. Using training data for validation.")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset if len(val_dataset) > 0 else train_dataset,
                         batch_size=16, shuffle=False, num_workers=0)


# =========================
# Model
# =========================
model = B1Classifier(num_classes=8).to(device)

# Loss + Optimizer (IMPORTANT: define ONCE)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


# =========================
# TensorBoard
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"./runs/baseline1_{timestamp}"
writer = SummaryWriter(log_dir)


# =========================
# Train One Epoch
# =========================
def train_one_epoch(model, loader):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)

        # Convert one-hot labels to class indices
        if labels.dim() > 1 and labels.size(1) > 1:
            labels = labels.argmax(dim=1)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# =========================
# Validation
# =========================
def evaluate(model, loader):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)

            # Convert one-hot labels to class indices
            if labels.dim() > 1 and labels.size(1) > 1:
                labels = labels.argmax(dim=1)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# =========================
# Training Loop
# =========================
EPOCHS = 5
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

    # TensorBoard logging
    writer.add_scalars("Loss", {
        "train": train_loss,
        "val": val_loss
    }, epoch)

    writer.add_scalars("Accuracy", {
        "train": train_acc,
        "val": val_acc
    }, epoch)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        save_dir = "./saved_models"
        os.makedirs(save_dir, exist_ok=True)

        model_path = os.path.join(save_dir, f"best_model_{timestamp}.pth")
        torch.save(model.state_dict(), model_path)

        print("Model saved:", model_path)


writer.close()