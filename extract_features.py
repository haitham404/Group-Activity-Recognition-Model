import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from constants import actions, num_features
from models.baseline3.model import Person_Activity_Classifier
from utils.data_utils import load_annotations, splits


# =========================
# Device setup
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Paths (UPDATED FOR YOUR MACHINE)
# =========================
base_dir = "//volleyball-datasets"

videos_dir = os.path.join(base_dir, "videos")
annotations_dir = os.path.join(base_dir, "volleyball_tracking_annotation")
features_dir = os.path.join(base_dir, "features")

# Load dataset annotations
annotations = load_annotations(videos_dir, annotations_dir)


# =========================
# Image preprocessing
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =========================
# Load pretrained model
# =========================
model = Person_Activity_Classifier(len(actions))

# Load trained weights
try:
    model.load_state_dict(
        torch.load(
            "/mnt/New Volume/Deep Learning. DR Mostafa/Group_Activity_Recognition/models/baseline2/model_20250309_005202_4",
            map_location=device
        )
    )
    print("Model weights loaded successfully.")
except FileNotFoundError:
    print("Warning: Model weights not found. Using randomly initialized weights.")

# Remove classification head → keep feature extractor only
model = nn.Sequential(*list(model.children())[:-1]).to(device)
model.eval()


# =========================
# Feature extraction
# =========================
with torch.no_grad():

    for split in splits:
        print(f"Processing split: {split}")

        split_features = defaultdict(dict)

        for video_id in splits[split]:
            if str(video_id) not in annotations:
                print(f"Warning: Video {video_id} not found in annotations. Skipping.")
                continue
            annotation = annotations[str(video_id)]

            for clip_id in annotation.clip_activities:

                clip_activity = annotation.clip_activities[clip_id]
                clip_features = []

                for frame_id, boxes in annotation.clip_annotations[clip_id].frame_annotations.items():

                    # Build image path
                    image_path = os.path.join(
                        videos_dir,
                        str(video_id),
                        clip_id,
                        f"{frame_id}.jpg"
                    )

                    image = Image.open(image_path).convert("RGB")

                    cropped_images = []

                    # Crop all players in frame
                    for box in boxes:
                        cropped_image = image.crop((
                            box.x,
                            box.y,
                            box.w,
                            box.h
                        ))

                        cropped_images.append(
                            transform(cropped_image).unsqueeze(0)
                        )

                    # Stack players in one tensor: [num_players, 3, 224, 224]
                    cropped_images = torch.cat(cropped_images).to(device)

                    # Extract features using CNN backbone
                    features = model(cropped_images)

                    # Flatten to [num_players, feature_dim]
                    features = features.view(cropped_images.size(0), -1)

                    # Aggregate player features (mean pooling)
                    features = features.mean(dim=0)

                    clip_features.append(features)

                # Stack frames → [num_frames, feature_dim]
                clip_features = torch.stack(clip_features)

                # Store features + label
                split_features[video_id][clip_id] = {
                    "features": clip_features.cpu(),
                    "label": clip_activity
                }

        # =========================
        # Save features to disk
        # =========================
        os.makedirs(features_dir, exist_ok=True)

        output_path = os.path.join(features_dir, f"{split}_features.pkl")

        with open(output_path, "wb") as f:
            pickle.dump(split_features, f)

        print(f"Saved: {output_path}")