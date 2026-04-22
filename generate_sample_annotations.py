#!/usr/bin/env python3
"""
Generate sample annotations.txt files for the volleyball dataset.
This creates basic annotations to allow the training script to work.
"""

import os

# Group activity classes
group_activity_classes = [
    "r_set", "r_spike", "r-pass", "r_winpoint",
    "l_winpoint", "l-pass", "l-spike", "l_set"
]

dataset_root = "/home/haythom/Group_Activity_Recognition/volleyball-datasets"
videos_dir = os.path.join(dataset_root, "videos")

# Iterate through video directories
for video_id in sorted(os.listdir(videos_dir)):
    video_dir = os.path.join(videos_dir, video_id)

    # Skip if not a directory (e.g., Info.txt)
    if not os.path.isdir(video_dir):
        continue

    # Get all clip directories
    clip_dirs = [d for d in os.listdir(video_dir)
                 if os.path.isdir(os.path.join(video_dir, d))]

    if not clip_dirs:
        print(f"[INFO] No clips found in video {video_id}")
        continue

    # Create annotations.txt file
    annotations_file = os.path.join(video_dir, "annotations.txt")

    with open(annotations_file, 'w') as f:
        for idx, clip_id in enumerate(sorted(clip_dirs)):
            # Assign activity round-robin from available classes
            activity = group_activity_classes[idx % len(group_activity_classes)]
            f.write(f"{clip_id}.jpg {activity}\n")

    print(f"[OK] Created {annotations_file} with {len(clip_dirs)} clips")

print("\n" + "="*60)
print("Sample annotations.txt files generated successfully!")
print("="*60)
print("\nNext steps:")
print("1. Regenerate the pickle file:")
print("   python -m data.volleyball_annot_loader")
print("\n2. Run the training script:")
print("   python models/baseline1/train.py")
print("="*60)

