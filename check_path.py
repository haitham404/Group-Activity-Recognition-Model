import os
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description="Check dataset paths")
parser.add_argument("--data-dir", default="volleyball-datasets", help="Dataset root directory")
args = parser.parse_args()

base_path = args.data_dir

print(f"Base directory exists: {os.path.exists(base_path)}")

if os.path.exists(base_path):
    print("\nContents:")
    print(os.listdir(base_path))


img_path = os.path.join(base_path, "videos", "4", "24745", "24740.jpg")

print(os.path.exists(img_path))

try:
    img = Image.open(img_path)
    print(f"Image loaded successfully, size: {img.size}")
except Exception as e:
    print(f"Error: {e}")
