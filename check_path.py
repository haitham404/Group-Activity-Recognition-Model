import os
from PIL import Image

base_path = "//volleyball-datasets"

print(f"Base directory exists: {os.path.exists(base_path)}")

if os.path.exists(base_path):
    print("\nContents:")
    print(os.listdir(base_path))


img_path = os.path.join(base_path, "videos", "4", "24745", "24740.jpg")

print(os.path.exists(img_path))

try:
    img = Image.open(img_path)
    img.show()
except Exception as e:
    print(f"Error: {e}")