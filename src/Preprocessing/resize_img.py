import os
from pathlib import Path
from PIL import Image, ImageOps

# Paths are built from this file's location, so it works from any working directory
BASE_DIR = Path(__file__).resolve().parent           # .../src/scripts/Preprocessing
REPO_ROOT = BASE_DIR.parents[1]                      # .../CS4287
input_dir = REPO_ROOT / "CarDD_release" / "CarDD_SOD"
output_dir = BASE_DIR / "Output"
TARGET_SIZE = (384, 576)
EXTENSIONS = (".png", ".jpg")
EXCLUDE_DIRS = {"CarDD-TE-Edge" ,"CarDD-TR-Edge",  "CarDD-VAL-Edge"}

def resize_image(image_path, output_folder, size):
    try:
        with Image.open(image_path) as img:
            # Masks/edge maps are usually single-channel, so pad with an int; colour images use a tuple
            color = 0 if img.mode in ("L", "1", "P") else (0, 0, 0)
            img = ImageOps.pad(img, size, method=Image.Resampling.LANCZOS, color=color)
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            img.save(output_path)
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")

def create_new_output_dir(path):
    os.makedirs(path, exist_ok=True)

def get_output_directory(input_directory):
    relative_path = os.path.relpath(input_directory, input_dir)
    return os.path.join(output_dir, relative_path)

for subdir, dirs, files in os.walk(input_dir):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    for file in files:
        if file.lower().endswith(EXTENSIONS):
            input_file_path = os.path.join(subdir, file)
            output_directory = get_output_directory(subdir)
            create_new_output_dir(output_directory)
            output_file_path = os.path.join(output_directory, file)
            if not os.path.exists(output_file_path):
                resize_image(input_file_path, output_directory, TARGET_SIZE)