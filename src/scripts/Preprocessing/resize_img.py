import os
from PIL import Image, ImageOps

input_dir = "Input"
output_dir = "Output"

def resize_image(image_path, output_folder, size, color = (0, 0, 0)):
    try:
        with Image.open(image_path) as img:
            img = ImageOps.pad(img, size, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            img.save(output_path)
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")

