import os
from PIL import Image

# Path to your dataset
input_dir = "C:\\Users\\acer\\OneDrive\\Desktop\\Road Images\\road_img\\resized_potholes"
output_dir = "C:\\Users\\acer\\OneDrive\\Desktop\\Road Images\\road_img\\more_pots"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define rotation angles
angles = [45, 135, 225, 315]

# Valid image extensions
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

count = 0

for filename in os.listdir(input_dir):
    if filename.lower().endswith(valid_extensions):
        input_path = os.path.join(input_dir, filename)
        with Image.open(input_path) as img:
            # Ensure consistent format (RGB)
            img = img.convert("RGB")

            for angle in angles:
                rotated = img.rotate(angle, expand=True)
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_rot{angle}{ext}"
                output_path = os.path.join(output_dir, new_filename)

                rotated.save(output_path)
                count += 1

print(f"✅ Done! Created {count} rotated images from {len(os.listdir(input_dir))} originals.")
print(f"📁 Rotated images saved in: {output_dir}")
