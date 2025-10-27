import os
from PIL import Image

# Path to the directory containing your images
input_dir = "C:\\Users\\acer\\OneDrive\\Desktop\\Road Images\\road_img\\Pothole"
output_dir = "C:\\Users\\acer\\OneDrive\\Desktop\\Road Images\\road_img\\resized_potholes"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Desired size
new_size = (800, 800)

# Supported image formats
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

for filename in os.listdir(input_dir):
    if filename.lower().endswith(valid_extensions):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Open image
        with Image.open(input_path) as img:
            # Resize (will distort aspect ratio)
            resized_img = img.resize(new_size)

            # Save resized image
            resized_img.save(output_path)

        print(f"Resized: {filename}")

print("✅ All images have been resized to 800x800 pixels.")
