from PIL import Image
import os

def crop_images_patch(image_paths, output_folder, crop_area):
    """
    Crops a specific patch from a list of images and saves the results.

    Args:
        image_paths (list): A list of full paths to the input images.
        output_folder (str): The folder where the cropped images will be saved.
        crop_area (tuple): A 4-tuple (left, upper, right, lower) defining the
                           crop region in pixels.
    """
    # 1. Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # 2. Iterate through each image path
    for i, path in enumerate(image_paths):
        try:
            # Open the image
            img = Image.open(path)
            
            # Crop the image using the defined coordinates
            cropped_img = img.crop(crop_area)
            
            # Create a unique filename for the output
            base_name = os.path.basename(path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(output_folder, f"{name}_patch{ext}")
            
            # Save the cropped image
            cropped_img.save(output_path)
            print(f"Successfully cropped and saved: {output_path}")

        except FileNotFoundError:
            print(f"Error: Image not found at {path}")
        except Exception as e:
            print(f"An error occurred while processing {path}: {e}")

# --- Configuration ---

# 1. Define the coordinates for the patch
# The format is a tuple: (left_x, top_y, right_x, bottom_y)
# Example: a 100x100 patch starting at (50, 50)
CROP_COORDINATES = (390, 520, 470, 580)

# 2. Define the image paths
# REPLACE THESE WITH THE ACTUAL PATHS TO YOUR 4 IMAGES
IMAGE_FILES = [
    "/home/mohammad/Documents/uni/deeplearning/FinalProject/data/chosen2/011/00000005.png",
    "/home/mohammad/Documents/uni/deeplearning/FinalProject/data/chosen2/011/dig_00000005.png",
    "/home/mohammad/Documents/uni/deeplearning/FinalProject/data/chosen2/011/gen_00000005.png",
    "/home/mohammad/Documents/uni/deeplearning/FinalProject/data/chosen2/011/wgen_00000005.png",
]

# 3. Define the output folder
OUTPUT_DIRECTORY = "/home/mohammad/Documents/uni/deeplearning/FinalProject/data/chosen2/011/cropped_patches"

# --- Execute the function ---
crop_images_patch(IMAGE_FILES, OUTPUT_DIRECTORY, CROP_COORDINATES)