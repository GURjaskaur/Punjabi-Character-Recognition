import os
import cv2
import numpy as np

# Define input and output directories
input_dir = r"D:\Gurmukhi dataset\Gurmukhi dataset\archive\dataset\gurmukhi"
output_dir = r"D:\Gurmukhi dataset\Gurmukhi dataset\archive\prepossesed_data"
img_size = (128, 128)  # Set your desired image size

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to preprocess images: grayscale, resize, and normalize
def preprocess_image(img_path):
    # Read the image in grayscale mode
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if image was loaded properly
    if img is None:
        print(f"Error: Unable to load image at {img_path}")
        return None  # Skip this image if it couldn't be loaded
    
    # Resize the image to the desired size (128x128)
    resized_img = cv2.resize(img, img_size)
    
    # Normalize the image pixel values to [0, 1]
    normalized_img = resized_img / 255.0
    
    return normalized_img

# List of valid image extensions
valid_extensions = ['.jpg', '.jpeg', '.png']

# Preprocess each image and save the result
for folder in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder)
    output_folder_path = os.path.join(output_dir, folder)
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # Iterate over each file in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is an image based on its extension
        if not any(file_name.lower().endswith(ext) for ext in valid_extensions):
            print(f"Skipping non-image file: {file_name}")
            continue  # Skip non-image files
        
        img_path = os.path.join(folder_path, file_name)
        
        # Preprocess the image
        processed_img = preprocess_image(img_path)
        
        # If image was loaded and processed, save it
        if processed_img is not None:
            save_path = os.path.join(output_folder_path, file_name)
            cv2.imwrite(save_path, (processed_img * 255).astype(np.uint8))  # Save as 8-bit grayscale image

print("Image preprocessing complete!")
