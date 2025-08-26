# src/preprocess.py

import cv2
import re
from skimage import exposure

def process_image(image_path, target_size):
    """
    Loads, enhances with CLAHE, resizes, and normalizes a single image.

    Args:
        image_path (str): The full path to the image file.
        target_size (tuple): The target (width, height) for the image.

    Returns:
        np.ndarray: The processed image as a NumPy array, or None if loading fails.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not read image at {image_path}. Skipping.")
            return None

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)

        # Resize and normalize
        resized_image = cv2.resize(enhanced_image, target_size)
        normalized_image = resized_image / 255.0
        
        return normalized_image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def clean_text(text):
    """
    Cleans and normalizes a string of text. Steps include:
    1. Convert to lowercase.
    2. Remove special characters, numbers, and punctuation.
    3. Remove extra whitespace.

    Args:
        text (str): The raw report text.

    Returns:
        str: The cleaned report text.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text