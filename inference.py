# src/inference.py

import torch
import cv2
import argparse
import os

# Import from our project files
from src.llm_config import *
from src.llm_model import LLM_Vision_Model # Import the new model
from src.transforms import get_transforms

def run_inference(image_path, checkpoint_path):
    """
    Loads the trained LLM-based model and generates a radiology report for a single image.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # 1. Load Model
    device = DEVICE
    print("Loading model...")
    model = LLM_Vision_Model().to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return
        
    # Load the saved LoRA adapter weights
    model.llm.load_adapter(checkpoint_path, adapter_name="default")
    tokenizer = model.tokenizer
    
    # 2. Preprocess Image
    print("Preprocessing image...")
    transforms = get_transforms(is_train=False, image_size=IMAGE_SIZE)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    processed_image = transforms(image=image)['image'].unsqueeze(0).to(device)

    # 3. Generate Report
    print("Generating report...")
    generated_ids = model.generate(processed_image)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print("\n--- Generated Report ---")
    print(generated_text)

if __name__ == '__main__':
    # --- How to run this script from the terminal ---
    # Example:
    # python src/inference.py --image_path /path/to/your/image.png

    parser = argparse.ArgumentParser(description="Generate a radiology report for a single X-ray image.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input chest X-ray image.')
    
    args = parser.parse_args()

    # Make sure to use the correct checkpoint name
    checkpoint_file = os.path.join(CHECKPOINT_DIR, "best_llm_model.pth")
    run_inference(args.image_path, checkpoint_file)