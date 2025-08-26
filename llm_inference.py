# src/inference.py

import torch
import cv2
import argparse
import os

# Import from our project files
# --- CHANGE #1: Import from llm_config and the correct llm_model ---
from src.llm_config import DEVICE, IMAGE_SIZE, CHECKPOINT_DIR
from src.llm_model import LLM_Vision_Model 
from src.transforms import get_transforms

def run_inference(image_path, checkpoint_dir):
    """
    Loads the trained LLM-based model and generates a radiology report for a single image.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # 1. Load the base model
    device = DEVICE
    print("Initializing base LLM Vision Model...")
    model = LLM_Vision_Model().to(device)
    
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Model checkpoint directory not found at {checkpoint_dir}")
        return
        
    # Load the fine-tuned LoRA adapter weights into the model
    print(f"Loading fine-tuned adapters from {checkpoint_dir}...")
    # The load_adapter method from PEFT handles loading the weights from the directory
    model.llm.load_adapter(checkpoint_dir, adapter_name="default")
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
    parser = argparse.ArgumentParser(description="Generate a radiology report for a single X-ray image.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input chest X-ray image.')
    args = parser.parse_args()

    # --- CHANGE #2: The path should be to the directory, not a .pth file ---
    checkpoint_directory = os.path.join(CHECKPOINT_DIR, "best_llm_model")
    
    run_inference(args.image_path, checkpoint_directory)