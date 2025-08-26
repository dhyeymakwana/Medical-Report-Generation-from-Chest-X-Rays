# src/run_evaluation.py

import torch
import pandas as pd
from tqdm import tqdm
import os
from evaluate import load

# Import from our project files
from src.llm_config import *
# --- CHANGE: Import the new model ---
from src.llm_model import LLM_Vision_Model 
from src.dataloader import create_dataloader

def evaluate_model():
    """
    Loads the best LLM-based model, generates predictions on the test set,
    and computes BLEU and ROUGE scores.
    """
    print("--- Starting Evaluation ---")
    
    # 1. Load Metrics
    bleu_metric = load('bleu')
    rouge_metric = load('rouge')

    # 2. Load Model
    device = DEVICE
    # --- CHANGE: Initialize the new model ---
    model = LLM_Vision_Model().to(device)
    
    # Load the saved LoRA adapter weights
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_llm_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        return
        
    # For PEFT models, you load the adapters like this
    model.llm.load_adapter(checkpoint_path, adapter_name="default")
    # The tokenizer is now part of the main model
    tokenizer = model.tokenizer
    
    # 3. Create Test DataLoader
    test_csv_path = os.path.join(PROCESSED_DATA_DIR, 'test_split.csv')
    test_loader = create_dataloader(csv_path=test_csv_path, is_train=False)

    # 4. Generate Predictions
    predictions = []
    references = []
    
    print("Generating predictions on the test set...")
    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch['image'].to(device)
        ground_truth_ids = batch['input_ids']

        generated_ids = model.generate(images)
        
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        reference_texts = tokenizer.batch_decode(ground_truth_ids, skip_special_tokens=True)
        
        processed_predictions = [" " if not pred.strip() else pred for pred in generated_texts]

        predictions.extend(processed_predictions)
        references.extend([[r] for r in reference_texts])

    # 5. Compute and Print Scores with Robust Error Handling
    print("\n--- Evaluation Results ---")
    
    try:
        bleu_score = bleu_metric.compute(predictions=predictions, references=references)
        print(f"BLEU Score: {bleu_score['bleu']:.4f}")
    except ZeroDivisionError:
        print("BLEU Score calculation failed (ZeroDivisionError).")

    try:
        rouge_score = rouge_metric.compute(predictions=predictions, references=references)
        print("\nROUGE Scores:")
        print(f"  ROUGE-1: {rouge_score['rouge1']:.4f}")
        print(f"  ROUGE-2: {rouge_score['rouge2']:.4f}")
        print(f"  ROUGE-L: {rouge_score['rougeL']:.4f}")
    except Exception as e:
        print(f"\nROUGE Score calculation failed: {e}")

    print("\n--- Evaluation Complete ---")

if __name__ == '__main__':
    evaluate_model()