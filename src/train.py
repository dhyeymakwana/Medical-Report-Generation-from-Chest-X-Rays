# src/train.py

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Import from our other project files
from src.config import *
from src.dataloader import create_dataloader
# --- CHANGE: Import the new model ---
from src.model import LLM_Vision_Model

# (The train_one_epoch and validate_one_epoch functions are the same as the corrected version)
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        decoder_inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        optimizer.zero_grad()
        outputs = model(images, decoder_inputs)
        loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), targets.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            decoder_inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            outputs = model(images, decoder_inputs)
            loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), targets.contiguous().view(-1))
            running_loss += loss.item()
    return running_loss / len(dataloader)

def main():
    """
    Main function to orchestrate the training and validation process.
    """
    print(f"--- Starting Model Training on {DEVICE} ---")

    # (Data splitting is the same)
    df = pd.read_csv(OUTPUT_CSV_PATH)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_split.csv'), index=False)
    print(f"Data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test.")

    # Create DataLoaders
    train_loader = create_dataloader(train_df, tokenizer=model.tokenizer, is_train=True)
    val_loader = create_dataloader(val_df, tokenizer=model.tokenizer, is_train=False)
    # --- CHANGE: Initialize the new LLM_Vision_Model ---
    print("Initializing LLM Vision Model...")
    model = LLM_Vision_Model().to(DEVICE)
    
    # The tokenizer is now part of the model
    tokenizer = model.tokenizer
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # (The training loop is the same)
    best_val_loss = float('inf')
    print("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate_one_epoch(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(CHECKPOINT_DIR, "best_llm_model.pth")
            # Save only the trainable LoRA parameters
            model.llm.save_pretrained(save_path)
            print(f"Validation loss improved. Model saved to {save_path}")

    print("--- Training Complete ---")

if __name__ == '__main__':
    main()