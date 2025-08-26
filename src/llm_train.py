# src/llm_train.py
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from tqdm import tqdm
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.llm_config import (
    DEVICE, OUTPUT_CSV_PATH, BATCH_SIZE, NUM_EPOCHS,
    LLM_LEARNING_RATE, PROJECTION_LEARNING_RATE, CHECKPOINT_DIR
)
from src.dataloader import create_dataloader
from src.llm_model import LLM_Vision_Model
from transformers import get_cosine_schedule_with_warmup

# ... (train_one_epoch and validate_one_epoch functions remain the same) ...
def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, scheduler):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        if not batch: continue
        
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        decoder_inputs, targets = input_ids[:, :-1], input_ids[:, 1:]
        decoder_attention_mask = attention_mask[:, :-1]

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(images, decoder_inputs, decoder_attention_mask)
            loss = criterion(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1))
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add gradient clipping
        scaler.step(optimizer)
        scaler.update()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item()
        
    return running_loss / len(dataloader) if len(dataloader) > 0 else 0.0

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            if not batch: continue

            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            decoder_inputs, targets = input_ids[:, :-1], input_ids[:, 1:]
            decoder_attention_mask = attention_mask[:, :-1]

            with torch.amp.autocast('cuda'):
                logits = model(images, decoder_inputs, decoder_attention_mask)
                loss = criterion(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1))
            
            running_loss += loss.item()

            # --- SANITY CHECK: Check the model's raw output ---
            if batch_idx == 0:
                print("\n--- SANITY CHECK: Raw Generated IDs ---")
                try:
                    # Generate a report for the first image in the batch
                    # This uses the generate method we fixed earlier
                    generated_ids = model.generate(images[0].unsqueeze(0))
                    print(generated_ids)
                    decoded_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    print("\n--- SANITY CHECK: Decoded Text ---")
                    print(decoded_text)
                except Exception as e:
                    print(f"Error during sanity check: {e}")

    return running_loss / len(dataloader) if len(dataloader) > 0 else 0.0


def main():
    print(f"--- Starting Model Training on {DEVICE} ---")
    df = pd.read_csv(OUTPUT_CSV_PATH)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    
    print("Initializing LLM Vision Model...")
    model = LLM_Vision_Model().to(DEVICE)
    
    train_loader = create_dataloader(train_df, model.tokenizer, is_train=True)
    val_loader = create_dataloader(val_df, model.tokenizer, is_train=False)
    
    optimizer = torch.optim.AdamW([
        {'params': model.llm.parameters(), 'lr': LLM_LEARNING_RATE},
        {'params': model.projection.parameters(), 'lr': PROJECTION_LEARNING_RATE}
    ])
    
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    scaler = torch.amp.GradScaler('cuda')
    
    print("Creating learning rate scheduler...")
    num_training_steps = NUM_EPOCHS * len(train_loader)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # --- CHANGE #1: ADD EARLY STOPPING PARAMETERS ---
    best_val_loss = float('inf')
    patience = 3  # Number of epochs to wait for improvement
    epochs_no_improve = 0
    
    print("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler, scheduler)
        val_loss = validate_one_epoch(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # --- CHANGE #2: IMPLEMENT EARLY STOPPING LOGIC ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0 # Reset counter
            save_path = os.path.join(CHECKPOINT_DIR, "best_llm_model")
            print(f"Validation loss improved. Saving LoRA adapters to {save_path}")
            model.llm.save_pretrained(save_path)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break # Exit the training loop

if __name__ == '__main__':
    main()
