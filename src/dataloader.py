# src/dataloader.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# Import from our other project files
from src.config import TOKENIZER_MODEL, MAX_TEXT_LENGTH, BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE
from src.transforms import get_transforms

class CXRDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the IU X-Ray data.
    This dataset returns the processed image and the raw text report.
    Tokenization is handled by the collate_fn in the DataLoader.
    """
    def __init__(self, df, is_train=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the image paths and reports.
            is_train (bool): Flag to determine whether to apply training augmentations.
        """
        self.df = df
        self.transforms = get_transforms(is_train, image_size=IMAGE_SIZE)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- Image Processing ---
        image_path = row['image_path']
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"\nWarning: Could not read image at {image_path}. Returning None.")
            return None 
            
        # Apply transforms if available
        if self.transforms:
            # For albumentations transforms
            if hasattr(self.transforms, '__call__'):
                augmented = self.transforms(image=image)
                image = augmented['image']
            # For torchvision transforms
            else:
                image = self.transforms(image)
        
        # Ensure image is a tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
            # Add channel dimension if missing (H, W) -> (1, H, W)
            if image.dim() == 2:
                image = image.unsqueeze(0)
            
        # --- Text ---
        # Get the raw cleaned report text and ADD A PREFIX
        report = row['cleaned_report']
        
        # CRITICAL: Add a prefix to guide the generation
        prefix = "Generate a chest X-ray report: "
        full_text = prefix + report
        
        return {
            "image": image,
            "report": full_text  # Return the prefixed string
        }

def collate_fn(batch, tokenizer):
    """
    Custom collate function.
    It tokenizes the text reports for the entire batch at once.
    """
    # Filter out any None items that may have resulted from bad image paths
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Stack images (they should already be tensors from __getitem__)
    images = torch.stack([item['image'] for item in batch])
    reports = [item['report'] for item in batch]
    
    # Tokenize the reports
    tokenized_output = tokenizer(
        reports,
        padding='max_length', 
        truncation=True, 
        max_length=MAX_TEXT_LENGTH, 
        return_tensors='pt'
    )
    
    input_ids = tokenized_output['input_ids']
    attention_mask = tokenized_output['attention_mask']

    return {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

def create_dataloader(df, tokenizer, is_train=True):
    """
    Utility function to create a PyTorch DataLoader.
    """
    dataset = CXRDataset(df, is_train)
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=is_train,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        # Use a lambda function to pass the tokenizer to the collate_fn
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    return dataloader