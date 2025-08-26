# src/config.py

import os
import torch

# --- Project Root ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
IMAGE_DIR = os.path.join(RAW_DATA_DIR, 'images', 'images_normalized')
REPORTS_CSV_PATH = os.path.join(RAW_DATA_DIR, 'indiana_reports.csv')
PROJECTIONS_CSV_PATH = os.path.join(RAW_DATA_DIR, 'indiana_projections.csv')

# Processed data path
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'preprocessed_data.csv')
RAG_INDEX_PATH = os.path.join(PROCESSED_DATA_DIR, 'rag_index.faiss')

# --- Model Checkpoint Path ---
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'models')

# --- Preprocessing Parameters ---
IMAGE_SIZE = (224, 224)
TOKENIZER_MODEL = 'dmis-lab/biobert-v1.1'
MAX_TEXT_LENGTH = 256

# --- Model & Training Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16       # Number of samples per batch
NUM_WORKERS = 2       # Number of CPU workers for data loading
LEARNING_RATE = 5e-5  # Optimizer learning rate
NUM_EPOCHS = 50       # Number of times to train on the entire dataset

# --- Ensure directories exist ---
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)