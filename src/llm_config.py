# src/llm_config.py
import os
import torch

# This automatically finds your project's root folder, no matter the OS
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# All other paths are built correctly from the project root
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
IMAGE_DIR = os.path.join(RAW_DATA_DIR, 'images', 'images_normalized')
REPORTS_CSV_PATH = os.path.join(RAW_DATA_DIR, 'indiana_reports.csv')
PROJECTIONS_CSV_PATH = os.path.join(RAW_DATA_DIR, 'indiana_projections.csv')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'validated_data.csv')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'models')

# --- Model & Preprocessing Parameters ---
IMAGE_MODEL = 'vit_base_patch16_224_in21k'
IMAGE_EMBED_DIM = 768 # ViT-Tiny outputs 192
IMAGE_SIZE = (224, 224)

TOKENIZER_MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' # Efficient and powerful model
MAX_TEXT_LENGTH = 256

# --- Training Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32       # A safe and effective size for a good GPU
NUM_WORKERS = 4
LLM_LEARNING_RATE = 1e-5        # A smaller LR for the pre-trained LLM
PROJECTION_LEARNING_RATE = 1e-4 
NUM_EPOCHS = 50

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)