# src/build_rag_index.py

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from tqdm import tqdm

from src.config import PROCESSED_DATA_DIR, RAG_INDEX_PATH

def build_index():
    """
    Creates a FAISS vector index from the training reports for RAG.
    """
    print("--- Building RAG Knowledge Base ---")
    
    # 1. Load the training data
    train_csv_path = os.path.join(PROCESSED_DATA_DIR, 'train_split.csv')
    if not os.path.exists(train_csv_path):
        print(f"Error: Training data not found at {train_csv_path}. Please run the training script once to create it.")
        return
    
    train_df = pd.read_csv(train_csv_path)
    reports = train_df['cleaned_report'].dropna().tolist()
    
    # 2. Load a sentence embedding model
    print("Loading sentence embedding model...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Encode all reports into vectors
    print("Encoding reports into vectors...")
    report_vectors = encoder.encode(reports, show_progress_bar=True, convert_to_numpy=True)
    
    # 4. Build and save the FAISS index
    index = faiss.IndexFlatL2(report_vectors.shape[1])
    index.add(report_vectors.astype(np.float32))
    
    faiss.write_index(index, RAG_INDEX_PATH)
    
    print(f"\n--- RAG Index Building Complete ---")
    print(f"Index with {len(reports)} reports saved to: {RAG_INDEX_PATH}")

if __name__ == '__main__':
    build_index()