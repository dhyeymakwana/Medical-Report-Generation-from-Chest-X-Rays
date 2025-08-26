# src/run_preprocessing.py

import pandas as pd
from tqdm import tqdm
import os

# Import from other modules
from src.llm_config import IMAGE_DIR, REPORTS_CSV_PATH, PROJECTIONS_CSV_PATH, OUTPUT_CSV_PATH
from src.dataset import create_master_df
from src.preprocess import clean_text

def main():
    """
    Main function to execute the full data preprocessing pipeline in the correct order.
    """
    print("--- Starting Data Preprocessing Pipeline ---")

    # --- Step 1: Create the initial DataFrame ---
    try:
        master_df = create_master_df(
            reports_csv_path=REPORTS_CSV_PATH,
            projections_csv_path=PROJECTIONS_CSV_PATH,
            image_dir=IMAGE_DIR
        )
        print(f"Successfully created initial DataFrame with {len(master_df)} pairs.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # --- Step 2: Verify all image paths exist and filter out invalid ones ---
    print("\nVerifying image paths exist...")
    initial_count = len(master_df)
    
    # We use .apply here, progress_apply is not necessary as this is very fast
    master_df = master_df[master_df['image_path'].apply(os.path.exists)].copy()
    
    final_count = len(master_df)
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} records with missing image files.")

    # --- Step 3: Clean the report text (only once) ---
    print("\nCleaning report text...")
    master_df['cleaned_report'] = master_df['report_text'].apply(clean_text)
    
    # --- Step 4: Filter out excessively short reports ---
    print("\nFiltering out excessively short reports...")
    initial_count = len(master_df)
    
    master_df['report_word_count'] = master_df['cleaned_report'].str.split().str.len()
    master_df = master_df[master_df['report_word_count'] >= 5].copy()
    
    final_count = len(master_df)
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} records with reports shorter than 5 words.")

    # --- Step 5: Save the final preprocessed data ---
    try:
        # Drop the temporary 'report_word_count' column before saving
        final_df_to_save = master_df[['image_path', 'report_text', 'cleaned_report']]
        final_df_to_save.to_csv(OUTPUT_CSV_PATH, index=False)
        
        print(f"\n--- Preprocessing Complete ---")
        print(f"Final data saved to: {OUTPUT_CSV_PATH}")
        print(f"Total processed records: {len(final_df_to_save)}")
    except Exception as e:
        print(f"Error saving the final CSV file: {e}")

if __name__ == '__main__':
    main()