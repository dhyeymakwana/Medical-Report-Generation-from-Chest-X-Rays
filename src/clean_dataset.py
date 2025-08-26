# src/clean_dataset.py
import pandas as pd
import re
from tqdm import tqdm

# Import paths from your config file
from src.config import REPORTS_CSV_PATH, PROJECTIONS_CSV_PATH, OUTPUT_CSV_PATH, IMAGE_DIR

def clean_text(text):
    """
    A more robust function to clean the radiology reports.
    """
    if not isinstance(text, str):
        return ""

    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove all characters except letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Remove common medical abbreviations and noise words
    noise_words = ['xxxx', 'xxx', 'xx', 'xr', 'pa', 'lat', 'ap', 'portable', 'chest']
    for word in noise_words:
        text = text.replace(word, '')

    # 4. Remove single-letter words
    text = re.sub(r'\b[a-z]\b', '', text)
    
    # 5. Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_final_dataframe():
    """
    Loads, merges, cleans, and filters the datasets to create the final preprocessed CSV.
    """
    reports_df = pd.read_csv(REPORTS_CSV_PATH)
    projections_df = pd.read_csv(PROJECTIONS_CSV_PATH)

    merged_df = projections_df.merge(reports_df, on='uid')
    merged_df['image_path'] = IMAGE_DIR + '/' + merged_df['filename']

    print("Cleaning report text...")
    tqdm.pandas()
    merged_df['cleaned_report'] = merged_df['impression'].progress_apply(clean_text)

    # --- FINAL FIX: Filter out very short reports ---
    # We will only keep reports that have more than 5 words after cleaning.
    # This ensures the model trains on meaningful, descriptive examples.
    MIN_WORDS = 5
    merged_df = merged_df[merged_df['cleaned_report'].apply(lambda x: len(x.split()) > MIN_WORDS)]

    final_df = merged_df[['image_path', 'impression', 'cleaned_report']].copy()
    final_df.rename(columns={'impression': 'report_text'}, inplace=True)

    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Final cleaned and filtered data saved to {OUTPUT_CSV_PATH}")
    print(f"Total high-quality samples for training: {len(final_df)}")

if __name__ == '__main__':
    create_final_dataframe()