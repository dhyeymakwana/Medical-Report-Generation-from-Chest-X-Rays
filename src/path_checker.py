# src/path_checker.py
import pandas as pd
import cv2
import os
from tqdm import tqdm

# Import the paths from your configuration
from src.config import PROCESSED_DATA_DIR

def validate_dataset(input_csv_path, output_csv_path):
    """
    Reads a CSV file, checks if each image path is valid and readable,
    and saves a new CSV with only the valid entries.
    """
    if not os.path.exists(input_csv_path):
        print(f"Error: Input file not found at {input_csv_path}")
        return

    df = pd.read_csv(input_csv_path)
    valid_rows = []
    
    print(f"Validating {len(df)} image paths...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        image_path = row['image_path']
        # Check if the file exists and can be opened by OpenCV
        if os.path.exists(image_path):
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    valid_rows.append(row)
            except Exception as e:
                # This will catch other potential read errors
                print(f"Skipping corrupt file at {image_path}: {e}")
    
    validated_df = pd.DataFrame(valid_rows)
    validated_df.to_csv(output_csv_path, index=False)
    
    print("\nValidation complete.")
    print(f"Original samples: {len(df)}")
    print(f"Valid samples: {len(validated_df)}")
    print(f"Removed samples: {len(df) - len(validated_df)}")
    print(f"Validated data saved to: {output_csv_path}")

if __name__ == '__main__':
    # Using the preprocessed file you last showed me
    input_path = os.path.join(PROCESSED_DATA_DIR, 'preprocessed_data.csv')
    output_path = os.path.join(PROCESSED_DATA_DIR, 'validated_data.csv')
    validate_dataset(input_path, output_path)