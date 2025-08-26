# In src/dataset.py

import pandas as pd
import os

def create_master_df(reports_csv_path, projections_csv_path, image_dir):
    """
    Creates a master DataFrame by merging report and projection CSVs,
    and correctly constructs the '.dcm.png' image file paths.
    """
    reports_df = pd.read_csv(reports_csv_path)
    projections_df = pd.read_csv(projections_csv_path)

    # Filter for frontal images and merge
    frontal_projections_df = projections_df[projections_df['projection'] == 'Frontal'].copy()
    merged_df = pd.merge(reports_df, frontal_projections_df, on='uid')

    # --- THIS IS THE FIX ---
    # Directly replace the .png extension with .dcm.png to match the actual files
    merged_df['image_path'] = merged_df['filename'].apply(
        lambda x: os.path.join(image_dir, x.replace('.png', '.dcm.png'))
    )
    
    # Combine impression and findings into the 'report_text' column
    merged_df['report_text'] = merged_df['impression'].fillna('') + " " + merged_df['findings'].fillna('')
    
    # Select and clean up the final DataFrame
    final_df = merged_df[['image_path', 'report_text']].copy()
    final_df = final_df[final_df['report_text'].str.strip() != '']
    final_df.dropna(subset=['image_path'], inplace=True)

    return final_df