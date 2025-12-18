# File: data_preprocessing/preprocess_images.py
# Author: EA-RTB Research Implementation
# Purpose: Preprocess ad images and prepare metadata for FoMO detection

import os
from PIL import Image
import pandas as pd

# Directories
RAW_IMAGE_DIR = "../dataset/images/"                # downloaded images from landing pages
PREPROCESS_DIR = "../dataset/images_preprocessed/" # preprocessed images
os.makedirs(PREPROCESS_DIR, exist_ok=True)

# Load metadata containing URLs, filenames, and optional ad text
METADATA_CSV = "../dataset/metadata.csv"
metadata = pd.read_csv(METADATA_CSV)

processed_records = []

# Loop through all images
for idx, row in metadata.iterrows():
    filename = row['filename']
    url = row['url']
    text = row.get('text', '')  # Optional ad text

    raw_path = os.path.join(RAW_IMAGE_DIR, filename)
    preprocess_path = os.path.join(PREPROCESS_DIR, filename)

    try:
        # Open image
        img = Image.open(raw_path).convert('RGB')
        
        # Resize to 224x224 (ViT/ResNet input size)
        img = img.resize((224, 224))
        
        # Normalize pixel values [0-1] and save
        img.save(preprocess_path)

        # Add to processed metadata
        processed_records.append({
            'url': url,
            'image_path': preprocess_path,
            'text': text
        })

    except Exception as e:
        print(f"Failed to process {filename}: {e}")

# Save processed metadata
processed_df = pd.DataFrame(processed_records)
processed_df.to_csv("../dataset/metadata_preprocessed.csv", index=False)

print("✅ Preprocessing complete: images resized and metadata saved.")
