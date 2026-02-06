#!/usr/bin/env python
# coding: utf-8
"""
#**Script to pool images**
Author : Ana Valladares
"""
# Libraries
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np

# Get source' paths
BASE_DIR = Path("/content/drive/MyDrive/Fotos_Scanner")
INPUT_DIR = Path("/content/drive/MyDrive/Fotos_Scanner/Analyze/inputs")
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# Get folder's date
def parse_date_from_folder(folder_name):
    match = re.search(r"\d{2}-\d{2}-\d{2}", folder_name)
    if match:
        return datetime.strptime(match.group(), "%d-%m-%y")
    return None

# Get folders in date range
def get_folders_in_date_range(base_dir, start_date, end_date):
    selected = []

    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue

        folder_date = parse_date_from_folder(folder.name)
        if folder_date and start_date <= folder_date <= end_date:
            selected.append(folder)

    return selected

# Function to clean
def clear_directory(folder):
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

# Function for counting images 
def count_images(folders):
    total = 0
    for folder in folders:
        total += len(list(folder.glob("*.jpg")))
    return total

# Function to deal with different scanner DPI
def normalize_scanner_image(img_path, target_px_per_cm=119):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]

    # --- try reading DPI metadata ---
    dpi = None
    try:
        with Image.open(img_path) as pil_img:
            dpi = pil_img.info.get("dpi", None)
    except:
        pass

    scanner_type = "old"
    scale_factor = 1.0

    # --- detect scanner ---
    if dpi is not None:
        if dpi[0] >= 1000:   # new scanner ~1200 dpi
            scanner_type = "new"
    else:
        # fallback using dimension heuristic
        if max(h, w) > 2000:
            scanner_type = "new"

    # --- apply scaling ---
    if scanner_type == "new":
        scale_factor = 119 / 471

        img = cv2.resize(
            img,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_AREA
        )

    return img, scanner_type, scale_factor

# Function to download images
def pool_images(folders, input_dir, pbar):
    for folder in folders:
        N = 8
        new_folder_name =  folder.name[-N:]

        for img_path in folder.glob("*"):
            if img_path.suffix.lower() not in [".jpg"]:
                continue

            new_name = f"{new_folder_name}_{img_path.name}"
            out_path = input_dir / new_name

            # Read image
            img_resized, scanner, factor = normalize_scanner_image(img_path)
                
            # Save image
            cv2.imwrite(str(out_path), img_resized)

            pbar.update(1)
            
# Ask user for start and end dates
def main():
    start_date_str = input("Enter start date (dd-mm-yy): ")
    end_date_str = input("Enter end date (dd-mm-yy): ")

    start_date = datetime.strptime(start_date_str, "%d-%m-%y")
    end_date = datetime.strptime(end_date_str, "%d-%m-%y")

    folders = get_folders_in_date_range(BASE_DIR, start_date, end_date)

    print("Selected folders:")
    for f in folders:
        print(" -", f.name)
        
    print("Clearing previous inputs...")
    clear_directory(INPUT_DIR)

    total_images = count_images(folders)
    print(f"\n Pooling {total_images} images...\n")

    with tqdm(total=total_images, desc="Downloading images", unit="img") as pbar:
        pool_images(folders, INPUT_DIR, pbar)

    print("\n✅ Images pooled into Analyze/inputs")

if __name__ == "__main__":
    main()