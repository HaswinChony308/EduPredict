"""
download_oulad.py — Download the OULAD Dataset
================================================
This script downloads the Open University Learning Analytics Dataset
(OULAD) and extracts it into the data/ folder.

OULAD is a publicly available dataset containing data about
students, their interactions with the VLE (Virtual Learning
Environment), assessments, and final results.

Source: https://analyse.kmi.open.ac.uk/open_dataset
License: CC-BY 4.0

Usage:
  python data/download_oulad.py
"""

import os
import sys
import urllib.request
import zipfile
import shutil

# Where to save the dataset
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# OULAD download URL (hosted on the Open University analytics site)
OULAD_URL = "https://analyse.kmi.open.ac.uk/open_dataset/download"

# Alternative mirror (in case the main link changes)
OULAD_MIRROR = "https://analyse.kmi.open.ac.uk/open_dataset/downloadDataset"


def download_oulad():
    """
    Download and extract the OULAD dataset.
    
    The dataset is ~60MB zipped, ~400MB unzipped.
    It contains these CSV files:
      - studentInfo.csv       (student demographics + final result)
      - assessments.csv       (assessment metadata)
      - studentAssessment.csv (student scores)
      - studentVle.csv        (VLE interaction clicks)
      - studentRegistration.csv (registration dates)
      - courses.csv           (course info)
      - vle.csv               (VLE activity types)
    """
    zip_path = os.path.join(DATA_DIR, "oulad.zip")
    
    # Check if data already exists
    required_files = ["studentInfo.csv", "studentAssessment.csv",
                      "assessments.csv", "studentVle.csv"]
    all_exist = all(
        os.path.exists(os.path.join(DATA_DIR, f)) for f in required_files
    )
    
    if all_exist:
        print("✓ OULAD dataset files already exist in data/ folder.")
        print("  To re-download, delete the CSV files first.")
        return True
    
    print("=" * 60)
    print("AAPT — Downloading OULAD Dataset")
    print("=" * 60)
    print(f"\nSource: {OULAD_URL}")
    print(f"Destination: {DATA_DIR}")
    print(f"Size: ~60MB (zipped)")
    
    # Try downloading
    urls_to_try = [OULAD_URL, OULAD_MIRROR]
    downloaded = False
    
    for url in urls_to_try:
        try:
            print(f"\n⏳ Downloading from {url}...")
            
            # Show progress during download
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    pct = min(100, block_num * block_size * 100 / total_size)
                    print(f"\r  Progress: {pct:.1f}%", end="", flush=True)
            
            urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
            print("\n  ✓ Download complete!")
            downloaded = True
            break
            
        except Exception as e:
            print(f"\n  ✗ Failed: {e}")
            continue
    
    if not downloaded:
        print("\n❌ Could not download OULAD dataset automatically.")
        print("   Please download manually from:")
        print("   https://analyse.kmi.open.ac.uk/open_dataset")
        print(f"   Extract CSVs to: {DATA_DIR}")
        return False
    
    # Extract the zip file
    try:
        print("\n⏳ Extracting files...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Extract all files to a temp directory first
            temp_dir = os.path.join(DATA_DIR, "_temp_oulad")
            zf.extractall(temp_dir)
            
            # Move CSV files to the data/ directory
            # (the zip might have a subdirectory structure)
            for root, dirs, files in os.walk(temp_dir):
                for fname in files:
                    if fname.endswith(".csv"):
                        src = os.path.join(root, fname)
                        dst = os.path.join(DATA_DIR, fname)
                        shutil.move(src, dst)
                        print(f"  ✓ Extracted: {fname}")
            
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Remove the zip file
        os.remove(zip_path)
        print("\n✅ OULAD dataset ready!")
        print("   Next step: python ml/preprocess.py")
        return True
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        print("   Please extract manually from the zip file.")
        return False


if __name__ == "__main__":
    download_oulad()
