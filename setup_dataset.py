"""
Dataset Setup & Verification Utility for FER-2013 Facial Emotion Recognition
=============================================================================
This script helps you set up, verify, or generate a sample dataset for the
Facial Emotion Recognition system.

Usage:
    python setup_dataset.py --check          # Check if dataset is properly organized
    python setup_dataset.py --download       # Attempt automatic Kaggle API download
    python setup_dataset.py --create-dummy   # Create synthetic demo images for quick testing
"""

import os
import sys
import shutil
import zipfile
import argparse
import numpy as np
from PIL import Image, ImageDraw

# Ensure Windows console handles UTF-8 output safely
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Standard 7 emotion categories in FER-2013
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")


def check_dataset_layout(print_summary=True):
    """
    Checks if dataset/train and dataset/test exist and contains the 7 emotion classes.
    Returns:
        bool: True if dataset is valid and non-empty, False otherwise.
    """
    if print_summary:
        print("\n==================================================")
        print("         FER-2013 DATASET LAYOUT CHECK           ")
        print("==================================================")
        print(f"Target dataset directory: {DATASET_DIR}")

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        if print_summary:
            print("[!] Missing folders: 'dataset/train' or 'dataset/test' does not exist.")
        return False

    train_counts = {}
    test_counts = {}
    total_train = 0
    total_test = 0

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # Count images per class
    for emotion in EMOTION_CLASSES:
        emotion_train_dir = os.path.join(TRAIN_DIR, emotion)
        emotion_test_dir = os.path.join(TEST_DIR, emotion)

        # Train count
        if os.path.exists(emotion_train_dir):
            c_train = len([
                f for f in os.listdir(emotion_train_dir)
                if os.path.splitext(f.lower())[1] in valid_extensions
            ])
        else:
            c_train = 0
        train_counts[emotion] = c_train
        total_train += c_train

        # Test count
        if os.path.exists(emotion_test_dir):
            c_test = len([
                f for f in os.listdir(emotion_test_dir)
                if os.path.splitext(f.lower())[1] in valid_extensions
            ])
        else:
            c_test = 0
        test_counts[emotion] = c_test
        total_test += c_test

    if print_summary:
        print(f"\n{'Class Name':<12} | {'Train Count':<12} | {'Test Count':<12}")
        print("-" * 42)
        for emotion in EMOTION_CLASSES:
            print(f"{emotion.capitalize():<12} | {train_counts[emotion]:<12} | {test_counts[emotion]:<12}")
        print("-" * 42)
        print(f"{'TOTAL':<12} | {total_train:<12} | {total_test:<12}\n")

    if total_train == 0 or total_test == 0:
        if print_summary:
            print("[!] Dataset folders exist but are currently empty.")
            print("[!] Follow the instructions below to download FER-2013.")
        return False

    if print_summary:
        print("[OK] Dataset successfully detected and ready for training!")
    return True


def download_kaggle_dataset():
    """
    Downloads and extracts FER-2013 using the official Kaggle CLI.
    Requires kaggle.json to be placed in ~/.kaggle/ (Linux/macOS) or C:\\Users\\<User>\\.kaggle\\ (Windows).
    """
    print("\nAttempting download from Kaggle (dataset: msambare/fer2013)...")
    try:
        import kaggle
    except ImportError:
        print("[!] The 'kaggle' library is not installed.")
        print("    Run: pip install kaggle")
        print("    Make sure your kaggle.json API token is in C:\\Users\\<Username>\\.kaggle\\kaggle.json")
        return False
    except Exception as e:
        print(f"[!] Error accessing Kaggle API credentials: {e}")
        print("    Please follow the manual download steps in the README.")
        return False

    try:
        print("[*] Downloading dataset archive...")
        kaggle.api.dataset_download_files(
            "msambare/fer2013",
            path=DATASET_DIR,
            unzip=True
        )
        print("[OK] Download and extraction complete!")
        check_dataset_layout()
        return True
    except Exception as err:
        print(f"[!] Kaggle API download failed: {err}")
        print("    Tip: Check your internet connection or use manual download.")
        return False


def create_dummy_dataset(train_per_class=120, test_per_class=30):
    """
    Generates synthetic 48x48 grayscale face-like images for rapid sanity testing.
    This lets you test train.py and the web app immediately without waiting
    for the ~300MB Kaggle download.
    """
    print(f"\nGenerating synthetic test dataset ({train_per_class} train, {test_per_class} test per class)...")

    np.random.seed(42)

    for split, count in [("train", train_per_class), ("test", test_per_class)]:
        split_dir = os.path.join(DATASET_DIR, split)
        for class_idx, emotion in enumerate(EMOTION_CLASSES):
            class_folder = os.path.join(split_dir, emotion)
            os.makedirs(class_folder, exist_ok=True)

            for i in range(count):
                # Generate 48x48 grayscale image with distinct patterns per emotion class
                # (to ensure the CNN has learnable features even in dummy test mode)
                img = Image.new("L", (48, 48), color=128)
                draw = ImageDraw.Draw(img)

                # Face outline circle
                draw.ellipse([4, 4, 44, 44], outline=220, fill=160 + (class_idx * 10) % 60)

                # Eyes
                draw.ellipse([14, 16, 18, 20], fill=40)
                draw.ellipse([30, 16, 34, 20], fill=40)

                # Class-specific mouth shape
                if emotion == "happy":
                    draw.arc([16, 26, 32, 38], start=0, end=180, fill=30, width=2)
                elif emotion == "sad":
                    draw.arc([16, 30, 32, 42], start=180, end=360, fill=30, width=2)
                elif emotion == "surprise":
                    draw.ellipse([21, 28, 27, 36], outline=30, fill=50)
                elif emotion == "angry":
                    draw.line([12, 14, 18, 17], fill=20, width=2)
                    draw.line([36, 14, 30, 17], fill=20, width=2)
                    draw.line([16, 32, 32, 32], fill=30, width=2)
                elif emotion == "disgust":
                    draw.line([16, 33, 24, 29], fill=30, width=2)
                    draw.line([24, 29, 32, 33], fill=30, width=2)
                elif emotion == "fear":
                    draw.ellipse([12, 14, 20, 22], fill=20)
                    draw.ellipse([28, 14, 36, 22], fill=20)
                    draw.arc([18, 30, 30, 38], start=0, end=180, fill=30, width=2)
                else:  # neutral
                    draw.line([18, 32, 30, 32], fill=40, width=2)

                # Add light Gaussian noise
                arr = np.array(img, dtype=np.int16)
                noise = np.random.normal(0, 8, arr.shape)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                noisy_img = Image.fromarray(arr)

                img_path = os.path.join(class_folder, f"sample_{i:04d}.png")
                noisy_img.save(img_path)

    print("[OK] Synthetic test dataset successfully created!")
    check_dataset_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FER-2013 Dataset Setup & Check Utility")
    parser.add_argument("--check", action="store_true", help="Check current dataset folder and class counts")
    parser.add_argument("--download", action="store_true", help="Download FER-2013 using Kaggle API")
    parser.add_argument("--create-dummy", action="store_true", help="Generate synthetic 48x48 images for quick dry-run")
    args = parser.parse_args()

    if args.create_dummy:
        create_dummy_dataset()
    elif args.download:
        download_kaggle_dataset()
    else:
        # Default behavior: run layout check
        check_dataset_layout()
