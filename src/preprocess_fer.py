# src/preprocess_fer.py

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Import config in a way that works BOTH:
# - when run as: python -m src.preprocess_fer   (recommended)
# - AND when run directly: python src/preprocess_fer.py
# ------------------------------------------------------------------
if __package__ is None or __package__ == "":
    # Running as a script, fix sys.path to include project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.config import FER_CSV_PATH, FER_PRE_DIR
else:
    # Running as part of src package (python -m src.preprocess_fer)
    from .config import FER_CSV_PATH, FER_PRE_DIR


def process_pixels(pixels: str):
    """
    Convert the 'pixels' string from FER2013 into a 48x48 normalized image array.
    Returns None if the row is corrupted.
    """
    try:
        arr = np.array(pixels.split(), dtype="float32")
        if arr.size != 48 * 48:
            return None
        img = arr.reshape(48, 48)
        img = img / 255.0  # normalize [0, 1]
        return img
    except Exception:
        return None


def prepare_data(df: pd.DataFrame):
    """
    Turn a FER dataframe with 'pixels' and 'emotion' into
    X (N, 48, 48, 1) and y (N,) arrays.
    """
    X = np.stack(df["pixels"].values)
    y = df["emotion"].values
    X = np.expand_dims(X, -1)  # (N,48,48,1)
    return X, y


def main():
    # ensure output directory exists
    os.makedirs(FER_PRE_DIR, exist_ok=True)

    print("📂 Loading FER2013 from:", FER_CSV_PATH)
    data = pd.read_csv(FER_CSV_PATH)
    print("Original shape:", data.shape)
    print("Columns:", data.columns.tolist())

    # process pixels
    print("\n🔧 Processing pixel strings into 48x48 images...")
    data["pixels"] = data["pixels"].apply(process_pixels)
    data = data.dropna(subset=["pixels"])
    print("After cleaning, shape:", data.shape)

    # split using Usage if possible
    if "Usage" in data.columns:
        print("\n📊 Splitting using 'Usage' column...")
        train_data = data[data["Usage"].str.contains("Train", case=False, na=False)]
        test_data = data[data["Usage"].str.contains("PrivateTest", case=False, na=False)]

        if test_data.empty:
            print("No 'PrivateTest' found. Using random stratified split instead.")
            train_data, test_data = train_test_split(
                data,
                test_size=0.2,
                random_state=42,
                stratify=data["emotion"],
            )
    else:
        print("\n⚠ No 'Usage' column found. Using random stratified split.")
        train_data, test_data = train_test_split(
            data,
            test_size=0.2,
            random_state=42,
            stratify=data["emotion"],
        )

    # validation split from train
    train_data, val_data = train_test_split(
        train_data,
        test_size=0.1,
        random_state=42,
        stratify=train_data["emotion"],
    )

    print(f"\n✅ Final splits:")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples:   {len(val_data)}")
    print(f"Test samples:  {len(test_data)}")

    # prepare numpy arrays
    print("\n📦 Converting to numpy arrays...")
    X_train, y_train = prepare_data(train_data)
    X_val, y_val = prepare_data(val_data)
    X_test, y_test = prepare_data(test_data)

    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    print("X_val shape:  ", X_val.shape, "y_val shape:  ", y_val.shape)
    print("X_test shape: ", X_test.shape, "y_test shape: ", y_test.shape)

    # save arrays
    print("\n💾 Saving preprocessed FER data to:", FER_PRE_DIR)
    np.save(os.path.join(FER_PRE_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(FER_PRE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(FER_PRE_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(FER_PRE_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(FER_PRE_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(FER_PRE_DIR, "y_test.npy"), y_test)

    print("\n🎉 Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
