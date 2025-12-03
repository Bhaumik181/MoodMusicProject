# src/emotion_utils.py

import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# -------------------------------------------------------
# Robust import of config (works with -m and direct run)
# -------------------------------------------------------
if __package__ is None or __package__ == "":
    # Running as a script: python src/emotion_utils.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.config import (
        EMOTION_MAP,
        EMOTION_MODEL_PATHS,
        SONGS_CSV_PATH,
        EMOTION_TO_INT,
    )
else:
    # Running as package: python -m src.main_webcam
    from .config import (
        EMOTION_MAP,
        EMOTION_MODEL_PATHS,
        SONGS_CSV_PATH,
        EMOTION_TO_INT,
    )

# -------------------------------------------------------
# Load emotion models (ensemble)
# -------------------------------------------------------
MODELS = []

for path in EMOTION_MODEL_PATHS:
    if os.path.exists(path):
        print("📂 Loading emotion model:", path)
        MODELS.append(load_model(path))
    else:
        print("⚠ Model file not found, skipping:", path)

if not MODELS:
    raise RuntimeError(
        "No emotion models found. Train at least one (emotion_model_v1.h5 / v2 / v3)."
    )

print(f"✅ Loaded {len(MODELS)} emotion model(s) for ensemble.")

# -------------------------------------------------------
# Load songs dataset
# -------------------------------------------------------
print("📂 Loading songs from:", SONGS_CSV_PATH)
SONGS_DF = pd.read_csv(SONGS_CSV_PATH)
print("Songs columns:", SONGS_DF.columns.tolist())

# Ensure we have an integer emotion column
if "emotion_int" not in SONGS_DF.columns:
    if "emotion" in SONGS_DF.columns:
        print("🔢 Creating 'emotion_int' from 'emotion' text labels...")
        SONGS_DF["emotion_int"] = SONGS_DF["emotion"].str.lower().map(EMOTION_TO_INT)
    else:
        raise ValueError(
            "Songs CSV must have either 'emotion_int' or 'emotion' column."
        )

before = len(SONGS_DF)
SONGS_DF = SONGS_DF.dropna(subset=["emotion_int"])
SONGS_DF["emotion_int"] = SONGS_DF["emotion_int"].astype(int)
after = len(SONGS_DF)

print(f"✅ Valid songs with emotion_int: {after}/{before}")
print("Emotion distribution:\n", SONGS_DF["emotion_int"].value_counts())


def predict_emotion_from_pixels(pixels_48x48: np.ndarray, debug: bool = False):
    """
    Ensemble prediction from multiple models.
    Input: 48x48 grayscale numpy array (0–255 or 0–1)
    Output: (emotion_idx, emotion_name, avg_probs)

    If debug=True, prints each model's top prediction and the final ensemble result.
    """
    img = pixels_48x48.astype("float32")
    if img.max() > 1.0:
        img = img / 255.0

    img = np.expand_dims(img, axis=-1)  # (48,48,1)
    img = np.expand_dims(img, axis=0)   # (1,48,48,1)

    total_probs = None

    if debug:
        print(f"\n[DEBUG] Using {len(MODELS)} models for ensemble:")

    for i, m in enumerate(MODELS):
        preds = m.predict(img)
        if debug:
            model_top_idx = int(np.argmax(preds[0]))
            model_top_name = EMOTION_MAP.get(model_top_idx, "Unknown")
            print(f"[DEBUG] Model {i+1} top idx: {model_top_idx} -> {model_top_name}")

        if total_probs is None:
            total_probs = preds[0]
        else:
            total_probs += preds[0]

    avg_probs = total_probs / len(MODELS)
    emotion_idx = int(np.argmax(avg_probs))
    emotion_name = EMOTION_MAP.get(emotion_idx, "Unknown")

    if debug:
        print(f"[DEBUG] Ensemble final idx: {emotion_idx} -> {emotion_name}")

    return emotion_idx, emotion_name, avg_probs


def recommend_song(emotion_idx: int, n: int = 3):
    """
    Recommend up to n songs matching the given emotion index.
    Returns a list of dicts with keys: 'song', 'artist', 'link'.
    - If 'link' exists, use that.
    - Else if 'uri' exists, use that.
    - Else link = "".
    """
    subset = SONGS_DF[SONGS_DF["emotion_int"] == emotion_idx]
    if subset.empty:
        print("⚠ No songs found for emotion:", emotion_idx)
        return []

    picked = subset.sample(min(n, len(subset)))

    results = []
    for _, row in picked.iterrows():
        link = ""
        if "link" in SONGS_DF.columns and isinstance(row.get("link", ""), str):
            link = row.get("link", "") or ""
        elif "uri" in SONGS_DF.columns and isinstance(row.get("uri", ""), str):
            link = row.get("uri", "") or ""

        results.append(
            {
                "song": row.get("song", ""),
                "artist": row.get("artist", ""),
                "link": link,
            }
        )

    return results


if __name__ == "__main__":
    # Optional quick test
    print("Testing emotion_utils...")
    print("Number of songs loaded:", len(SONGS_DF))
