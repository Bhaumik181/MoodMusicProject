# src/preprocess_songs.py

import os
import sys
import re
import pandas as pd

# -------------------------------------------------------
# Robust import of config (works when run as script or -m)
# -------------------------------------------------------
if __package__ is None or __package__ == "":
    # Running directly: python src/preprocess_songs.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.config import SONGS_CSV_PATH, SONGS_PRE_DIR, EMOTION_TO_INT
else:
    # Running as module: python -m src.preprocess_songs
    from .config import SONGS_CSV_PATH, SONGS_PRE_DIR, EMOTION_TO_INT


def clean_lyrics(t: str) -> str:
    """Basic text cleaning for lyrics."""
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)              # remove URLs
    t = re.sub(r"[^a-z0-9',.!?;:\s]", " ", t)           # keep only letters, digits, basic punct
    t = re.sub(r"\s+", " ", t).strip()                  # remove extra spaces
    return t


def main():
    os.makedirs(SONGS_PRE_DIR, exist_ok=True)

    print("📂 Loading songs from:", SONGS_CSV_PATH)
    songs = pd.read_csv(SONGS_CSV_PATH)
    print("Original shape:", songs.shape)
    print("Columns:", songs.columns.tolist())

    # clean text column if exists
    if "text" in songs.columns:
        print("\n🔧 Cleaning lyrics text...")
        songs["text"] = songs["text"].fillna("").apply(clean_lyrics)
    else:
        print("\n⚠ No 'text' column found. Skipping lyrics cleaning.")

    # Ensure emotion_int column exists (mapping from emotion string if needed)
    if "emotion_int" not in songs.columns:
        if "emotion" in songs.columns:
            print("🔢 Creating 'emotion_int' from 'emotion' text column...")
            songs["emotion_int"] = songs["emotion"].str.lower().map(EMOTION_TO_INT)
        else:
            print("❌ No 'emotion' or 'emotion_int' column found in songs CSV!")
            print("   Make sure your CSV has an 'emotion' column like 'happy', 'sad', etc.")
            return

    # Drop invalid rows
    before = len(songs)
    songs = songs.dropna(subset=["emotion_int"])
    songs["emotion_int"] = songs["emotion_int"].astype(int)
    after = len(songs)

    print(f"\n✅ Kept {after} rows out of {before} with valid emotion_int.")
    print("Emotion distribution:\n", songs["emotion_int"].value_counts())

    # Save cleaned CSV
    output_path = os.path.join(SONGS_PRE_DIR, "songs_cleaned.csv")
    songs.to_csv(output_path, index=False)

    print("\n💾 Saved cleaned songs to:", output_path)
    print("🎉 Songs preprocessing completed!")


if __name__ == "__main__":
    main()
