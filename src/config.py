# src/config.py

import os

# ---------------------------
# BASE DIRECTORY (project root)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------
# DATA PATHS
# ---------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")

# FER2013 CSV
FER_CSV_PATH = os.path.join(DATA_DIR, "fer2013.csv")

# Songs CSV (MAKE SURE the name matches your actual file in /data)
SONGS_CSV_PATH = os.path.join(DATA_DIR, "spotify_millsongdata_7emotions_sample.csv")
# If your songs file is named differently, change only the "spotify_millsongdata_7emotions_full.csv" part.

# ---------------------------
# PREPROCESSED DATA PATHS
# ---------------------------
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed")

# FER preprocessed arrays
FER_PRE_DIR = os.path.join(PREPROCESSED_DIR, "fer")

# ✅ Songs preprocessed output folder (needed for preprocess_songs.py)
SONGS_PRE_DIR = os.path.join(PREPROCESSED_DIR, "songs")

# ---------------------------
# MODEL PATHS
# ---------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.h5")

# ---------------------------
# EMOTION MAPPINGS (7 classes)
# ---------------------------
EMOTION_MAP = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

EMOTION_TO_INT = {v.lower(): k for k, v in EMOTION_MAP.items()}

MODELS_DIR = os.path.join(BASE_DIR, "models")

# Old single-model path (keep if you want)
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.h5")

# ✅ New multi-model paths
EMOTION_MODEL_PATHS = [
    os.path.join(MODELS_DIR, "emotion_model_v1.h5"),
    os.path.join(MODELS_DIR, "emotion_model_v2.h5"),
    os.path.join(MODELS_DIR, "emotion_model_v3.h5"),
]
