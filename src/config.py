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

# FER-2013 CSV file (facial emotion dataset)
FER_CSV_PATH = os.path.join(DATA_DIR, "fer2013.csv")

# Songs CSV (make sure this name matches your file in /data)
SONGS_CSV_PATH = os.path.join(DATA_DIR, "spotify_millsongdata_7emotions_sample.csv")
# If your songs file has a different name, change ONLY the filename above.

# ---------------------------
# PREPROCESSED DATA PATHS
# ---------------------------
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed")

# Preprocessed FER numpy arrays
FER_PRE_DIR = os.path.join(PREPROCESSED_DIR, "fer")

# Preprocessed songs (optional cleaned CSV etc.)
SONGS_PRE_DIR = os.path.join(PREPROCESSED_DIR, "songs")

# ---------------------------
# MODEL PATHS
# ---------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ✅ Ensemble of emotion models
# These files should exist in:  E:\ML project\MoodMusicProject\models\
#   - emotion_model_v1.h5  (your first trained CNN)
#   - emotion_model_v2.h5  (your improved CNN with augmentation)
# You can add a v3 later if you want.
EMOTION_MODEL_PATHS = [
    os.path.join(MODELS_DIR, "emotion_model_v1.h5"),
    os.path.join(MODELS_DIR, "emotion_model_v2.h5"),
    # os.path.join(MODELS_DIR, "emotion_model_v3.h5"),
]

# ---------------------------
# EMOTION LABELS (7 classes from FER2013)
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

# Reverse: string -> int (used for songs/emotion mapping)
EMOTION_TO_INT = {name.lower(): idx for idx, name in EMOTION_MAP.items()}
