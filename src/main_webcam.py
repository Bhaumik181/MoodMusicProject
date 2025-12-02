# src/main_webcam.py

import cv2
import numpy as np

from src.emotion_utils import predict_emotion_from_pixels, recommend_song
from src.config import EMOTION_MAP


def detect_face(frame_bgr):
    """Detect faces and return (faces, gray_image)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces, gray


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 's' to get song recommendations for current emotion.")
    print("Press 'q' to quit.")

    last_emotion_idx = None
    current_songs = []  # list of {'song', 'artist', 'link'}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        faces, gray = detect_face(frame)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))

            # Ensemble emotion prediction
            emotion_idx, emotion_name, _ = predict_emotion_from_pixels(face)
            last_emotion_idx = emotion_idx

            # Draw face box + emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                emotion_name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # ─────────────────────────────────────────────
        #  Draw current recommended songs on the frame
        # ─────────────────────────────────────────────
        if current_songs:
            h, w, _ = frame.shape
            y0 = h - 60  # starting vertical position
            dy = 20      # line spacing

            cv2.putText(
                frame,
                "Songs:",
                (10, y0 - dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            for i, s in enumerate(current_songs[:3]):  # show at most 3
                text = f"{i+1}. {s['song']} - {s['artist']}"
                if len(text) > 40:
                    text = text[:37] + "..."
                cv2.putText(
                    frame,
                    text,
                    (10, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("Mood Music AI (Ensemble)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s'):
            if last_emotion_idx is None:
                print("No emotion detected yet. Look at the camera :)")
            else:
                emo_name = EMOTION_MAP[last_emotion_idx]
                print(f"\nEmotion: {emo_name}")
                recs = recommend_song(last_emotion_idx, n=3)
                if not recs:
                    print("No songs found for this emotion.")
                    current_songs = []
                else:
                    print("Recommended songs:")
                    for i, s in enumerate(recs, 1):
                        print(f"{i}. {s['song']} - {s['artist']}")
                        print("   Link:", s['link'])
                    current_songs = recs

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
