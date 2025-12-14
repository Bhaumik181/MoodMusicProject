# src/train_emotion_model.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# absolute import from src package
from src.config import FER_PRE_DIR, MODELS_DIR, EMOTION_MODEL_PATH, EMOTION_MAP


def build_model(num_classes: int):
    #CNN with more regularization to reduce overfitting on FER2013.
    weight_decay = 1e-4  # L2 regularization factor

    model = Sequential(
        [
            # Block 1
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=l2(weight_decay),
                input_shape=(48, 48, 1),
            ),
            BatchNormalization(),
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=l2(weight_decay),
            ),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Block 2
            Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=l2(weight_decay),
            ),
            BatchNormalization(),
            Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=l2(weight_decay),
            ),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Block 3
            Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=l2(weight_decay),
            ),
            BatchNormalization(),
            Conv2D(
                128,
                (3, 3),
                activation="relu",
                padding="same",
                kernel_regularizer=l2(weight_decay),
            ),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.4),

            Flatten(),
            Dense(256, activation="relu", kernel_regularizer=l2(weight_decay)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # -----------------------------
    # Load preprocessed FER arrays
    # -----------------------------
    X_train = np.load(os.path.join(FER_PRE_DIR, "X_train.npy"))
    X_val = np.load(os.path.join(FER_PRE_DIR, "X_val.npy"))
    X_test = np.load(os.path.join(FER_PRE_DIR, "X_test.npy"))

    y_train_raw = np.load(os.path.join(FER_PRE_DIR, "y_train.npy"))
    y_val_raw = np.load(os.path.join(FER_PRE_DIR, "y_val.npy"))
    y_test_raw = np.load(os.path.join(FER_PRE_DIR, "y_test.npy"))

    num_classes = len(np.unique(y_train_raw))
    y_train = to_categorical(y_train_raw, num_classes)
    y_val = to_categorical(y_val_raw, num_classes)
    y_test = to_categorical(y_test_raw, num_classes)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
    print("X_test: ", X_test.shape, "y_test: ", y_test.shape)

    # -----------------------------
    # Data Augmentation
    # -----------------------------
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    train_datagen.fit(X_train)

    # -----------------------------
    # Build model
    # -----------------------------
    model = build_model(num_classes)
    model.summary()

    # -----------------------------
    # Callbacks
    # -----------------------------
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1,
    )

    batch_size = 64
    epochs = 40

    # -----------------------------
    # Train model with augmentation
    # -----------------------------
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[early_stop, reduce_lr],
    )

    # -----------------------------
    # Evaluate on test set
    # -----------------------------
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    target_names = [EMOTION_MAP[i] for i in range(len(EMOTION_MAP))]

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=target_names,
        yticklabels=target_names,
        cmap="magma",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - FER2013 Emotion Model")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot training curves
    # -----------------------------
    hist = history.history

    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(hist["accuracy"], label="Train Acc")
    plt.plot(hist["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(hist["loss"], label="Train Loss")
    plt.plot(hist["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Save model
    # -----------------------------
    model.save(EMOTION_MODEL_PATH)
    print("\nModel saved to:", EMOTION_MODEL_PATH)


if __name__ == "__main__":
    main()