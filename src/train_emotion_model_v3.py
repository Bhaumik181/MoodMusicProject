# src/train_emotion_model_v3.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    Add,
    Activation,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from src.config import FER_PRE_DIR, MODELS_DIR, EMOTION_MAP


def residual_block(x, filters, weight_decay, strides=(1, 1)):
    """A small ResNet-style block: Conv-BN-ReLU-Conv-BN + shortcut."""
    shortcut = x

    # First conv
    x = Conv2D(
        filters,
        (3, 3),
        strides=strides,
        padding="same",
        kernel_regularizer=l2(weight_decay),
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Second conv
    x = Conv2D(
        filters,
        (3, 3),
        padding="same",
        kernel_regularizer=l2(weight_decay),
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)

    # If dimensions change, project shortcut
    if shortcut.shape[-1] != filters or strides != (1, 1):
        shortcut = Conv2D(
            filters,
            (1, 1),
            strides=strides,
            padding="same",
            kernel_regularizer=l2(weight_decay),
            use_bias=False,
        )(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add + ReLU
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def build_model(num_classes: int):
    weight_decay = 1e-4

    inp = Input(shape=(48, 48, 1))

    # Stem
    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )(inp)
    x = BatchNormalization()(x)

    # Block group 1 (32 filters)
    x = residual_block(x, 32, weight_decay)
    x = residual_block(x, 32, weight_decay)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Block group 2 (64 filters)
    x = residual_block(x, 64, weight_decay, strides=(2, 2))
    x = residual_block(x, 64, weight_decay)
    x = Dropout(0.3)(x)

    # Block group 3 (128 filters)
    x = residual_block(x, 128, weight_decay, strides=(2, 2))
    x = residual_block(x, 128, weight_decay)
    x = Dropout(0.4)(x)

    # Global pooling instead of Flatten
    x = GlobalAveragePooling2D()(x)

    # Dense head
    x = Dense(256, activation="relu", kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load preprocessed FER data (same as v2)
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
    print("X_test: ", X_test.shape, "y_test:", y_test.shape)

    # Data augmentation (same style as v2)
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )

    train_datagen.fit(X_train)

    # Build v3 model (ResNet-style)
    model = build_model(num_classes)
    model.summary()

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

    print("Starting training for model v3...")
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[early_stop, reduce_lr],
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Test] Loss (v3): {test_loss:.4f}")
    print(f"[Test] Accuracy (v3): {test_acc:.4f}")

    # Classification report
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    target_names = [EMOTION_MAP[i] for i in range(len(EMOTION_MAP))]
    print("\nClassification Report (v3):")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Confusion matrix
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
    plt.title("Confusion Matrix - FER2013 Emotion Model (v3)")
    plt.tight_layout()
    plt.show()

    # Training curves
    hist = history.history

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist["accuracy"], label="Train Acc")
    plt.plot(hist["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy (v3)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist["loss"], label="Train Loss")
    plt.plot(hist["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss (v3)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save as v3
    save_path = os.path.join(MODELS_DIR, "emotion_model_v3.h5")
    model.save(save_path)
    print("\nModel v3 saved to:", save_path)


if __name__ == "__main__":
    main()
