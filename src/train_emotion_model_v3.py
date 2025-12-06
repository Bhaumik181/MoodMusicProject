# src/train_emotion_model_v3.py  (you can copy your v2 file and edit)

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
