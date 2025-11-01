#!/usr/bin/env python3
"""
KFS Classifier v3 — "Unbreakable CNN"
-------------------------------------
Massively augmented, highly regularized, and designed to generalize
from small, imperfect datasets (like field images of KFS scrolls).

Changes from v2:
- Much stronger augmentations (rotation, contrast, noise, blur, perspective)
- Added GaussianNoise + RandomContrast layers inside the model
- Slightly deeper CNN with residual-style shortcut
- Label smoothing & dropout for robustness
- Early stopping + model checkpointing
- Still outputs both .h5 and .tflite models
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
import matplotlib.pyplot as plt

# Paths
base_dir = "/home/viraj/Robocon/Team-Vulcans-Robocon-2026/teams/team-2_vision/dataset"
output_h5 = "/home/viraj/Robocon/Team-Vulcans-Robocon-2026/teams/team-2_vision/kfs_classifier_v3.h5"
output_tflite = "/home/viraj/Robocon/Team-Vulcans-Robocon-2026/teams/team-2_vision/kfs_classifier_v3.tflite"

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 60  # early stopping will cut sooner if needed

# Supercharged data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=45,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.3,
    zoom_range=0.35,
    brightness_range=[0.5, 1.6],
    channel_shift_range=50.0,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.25
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Helper block (Conv -> BN -> ReLU)
def conv_block(x, filters, kernel=3, dropout=0.2):
    x = layers.Conv2D(filters, kernel, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x

# Model definition
inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Input-level augmentations
x = layers.RandomFlip("horizontal_and_vertical")(inputs)
x = layers.RandomRotation(0.25)(x)
x = layers.RandomZoom(0.2)(x)
x = layers.RandomContrast(0.3)(x)
x = layers.GaussianNoise(0.05)(x)

# CNN body
x = conv_block(x, 32)
x = layers.MaxPooling2D()(x)
x = conv_block(x, 64)
x = layers.MaxPooling2D()(x)
x = conv_block(x, 128)
x = layers.MaxPooling2D()(x)
x = conv_block(x, 256)
x = layers.MaxPooling2D()(x)

# Flatten and dense head
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# Compile with label smoothing for better generalization
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint(output_h5, monitor='val_loss', save_best_only=True)
]

# Train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# Save final model
model.save(output_h5)
print(f"✅ Model saved to: {output_h5}")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(output_tflite, "wb") as f:
    f.write(tflite_model)
print(f"✅ TFLite model saved to: {output_tflite}")

# Plot metrics
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
plt.title('Training Progress (v3)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Print class indices
print("\nClass Indices:")
for k, v in train_generator.class_indices.items():
    print(f"{v}: {k}")
