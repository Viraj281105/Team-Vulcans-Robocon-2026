#!/usr/bin/env python3
"""
KFS Classifier v4 — Transfer Learning + Massive Augmentation
------------------------------------------------------------
- Uses MobileNetV3Small (pretrained on ImageNet)
- Auto-augments dataset up to thousands of variations per class
- Skips missing folders safely
- Optimized for CPU (no GPU required)
"""

import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras import layers, models
from tqdm import tqdm

# --- Paths ---
BASE_DIR = "/home/meherdeep/rpi-github/Team-Vulcans-Robocon-2026/teams/team-2_vision/dataset"
EXPANDED_DIR = os.path.join(BASE_DIR, "expanded")

print(EXPANDED_DIR)
MODEL_NAME = "kfs_classifier_v4_transfer"
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 40
EXPANDED_COUNT = 10000  # per class


# --- Utility: Advanced augmentation function ---
def advanced_augment(img):
    img = img.astype(np.uint8)
    # Random flips
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    if random.random() > 0.7:
        img = cv2.flip(img, 0)

    # Random rotation
    angle = random.uniform(-25, 25)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, random.uniform(0.9, 1.1))
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Random brightness/contrast
    alpha = random.uniform(0.6, 1.6)  # contrast
    beta = random.randint(-40, 40)    # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Random blur or noise
    if random.random() > 0.7:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() > 0.7:
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Random color shift (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + random.randint(-20, 20)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-30, 30), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-40, 40), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img


# --- Dataset expansion ---
def expand_dataset(base_dir):
    os.makedirs(EXPANDED_DIR, exist_ok=True)
    total_generated = 0

    print("📸 Expanding dataset...")
    for cls in os.listdir(base_dir):
        src = os.path.join(base_dir, cls)
        if not os.path.isdir(src) or cls == "expanded":
            continue

        images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not images:
            print(f"⚠️ Skipping '{cls}' — no images found.")
            continue

        dst = os.path.join(EXPANDED_DIR, cls)
        os.makedirs(dst, exist_ok=True)

        print(f"→ {cls}: {len(images)} originals → generating up to {EXPANDED_COUNT} images")
        for i in tqdm(range(EXPANDED_COUNT), desc=f"Augmenting {cls}", ncols=80):
            img_path = os.path.join(src, random.choice(images))
            img = cv2.imread(img_path)
            if img is None:
                continue
            aug_img = advanced_augment(img)
            out_path = os.path.join(dst, f"{cls}_{i:05d}.jpg")
            cv2.imwrite(out_path, aug_img)
            total_generated += 1

    print(f"✅ Total augmented images: {total_generated}")
    return EXPANDED_DIR


# --- Expand dataset first ---
expanded_dir = expand_dataset(BASE_DIR)

# --- Data preparation ---
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    brightness_range=[0.5, 1.5],
    channel_shift_range=30,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_generator = train_datagen.flow_from_directory(
    expanded_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    expanded_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# --- Model ---
base_model = MobileNetV3Small(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

for layer in base_model.layers[:-30]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n🚀 Starting training...\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# --- Save model ---
save_dir = os.path.dirname(BASE_DIR)
keras_path = os.path.join(save_dir, f"{MODEL_NAME}.keras")
tflite_path = os.path.join(save_dir, f"{MODEL_NAME}.tflite")

model.save(keras_path)
print(f"✅ Saved Keras model to: {keras_path}")

# --- Convert to TFLite ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"✅ Saved TFLite model to: {tflite_path}")

# --- Class Indices ---
print("\n📋 Class Indices:")
for k, v in train_generator.class_indices.items():
    print(f"{v}: {k}")
