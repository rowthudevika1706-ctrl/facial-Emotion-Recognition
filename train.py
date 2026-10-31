"""
=============================================================================
Facial Emotion Recognition (FER-2013) - CNN Training & Evaluation Pipeline
=============================================================================
Deep Learning Assignment | College-Level Project
Compatible with Python 3.10 / 3.11, TensorFlow 2.15+, and Windows

CONCEPTS EXPLAINED FOR STUDENTS:
--------------------------------
1. WHAT IS AN EPOCH?
   One epoch represents one complete pass of the entire training dataset
   through the neural network. During each epoch, the model processes images
   in small batches, calculates the loss (prediction error), and updates its
   internal weights via backpropagation.

2. WHAT IS OVERFITTING?
   Overfitting occurs when a neural network memorizes the training data too
   closely (including noise and background details) instead of learning general
   facial features (like smile curvature or eyebrow tilt). A classic sign of
   overfitting is when Training Accuracy continues to rise (e.g. 95%), but
   Validation Accuracy plateaus or drops (e.g. 58%).

3. WHAT IS DROPOUT?
   Dropout is a regularization technique. During training, a specified percentage
   of neurons (e.g. 25% or 50%) are randomly "turned off" at each step. This
   prevents neurons from co-adapting too closely, forcing the network to learn
   redundant, robust feature representations.

4. WHAT IS EARLY STOPPING?
   Rather than training blindly for a fixed number of epochs, EarlyStopping
   monitors validation loss. If the validation loss fails to improve for several
   consecutive epochs ('patience'), training terminates automatically and restores
   the best model weights found so far.

5. WHY USE CLASS WEIGHTS?
   In FER-2013, classes are imbalanced: "Happy" has over 7,000 images, while
   "Disgust" has only ~436. Without class weighting, the CNN would prioritize the
   majority classes and virtually ignore "Disgust". Class weights penalize errors
   on rare classes more heavily, balancing the gradient updates.
=============================================================================
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Ensure Windows console handles UTF-8 output safely
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Silence TensorFlow verbose C++ logs while retaining error messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers


# =============================================================================
# 1. CONSTANTS & CONFIGURATION
# =============================================================================
IMG_HEIGHT = 48
IMG_WIDTH = 48
CHANNELS = 1              # Grayscale input (48x48x1)
NUM_CLASSES = 7
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.0005

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


# =============================================================================
# 2. DATASET VERIFICATION & LOADING UTILITIES
# =============================================================================
def scan_dataset_directory(data_path):
    """
    Auto-detects train/ and test/ folders inside the dataset directory and
    prints class distribution counts.
    """
    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "test")

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        print("\n" + "=" * 65)
        print("[ERROR] Missing dataset layout!")
        print(f"Looked in: {data_path}")
        print("Expected structure:")
        print("  dataset/")
        print("    ├── train/")
        print("    │   ├── angry/")
        print("    │   ├── disgust/")
        print("    │   └── ... (all 7 classes)")
        print("    └── test/")
        print("        └── ... (all 7 classes)")
        print("Run 'python setup_dataset.py --create-dummy' for immediate testing")
        print("or download the full FER-2013 dataset as shown in the README.")
        print("=" * 65 + "\n")
        sys.exit(1)

    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith(".")
    ])

    if len(classes) == 0:
        print(f"[ERROR] No class subdirectories found in: {train_dir}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("                  FER-2013 DATASET LAYOUT CHECK                  ")
    print("=" * 65)
    print(f"{'Class Index':<12} | {'Class Name':<12} | {'Train Count':<12} | {'Test Count':<12}")
    print("-" * 65)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    train_totals = 0
    test_totals = 0

    for idx, cname in enumerate(classes):
        tr_count = len([
            f for f in os.listdir(os.path.join(train_dir, cname))
            if os.path.splitext(f.lower())[1] in valid_exts
        ])
        te_path = os.path.join(test_dir, cname)
        te_count = len([
            f for f in os.listdir(te_path)
            if os.path.splitext(f.lower())[1] in valid_exts
        ]) if os.path.exists(te_path) else 0

        train_totals += tr_count
        test_totals += te_count
        print(f"{idx:<12} | {cname.capitalize():<12} | {tr_count:<12} | {te_count:<12}")

    print("-" * 65)
    print(f"{'TOTAL':<12} | {'':<12} | {train_totals:<12} | {test_totals:<12}")
    print("=" * 65 + "\n")

    if train_totals == 0:
        print("[ERROR] Training directory contains 0 images. Please populate the dataset.")
        sys.exit(1)

    return classes


def load_dataset_splits(data_path, sample_limit=None, batch_size=DEFAULT_BATCH_SIZE, val_split=0.2, seed=42):
    """
    Loads images from train/ and test/ folders.
    - If sample_limit is set (e.g. --sample 500), samples at most N images per class
      for rapid testing while maintaining all 7 classes.
    - Divides training data into 80% train and 20% validation.
    - Loads test folder purely for final evaluation.
    - Rescales pixel values from [0, 255] to [0.0, 1.0].
    """
    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "test")
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith(".")
    ])
    class_to_idx = {cname: i for i, cname in enumerate(classes)}

    def gather_paths(root_dir, limit_per_class=None):
        file_paths = []
        labels = []
        for cname in classes:
            folder = os.path.join(root_dir, cname)
            if not os.path.exists(folder):
                continue
            images = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.splitext(f.lower())[1] in valid_exts
            ]
            if limit_per_class and len(images) > limit_per_class:
                random.seed(seed)
                images = random.sample(images, limit_per_class)
            file_paths.extend(images)
            labels.extend([class_to_idx[cname]] * len(images))
        return file_paths, labels

    train_val_paths, train_val_labels = gather_paths(train_dir, sample_limit)
    test_paths, test_labels = gather_paths(test_dir, sample_limit if sample_limit else None)

    # Shuffle training paths and split into Train and Validation sets
    combined = list(zip(train_val_paths, train_val_labels))
    random.seed(seed)
    random.shuffle(combined)
    train_val_paths, train_val_labels = zip(*combined)
    train_val_paths, train_val_labels = list(train_val_paths), list(train_val_labels)

    split_idx = int(len(train_val_paths) * (1.0 - val_split))
    train_paths = train_val_paths[:split_idx]
    train_labels = train_val_labels[:split_idx]
    val_paths = train_val_paths[split_idx:]
    val_labels = train_val_labels[split_idx:]

    print(f"Loaded {len(train_paths)} training samples, {len(val_paths)} validation samples, and {len(test_paths)} test samples.")
    if sample_limit:
        print(f"[*] Sample mode active: maximum {sample_limit} images per class selected.")

    # Helper function to read and preprocess a single image file
    def parse_image_grayscale(file_path, label):
        # 1. Read binary contents
        img_bytes = tf.io.read_file(file_path)
        # 2. Decode into 1-channel grayscale tensor
        img = tf.io.decode_image(img_bytes, channels=1, expand_animations=False)
        # 3. Resize to exact target 48x48 resolution
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        # 4. Normalize pixel values from [0, 255] to [0.0, 1.0]
        img = tf.cast(img, tf.float32) / 255.0
        # 5. One-hot encode label for categorical cross-entropy
        one_hot_label = tf.one_hot(label, depth=len(classes))
        return img, one_hot_label

    def build_tf_dataset(paths, labels, is_training=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if is_training:
            ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=seed)
        ds = ds.map(parse_image_grayscale, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    train_ds = build_tf_dataset(train_paths, train_labels, is_training=True)
    val_ds = build_tf_dataset(val_paths, val_labels, is_training=False)
    test_ds = build_tf_dataset(test_paths, test_labels, is_training=False)

    # Compute class weights for the training subset to counterbalance underrepresented emotions
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=np.array(train_labels)
    )
    # Smooth / clip extreme weights (max 3.5x) to prevent minority classes (e.g. disgust) from overpowering the loss
    class_weights_dict = {
        int(i): float(np.clip(w, 0.4, 3.5))
        for i, w in zip(np.unique(train_labels), class_weights_arr)
    }

    print("\nCalculated Smoothed Class Weights (dampened to avoid minority-class bias):")
    for idx, cname in enumerate(classes):
        w = class_weights_dict.get(idx, 1.0)
        print(f"  Class {idx} ({cname.capitalize()}): weight = {w:.3f}")
    print()

    return train_ds, val_ds, test_ds, classes, class_weights_dict, (test_paths, test_labels)


# =============================================================================
# 3. CNN ARCHITECTURE DEFINITION
# =============================================================================
def build_emotion_cnn(num_classes=NUM_CLASSES, input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)):
    """
    Constructs a modern, robust Convolutional Neural Network from scratch for FER-2013.

    Layer Hierarchy:
    ----------------
    - Data Augmentation (RandomFlip, RandomRotation, RandomZoom):
        Executed only during training mode to artificially broaden dataset variance.
    - Block 1 (Low-level edges and textures):
        Conv2D(32) -> BatchNorm -> Conv2D(64) -> BatchNorm -> MaxPool(2x2) -> Dropout(0.25)
    - Block 2 (Mid-level facial parts: eyes, nose, mouth curvature):
        Conv2D(128) -> BatchNorm -> Conv2D(128) -> BatchNorm -> MaxPool(2x2) -> Dropout(0.25)
    - Block 3 (High-level holistic emotion features):
        Conv2D(256) -> BatchNorm -> MaxPool(2x2) -> Dropout(0.25)
    - Classification Head:
        Flatten -> Dense(256) -> BatchNorm -> Dropout(0.5) -> Dense(7, Softmax)
    """
    inputs = layers.Input(shape=input_shape, name="input_grayscale_face")

    # Light on-the-fly data augmentation applied only during training
    x = layers.RandomFlip("horizontal", name="aug_horizontal_flip")(inputs)
    x = layers.RandomRotation(0.08, fill_mode="nearest", name="aug_rotation")(x)
    x = layers.RandomZoom(0.08, fill_mode="nearest", name="aug_zoom")(x)

    # --- Feature Extraction Block 1 ---
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(x)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="dropout1")(x)

    # --- Feature Extraction Block 2 ---
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="dropout2")(x)

    # --- Feature Extraction Block 3 ---
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.25, name="dropout3")(x)

    # --- Classification Dense Head ---
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.50, name="dropout_head")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="emotion_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="FER2013_Custom_CNN")
    return model


# =============================================================================
# 4. EVALUATION & METRIC VISUALIZATION
# =============================================================================
def plot_training_history(history, save_path):
    """
    Plots dual subplots:
    1. Training vs Validation Accuracy
    2. Training vs Validation Loss
    """
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # --- Subplot 1: Accuracy ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2563eb", linewidth=2.2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#16a34a", linewidth=2.2, linestyle="--")
    plt.title("Model Accuracy Across Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")

    # --- Subplot 2: Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#dc2626", linewidth=2.2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#d97706", linewidth=2.2, linestyle="--")
    plt.title("Categorical Cross-Entropy Loss Across Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Saved training accuracy and loss curve to: {save_path}")


def evaluate_and_generate_reports(model, test_paths, test_labels, class_names, outputs_dir):
    """
    Evaluates the model on the unseen test set, generates a normalized Confusion
    Matrix image and writes a comprehensive Classification Report text file.
    """
    print("\n" + "=" * 65)
    print("                 FINAL EVALUATION ON TEST SET                     ")
    print("=" * 65)

    y_true = []
    y_pred = []

    # Batch test inference to avoid memory spikes
    batch_size = 64
    total_samples = len(test_paths)

    for i in range(0, total_samples, batch_size):
        batch_paths = test_paths[i:i + batch_size]
        batch_true = test_labels[i:i + batch_size]

        batch_imgs = []
        for p in batch_paths:
            img_bytes = tf.io.read_file(p)
            img = tf.io.decode_image(img_bytes, channels=1, expand_animations=False)
            img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
            img = tf.cast(img, tf.float32) / 255.0
            batch_imgs.append(img.numpy())

        batch_imgs = np.array(batch_imgs)
        preds = model.predict(batch_imgs, verbose=0)
        batch_pred_classes = np.argmax(preds, axis=1)

        y_true.extend(batch_true)
        y_pred.extend(batch_pred_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Exact Test Accuracy (unfabricated real measurement)
    test_acc = np.mean(y_true == y_pred) * 100.0
    print(f"\n>> REAL TEST ACCURACY: {test_acc:.2f}%\n")

    # 2. Classification Report (Precision, Recall, F1-Score)
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=[c.capitalize() for c in class_names],
        digits=4
    )
    print(report_text)

    report_path = os.path.join(outputs_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("      FER-2013 FACIAL EMOTION RECOGNITION TEST REPORT       \n")
        f.write("============================================================\n\n")
        f.write(f"Overall Test Accuracy: {test_acc:.2f}%\n\n")
        f.write("Classification Metrics by Emotion Class:\n")
        f.write(report_text)
        f.write("\n\nMETRIC GUIDE FOR STUDENTS:\n")
        f.write("- Precision: When the model predicts emotion X, how often is it correct?\n")
        f.write("- Recall: Out of all real instances of emotion X, how many did the model find?\n")
        f.write("- F1-Score: Harmonic balance between Precision and Recall.\n")
    print(f"[OK] Saved classification report to: {report_path}")

    # 3. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(9, 7.5))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[c.capitalize() for c in class_names],
        yticklabels=[c.capitalize() for c in class_names],
        cbar_kws={"label": "Normalized Ratio"}
    )
    plt.title(f"Normalized Confusion Matrix (Test Accuracy: {test_acc:.2f}%)", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Emotion", fontsize=11)
    plt.ylabel("Actual True Emotion", fontsize=11)
    plt.tight_layout()

    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Saved confusion matrix visualization to: {cm_path}")


# =============================================================================
# 5. MAIN TRAINING WORKFLOW
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train CNN for Facial Emotion Recognition on FER-2013")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size (default: 64)")
    parser.add_argument("--sample", type=int, default=None, help="Sample limit per class for quick dry runs (e.g. --sample 500)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Initial Adam learning rate (default: 0.0005)")
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR, help="Path to dataset directory containing train/ and test/")
    args = parser.parse_args()

    # Ensure output and model directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Step 1: Scan and verify dataset layout
    detected_classes = scan_dataset_directory(args.dataset_dir)

    # Step 2: Save class names mapping immediately for app.py alignment
    class_names_path = os.path.join(MODELS_DIR, "class_names.json")
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump([c.capitalize() for c in detected_classes], f, indent=2)
    print(f"[OK] Saved class labels to: {class_names_path}")

    # Step 3: Load Train, Validation, and Test datasets
    train_ds, val_ds, test_ds, class_names, class_weights_dict, (test_paths, test_labels) = load_dataset_splits(
        data_path=args.dataset_dir,
        sample_limit=args.sample,
        batch_size=args.batch_size
    )

    # Step 4: Build CNN Architecture
    model = build_emotion_cnn(num_classes=len(class_names))
    model.summary()

    # Step 5: Compile with Adam optimizer and Categorical Cross-Entropy
    optimizer = optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Step 6: Configure Training Callbacks
    model_save_path = os.path.join(MODELS_DIR, "emotion_model.keras")

    training_callbacks = [
        # Save only when validation loss reaches a new low
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        # Stop early if validation loss fails to improve for 6 epochs
        callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when learning plateaus
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.25,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n" + "=" * 65)
    print("                     STARTING MODEL TRAINING                      ")
    print("=" * 65)
    print(f"Target Epochs : {args.epochs}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Target Output : {model_save_path}")
    print("=" * 65 + "\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights_dict,
        callbacks=training_callbacks,
        verbose=1
    )

    print("\n[OK] Training complete!")

    # Step 7: Plot Accuracy and Loss Curves
    plot_training_history(history, os.path.join(OUTPUTS_DIR, "accuracy_loss.png"))

    # Step 8: Evaluate on Test Set & Generate Confusion Matrix & Report
    evaluate_and_generate_reports(model, test_paths, test_labels, class_names, OUTPUTS_DIR)

    print("\n" + "=" * 65)
    print("                     TRAINING PIPELINE SUMMARY                   ")
    print("=" * 65)
    print(f"1. Model saved at            : {model_save_path}")
    print(f"2. Class labels saved at     : {class_names_path}")
    print(f"3. Metric curves saved at    : {os.path.join(OUTPUTS_DIR, 'accuracy_loss.png')}")
    print(f"4. Confusion matrix saved at : {os.path.join(OUTPUTS_DIR, 'confusion_matrix.png')}")
    print(f"5. Metrics report saved at   : {os.path.join(OUTPUTS_DIR, 'classification_report.txt')}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
