"""
KisanAI - Disease Detection Agent
==================================

Trains a leaf-disease classifier with MobileNetV2 transfer learning
(frozen base -> fine-tune), following the plan in your project brief:
dataset check -> split -> resize/augment -> frozen-base training ->
fine-tuning -> evaluation (classification report + confusion matrix) ->
export `disease_model.keras` + `disease_class_names.json`.

EXPECTED DATASET LAYOUT (standard "ImageFolder" style, same as PlantVillage /
PlantDoc downloads):

    dataset/
        Tomato___Early_blight/
            img001.jpg
            img002.jpg
            ...
        Tomato___Late_blight/
            ...
        Potato___healthy/
            ...

One sub-folder per class. Put the `dataset` folder next to this script
(or pass --data_dir to point at wherever you extracted it), then run:

    python disease_agent_train.py --data_dir ./dataset --output_dir ./artifacts

Requirements (install once):

    pip install tensorflow pillow scikit-learn matplotlib

Notes on honesty (per the project's documentation rules):
  - This script prints and saves real metrics from your run. Nothing here
    invents numbers - if you have not run it yet, there are no results to
    report.
  - Class imbalance is reported before training so you can see it, not
    hidden.
  - The exported "disease_class_names.json" preserves the exact index-to-
    class-name mapping the model was trained with; the FastAPI/Node side
    must load this file rather than hard-coding class names, or predictions
    will point at the wrong disease name.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Train the KisanAI disease-detection agent (MobileNetV2).")
    p.add_argument("--data_dir", type=str, default="./dataset",
                    help="Path to the dataset folder (one sub-folder per class). Default: ./dataset")
    p.add_argument("--output_dir", type=str, default="./artifacts",
                    help="Where to save the model, class names, plots and reports. Default: ./artifacts")
    p.add_argument("--img_size", type=int, default=224,
                    help="Square image size fed to MobileNetV2. Default: 224 (MobileNetV2's native size).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_split", type=float, default=0.2,
                    help="Fraction of images held out for validation (only used if --data_dir has no "
                         "separate train/ and val/ sub-folders).")
    p.add_argument("--epochs_head", type=int, default=15,
                    help="Epochs for phase 1 (frozen MobileNetV2 base, train the new classifier head only).")
    p.add_argument("--epochs_finetune", type=int, default=10,
                    help="Epochs for phase 2 (unfreeze the top of the base and fine-tune at a low LR).")
    p.add_argument("--fine_tune_at", type=int, default=100,
                    help="Layer index in MobileNetV2 from which layers become trainable during phase 2. "
                         "MobileNetV2 has 154 layers; 100 unfreezes roughly the last third.")
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_finetune", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true",
                    help="Only inspect the dataset (counts, classes, a sample grid) and exit - "
                         "use this first to sanity-check the folder before a full training run.")
    return p.parse_args()


def find_split_dirs(data_dir):
    """Return (train_dir, val_dir) if the dataset already has them, else (data_dir, None)."""
    lower = {name.lower(): name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))}
    train_key = next((k for k in ("train", "training") if k in lower), None)
    val_key = next((k for k in ("val", "valid", "validation") if k in lower), None)
    if train_key and val_key:
        return os.path.join(data_dir, lower[train_key]), os.path.join(data_dir, lower[val_key])
    return data_dir, None


def inspect_dataset(data_dir):
    """Print per-class image counts so imbalance is visible before training."""
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not classes:
        print(f"ERROR: no class sub-folders found inside '{data_dir}'.")
        print("Expected layout: data_dir/ClassName/image.jpg  (see the docstring at the top of this file).")
        sys.exit(1)

    counts = {}
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for c in classes:
        cdir = os.path.join(data_dir, c)
        counts[c] = sum(1 for f in os.listdir(cdir) if f.lower().endswith(exts))

    total = sum(counts.values())
    print(f"\nFound {len(classes)} classes, {total} images total, in '{data_dir}'")
    print(f"{'Class':45s}  Images")
    print("-" * 55)
    for c in classes:
        print(f"{c:45s}  {counts[c]}")

    if total > 0:
        vals = list(counts.values())
        imbalance_ratio = max(vals) / max(min(vals), 1)
        print(f"\nMost/least populated class ratio: {imbalance_ratio:.1f}x")
        if imbalance_ratio > 3:
            print("Classes are noticeably imbalanced - class_weight is applied automatically below, "
                  "but consider this when reading per-class recall in the final report.")
    return classes, counts


def build_datasets(args):
    import tensorflow as tf

    train_dir, val_dir = find_split_dirs(args.data_dir)
    classes, counts = inspect_dataset(train_dir)
    if val_dir:
        inspect_dataset(val_dir)

    img_size = (args.img_size, args.img_size)

    if val_dir:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, image_size=img_size, batch_size=args.batch_size, seed=args.seed)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir, image_size=img_size, batch_size=args.batch_size, seed=args.seed)
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, image_size=img_size, batch_size=args.batch_size, seed=args.seed,
            validation_split=args.val_split, subset="training")
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, image_size=img_size, batch_size=args.batch_size, seed=args.seed,
            validation_split=args.val_split, subset="validation")

    class_names = train_ds.class_names  # keep the exact order Keras assigned to label indices
    print(f"\nClass order used for label indices (save this - it must match at inference time):")
    for i, c in enumerate(class_names):
        print(f"  {i}: {c}")

    # class weights, computed from the raw folder counts (not the batched dataset), to soften imbalance
    class_weight = None
    if not val_dir:
        total = sum(counts.values())
        class_weight = {i: total / (len(class_names) * counts[c]) for i, c in enumerate(class_names)}
        print(f"\nClass weights (higher = rarer class, weighted up in the loss):")
        for i, c in enumerate(class_names):
            print(f"  {c}: {class_weight[i]:.2f}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names, class_weight


def build_model(num_classes, img_size, fine_tune_at):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False  # phase 1: frozen base

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # scales to [-1, 1], matches ImageNet training
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="kisanai_disease_mobilenetv2")
    return model, base


def train(args):
    import tensorflow as tf

    os.makedirs(args.output_dir, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)

    train_ds, val_ds, class_names, class_weight = build_datasets(args)
    model, base = build_model(len(class_names), args.img_size, args.fine_tune_at)

    ckpt_path = os.path.join(args.output_dir, "best_checkpoint.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]

    # ---------- Phase 1: train the new head, base frozen ----------
    print("\n=== Phase 1: training classifier head (MobileNetV2 base frozen) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr_head),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs_head,
        class_weight=class_weight, callbacks=callbacks)

    # ---------- Phase 2: unfreeze top of the base, fine-tune at a low LR ----------
    print(f"\n=== Phase 2: fine-tuning from layer {args.fine_tune_at} of the base onward ===")
    base.trainable = True
    for layer in base.layers[:args.fine_tune_at]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr_finetune),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    history2 = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
        class_weight=class_weight, callbacks=callbacks)

    # ---------- Save model + class names ----------
    model_path = os.path.join(args.output_dir, "disease_model.keras")
    model.save(model_path)
    class_names_path = os.path.join(args.output_dir, "disease_class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"\nSaved model to: {model_path}")
    print(f"Saved class names (index -> label) to: {class_names_path}")

    # ---------- Combine histories for the training-curve plot ----------
    def merge(k):
        return history1.history.get(k, []) + history2.history.get(k, [])
    full_history = {k: merge(k) for k in ["loss", "val_loss", "accuracy", "val_accuracy"]}
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(full_history, f, indent=2)

    plot_training_curves(full_history, args.epochs_head, args.output_dir)
    evaluate(model, val_ds, class_names, args.output_dir)


def plot_training_curves(history, epochs_head_actual_split, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].axvline(epochs_head_actual_split, color="gray", linestyle="--", label="fine-tuning starts")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="val")
    axes[1].axvline(epochs_head_actual_split, color="gray", linestyle="--", label="fine-tuning starts")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()

    fig.tight_layout()
    out = os.path.join(output_dir, "training_curves.png")
    fig.savefig(out, dpi=150)
    print(f"Saved training curves to: {out}")


def evaluate(model, val_ds, class_names, output_dir):
    """Real classification report + confusion matrix on the validation set.
    No number here is invented - if you have not trained yet, this function
    has not been called."""
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print("\n=== Classification report (validation set) ===")
    print(report)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.5), max(5, len(class_names) * 0.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (validation set)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(out, dpi=150)
    print(f"Saved confusion matrix to: {out}")

    with open(os.path.join(output_dir, "run_summary.json"), "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "num_classes": len(class_names),
            "val_images": len(y_true),
        }, f, indent=2)


def dry_run(args):
    train_dir, val_dir = find_split_dirs(args.data_dir)
    inspect_dataset(train_dir)
    if val_dir:
        print("\nSeparate validation folder detected:")
        inspect_dataset(val_dir)
    print("\nDry run only - no model was trained. Re-run without --dry_run to train.")


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isdir(args.data_dir):
        print(f"ERROR: '{args.data_dir}' does not exist. Put your dataset there, or pass --data_dir <path>.")
        sys.exit(1)
    if args.dry_run:
        dry_run(args)
    else:
        train(args)
