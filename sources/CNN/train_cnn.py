"""
Car image classifier - CNN (7 classes) built with TensorFlow/Keras.

This is the single source of truth for the CNN part of the project. The Jupyter
notebook ``main.ipynb`` is only a thin wrapper around this module, so both
entry points always behave identically.

What this script fixes compared to the original notebook
--------------------------------------------------------
1. The test set is NEVER used for validation (the notebook passed the test
   generator as ``validation_data``, which is data leakage).
2. A dedicated validation split is carved out of ``Dataset/Train``.
3. The classifier head no longer flattens 123 008 features into a Dense layer.
   ``GlobalAveragePooling2D`` + ``BatchNormalization`` + ``Dropout`` keep the
   parameter count small.
4. Two architectures are available:
      * ``scratch``      - the small CNN trained from random init (~103k params)
      * ``mobilenetv2`` - MobileNetV2 pre-trained on ImageNet (default).
                          This is what makes the predictions *balanced*: a
                          dataset of only 3 352 images is far too small to
                          train a CNN from scratch, and the from-scratch model
                          collapses onto the majority class (Audi).
5. Training is done in two stages: the backbone is frozen first, then
   fine-tuned with a small learning rate.
6. Class imbalance is handled with *effective-number* class weights (much less
   aggressive than plain inverse frequency) plus a **prior correction** at
   inference time, which removes the "everything is Audi" bias.
7. ``EarlyStopping`` / ``ModelCheckpoint`` / ``ReduceLROnPlateau`` are used, and
   the *best* checkpoint (by validation accuracy) is the one that gets saved.
8. The class order is written to ``labels.json`` next to the model instead of
   being hard-coded inside the prediction function.
9. A confusion matrix, a per-class classification report and the balanced
   accuracy are produced for every run.

Usage
-----
    python train_cnn.py --mode train                       # train + evaluate
    python train_cnn.py --mode train --arch mobilenetv2    # default
    python train_cnn.py --mode evaluate                    # evaluate a saved model
    python train_cnn.py --mode predict --image a.jpg       # predict on one image
    python train_cnn.py --mode serve                       # launch the Gradio UI
    python train_cnn.py --mode train --smoke               # 1-epoch sanity check
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent          # sources/CNN
DATASET_DIR = HERE.parent / "Dataset"           # sources/Dataset
TRAIN_DIR = DATASET_DIR / "Train"
TEST_DIR = DATASET_DIR / "Test"
OUTPUT_DIR = HERE / "outputs" / "car_cnn"

MODEL_PATH = OUTPUT_DIR / "model.keras"
LABELS_PATH = OUTPUT_DIR / "labels.json"
META_PATH = OUTPUT_DIR / "meta.json"
HISTORY_PATH = OUTPUT_DIR / "history.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

CLASS_NAMES = [
    "Audi",
    "Hyundai Creta",
    "Mahindra Scorpio",
    "Rolls Royce",
    "Maruti Suzuki Swift",
    "Tata Safari",
    "Toyota Innova",
]

# Folder name on disk -> display name. ``Dataset/Train/Swift`` but the class is
# called "Maruti Suzuki Swift" in the reports.
FOLDER_TO_CLASS = {
    "Swift": "Maruti Suzuki Swift",
}

ARCHITECTURES = ("scratch", "mobilenetv2")
DEFAULT_IMAGE_SIZE = {"scratch": 256, "mobilenetv2": 224}
BATCH_SIZE = 32
NUM_CLASSES = len(CLASS_NAMES)
SEED = 42


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def set_seed(seed: int = SEED) -> None:
    """Make the run reproducible (numpy + tensorflow + python random)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:  # pragma: no cover - tensorflow is a hard requirement
        pass


def log(message: str) -> None:
    """Print a timestamped line so the console output is easy to follow."""
    from datetime import datetime

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _configure_ssl() -> None:
    """
    Make the Keras weight download work on macOS/Windows Python builds whose
    bundled CA store is broken (raises ``CERTIFICATE_VERIFY_FAILED``).
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except Exception:  # pragma: no cover - certifi is optional
        pass


def check_dataset() -> None:
    """Fail fast with a helpful message if the dataset is missing."""
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(
            f"Training folder not found: {TRAIN_DIR}\n"
            "Expected structure:\n"
            "  sources/Dataset/Train/<ClassName>/*.jpg"
        )
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    for base in (TRAIN_DIR, TEST_DIR):
        folders = sorted(p for p in base.iterdir() if p.is_dir())
        if not folders:
            raise RuntimeError(f"No class sub-folders inside {base}")
        log(f"{base.name}: {len(folders)} classes -> {[p.name for p in folders]}")


def class_counts(dataset_dir: Path) -> Counter:
    """Count how many images each class folder holds."""
    counter: Counter = Counter()
    for folder in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        counter[folder.name] = sum(1 for f in folder.iterdir() if f.is_file())
    return counter


def sorted_class_folders(dataset_dir: Path) -> list[str]:
    """Folder names in the exact order ``image_dataset_from_directory`` uses."""
    return sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())


def compute_class_weight(dataset_dir: Path, mode: str = "effective") -> dict[int, float]:
    """
    Weight every class so the rare brands are not drowned out by the frequent
    ones (Audi has ~3x more images than Hyundai Creta).

    Two strategies:

    ``balanced``    plain inverse frequency
                    ``w_c = N / (K * n_c)`` - very aggressive, it over-boosts
                    the rare classes and destabilises early training.
    ``effective``  effective number of samples (Cui et al. 2019)
                    ``w_c = (1 - beta) / (1 - beta ** n_c)`` - much gentler and
                    it works far better with transfer learning.
                    This is the default.

    Keras expects the keys to be the *integer class indices*, which follow the
    sorted folder order.

    Every weight is finally rescaled so the **mean weight is 1.0**. Without
    that normalisation the effective-number weights are around 0.002, which
    silently shrinks every gradient by ~500x and makes the model learn almost
    nothing within the epoch budget.
    """
    counts = class_counts(dataset_dir)
    folder_order = sorted_class_folders(dataset_dir)
    total = sum(counts.values())
    n_classes = len(counts)

    if mode == "none":
        return {i: 1.0 for i in range(n_classes)}

    if mode == "balanced":
        raw = {
            folder_order.index(name): total / (n_classes * count)
            for name, count in counts.items()
        }
    else:  # "effective"
        beta = 0.999
        raw = {}
        for i, name in enumerate(folder_order):
            n = counts.get(name, 0)
            raw[i] = (1.0 - beta) / (1.0 - beta**n) if n > 0 else 1.0

    # Normalise so that the average sample weight is exactly 1.0.
    mean = sum(raw.values()) / len(raw)
    return {i: w / mean for i, w in raw.items()}


def class_priors(dataset_dir: Path) -> np.ndarray:
    """Relative frequency of every class - used for the prior correction."""
    counts = class_counts(dataset_dir)
    folder_order = sorted_class_folders(dataset_dir)
    total = sum(counts.values())
    return np.array([counts.get(name, 0) / total for name in folder_order])


# --------------------------------------------------------------------------------------
# Data pipeline
# --------------------------------------------------------------------------------------


def build_datasets(
    image_size: int = 224,
    batch_size: int = BATCH_SIZE,
    validation_split: float = 0.2,
    smoke: bool = False,
):
    """
    Build the train / validation / test ``tf.data.Dataset`` objects.

    * Train folder -> split into train (80%) and validation (20%).
    * Test folder  -> used only for the final evaluation.
    * Augmentation (flip / rotation / zoom / brightness) is applied to the
      training images only.
    """
    import tensorflow as tf

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        validation_split=validation_split,
        subset="training",
        seed=SEED,
        labels="inferred",
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        validation_split=validation_split,
        subset="validation",
        seed=SEED,
        labels="inferred",
        shuffle=True,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        labels="inferred",
        shuffle=False,
    )

    if smoke:
        # Keep only a few batches so the sanity check finishes in < 1 minute.
        train_ds = train_ds.take(4)
        val_ds = val_ds.take(2)
        test_ds = test_ds.take(2)
        log("SMOKE MODE: using only a handful of batches")

    # --- Normalisation (ALL splits) ----------------------------------------------
    # IMPORTANT: the model is fed pixels scaled to [0, 1] everywhere. The same
    # scaling must be applied here and in ``preprocess_image()``; if the two
    # disagree the network sees a completely different input distribution at
    # inference time and every prediction becomes wrong.
    normalize = tf.keras.Sequential(
        [tf.keras.layers.Rescaling(1.0 / 255.0)],
        name="normalize",
    )

    # --- Augmentation (training only) ---------------------------------------------
    # Kept deliberately mild: cars are roughly symmetric so a horizontal flip is
    # valid, but a large rotation/zoom would crop out the very details
    # (grille, badge, headlight shape) that separate the seven brands.
    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomZoom(height_factor=0.10, width_factor=0.10),
            tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
            tf.keras.layers.RandomBrightness(factor=0.10),
        ],
        name="augmentation",
    )
    train_ds = train_ds.map(
        lambda x, y: (normalize(augment(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda x, y: (normalize(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_ds = test_ds.map(
        lambda x, y: (normalize(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    # Sanity check: make sure the values really are in [0, 1].
    sample_x, _ = next(iter(train_ds))
    log(
        f"Pixel range after normalisation: "
        f"[{float(tf.reduce_min(sample_x)):.4f}, {float(tf.reduce_max(sample_x)):.4f}]"
    )

    log(
        f"Train batches: {train_ds.cardinality().numpy()} | "
        f"Val batches: {val_ds.cardinality().numpy()} | "
        f"Test batches: {test_ds.cardinality().numpy()}"
    )
    return train_ds, val_ds, test_ds


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


def build_scratch_model(num_classes: int = NUM_CLASSES, image_size: int = 256):
    """
    The original notebook did:
        Conv(32) -> Pool -> Conv(32) -> Pool -> Flatten(123008) -> Dense(96)
    which put 99.9% of all parameters into a single Dense layer (11.8 M) and
    overfitted badly. Here we keep the convolutional feature extractor but
    replace the huge Flatten + Dense block with a GlobalAveragePooling +
    BatchNorm + Dropout head, which generalises much better.
    """
    from tensorflow.keras import layers
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            # --- Block 1: edges, corners, colour gradients -----------------------
            layers.Input(shape=(image_size, image_size, 3)),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.10),

            # --- Block 2: textures, grilles, headlights -------------------------
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.15),

            # --- Block 3: car parts ---------------------------------------------
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.20),

            # --- Head: 128 features instead of 32768 -----------------------------
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.50),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="car_cnn",
    )
    return model


def build_mobilenetv2_model(num_classes: int = NUM_CLASSES, image_size: int = 224):
    """
    MobileNetV2 pre-trained on ImageNet with a fresh classification head.

    Why this is the default
    -----------------------
    The dataset only holds 3 352 images. A CNN trained from random
    initialisation cannot learn "this is a Rolls Royce grille" from that
    little data - it just memorises the majority class and predicts Audi for
    everything. A network pre-trained on 1.2 M ImageNet images already knows
    about shapes, textures and materials, so a couple of epochs of fine-tuning
    are enough to get a *balanced* model.

    The ImageNet weights are downloaded once and then cached in ``~/.keras``.
    """
    _configure_ssl()
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    import tensorflow as tf

    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )
    base.trainable = False

    model = tf.keras.Sequential(
        [
            base,
            layers.Dropout(0.30),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="car_mobilenetv2",
    )
    return model


def build_model(
    arch: str = "mobilenetv2",
    num_classes: int = NUM_CLASSES,
    image_size: int = 224,
):
    """Factory that returns the requested architecture."""
    if arch == "scratch":
        return build_scratch_model(num_classes, image_size)
    if arch == "mobilenetv2":
        return build_mobilenetv2_model(num_classes, image_size)
    raise ValueError(f"Unknown architecture '{arch}'. Choose from {ARCHITECTURES}.")


def _iter_layers(layer):
    """Yield a layer and all of its nested sub-layers (depth first)."""
    yield layer
    for sub in getattr(layer, "layers", []) or []:
        yield from _iter_layers(sub)


def unfreeze_backbone(model) -> None:
    """
    Second fine-tuning stage: unfreeze the MobileNetV2 backbone.

    The BatchNorm layers stay frozen - their running statistics were estimated
    on ImageNet and letting them drift on 2 682 images destroys them.
    """
    from tensorflow.keras import layers

    # Unfreeze everything first (setting ``trainable`` on a nested model
    # propagates down to its sub-layers).
    for layer in model.layers:
        layer.trainable = True

    # Then re-freeze every BatchNorm inside the tree.
    frozen_bn = 0
    for layer in _iter_layers(model):
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1

    log(f"Backbone unfrozen ({frozen_bn} BatchNorm layers kept frozen)")


def compile_model(model, learning_rate: float = 1e-3):
    """Adam + sparse categorical cross-entropy (labels are integers 0..6)."""
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------------------
# Prior correction (removes the "everything is Audi" bias)
# --------------------------------------------------------------------------------------


def apply_prior_correction(probs: np.ndarray, priors: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Re-balance the predicted probabilities by the training prior.

    This is the classic *logit adjustment* / *prior correction* trick. The
    network is trained with class weights, so it already compensates for the
    imbalance to some degree; this step removes the remaining bias in a
    principled way:

        q_c  proportional to  p_c / prior_c ** alpha

    ``alpha = 0`` keeps the raw output, ``alpha = 1`` fully removes the prior.
    The value is picked on the **validation** split (never on the test set) by
    :func:`select_prior_alpha`.
    """
    if alpha == 0:
        return probs
    p = np.clip(np.asarray(probs, dtype="float64"), 1e-9, 1.0)
    logits = np.log(p) - alpha * np.log(np.clip(priors, 1e-9, 1.0))
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype("float32")


def _predict_probas(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, probabilities) for a whole dataset."""
    y_true, probs = [], []
    for images, labels in dataset:
        probs.append(model.predict(images, verbose=0))
        y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.concatenate(probs, axis=0)


def select_prior_alpha(model, val_ds, priors: np.ndarray) -> float:
    """
    Pick the prior-correction strength that gives the best **balanced**
    accuracy on the validation split.

    Balanced accuracy (the mean of the per-class recalls) is used instead of
    plain accuracy because plain accuracy is exactly the metric that is
    dominated by the big classes - the thing we are trying to fix.
    """
    from sklearn.metrics import balanced_accuracy_score

    y_true, probs = _predict_probas(model, val_ds)
    if len(y_true) == 0:
        return 0.0

    best_alpha, best_score = 0.0, -1.0
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        corrected = apply_prior_correction(probs, priors, alpha)
        score = balanced_accuracy_score(y_true, np.argmax(corrected, axis=1))
        log(f"  prior alpha={alpha:.2f} -> val balanced accuracy {score:.4f}")
        if score > best_score:
            best_alpha, best_score = alpha, score
    log(f"Selected prior correction alpha = {best_alpha:.2f}")
    return best_alpha


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def train(args) -> dict:
    import tensorflow as tf

    set_seed(args.seed)
    ensure_output_dir()
    check_dataset()

    image_size = args.image_size or DEFAULT_IMAGE_SIZE[args.arch]

    counts = class_counts(TRAIN_DIR)
    log("Train images per class: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    class_weight = compute_class_weight(TRAIN_DIR, mode=args.class_weight)
    log(
        "Class weights ("
        + args.class_weight
        + "): "
        + ", ".join(
            f"{CLASS_NAMES[idx]}={w:.2f}" for idx, w in sorted(class_weight.items())
        )
    )

    train_ds, val_ds, test_ds = build_datasets(
        image_size=image_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        smoke=args.smoke,
    )

    model = build_model(args.arch, NUM_CLASSES, image_size)
    compile_model(model, args.learning_rate)
    model.summary(print_fn=log)

    def make_callbacks(patience: int):
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

    # One shared tracker across both stages: the checkpoint must keep the best
    # model of the *whole* run, even when the fine-tuning stage is worse than
    # the frozen stage.
    best = {"val_accuracy": -1.0, "epoch": 0}
    best_path = OUTPUT_DIR / "best_checkpoint.keras"

    class BestCheckpoint(tf.keras.callbacks.Callback):
        """Save the weights whenever validation accuracy reaches a new peak."""

        def on_epoch_end(self, epoch, logs=None):
            score = (logs or {}).get("val_accuracy")
            if score is None:
                return
            if score > best["val_accuracy"]:
                best["val_accuracy"] = float(score)
                best["epoch"] = int(epoch) + 1
                self.model.save(best_path)
                log(
                    f"New best validation accuracy {score:.4f} "
                    f"at epoch {best['epoch']} -> checkpoint saved"
                )

    def fit_epochs(epochs: int, initial_epoch: int = 0):
        return model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=initial_epoch + epochs,
            class_weight=class_weight,
            callbacks=make_callbacks(args.patience) + [BestCheckpoint()],
            verbose=2,
        )

    history_data: dict[str, list[float]] = {}

    # ---- Stage 1: frozen backbone -------------------------------------------------
    freeze_epochs = 0 if args.arch == "scratch" else args.freeze_epochs
    if freeze_epochs > 0:
        log(f"Stage 1/2: frozen backbone for {freeze_epochs} epochs @ lr={args.learning_rate}")
        history = fit_epochs(freeze_epochs)
        history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    else:
        log(f"Training from scratch for up to {args.epochs} epochs")

    # ---- Stage 2: fine-tune the whole network ------------------------------------
    epochs_done = len(history_data.get("loss", []))
    if args.arch == "mobilenetv2":
        remaining = args.epochs - epochs_done
        if remaining > 0:
            log(f"Stage 2/2: fine-tuning backbone for {remaining} epochs @ lr={args.fine_tune_lr}")
            unfreeze_backbone(model)
            compile_model(model, args.fine_tune_lr)
            history = fit_epochs(remaining, initial_epoch=epochs_done)
            for key, values in history.history.items():
                history_data.setdefault(key, []).extend(float(v) for v in values)
    elif not history_data:
        # ``scratch`` -> single run over all epochs.
        log(f"Training for up to {args.epochs} epochs (early stopping enabled)")
        history = fit_epochs(args.epochs)
        history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    HISTORY_PATH.write_text(json.dumps(history_data, indent=2))
    log(f"History written to {HISTORY_PATH}")

    # The checkpoint holds the best model (by validation accuracy) of the run.
    if best_path.is_file():
        model = tf.keras.models.load_model(best_path)
        log(
            f"Restored the best checkpoint "
            f"(val accuracy {best['val_accuracy']:.4f} at epoch {best['epoch']})"
        )
    model.save(MODEL_PATH)
    best_path.unlink(missing_ok=True)
    save_labels()
    log(f"Model saved to {MODEL_PATH}")

    # ---- Prior correction strength, chosen on the validation split ----------------
    priors = class_priors(TRAIN_DIR)
    alpha = select_prior_alpha(model, val_ds, priors)

    META_PATH.write_text(
        json.dumps(
            {
                "architecture": args.arch,
                "image_size": int(image_size),
                "class_names": CLASS_NAMES,
                "class_folders": sorted_class_folders(TRAIN_DIR),
                "priors": {CLASS_NAMES[i]: float(priors[i]) for i in range(NUM_CLASSES)},
                "prior_alpha": float(alpha),
            },
            indent=2,
        )
    )
    log(f"Metadata written to {META_PATH}")

    metrics = evaluate(model, test_ds, priors=priors, alpha=alpha, plot=True)

    best_epoch = int(best["epoch"]) if best["epoch"] else int(np.argmax(history_data["val_accuracy"])) + 1
    metrics["best_epoch"] = best_epoch
    metrics["best_val_accuracy"] = float(best["val_accuracy"])
    metrics["final_train_loss"] = history_data["loss"][-1]
    metrics["final_val_loss"] = history_data["val_loss"][-1]
    metrics["final_train_accuracy"] = history_data["accuracy"][-1]
    metrics["final_val_accuracy"] = history_data["val_accuracy"][-1]
    metrics["num_parameters"] = int(model.count_params())
    metrics["epochs_run"] = len(history_data["loss"])
    metrics["architecture"] = args.arch
    metrics["image_size"] = int(image_size)
    metrics["prior_alpha"] = float(alpha)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log(f"Metrics written to {METRICS_PATH}")
    return metrics


def save_labels() -> None:
    """Save the class order so predictions never depend on hard-coded strings."""
    indices = {name: i for i, name in enumerate(CLASS_NAMES)}
    LABELS_PATH.write_text(
        json.dumps({"class_indices": indices, "class_names": CLASS_NAMES}, indent=2)
    )
    log(f"Labels written to {LABELS_PATH}")


def load_labels() -> list[str]:
    """Read the class order back from disk (falls back to the constant list)."""
    if LABELS_PATH.is_file():
        data = json.loads(LABELS_PATH.read_text())
        return data["class_names"]
    return CLASS_NAMES


def load_meta() -> dict:
    """Read the training metadata (architecture, image size, prior alpha)."""
    if META_PATH.is_file():
        return json.loads(META_PATH.read_text())
    return {
        "architecture": "scratch",
        "image_size": DEFAULT_IMAGE_SIZE["scratch"],
        "class_names": load_labels(),
        "prior_alpha": 0.0,
    }


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def evaluate(
    model,
    test_ds,
    priors: np.ndarray | None = None,
    alpha: float = 0.0,
    plot: bool = True,
) -> dict:
    """Evaluate on the test set and build a confusion matrix."""
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )

    y_true, probs = _predict_probas(model, test_ds)
    if priors is None:
        priors = class_priors(TRAIN_DIR)
    corrected = apply_prior_correction(probs, priors, alpha)

    y_pred = np.argmax(corrected, axis=1)
    loss = float(
        -np.mean(
            np.log(
                np.clip(
                    corrected[np.arange(len(y_true)), y_true],
                    1e-7,
                    1,
                )
            )
        )
    )
    accuracy = float(np.mean(y_true == y_pred))
    balanced = float(balanced_accuracy_score(y_true, y_pred))

    # Always report all 7 classes, even when a subset of them appears in the
    # evaluated batch (this happens in --smoke mode).
    all_labels = list(range(NUM_CLASSES))

    print("\nClassification report")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=all_labels,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    log(
        f"Test loss: {loss:.4f} | Test accuracy: {accuracy:.4f} | "
        f"Balanced accuracy: {balanced:.4f}"
    )

    # How many predictions does each class receive? A healthy model spreads
    # them out; a collapsed model dumps everything into one class.
    pred_counts = np.bincount(y_pred, minlength=NUM_CLASSES)
    log("Predicted label distribution: " + ", ".join(
        f"{CLASS_NAMES[i]}={pred_counts[i]}" for i in range(NUM_CLASSES)
    ))

    if plot:
        plot_confusion_matrix(cm)
        plot_training_curves_from_history()

    return {
        "test_loss": loss,
        "test_accuracy": accuracy,
        "balanced_accuracy": balanced,
        "predicted_label_distribution": {
            CLASS_NAMES[i]: int(pred_counts[i]) for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm) -> None:
    """Render the confusion matrix as a heat-map PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test set)")

    # Write the raw counts in every cell.
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=9,
            )
    fig.tight_layout()
    path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Confusion matrix saved to {path}")


def plot_training_curves_from_history() -> None:
    """Re-draw the accuracy / loss curves from ``history.json``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not HISTORY_PATH.is_file():
        return
    history = json.loads(HISTORY_PATH.read_text())
    epochs = range(len(history["loss"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.plot(epochs, history["accuracy"], label="Training Accuracy")
    ax1.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    ax1.set_title("Training & Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["loss"], label="Training Loss")
    ax2.plot(epochs, history["val_loss"], label="Validation Loss")
    ax2.set_title("Training & Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"Training curves saved to {path}")


# --------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------


def preprocess_image(img, image_size: int = 224) -> np.ndarray:
    """
    Turn any input image into the exact tensor the model was trained on.

    The four steps must match training: RGB, resize, float32, /255, plus the
    batch dimension. Getting any of them wrong silently ruins the prediction.
    """
    from PIL import Image

    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    img = img.convert("RGB")                 # guarantees 3 channels
    img = img.resize((image_size, image_size))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)       # (1, H, W, 3)


def predict_image(img, model=None, labels: list[str] | None = None) -> dict:
    """
    Return a dict of {class_name: probability} for one image.

    This is the function handed to the Gradio interface. The prior correction
    learned during training is applied here as well, so the probabilities shown
    to the user are the balanced ones.
    """
    if model is None:
        model = load_trained_model()
    if labels is None:
        labels = load_labels()

    meta = load_meta()
    image_size = int(meta.get("image_size", 224))
    alpha = float(meta.get("prior_alpha", 0.0))

    probabilities = model.predict(preprocess_image(img, image_size), verbose=0)[0]

    if alpha > 0:
        priors = np.array(
            [meta.get("priors", {}).get(name, 1.0 / len(labels)) for name in labels],
            dtype="float64",
        )
        # ``probabilities`` is already 1-D (one image -> NUM_CLASSES), and
        # ``apply_prior_correction`` works on the last axis, so no extra
        # indexing is needed here.
        probabilities = apply_prior_correction(probabilities, priors, alpha)

    return {labels[i]: float(probabilities[i]) for i in range(len(labels))}


def load_trained_model():
    """Load the saved model, raising a clear error when it is missing."""
    import tensorflow as tf

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Train it first:  python train_cnn.py --mode train"
        )
    log(f"Loading model from {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------------------------------------------
# Gradio web interface
# --------------------------------------------------------------------------------------


def serve(args) -> None:
    import gradio as gr
    import pandas as pd

    model = load_trained_model()
    labels = load_labels()
    meta = load_meta()
    log(
        f"Serving architecture '{meta.get('architecture', '?')}' "
        f"with image size {meta.get('image_size', '?')} "
        f"and prior correction alpha={meta.get('prior_alpha', 0.0)}"
    )

    def _predict(img):
        """Return (label_dict, probability_dataframe) for the Gradio outputs."""
        if img is None:
            return {}, pd.DataFrame({"Class": [], "Probability": []})
        probs = predict_image(img, model=model, labels=labels)
        frame = pd.DataFrame(
            {"Class": list(probs.keys()), "Probability": list(probs.values())}
        )
        return probs, frame

    with gr.Blocks(title="PPR501 - Car Image Classifier") as demo:
        gr.Markdown(
            "# Car Image Classifier\n"
            "Phân loại ảnh xe thuộc 7 nhóm: Audi, Hyundai Creta, Mahindra "
            "Scorpio, Rolls Royce, Maruti Suzuki Swift, Tata Safari, Toyota Innova."
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Ảnh xe")
                submit = gr.Button("Phân loại", variant="primary")
            with gr.Column():
                output = gr.Label(num_top_classes=len(labels), label="Kết quả")
                # ``gr.Bar`` was removed in Gradio 5/6 -> ``gr.BarPlot``.
                if hasattr(gr, "BarPlot"):
                    prob_chart = gr.BarPlot(
                        x="Class",
                        y="Probability",
                        y_title="Probability",
                        sort="-y",
                        label="Xác suất từng lớp",
                    )
                else:  # pragma: no cover - very old Gradio
                    prob_chart = gr.Dataframe(label="Xác suất từng lớp")

        submit.click(_predict, inputs=image_input, outputs=[output, prob_chart])
        image_input.change(_predict, inputs=image_input, outputs=[output, prob_chart])

    log(f"Starting the Gradio interface on http://localhost:{args.port} ...")
    demo.launch(share=args.share, server_port=args.port)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="CNN car-image classifier (7 classes).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict", "serve"],
        default="train",
        help="What to do.",
    )
    parser.add_argument(
        "--arch",
        choices=list(ARCHITECTURES),
        default="mobilenetv2",
        help="Model architecture. 'mobilenetv2' is pre-trained on ImageNet.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input resolution. Defaults to 224 (mobilenetv2) / 256 (scratch).",
    )
    parser.add_argument("--epochs", type=int, default=20,
                        help="Total number of epochs (both stages combined).")
    parser.add_argument("--freeze-epochs", type=int, default=5,
                        help="Epochs with a frozen backbone (mobilenetv2 only).")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4,
                        help="Learning rate for the fine-tuning stage.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate of the first stage.")
    parser.add_argument("--patience", type=int, default=6,
                        help="Early-stopping patience on validation accuracy.")
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--class-weight",
        choices=["effective", "balanced", "none"],
        default="effective",
        help="Strategy used to rebalance the classes during training.",
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an image (used with --mode predict).")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link.")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run for a quick sanity check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        train(args)

    elif args.mode == "evaluate":
        model = load_trained_model()
        meta = load_meta()
        _, _, test_ds = build_datasets(
            image_size=int(meta.get("image_size", 224)),
            batch_size=args.batch_size,
            smoke=args.smoke,
        )
        evaluate(
            model,
            test_ds,
            priors=class_priors(TRAIN_DIR),
            alpha=float(meta.get("prior_alpha", 0.0)),
            plot=True,
        )

    elif args.mode == "predict":
        if not args.image:
            raise SystemExit("Please provide an image: --image path/to/photo.jpg")
        from PIL import Image

        model = load_trained_model()
        img = Image.open(args.image)
        result = predict_image(img, model=model)
        print("\nProbabilities:")
        for name, prob in sorted(result.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<22} {prob * 100:6.2f}%")
        best = max(result, key=result.get)
        print(f"\nPredicted class: {best}  ({result[best] * 100:.2f}%)")

    elif args.mode == "serve":
        serve(args)


if __name__ == "__main__":
    main()
