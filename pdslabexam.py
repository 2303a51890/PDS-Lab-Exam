# ============================================================
# APPLE PLANT DISEASE CLASSIFICATION
# MAIN TRAINING PIPELINE
#
# Dataset: Plant Pathology 2021 - FGVC8
# Models: ResNet50 + EfficientNetB4
# Split: 70 / 15 / 15 stratified
# ============================================================

import os
import json
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3

DATASET_ROOT = (
    "/kaggle/input/competitions/"
    "plant-pathology-2021-fgvc8"
)

CSV_PATH = os.path.join(
    DATASET_ROOT,
    "train.csv"
)

IMAGE_DIR = os.path.join(
    DATASET_ROOT,
    "train_images"
)

PROJECT_DIR = (
    "/kaggle/working/"
    "apple_plant_disease_project"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("APPLE PLANT DISEASE CLASSIFICATION")
print("=" * 70)

print("\nTensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")

print("GPUs detected:", len(gpus))

for i, gpu in enumerate(gpus):
    print(f"GPU {i}:", gpu)

# ============================================================
# MULTI-GPU STRATEGY
# ============================================================

if len(gpus) > 1:

    strategy = tf.distribute.MirroredStrategy()

else:

    strategy = tf.distribute.get_strategy()

print("\nNumber of replicas:", strategy.num_replicas_in_sync)

# ============================================================
# LOAD DATASET
# ============================================================

print("\n========== LOADING DATASET ==========")

df = pd.read_csv(CSV_PATH)

print("CSV rows:", len(df))
print("Columns:", list(df.columns))

# Verify images exist
df["image_path"] = df["image"].apply(
    lambda x: os.path.join(
        IMAGE_DIR,
        x
    )
)

missing = (
    ~df["image_path"].apply(os.path.isfile)
).sum()

print("Missing images:", missing)

if missing > 0:
    print("WARNING: Missing images detected.")

# ============================================================
# CLASS MAPPING
# ============================================================

classes = sorted(
    df["labels"].unique()
)

NUM_CLASSES = len(classes)

class_to_idx = {
    label: i
    for i, label in enumerate(classes)
}

idx_to_class = {
    i: label
    for label, i in class_to_idx.items()
}

df["label_id"] = df["labels"].map(
    class_to_idx
)

print("\n========== CLASSES ==========")
print("Number of classes:", NUM_CLASSES)

for i, label in idx_to_class.items():

    count = (
        df["labels"] == label
    ).sum()

    print(
        f"{i:02d} | "
        f"{label:35s} | "
        f"{count}"
    )

# ============================================================
# DATASET DISTRIBUTION
# ============================================================

df["labels"].value_counts().sort_index().to_csv(
    os.path.join(
        RESULT_DIR,
        "class_distribution.csv"
    )
)

# ============================================================
# STRATIFIED 70 / 15 / 15 SPLIT
# ============================================================

print("\n========== DATASET SPLIT ==========")

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label_id"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label_id"],
    random_state=SEED
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Total:", len(df))
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print(
    f"\nTrain: {len(train_df)/len(df)*100:.2f}%"
)

print(
    f"Validation: {len(val_df)/len(df)*100:.2f}%"
)

print(
    f"Test: {len(test_df)/len(df)*100:.2f}%"
)

# Save splits
train_df.to_csv(
    os.path.join(
        PROJECT_DIR,
        "train_split.csv"
    ),
    index=False
)

val_df.to_csv(
    os.path.join(
        PROJECT_DIR,
        "validation_split.csv"
    ),
    index=False
)

test_df.to_csv(
    os.path.join(
        PROJECT_DIR,
        "test_split.csv"
    ),
    index=False
)

# Save complete metadata
df.to_csv(
    os.path.join(
        PROJECT_DIR,
        "dataset_metadata.csv"
    ),
    index=False
)

# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

distribution = pd.DataFrame({

    "Train":
        train_df["labels"].value_counts(),

    "Validation":
        val_df["labels"].value_counts(),

    "Test":
        test_df["labels"].value_counts()

}).fillna(0).astype(int)

distribution["Total"] = (
    distribution["Train"] +
    distribution["Validation"] +
    distribution["Test"]
)

print("\n========== SPLIT DISTRIBUTION ==========")
print(distribution)

distribution.to_csv(
    os.path.join(
        RESULT_DIR,
        "split_distribution.csv"
    )
)

# ============================================================
# CLASS WEIGHTS
# ============================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["label_id"].values
)

class_weights = {
    i: float(w)
    for i, w in enumerate(weights)
}

print("\n========== CLASS WEIGHTS ==========")

for i, weight in class_weights.items():

    print(
        f"{idx_to_class[i]:35s}: "
        f"{weight:.4f}"
    )

with open(
    os.path.join(
        RESULT_DIR,
        "class_weights.json"
    ),
    "w"
) as f:

    json.dump(
        {
            idx_to_class[k]: v
            for k, v in class_weights.items()
        },
        f,
        indent=2
    )

# ============================================================
# DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


def create_dataset(
    dataframe,
    training=False
):

    paths = dataframe[
        "image_path"
    ].values

    labels = dataframe[
        "label_id"
    ].values.astype(
        np.int32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels
        )
    )

    if training:

        dataset = dataset.shuffle(
            len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


train_ds = create_dataset(
    train_df,
    training=True
)

val_ds = create_dataset(
    val_df
)

test_ds = create_dataset(
    test_df
)

print("\nTensorFlow datasets ready.")

# ============================================================
# AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(
    [

        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.10
        ),

        tf.keras.layers.RandomZoom(
            0.15
        ),

        tf.keras.layers.RandomContrast(
            0.10
        )

    ],
    name="augmentation"
)

# ============================================================
# MODEL BUILDERS
# ============================================================

def build_resnet50():

    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(
            224,
            224,
            3
        )
    )

    base.trainable = False

    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    x = augmentation(inputs)

    x = tf.keras.applications.resnet50.preprocess_input(
        x
    )

    x = base(
        x,
        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        256,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.30
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    return tf.keras.Model(
        inputs,
        outputs,
        name="ResNet50"
    )


def build_efficientnetb4():

    base = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(
            224,
            224,
            3
        )
    )

    base.trainable = False

    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    x = augmentation(inputs)

    x = base(
        x,
        training=False
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        256,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.30
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    return tf.keras.Model(
        inputs,
        outputs,
        name="EfficientNetB4"
    )

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    model_name,
    builder
):

    print("\n")
    print("=" * 70)
    print("TRAINING:", model_name)
    print("=" * 70)

    with strategy.scope():

        model = builder()

        model.compile(

            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LR
            ),

            loss="sparse_categorical_crossentropy",

            metrics=[
                "accuracy"
            ]
        )

    model.summary()

    checkpoint = os.path.join(
        MODEL_DIR,
        f"{model_name}_best.weights.h5"
    )

    full_model = os.path.join(
        MODEL_DIR,
        f"{model_name}_best.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=1,
            min_lr=1e-7,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.TerminateOnNaN()

    ]

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    start = time.time()

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1
    )

    training_time = time.time() - start

    # --------------------------------------------------------
    # RESTORE BEST
    # --------------------------------------------------------

    if os.path.exists(checkpoint):

        model.load_weights(
            checkpoint
        )

    model.save(
        full_model
    )

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        history.history
    )

    history_df.to_csv(
        os.path.join(
            RESULT_DIR,
            f"{model_name}_history.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating test predictions...")

    probabilities = model.predict(
        test_ds,
        verbose=1
    )

    y_true = test_df[
        "label_id"
    ].values

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    y_true_onehot = tf.keras.utils.to_categorical(
        y_true,
        NUM_CLASSES
    )

    try:

        roc_auc = roc_auc_score(
            y_true_onehot,
            probabilities,
            multi_class="ovr",
            average="macro"
        )

    except Exception as e:

        print("ROC-AUC error:", e)
        roc_auc = None

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    cm_df.to_csv(
        os.path.join(
            RESULT_DIR,
            f"{model_name}_confusion_matrix.csv"
        )
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(
        report
    ).transpose().to_csv(
        os.path.join(
            RESULT_DIR,
            f"{model_name}_classification_report.csv"
        )
    )

    # --------------------------------------------------------
    # INFERENCE SPEED
    # --------------------------------------------------------

    benchmark_count = min(
        200,
        len(test_df)
    )

    benchmark_paths = test_df[
        "image_path"
    ].values[:benchmark_count]

    benchmark_labels = np.zeros(
        benchmark_count,
        dtype=np.int32
    )

    benchmark_ds = tf.data.Dataset.from_tensor_slices(
        (
            benchmark_paths,
            benchmark_labels
        )
    )

    benchmark_ds = benchmark_ds.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    ).batch(
        BATCH_SIZE
    )

    # Warmup
    model.predict(
        benchmark_ds.take(1),
        verbose=0
    )

    start_inf = time.perf_counter()

    model.predict(
        benchmark_ds,
        verbose=0
    )

    elapsed = time.perf_counter() - start_inf

    inference_ms = (
        elapsed /
        benchmark_count
    ) * 1000

    throughput = (
        benchmark_count /
        elapsed
    )

    # --------------------------------------------------------
    # METRICS DICTIONARY
    # --------------------------------------------------------

    metrics = {

        "Model":
            model_name,

        "Accuracy":
            float(accuracy),

        "Precision":
            float(precision),

        "Recall":
            float(recall),

        "F1_Score":
            float(f1),

        "ROC_AUC":
            None
            if roc_auc is None
            else float(roc_auc),

        "Parameters":
            int(
                model.count_params()
            ),

        "Parameters_M":
            float(
                model.count_params() /
                1e6
            ),

        "Inference_ms_per_image":
            float(inference_ms),

        "Inference_images_sec":
            float(throughput),

        "Training_Time_sec":
            float(training_time)

    }

    with open(
        os.path.join(
            RESULT_DIR,
            f"{model_name}_metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # TRAINING CURVES
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        f"{model_name} Accuracy"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            f"{model_name}_accuracy.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"{model_name} Loss"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            f"{model_name}_loss.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(model_name, "RESULTS")
    print("=" * 70)

    for key, value in metrics.items():

        print(
            f"{key}: {value}"
        )

    return metrics


# ============================================================
# TRAIN MODEL 1
# ============================================================

resnet_metrics = train_model(
    "ResNet50",
    build_resnet50
)

# ============================================================
# TRAIN MODEL 2
# ============================================================

efficientnet_metrics = train_model(
    "EfficientNetB4",
    build_efficientnetb4
)

# ============================================================
# FINAL COMPARISON
# ============================================================

comparison = pd.DataFrame([
    resnet_metrics,
    efficientnet_metrics
])

comparison.to_csv(
    os.path.join(
        RESULT_DIR,
        "baseline_model_comparison.csv"
    ),
    index=False
)

print("\n")
print("=" * 100)
print("FINAL BASELINE MODEL COMPARISON")
print("=" * 100)

print(
    comparison[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "ROC_AUC",
            "Parameters_M",
            "Inference_ms_per_image",
            "Inference_images_sec",
            "Training_Time_sec"
        ]
    ].to_string(
        index=False
    )
)

# ============================================================
# CREATE PROJECT ZIP
# ============================================================

import shutil

archive_base = (
    "/kaggle/working/"
    "apple_plant_disease_project_final"
)

archive_path = shutil.make_archive(
    archive_base,
    "zip",
    PROJECT_DIR
)

print("\nProject archive created:")
print(archive_path)

print("\n")
print("=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)