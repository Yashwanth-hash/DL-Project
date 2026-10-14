
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
FRAME_DIR = ROOT / "dataset" / "frames"
CACHE_PATH = ROOT / "dataset" / "cnn_features_cache.npz"

MODEL_PATH = ROOT / "models" / "drowsiness_cnn_gru.keras"
BEST_MODEL_PATH = ROOT / "models" / "drowsiness_gru_best.keras"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["alert", "low_vigilance", "drowsy"]

TRAIN_IDS = {"03", "04", "06", "07", "08", "09", "10", "12"}
VAL_IDS = {"01", "05"}
TEST_IDS = {"02", "11"}

SEQUENCE_LENGTH = 5
EPOCHS = 20
BATCH_SIZE = 32
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------
# 1. LOAD CACHED CNN FEATURES
# --------------------------------------------------

print("Loading cached CNN features...")

if not CACHE_PATH.exists():
    raise FileNotFoundError(
        f"Feature cache not found: {CACHE_PATH}\n"
        "Run training/train_cnn_lstm.py first to generate it."
    )

cache = np.load(CACHE_PATH, allow_pickle=False)
cached_paths = cache["paths"]
features = cache["features"]

print("Cached features:", features.shape)


# --------------------------------------------------
# 2. RECONSTRUCT FRAME METADATA
# --------------------------------------------------

print("\nReconstructing frame metadata...")

class_label_from_folder = {
    "alert": 0,
    "low_vigilance": 1,
    "drowsy": 2,
}

records = []

# The cache was created in sorted class/participant/frame order.
for path_string in cached_paths:
    path = Path(str(path_string))

    try:
        relative = path.relative_to(FRAME_DIR.resolve())
    except ValueError:
        raise RuntimeError(
            f"Cached frame is outside expected directory: {path}"
        )

    if len(relative.parts) != 3:
        raise RuntimeError(f"Unexpected frame path: {path}")

    class_name, participant, filename = relative.parts

    if class_name not in class_label_from_folder:
        raise RuntimeError(f"Unknown class folder: {class_name}")

    # Filenames: 0_frame_0000060.jpg, 5_frame_..., 10_frame_...
    import re

    match = re.fullmatch(
        r"(0|5|10)_frame_(\d+)\.jpg",
        filename,
    )

    if match is None:
        raise RuntimeError(f"Unexpected frame filename: {filename}")

    video_label = int(match.group(1))
    frame_number = int(match.group(2))

    expected_label = {
        "alert": 0,
        "low_vigilance": 5,
        "drowsy": 10,
    }[class_name]

    if video_label != expected_label:
        raise RuntimeError(f"Class/filename mismatch: {path}")

    records.append({
        "participant": participant,
        "class_name": class_name,
        "label": class_label_from_folder[class_name],
        "frame_number": frame_number,
        "video_label": video_label,
    })

if len(records) != len(features):
    raise RuntimeError("Cached features and metadata counts differ.")

print(f"Metadata records: {len(records)}")


# --------------------------------------------------
# 3. BUILD CHRONOLOGICAL SEQUENCES
# --------------------------------------------------

print("\nBuilding sequences...")

groups = {}

for index, record in enumerate(records):
    key = (
        record["participant"],
        record["class_name"],
        record["video_label"],
    )
    groups.setdefault(key, []).append(index)

X_sequences = []
y_sequences = []
sequence_participants = []

for (participant, class_name, video_label), indices in groups.items():
    indices.sort(key=lambda i: records[i]["frame_number"])

    run = []

    def save_run(run_indices):
        for start in range(
            0,
            len(run_indices) - SEQUENCE_LENGTH + 1,
            SEQUENCE_LENGTH,
        ):
            selected = run_indices[
                start:start + SEQUENCE_LENGTH
            ]

            X_sequences.append(features[selected])
            y_sequences.append(
                class_label_from_folder[class_name]
            )
            sequence_participants.append(participant)

    previous_frame = None

    for index in indices:
        frame_number = records[index]["frame_number"]

        if (
            previous_frame is not None
            and frame_number - previous_frame != 60
        ):
            save_run(run)
            run = []

        run.append(index)
        previous_frame = frame_number

    save_run(run)

X = np.asarray(X_sequences, dtype=np.float32)
y = np.asarray(y_sequences, dtype=np.int32)
participants = np.asarray(sequence_participants)

if len(X) == 0:
    raise RuntimeError("No sequences could be constructed.")

print("Sequence tensor:", X.shape)

for i, name in enumerate(CLASSES):
    print(f"{name}: {int(np.sum(y == i))} sequences")


# --------------------------------------------------
# 4. SPLIT BY PARTICIPANT
# --------------------------------------------------

train_mask = np.isin(participants, list(TRAIN_IDS))
val_mask = np.isin(participants, list(VAL_IDS))
test_mask = np.isin(participants, list(TEST_IDS))

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

if min(len(X_train), len(X_val), len(X_test)) == 0:
    raise RuntimeError("One or more dataset splits are empty.")

print("\nSplit sizes:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# --------------------------------------------------
# 5. BUILD CNN-GRU MODEL
# --------------------------------------------------

print("\nBuilding CNN-GRU model...")

gru_model = models.Sequential([
    layers.Input(shape=(SEQUENCE_LENGTH, features.shape[-1])),
    layers.GRU(64, return_sequences=False),
    layers.Dropout(0.4),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(len(CLASSES), activation="softmax"),
])

gru_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

gru_model.summary()


# --------------------------------------------------
# 6. TRAIN
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
    ),
]

history = gru_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)


# --------------------------------------------------
# 7. EVALUATE ON HELD-OUT PARTICIPANTS
# --------------------------------------------------

print("\nEvaluating CNN-GRU...")

test_probabilities = gru_model.predict(X_test, batch_size=BATCH_SIZE)
y_pred = np.argmax(test_probabilities, axis=1)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=CLASSES,
    zero_division=0,
)

print(f"\nTest accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print("\nClassification report:")
print(report)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASSES,
    yticklabels=CLASSES,
    cmap="Blues",
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN-GRU Confusion Matrix")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "cnn_gru_confusion_matrix.png",
    dpi=150,
)
plt.close()

gru_model.save(MODEL_PATH)

with open(RESULTS_DIR / "cnn_gru_report.txt", "w") as file:
    file.write(f"Test accuracy: {accuracy:.4f}\n")
    file.write(f"Macro F1: {macro_f1:.4f}\n\n")
    file.write(report)

print("\nCNN-GRU training complete.")
print("Model:", MODEL_PATH)
print("Best checkpoint:", BEST_MODEL_PATH)
print("Report:", RESULTS_DIR / "cnn_gru_report.txt")
print("Confusion matrix:", RESULTS_DIR / "cnn_gru_confusion_matrix.png")
