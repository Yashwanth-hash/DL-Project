
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ---------------- CONFIGURATION ----------------

SEED = 42
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 5

PROJECT_DIR = Path(__file__).resolve().parent.parent
FRAME_DIR = PROJECT_DIR / "dataset" / "frames"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

CLASSES = ["alert", "low_vigilance", "drowsy"]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------- COLLECT DATA ----------------

image_paths = []
labels = []
participants = set()

for label_index, class_name in enumerate(CLASSES):
    class_dir = FRAME_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Missing class directory: {class_dir}"
        )

    for image_path in class_dir.rglob("*.jpg"):
        participant = image_path.parent.name

        image_paths.append(str(image_path))
        labels.append(label_index)
        participants.add(participant)

participants = sorted(participants)

if len(participants) < 6:
    raise ValueError(
        "Too few participants found. Check dataset/frames."
    )

print(f"Total images: {len(image_paths)}")
print(f"Participants: {participants}")

# ---------------- PARTICIPANT-LEVEL SPLIT ----------------
# All frames from one participant stay in a single split.

random.shuffle(participants)

train_participants = set(participants[:8])
val_participants = set(participants[8:10])
test_participants = set(participants[10:12])

print("\nTraining participants:", sorted(train_participants))
print("Validation participants:", sorted(val_participants))
print("Testing participants:", sorted(test_participants))

# Build lists using the participant ID stored in the path.
# Expected layout: dataset/frames/class_name/participant/file.jpg

train_paths, train_labels = [], []
val_paths, val_labels = [], []
test_paths, test_labels = [], []

for path, label in zip(image_paths, labels):
    participant = Path(path).parent.name

    if participant in train_participants:
        train_paths.append(path)
        train_labels.append(label)
    elif participant in val_participants:
        val_paths.append(path)
        val_labels.append(label)
    elif participant in test_participants:
        test_paths.append(path)
        test_labels.append(label)

print("\nSplit sizes:")
print(f"Train: {len(train_paths)}")
print(f"Validation: {len(val_paths)}")
print(f"Test: {len(test_paths)}")

# ---------------- TF.DATA PIPELINE ----------------

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )
    return image, label


def make_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, np.asarray(labels, dtype=np.int32))
    )

    if training:
        dataset = dataset.shuffle(
            min(len(paths), 2000),
            seed=SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

# ---------------- BUILD MODEL ----------------

print("\nLoading MobileNetV2 pretrained weights...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.05)(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(
    len(CLASSES),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAIN ----------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "drowsiness_mobilenetv2.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---------------- FINAL TEST EVALUATION ----------------

model = tf.keras.models.load_model(
    MODEL_DIR / "drowsiness_mobilenetv2.keras"
)

test_loss, test_accuracy = model.evaluate(test_ds)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

probabilities = model.predict(test_ds)
predictions = np.argmax(probabilities, axis=1)

print("\nClassification Report:")
print(
    classification_report(
        test_labels,
        predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0
    )
)

cm = confusion_matrix(
    test_labels,
    predictions,
    labels=list(range(len(CLASSES)))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

display.plot(cmap="Blues", xticks_rotation=20)
plt.title("MobileNetV2 - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=150
)
plt.close()

# Save training history for later analysis.
np.savez(
    RESULTS_DIR / "training_history.npz",
    accuracy=history.history["accuracy"],
    val_accuracy=history.history["val_accuracy"],
    loss=history.history["loss"],
    val_loss=history.history["val_loss"]
)

print("\nTraining complete!")
print("Model saved in:", MODEL_DIR)
print("Confusion matrix saved in:", RESULTS_DIR)
