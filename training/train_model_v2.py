
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

SEED = 42
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS_HEAD = 5
EPOCHS_FINE = 8

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "frames"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CLASSES = ["alert", "low_vigilance", "drowsy"]

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Collect frames and participant IDs.
paths, labels = [], []
participants = set()

for class_index, class_name in enumerate(CLASSES):
    class_dir = DATA_DIR / class_name

    for participant_dir in sorted(class_dir.iterdir()):
        if not participant_dir.is_dir():
            continue

        participants.add(participant_dir.name)

        for image_path in participant_dir.glob("*.jpg"):
            paths.append(str(image_path))
            labels.append(class_index)

participants = sorted(participants)
if len(participants) != 12:
    raise ValueError(f"Expected 12 participants, got {len(participants)}")

# Keep the same participant split as earlier.
shuffled = participants.copy()
random.shuffle(shuffled)

train_ids = set(shuffled[:8])
val_ids = set(shuffled[8:10])
test_ids = set(shuffled[10:12])

print("Train:", sorted(train_ids))
print("Validation:", sorted(val_ids))
print("Test:", sorted(test_ids))

train_paths, train_labels = [], []
val_paths, val_labels = [], []
test_paths, test_labels = [], []

for path, label in zip(paths, labels):
    participant = Path(path).parent.name

    if participant in train_ids:
        train_paths.append(path)
        train_labels.append(label)
    elif participant in val_ids:
        val_paths.append(path)
        val_labels.append(label)
    else:
        test_paths.append(path)
        test_labels.append(label)

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (paths, np.asarray(labels, dtype=np.int32))
    )
    if training:
        ds = ds.shuffle(min(len(paths), 2000), seed=SEED)

    return (
        ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE)
    )

train_ds = make_dataset(train_paths, train_labels, True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

# Build model.
base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.08)(x)
x = tf.keras.layers.RandomZoom(0.1)(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model_path = MODEL_DIR / "drowsiness_mobilenetv2_v2.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(model_path), monitor="val_loss", save_best_only=True
    )
]

print("\nPhase 1: Train classification head")
history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks
)

# Fine-tune the later feature layers.
base.trainable = True

for layer in base.layers[:-30]:
    layer.trainable = False

# Keep BatchNorm layers frozen for stability.
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nPhase 2: Fine-tune last MobileNetV2 layers")
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=callbacks
)

# Evaluate best checkpoint on held-out participants.
model = tf.keras.models.load_model(model_path)
_, accuracy = model.evaluate(test_ds)
probabilities = model.predict(test_ds)
predictions = np.argmax(probabilities, axis=1)

print(f"\nModel 4 test accuracy: {accuracy:.4f}")
print(classification_report(
    test_labels,
    predictions,
    labels=[0, 1, 2],
    target_names=CLASSES,
    zero_division=0
))

cm = confusion_matrix(test_labels, predictions, labels=[0, 1, 2])
display = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
display.plot(xticks_rotation=20)
plt.title("Fine-tuned MobileNetV2 - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "fine_tuned_confusion_matrix.png", dpi=150)
plt.close()

print("\nModel saved to:", model_path)
