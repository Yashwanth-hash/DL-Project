
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
EPOCHS = 5

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "face_crops"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CLASSES = ["alert", "low_vigilance", "drowsy"]

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Collect images and participant IDs.
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
    raise ValueError(
        f"Expected 12 participants, found {len(participants)}"
    )

print("Total images:", len(paths))
print("Participants:", participants)

# Reproduce the previous split.
shuffled = participants.copy()
random.shuffle(shuffled)

train_ids = set(shuffled[:8])
val_ids = set(shuffled[8:10])
test_ids = set(shuffled[10:12])

print("Train IDs:", sorted(train_ids))
print("Validation IDs:", sorted(val_ids))
print("Test IDs:", sorted(test_ids))

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

print("Train:", len(train_paths))
print("Validation:", len(val_paths))
print("Test:", len(test_paths))

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )
    return image, label

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (paths, np.asarray(labels, dtype=np.int32))
    )

    if training:
        ds = ds.shuffle(
            min(len(paths), 2000), seed=SEED
        )

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

# Build MobileNetV2 transfer-learning model.
base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.05)(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(
    len(CLASSES), activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model_path = MODEL_DIR / "drowsiness_face_crops.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(model_path),
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

# Evaluate on held-out participants.
model = tf.keras.models.load_model(model_path)
loss, accuracy = model.evaluate(test_ds)
probabilities = model.predict(test_ds)
predictions = np.argmax(probabilities, axis=1)

print(f"\nFull-face test accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(
    test_labels,
    predictions,
    labels=[0, 1, 2],
    target_names=CLASSES,
    zero_division=0
))

cm = confusion_matrix(
    test_labels, predictions, labels=[0, 1, 2]
)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=CLASSES
)
display.plot(xticks_rotation=20)
plt.title("Full-Face Model - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "face_crop_confusion_matrix.png",
    dpi=150
)
plt.close()

np.savez(
    RESULTS_DIR / "face_crop_training_history.npz",
    accuracy=history.history["accuracy"],
    val_accuracy=history.history["val_accuracy"],
    loss=history.history["loss"],
    val_loss=history.history["val_loss"]
)

print("\nModel saved to:", model_path)
print("Confusion matrix saved to:", RESULTS_DIR)
