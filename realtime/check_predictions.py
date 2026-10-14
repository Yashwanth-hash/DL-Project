
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "drowsiness_mobilenetv2.keras"
FRAME_DIR = ROOT / "dataset" / "frames"

CLASSES = ["alert", "low_vigilance", "drowsy"]
IMAGE_SIZE = (160, 160)

model = tf.keras.models.load_model(MODEL_PATH)

# Select a saved UTA-RLDD alert frame.
alert_dir = FRAME_DIR / "alert"
images = list(alert_dir.rglob("*.jpg"))

if not images:
    raise FileNotFoundError("No alert frames found.")

image_path = images[0]
print("Testing dataset image:", image_path)

def predict(image_bgr):
    image = cv2.resize(image_bgr, IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    probabilities = model.predict(image, verbose=0)[0]

    for name, probability in zip(CLASSES, probabilities):
        print(f"{name}: {probability:.4f}")

    print("Predicted class:", CLASSES[np.argmax(probabilities)])
    print("-" * 35)

image = cv2.imread(str(image_path))

if image is None:
    raise RuntimeError("Could not read the selected image.")

predict(image)

# Now test a live webcam frame.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press SPACE to capture a frame; press Q to quit.")

while True:
    ok, frame = cap.read()

    if not ok:
        break

    cv2.imshow("Capture webcam frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        print("Webcam prediction:")
        predict(frame)
        break

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
