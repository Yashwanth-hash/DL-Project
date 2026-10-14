
import cv2
import time
import math
import numpy as np
import mediapipe as mp
import tensorflow as tf

from pathlib import Path
from collections import deque
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- CONFIGURATION ----------------
ROOT_DIR = Path(__file__).resolve().parent.parent

CNN_PATH = ROOT_DIR / "models" / "drowsiness_mobilenetv2.keras"
LSTM_PATH = ROOT_DIR / "models" / "drowsiness_cnn_lstm.keras"

IMAGE_SIZE = (160, 160)
CLOSURE_THRESHOLD = 2.5
EYE_THRESHOLD = 0.20
PREDICTION_INTERVAL = 0.5
SEQUENCE_LENGTH = 5

CLASS_NAMES = ["alert", "low_vigilance", "drowsy"]

# ---------------- LOAD MODELS ----------------
print("Loading MobileNetV2 CNN...")
cnn = tf.keras.models.load_model(CNN_PATH)

feature_extractor = Model(
    inputs=cnn.input,
    outputs=cnn.get_layer("global_average_pooling2d").output,
)

print("Loading CNN-LSTM...")
lstm_model = tf.keras.models.load_model(LSTM_PATH)

print("Both models loaded successfully.")

# ---------------- INITIALIZE CAMERA ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions.")

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(landmarks, indices, width, height):
    points = []

    for index in indices:
        lm = landmarks[index]
        points.append((int(lm.x * width), int(lm.y * height)))

    vertical1 = math.dist(points[1], points[5])
    vertical2 = math.dist(points[2], points[4])
    horizontal = math.dist(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def extract_cnn_features(frame):
    image = cv2.resize(frame, IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    features = feature_extractor(image, training=False).numpy()
    return features[0]


def predict_lstm(sequence):
    # Expected input shape: (1, 5, 1280)
    sequence = np.asarray(sequence, dtype=np.float32)
    sequence = np.expand_dims(sequence, axis=0)

    probabilities = lstm_model.predict(sequence, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    print(
        f"LSTM — Alert: {probabilities[0]:.3f} | "
        f"Low vigilance: {probabilities[1]:.3f} | "
        f"Drowsy: {probabilities[2]:.3f}"
    )

    return predicted_class, confidence


# ---------------- LIVE STATE ----------------
feature_sequence = deque(maxlen=SEQUENCE_LENGTH)

closed_since = None
last_prediction_time = 0.0
last_lstm_class = "Collecting frames..."
last_lstm_confidence = 0.0

previous_frame_time = time.time()
fps = 0.0

try:
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            now = time.time()

            elapsed = now - previous_frame_time
            if elapsed > 0:
                fps = 1.0 / elapsed
            previous_frame_time = now

            # -------- CNN FEATURE EXTRACTION + LSTM --------
            if now - last_prediction_time >= PREDICTION_INTERVAL:
                try:
                    features = extract_cnn_features(frame)
                    feature_sequence.append(features)

                    if len(feature_sequence) == SEQUENCE_LENGTH:
                        (
                            last_lstm_class,
                            last_lstm_confidence,
                        ) = predict_lstm(list(feature_sequence))

                except Exception as error:
                    print("CNN-LSTM prediction error:", error)
                    last_lstm_class = "MODEL ERROR"

                last_prediction_time = now

            # ---------------- FACE AND EYE TRACKING ----------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            eye_status = "UNKNOWN"
            closure_time = 0.0
            warning = False

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_ear = eye_aspect_ratio(
                    landmarks, LEFT_EYE, width, height
                )
                right_ear = eye_aspect_ratio(
                    landmarks, RIGHT_EYE, width, height
                )
                ear = (left_ear + right_ear) / 2.0

                eyes_closed = ear < EYE_THRESHOLD

                if eyes_closed:
                    if closed_since is None:
                        closed_since = now

                    closure_time = now - closed_since
                    eye_status = "CLOSED"

                    if closure_time >= CLOSURE_THRESHOLD:
                        live_status = "TRIGGER WARNING"
                        warning = True
                    else:
                        live_status = "DROWSY (MAYBE)"
                else:
                    closed_since = None
                    closure_time = 0.0
                    eye_status = "OPEN"
                    live_status = "ALERT"

            else:
                # Missing face data must not count as eye closure.
                closed_since = None
                live_status = "FACE NOT DETECTED"

            # ---------------- DRAW DASHBOARD ----------------
            cv2.rectangle(
                frame, (0, 0), (width, 195), (25, 25, 25), -1
            )

            status_color = (0, 200, 0)

            if live_status == "DROWSY (MAYBE)":
                status_color = (0, 200, 255)
            elif warning:
                status_color = (0, 0, 255)
            elif live_status == "FACE NOT DETECTED":
                status_color = (0, 165, 255)

            cv2.putText(
                frame,
                f"Live eye status: {live_status}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                status_color,
                2,
            )

            cv2.putText(
                frame,
                f"CNN-LSTM class: {last_lstm_class}",
                (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"LSTM confidence: {last_lstm_confidence:.1%}",
                (15, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                2,
            )

            cv2.putText(
                frame,
                f"Sequence: {len(feature_sequence)}/{SEQUENCE_LENGTH} frames",
                (15, 122),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                2,
            )

            cv2.putText(
                frame,
                f"Eyes: {eye_status} | Closure: {closure_time:.1f}s",
                (15, 152),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (width - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            if warning:
                cv2.rectangle(
                    frame, (0, 205), (width, 265), (0, 0, 220), -1
                )
                cv2.putText(
                    frame,
                    "WARNING: PROLONGED EYE CLOSURE!",
                    (15, 243),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Driver Drowsiness Detection", frame)

            # Press Q to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
