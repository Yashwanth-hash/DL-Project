
from pathlib import Path
import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
FRAME_DIR = ROOT / "dataset" / "frames"
OUTPUT_DIR = ROOT / "dataset" / "face_crops"
MODEL_PATH = ROOT / "models" / "blaze_face_short_range.tflite"

CLASSES = ["alert", "low_vigilance", "drowsy"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing detector model: {MODEL_PATH}")

options = mp.tasks.vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    min_detection_confidence=0.5
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

processed = 0
saved = 0
skipped = 0
failed = 0

with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
    for class_name in CLASSES:
        class_dir = FRAME_DIR / class_name

        if not class_dir.exists():
            print(f"Missing class folder: {class_dir}")
            continue

        images = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in images:
            participant = image_path.parent.name
            output_folder = OUTPUT_DIR / class_name / participant
            output_folder.mkdir(parents=True, exist_ok=True)

            output_path = output_folder / f"{image_path.stem}.jpg"

            if output_path.exists():
                skipped += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                failed += 1
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(mp_image)
            processed += 1

            if not result.detections:
                failed += 1
                continue

            detection = max(
                result.detections,
                key=lambda d: (
                    d.bounding_box.width * d.bounding_box.height
                )
            )

            box = detection.bounding_box
            h, w = image.shape[:2]

            # Add 15% padding around the face.
            pad_x = int(box.width * 0.15)
            pad_y = int(box.height * 0.15)

            x1 = max(0, box.origin_x - pad_x)
            y1 = max(0, box.origin_y - pad_y)
            x2 = min(w, box.origin_x + box.width + pad_x)
            y2 = min(h, box.origin_y + box.height + pad_y)

            face = image[y1:y2, x1:x2]

            if face.size == 0:
                failed += 1
                continue

            face = cv2.resize(
                face,
                (160, 160),
                interpolation=cv2.INTER_AREA
            )

            if cv2.imwrite(str(output_path), face):
                saved += 1
            else:
                failed += 1

            if processed % 500 == 0:
                print(
                    f"Processed: {processed} | "
                    f"Saved: {saved} | Failed: {failed}"
                )

print("\nFull-face cropping complete!")
print(f"Processed: {processed}")
print(f"Saved: {saved}")
print(f"Skipped existing: {skipped}")
print(f"Failed/no face: {failed}")
print(f"Output: {OUTPUT_DIR}")
