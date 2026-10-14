
import cv2
from pathlib import Path

DATASET_DIR = Path("dataset/raw")
OUTPUT_DIR = Path("dataset/frames")

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv",
    ".wmv", ".webm", ".mpeg", ".mpg"
}

LABELS = {
    "0": "alert",
    "5": "low_vigilance",
    "10": "drowsy"
}

SAMPLE_EVERY_SECONDS = 2
IMAGE_SIZE = (224, 224)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

videos = sorted(
    path for path in DATASET_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in VIDEO_EXTENSIONS
)

if not videos:
    raise SystemExit("No videos found. Check dataset/raw.")

for video_index, video_path in enumerate(videos, start=1):
    label = LABELS.get(video_path.stem)

    if label is None:
        print(f"Skipping unknown label: {video_path.name}")
        continue

    participant = video_path.parent.name
    output_folder = OUTPUT_DIR / label / participant
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print(f"Invalid FPS: {video_path}")
        cap.release()
        continue

    interval = max(1, round(fps * SAMPLE_EVERY_SECONDS))
    frame_index = 0
    saved_count = 0

    print(f"[{video_index}/{len(videos)}] Processing {video_path.name}")

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % interval == 0:
            resized = cv2.resize(frame, IMAGE_SIZE)

            output_path = output_folder / (
                f"{video_path.stem}_frame_{frame_index:07d}.jpg"
            )

            if cv2.imwrite(str(output_path), resized):
                saved_count += 1

        frame_index += 1

    cap.release()

    print(f"  Saved {saved_count} frames for {label}")

print("\nFrame extraction complete!")
print(f"Frames saved under: {OUTPUT_DIR.resolve()}")
