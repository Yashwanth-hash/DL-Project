
from pathlib import Path

DATASET_DIR = Path("dataset/raw")

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv",
    ".wmv", ".webm", ".mpeg", ".mpg"
}

class_names = {
    "0": "Alert",
    "5": "Low Vigilance",
    "10": "Drowsy"
}

if not DATASET_DIR.exists():
    print(f"Dataset folder not found: {DATASET_DIR.resolve()}")
    raise SystemExit(1)

videos = sorted(
    p for p in DATASET_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
)

print(f"Total videos found: {len(videos)}")
print("=" * 60)

for video in videos:
    label = video.stem
    size_gb = video.stat().st_size / (1024 ** 3)

    print(
        f"Participant: {video.parent.name} | "
        f"Label: {label} | "
        f"Class: {class_names.get(label, 'Unknown')} | "
        f"Size: {size_gb:.2f} GB"
    )

print("=" * 60)
print("Inspection complete!")
