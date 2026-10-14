
from pathlib import Path

FRAME_DIR = Path("dataset/frames")

classes = ["alert", "low_vigilance", "drowsy"]

total = 0

for class_name in classes:
    class_dir = FRAME_DIR / class_name

    if not class_dir.exists():
        print(f"{class_name}: FOLDER NOT FOUND")
        continue

    images = list(class_dir.rglob("*.jpg"))
    count = len(images)
    total += count

    print(f"{class_name}: {count} frames")

print("-" * 30)
print(f"TOTAL: {total} frames")
