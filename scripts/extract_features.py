from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nrityalens.analyzer import FEATURE_COLUMNS, extract_landmark_features


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def iter_images(dataset_dir: Path):
    for label_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        for image_path in sorted(label_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield label_dir.name, image_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmark features from labeled images.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "features.csv")
    args = parser.parse_args()

    rows = []
    skipped = []

    for label, image_path in iter_images(args.dataset_dir):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            skipped.append((image_path, "could not read image"))
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        features = extract_landmark_features(image_rgb)
        if features is None:
            skipped.append((image_path, "no full-body pose detected"))
            continue

        rows.append({"label": label, "image_path": str(image_path.relative_to(ROOT)), **features})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["label", "image_path", *FEATURE_COLUMNS]
    pd.DataFrame(rows, columns=columns).to_csv(args.output, index=False)

    print(f"Wrote {len(rows)} rows to {args.output}")
    if skipped:
        print(f"Skipped {len(skipped)} images:")
        for path, reason in skipped[:20]:
            print(f"- {path}: {reason}")
        if len(skipped) > 20:
            print(f"- ... {len(skipped) - 20} more")


if __name__ == "__main__":
    main()
