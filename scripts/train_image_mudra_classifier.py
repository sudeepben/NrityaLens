from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nrityalens.analyzer import image_feature_vector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_DATASET = ROOT / "data" / "external" / "Bharatanatyam-Mudra-Dataset"
DEFAULT_MODEL = ROOT / "models" / "mudra_image_classifier.joblib"
DEFAULT_LABELS: list[str] = []


def normalize_label(folder_name: str) -> str:
    cleaned = re.sub(r"\(\d+\)$", "", folder_name).strip()
    return cleaned.lower().replace(" ", "_")


def iter_images(dataset_dir: Path, wanted_labels: set[str]):
    for label_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        label = normalize_label(label_dir.name)
        if wanted_labels and label not in wanted_labels:
            continue
        for image_path in sorted(label_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield label, image_path


def read_image_rgb(image_path: Path):
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deployment-safe image mudra classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS, help="Labels to include. Default: all folders.")
    args = parser.parse_args()

    if not args.dataset_dir.exists():
        raise SystemExit(f"Dataset folder not found: {args.dataset_dir}")

    wanted_labels = {normalize_label(label) for label in args.labels}
    rows = []
    labels = []
    counts: dict[str, int] = {}
    skipped = 0

    for label, image_path in iter_images(args.dataset_dir, wanted_labels):
        if args.limit_per_class and counts.get(label, 0) >= args.limit_per_class:
            continue

        image_rgb = read_image_rgb(image_path)
        if image_rgb is None:
            skipped += 1
            continue

        rows.append(image_feature_vector(image_rgb, image_size=args.image_size))
        labels.append(label)
        counts[label] = counts.get(label, 0) + 1

    if not rows:
        raise SystemExit("No images were loaded for training.")

    X = np.vstack(rows)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=300, random_state=42)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    sorted_labels = sorted(set(labels))
    print("Training samples:")
    for label, count in sorted(counts.items()):
        print(f"- {label}: {count}")
    print(f"Skipped unreadable images: {skipped}")
    print("Classification report:")
    print(classification_report(y_test, predictions, labels=sorted_labels, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=sorted_labels))

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "image_size": args.image_size,
            "labels": sorted_labels,
        },
        args.model_output,
        compress=3,
    )
    print(f"Saved model to {args.model_output}")


if __name__ == "__main__":
    main()
