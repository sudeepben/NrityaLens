from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "data" / "mudra_features.csv"
DEFAULT_MODEL = ROOT / "models" / "mudra_classifier.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hand-landmark Bharatanatyam mudra classifier.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.features)
    except (FileNotFoundError, EmptyDataError):
        raise SystemExit("No mudra features found. Run scripts/extract_mudra_features.py first.")

    if df.empty:
        raise SystemExit("No mudra features found. Check whether MediaPipe detected hands in the dataset.")

    feature_columns = [column for column in df.columns if column not in {"label", "image_path"}]
    X = df[feature_columns]
    y = df["label"]

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise SystemExit("Need at least two mudra labels to train a classifier.")

    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    labels = sorted(y.unique())
    print("Classification report:")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=labels))
    print("Labels:", ", ".join(labels))

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_columns": feature_columns}, args.model_output)
    print(f"Saved model to {args.model_output}")


if __name__ == "__main__":
    main()
