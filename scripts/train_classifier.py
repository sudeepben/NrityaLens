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
DEFAULT_FEATURES = ROOT / "data" / "features.csv"
DEFAULT_MODEL = ROOT / "models" / "pose_classifier.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NrityaLens landmark classifier.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.features)
    except EmptyDataError:
        raise SystemExit("No features found. Add labeled images and run extract_features.py first.")
    if df.empty:
        raise SystemExit("No features found. Add labeled images and run extract_features.py first.")

    feature_columns = [column for column in df.columns if column not in {"label", "image_path"}]
    class_counts = df["label"].value_counts()
    if len(class_counts) < 2:
        raise SystemExit("Need images from at least two labels to train a classifier.")

    X = df[feature_columns]
    y = df["label"]
    stratify = y if class_counts.min() >= 2 else None

    if len(df) >= 10 and stratify is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=42,
                    class_weight="balanced",
                    min_samples_leaf=1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=sorted(y.unique())))
    print("Labels:", ", ".join(sorted(y.unique())))

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_columns": feature_columns}, args.model_output)
    print(f"Saved model to {args.model_output}")


if __name__ == "__main__":
    main()
