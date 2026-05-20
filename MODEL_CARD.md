# Model Card

## Current Model

`models/mudra_classifier.joblib`

This model is committed so the deployed app can run the trained mudra classifier without downloading the external dataset at startup.

## Task

Classify Bharatanatyam hand mudras from MediaPipe hand landmark features.

## Supported Labels

- Alapadmam
- Anjali
- Ardhapathaka
- Mayura
- Mrigasirsha
- Mushti
- Padmakosha
- Pathaka
- Shukatundam
- Sikharam
- Suchi
- Tripathaka

## Features

The model uses engineered hand-landmark features such as:

- hand presence
- bounding box width and height
- wrist distance
- fingertip distances
- finger extension ratios
- palm spread

## Training

```powershell
cd D:\NrityaLens
.\.venv\Scripts\python.exe scripts\extract_mudra_features.py
.\.venv\Scripts\python.exe scripts\train_mudra_classifier.py
```

## Latest Local Evaluation

The latest local 12-class mudra model used 4,918 MediaPipe-detected samples and reached 98 percent accuracy on a held-out test split.

The most visible confusion was around visually similar shapes, especially Ardhapathaka versus Mayura or Mrigasirsha.

## Limitations

- Works best on clear hand images.
- Full-body posture scoring still depends on MediaPipe pose detection and rule-based geometry.
- Mudra meaning depends on performance context, so explanations should be treated as cultural guidance rather than a final interpretation.
- The model may not generalize well to images outside the dataset style without additional data.
