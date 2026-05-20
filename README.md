# NrityaLens

AI-powered Indian classical dance movement and meaning analyzer.

This MVP focuses on Bharatanatyam image analysis:

- Upload a dance image
- Extract body and hand landmarks with MediaPipe
- Estimate likely pose or mudra from curated rules
- Score posture, symmetry, and pose match
- Explain cultural meaning from a local knowledge base
- Save analysis history locally

## Run Locally

```powershell
cd D:\NrityaLens
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If MediaPipe is not installed, the app still opens and explains what dependency is missing, but image analysis requires the full requirements.

## Project Layout

```text
nrityalens/
  app.py
  requirements.txt
  models/
    pose_classifier.joblib
  scripts/
    extract_features.py
    train_classifier.py
    extract_mudra_features.py
    train_mudra_classifier.py
  data/
    mudra_meanings.json
    features.csv
    dataset/
      anjali/
      araimandi/
      nataraja_pose/
      pataka/
      tripataka/
  nrityalens/
    analyzer.py
    knowledge.py
    storage.py
```

## MVP Scope

The app currently supports a trained hand-landmark mudra classifier for 12 Bharatanatyam mudras, plus a rule-based baseline for body posture scoring. Full-body pose classification is scaffolded but still needs a labeled full-body dataset.

## Training Workflow

Add labeled images to the matching folder under `data/dataset`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\extract_features.py
.\.venv\Scripts\python.exe scripts\train_classifier.py
```

The trained classifier is saved to `models/pose_classifier.joblib`. The app uses that model automatically when it exists.

For the GitHub Bharatanatyam mudra dataset:

```powershell
.\.venv\Scripts\python.exe scripts\extract_mudra_features.py
.\.venv\Scripts\python.exe scripts\train_mudra_classifier.py
```

The trained mudra classifier is saved to `models/mudra_classifier.joblib`. Raw external datasets should stay local under `data/external/` and should not be committed to GitHub.

The default mudra training set includes: `Alapadmam`, `Anjali`, `Ardhapathaka`, `Mayura`, `Mrigasirsha`, `Mushti`, `Padmakosha`, `Pathaka`, `Shukatundam`, `Sikharam`, `Suchi`, and `Tripathaka`.

## GitHub Notes

The repository is designed to commit source code, scripts, curated meanings, and documentation. It does not commit:

- External datasets under `data/external/`
- Generated feature CSV files
- Trained `.joblib` model files
- Local SQLite history
- The local `.venv`

See `DATASETS.md` and `MODEL_CARD.md` for dataset attribution and model reproduction notes.
