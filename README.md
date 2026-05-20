# NrityaLens

AI-powered Indian classical dance movement and meaning analyzer.

Live app: https://nrityalens-3dmmccjdxcgvvrwc2henez.streamlit.app/

This MVP focuses on Bharatanatyam image analysis:

- Upload a dance image
- Detect mudras from the Bharatanatyam mudra dataset in the deployed app
- Explain cultural meaning from a local knowledge base
- Download an analysis report
- Run optional MediaPipe landmark analysis locally
- Save analysis history locally

## Run Locally

```powershell
cd D:\NrityaLens
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The deployed app uses a lightweight image classifier. For local MediaPipe landmark training and posture scoring utilities, install:

```powershell
pip install -r requirements-training.txt
```

## Project Layout

```text
nrityalens/
  app.py
  requirements.txt
  requirements-training.txt
  models/
    mudra_image_classifier.joblib
    mudra_classifier.joblib
  scripts/
    extract_features.py
    train_classifier.py
    extract_mudra_features.py
    train_mudra_classifier.py
    train_image_mudra_classifier.py
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

The deployed app supports a trained image-based classifier for all 50 mudra folders in the Bharatanatyam mudra dataset. Local runs can also use a MediaPipe hand-landmark mudra classifier when `requirements-training.txt` is installed. Full-body pose classification is scaffolded but still needs a labeled full-body dataset.

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

The trained image classifier is saved to `models/mudra_image_classifier.joblib`. The MediaPipe landmark classifier is saved to `models/mudra_classifier.joblib`. Raw external datasets should stay local under `data/external/` and should not be committed to GitHub.

The default mudra training set includes: `Alapadmam`, `Anjali`, `Ardhapathaka`, `Mayura`, `Mrigasirsha`, `Mushti`, `Padmakosha`, `Pathaka`, `Shukatundam`, `Sikharam`, `Suchi`, and `Tripathaka`.

## GitHub Notes

The repository is designed to commit source code, scripts, curated meanings, and documentation. It does not commit:

- External datasets under `data/external/`
- Generated feature CSV files
- Local SQLite history
- The local `.venv`

See `DATASETS.md` and `MODEL_CARD.md` for dataset attribution and model reproduction notes.

See `DEPLOYMENT.md` for Streamlit Community Cloud deployment settings.
