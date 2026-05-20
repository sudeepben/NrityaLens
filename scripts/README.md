# NrityaLens Training Scripts

Use these scripts after adding labeled images under `data/dataset`.

Expected folder layout:

```text
data/dataset/
  anjali/
  araimandi/
  nataraja_pose/
  pataka/
  tripataka/
```

Run from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\extract_features.py
.\.venv\Scripts\python.exe scripts\train_classifier.py
```

The trained model is saved to `models/pose_classifier.joblib`. The Streamlit app uses it automatically when the file exists.

## Mudra Classifier

The Bharatanatyam mudra dataset is expected at:

```text
data/external/Bharatanatyam-Mudra-Dataset/
```

Train the focused mudra classifier:

```powershell
.\.venv\Scripts\python.exe scripts\extract_mudra_features.py
.\.venv\Scripts\python.exe scripts\train_mudra_classifier.py
```

By default this trains on the app-supported mudras: `Alapadmam`, `Anjali`, `Ardhapathaka`, `Mayura`, `Mrigasirsha`, `Mushti`, `Padmakosha`, `Pathaka`, `Shukatundam`, `Sikharam`, `Suchi`, and `Tripathaka`. Add `--labels` to choose different classes.

Train the deployment-safe image classifier:

```powershell
.\.venv\Scripts\python.exe scripts\train_image_mudra_classifier.py
```
