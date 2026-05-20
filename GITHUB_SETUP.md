# GitHub Setup

After the project is ready to publish, create an empty GitHub repository named `NrityaLens`, then run:

```powershell
cd D:\NrityaLens
git remote add origin https://github.com/YOUR_USERNAME/NrityaLens.git
git branch -M main
git push -u origin main
```

Do not commit raw datasets, generated feature files, local history databases, virtual environments, or model artifacts unless you intentionally decide to publish them separately.

## Reproduce the Mudra Model

```powershell
cd D:\NrityaLens
git clone --depth 1 https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset.git data\external\Bharatanatyam-Mudra-Dataset
.\.venv\Scripts\python.exe scripts\extract_mudra_features.py
.\.venv\Scripts\python.exe scripts\train_mudra_classifier.py
```
