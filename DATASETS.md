# Dataset Notes

NrityaLens uses external datasets locally for training, but raw dataset files are not committed to this repository.

## Bharatanatyam Mudra Dataset

Source: https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset

Use in this project:

- Hand/mudra classification
- MediaPipe hand landmark feature extraction
- Training `models/mudra_classifier.joblib`

The dataset contains Bharatanatyam hand gesture images collected from volunteers. The local training workflow currently uses these 12 classes:

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

Setup:

```powershell
cd D:\NrityaLens
git clone --depth 1 https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset.git data\external\Bharatanatyam-Mudra-Dataset
```

The raw dataset path is ignored by Git:

```text
data/external/
```

## Indian Dance Form Recognition Dataset

Source: https://www.kaggle.com/datasets/somnath796/indian-dance-form-recognition

Planned use:

- Dance-form classification across Indian classical dance forms

This dataset has not been integrated yet. Keep Kaggle data local and do not commit it to GitHub.
