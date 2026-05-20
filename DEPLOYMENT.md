# Deployment

NrityaLens is prepared for Streamlit Community Cloud.

## Streamlit Community Cloud

Official docs:

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

Deploy settings:

```text
Repository: sudeepben/NrityaLens
Branch: main
Main file path: app.py
Python version: 3.10
```

Use **Advanced settings** during deployment to select Python 3.10. Streamlit Community Cloud defaults to Python 3.12, and MediaPipe compatibility is more predictable with the Python 3.10 environment used locally for this project.

## Included Model Artifact

The trained mudra classifier is committed at:

```text
models/mudra_classifier.joblib
```

This lets the deployed app run the 12-mudra classifier immediately. The raw external dataset and generated feature CSVs are still ignored and are not pushed.

## System Packages

`packages.txt` installs the Linux package commonly needed by OpenCV/MediaPipe on Streamlit Cloud:

```text
libgl1
```

## After Deployment

Once the Streamlit app is live:

1. Copy the deployed app URL.
2. Add the URL to `README.md`.
3. Add final screenshots or demo GIFs.
4. Commit and push the README update.
