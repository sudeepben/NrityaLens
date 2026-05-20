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

## Included Model Artifacts

The deployment-safe image classifier is committed at:

```text
models/mudra_image_classifier.joblib
```

The MediaPipe landmark classifier is also committed for local use:

```text
models/mudra_classifier.joblib
```

The deployed app can run the image classifier immediately without installing MediaPipe. The raw external dataset and generated feature CSVs are still ignored and are not pushed.

## System Packages

The app uses `opencv-python-headless`, so no extra apt packages are required for Streamlit Cloud.

## After Deployment

Once the Streamlit app is live:

1. Copy the deployed app URL.
2. Add the URL to `README.md`.
3. Add final screenshots or demo GIFs.
4. Commit and push the README update.
