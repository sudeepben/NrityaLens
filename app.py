from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from nrityalens.analyzer import (
    analyze_image,
    dependencies_available,
    image_analysis_available,
    image_mudra_model_available,
    model_available,
    mudra_model_available,
)
from nrityalens.knowledge import find_meaning, load_meanings
from nrityalens.storage import init_db, recent_analyses, save_analysis


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "mudra_meanings.json"
DB_PATH = ROOT / "data" / "analysis_history.sqlite3"


st.set_page_config(page_title="NrityaLens", page_icon="NL", layout="wide")


@st.cache_data
def cached_meanings() -> list:
    return load_meanings(DATA_PATH)


def metric_card(label: str, value: str) -> None:
    st.metric(label, value)


def build_report(file_name: str, result, meaning) -> str:
    meaning_text = meaning.meaning if meaning else "No curated meaning available yet."
    keywords = ", ".join(meaning.keywords) if meaning else "Not available"
    feedback = "\n".join(f"- {item}" for item in result.feedback)
    posture_lines = ""
    if result.metrics:
        posture_lines = f"""Posture score: {result.posture_score:.0f}/100
Symmetry score: {result.symmetry_score:.0f}/100
Pose match score: {result.pose_match_score:.0f}/100
"""
    else:
        posture_lines = "Posture score: Not available in mudra-only mode\n"
    return f"""NrityaLens Analysis Report

Uploaded file: {file_name}
Detected dance form: {result.dance_form}
Detected pose/mudra: {result.detected_label or "Unknown"}
Confidence: {result.confidence * 100:.0f}%
{posture_lines}

Cultural meaning:
{meaning_text}

Possible symbols:
{keywords}

Feedback:
{feedback}
"""


def main() -> None:
    init_db(DB_PATH)
    meanings = cached_meanings()

    st.title("NrityaLens")
    st.caption("AI-powered Indian classical dance movement and meaning analyzer")

    with st.sidebar:
        st.header("MVP Scope")
        st.write("Dance form: Bharatanatyam")
        st.write("Input: still image")
        if model_available() and (mudra_model_available() or image_mudra_model_available()):
            st.write("Analysis: trained pose and mudra classifiers")
        elif image_mudra_model_available() and not dependencies_available():
            st.write("Analysis: deployment image classifier")
        elif mudra_model_available():
            st.write("Analysis: trained mudra classifier plus pose baseline")
        elif model_available():
            st.write("Analysis: trained pose classifier plus mudra baseline")
        else:
            st.write("Analysis: MediaPipe landmarks plus curated baseline rules")
        st.divider()
        st.subheader("Supported Mudras")
        supported = sorted(entry.label for entry in meanings if entry.type == "mudra")
        st.write(", ".join(supported))
        st.divider()
        st.subheader("Roadmap")
        st.write("1. Collect labeled mudra images")
        st.write("2. Train landmark classifier")
        st.write("3. Add short video frame analysis")
        st.write("4. Replace local lookup with RAG")
        st.divider()
        st.caption("Training data attribution is documented in DATASETS.md.")

    uploaded = st.file_uploader("Upload a Bharatanatyam pose or mudra image", type=["jpg", "jpeg", "png"])
    st.caption("For best mudra detection, use a clear hand image. For posture scoring, use a full-body dance image.")

    if not dependencies_available() and not image_analysis_available():
        st.warning(
            "OpenCV or model files are missing. Install requirements.txt and ensure model artifacts are present."
        )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        image_array = np.array(image)

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        if dependencies_available() or image_analysis_available():
            with st.spinner("Analyzing pose landmarks..."):
                result = analyze_image(image_array)

            meaning = find_meaning(meanings, result.detected_label)
            save_analysis(
                DB_PATH,
                {
                    "file_name": uploaded.name,
                    "dance_form": result.dance_form,
                    "detected_label": result.detected_label,
                    "confidence": result.confidence,
                    "posture_score": result.posture_score,
                    "symmetry_score": result.symmetry_score,
                    "pose_match_score": result.pose_match_score,
                    "feedback": result.feedback,
                },
            )

            with right:
                st.subheader("Landmark View")
                if result.annotated_image is not None:
                    st.image(result.annotated_image, use_container_width=True)

            st.subheader("Analysis")
            if model_available() or mudra_model_available() or image_mudra_model_available():
                active_models = []
                if model_available():
                    active_models.append("pose")
                if mudra_model_available():
                    active_models.append("landmark mudra")
                if image_mudra_model_available():
                    active_models.append("image mudra")
                st.success(f"Using trained {' and '.join(active_models)} classifier.")
            else:
                st.info("Using rule-based baseline. Train a model to improve detection.")

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Detected Dance Form", result.dance_form)
            with c2:
                metric_card("Detected Pose/Mudra", result.detected_label or "Unknown")
            with c3:
                metric_card("Confidence", f"{result.confidence * 100:.0f}%")

            if result.metrics:
                c4, c5, c6 = st.columns(3)
                with c4:
                    metric_card("Posture Score", f"{result.posture_score:.0f}/100")
                with c5:
                    metric_card("Symmetry Score", f"{result.symmetry_score:.0f}/100")
                with c6:
                    metric_card("Pose Match Score", f"{result.pose_match_score:.0f}/100")
            else:
                st.info("Mudra-only mode is active for this analysis. Posture and symmetry scores require MediaPipe landmarks in a local run.")

            st.subheader("Cultural Meaning")
            if meaning:
                st.write(meaning.meaning)
                st.caption("Possible symbols: " + ", ".join(meaning.keywords))
            else:
                st.write("No curated meaning was found for this detected label yet.")

            st.subheader("Alignment Feedback")
            for item in result.feedback:
                st.write(f"- {item}")

            report = build_report(uploaded.name, result, meaning)
            st.download_button(
                "Download Analysis Report",
                data=report,
                file_name=f"nrityalens_{Path(uploaded.name).stem}_report.txt",
                mime="text/plain",
            )

            if result.metrics:
                st.subheader("Landmark Metrics")
                metrics_df = pd.DataFrame(
                    [{"metric": key.replace("_", " "), "value": round(value, 3)} for key, value in result.metrics.items()]
                )
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            with right:
                st.subheader("Analysis unavailable")
                st.write("Install the project dependencies, then run the app again.")

    history = recent_analyses(DB_PATH)
    if history:
        st.divider()
        st.subheader("Recent Analysis History")
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
