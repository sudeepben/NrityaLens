from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
    import joblib
    import mediapipe as mp
    import pandas as pd
except ImportError:
    cv2 = None
    joblib = None
    mp = None
    pd = None


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pose_classifier.joblib"
MUDRA_MODEL_PATH = ROOT / "models" / "mudra_classifier.joblib"
LABEL_DISPLAY = {
    "alapadmam": "Alapadmam",
    "anjali": "Anjali",
    "ardhapathaka": "Ardhapathaka",
    "araimandi": "Araimandi",
    "mayura": "Mayura",
    "mrigasirsha": "Mrigasirsha",
    "mushti": "Mushti",
    "nataraja_pose": "Nataraja Pose",
    "padmakosha": "Padmakosha",
    "pathaka": "Pataka",
    "pataka": "Pataka",
    "shukatundam": "Shukatundam",
    "sikharam": "Sikharam",
    "suchi": "Suchi",
    "tripathaka": "Tripataka",
    "tripataka": "Tripataka",
}
FEATURE_COLUMNS = [
    "left_knee_angle",
    "right_knee_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "shoulder_level_delta",
    "hip_level_delta",
    "wrist_level_delta",
    "knee_balance_delta",
    "elbow_balance_delta",
    "hands_distance",
    "hand_count",
]
HAND_FEATURE_COLUMNS = [
    "left_hand_present",
    "right_hand_present",
    "hand_count",
    "wrist_distance",
    "left_bbox_width",
    "left_bbox_height",
    "left_bbox_ratio",
    "left_thumb_index_distance",
    "left_index_middle_distance",
    "left_middle_ring_distance",
    "left_ring_pinky_distance",
    "left_palm_spread",
    "left_index_extension",
    "left_middle_extension",
    "left_ring_extension",
    "left_pinky_extension",
    "right_bbox_width",
    "right_bbox_height",
    "right_bbox_ratio",
    "right_thumb_index_distance",
    "right_index_middle_distance",
    "right_middle_ring_distance",
    "right_ring_pinky_distance",
    "right_palm_spread",
    "right_index_extension",
    "right_middle_extension",
    "right_ring_extension",
    "right_pinky_extension",
]


@dataclass(frozen=True)
class AnalysisResult:
    dance_form: str
    detected_label: str | None
    confidence: float
    posture_score: float
    symmetry_score: float
    pose_match_score: float
    feedback: list[str]
    annotated_image: np.ndarray | None
    metrics: dict[str, float]


def dependencies_available() -> bool:
    return cv2 is not None and mp is not None


def model_available() -> bool:
    return joblib is not None and MODEL_PATH.exists()


def mudra_model_available() -> bool:
    return joblib is not None and MUDRA_MODEL_PATH.exists()


def analyze_image(image_rgb: np.ndarray) -> AnalysisResult:
    if not dependencies_available():
        raise RuntimeError("MediaPipe and OpenCV are required for image analysis.")

    pose_model = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
    hands_model = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)
    pose_result = pose_model.process(image_rgb)
    hands_result = hands_model.process(image_rgb)

    annotated = image_rgb.copy()
    drawing = mp.solutions.drawing_utils
    if pose_result.pose_landmarks:
        drawing.draw_landmarks(annotated, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    if hands_result.multi_hand_landmarks:
        for hand_landmarks in hands_result.multi_hand_landmarks:
            drawing.draw_landmarks(annotated, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    if not pose_result.pose_landmarks:
        mudra_prediction = _classify_mudra_from_hands(hands_result)
        if mudra_prediction:
            label, confidence = mudra_prediction
            return AnalysisResult(
                dance_form="Bharatanatyam",
                detected_label=label,
                confidence=confidence,
                posture_score=0.0,
                symmetry_score=0.0,
                pose_match_score=0.0,
                feedback=[
                    "Hand mudra was detected, but no full-body pose was found. Use a full-body image for posture and symmetry scores."
                ],
                annotated_image=annotated,
                metrics={},
            )
        return AnalysisResult(
            dance_form="Bharatanatyam",
            detected_label=None,
            confidence=0.0,
            posture_score=0.0,
            symmetry_score=0.0,
            pose_match_score=0.0,
            feedback=["No full-body pose was detected. Try a clearer image with the dancer visible from head to feet."],
            annotated_image=annotated,
            metrics={},
        )

    metrics = _calculate_metrics_from_landmarks(
        pose_result.pose_landmarks.landmark,
        len(hands_result.multi_hand_landmarks or []),
    )
    label, confidence = _classify(metrics)
    mudra_prediction = _classify_mudra_from_hands(hands_result)
    if mudra_prediction and mudra_prediction[1] >= confidence:
        label, confidence = mudra_prediction
    posture_score, symmetry_score, pose_match_score = _score_pose(metrics, label)
    feedback = _feedback(metrics, label)

    return AnalysisResult(
        dance_form="Bharatanatyam",
        detected_label=label,
        confidence=confidence,
        posture_score=posture_score,
        symmetry_score=symmetry_score,
        pose_match_score=pose_match_score,
        feedback=feedback,
        annotated_image=annotated,
        metrics=metrics,
    )


def extract_landmark_features(image_rgb: np.ndarray) -> dict[str, float] | None:
    if not dependencies_available():
        raise RuntimeError("MediaPipe and OpenCV are required for feature extraction.")

    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose_model:
        pose_result = pose_model.process(image_rgb)
    if not pose_result.pose_landmarks:
        return None

    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2) as hands_model:
        hands_result = hands_model.process(image_rgb)

    return _calculate_metrics_from_landmarks(
        pose_result.pose_landmarks.landmark,
        len(hands_result.multi_hand_landmarks or []),
    )


def extract_hand_features(image_rgb: np.ndarray) -> dict[str, float] | None:
    if not dependencies_available():
        raise RuntimeError("MediaPipe and OpenCV are required for hand feature extraction.")

    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2) as hands_model:
        hands_result = hands_model.process(image_rgb)

    if not hands_result.multi_hand_landmarks:
        return None
    return _calculate_hand_features(hands_result)


def _calculate_metrics_from_landmarks(landmarks: Any, hand_count: int) -> dict[str, float]:
    points = _pose_points(landmarks)
    return _calculate_metrics(points, hand_count)


def _pose_points(landmarks: Any) -> dict[str, np.ndarray]:
    pose = mp.solutions.pose.PoseLandmark
    names = {
        "left_shoulder": pose.LEFT_SHOULDER,
        "right_shoulder": pose.RIGHT_SHOULDER,
        "left_elbow": pose.LEFT_ELBOW,
        "right_elbow": pose.RIGHT_ELBOW,
        "left_wrist": pose.LEFT_WRIST,
        "right_wrist": pose.RIGHT_WRIST,
        "left_hip": pose.LEFT_HIP,
        "right_hip": pose.RIGHT_HIP,
        "left_knee": pose.LEFT_KNEE,
        "right_knee": pose.RIGHT_KNEE,
        "left_ankle": pose.LEFT_ANKLE,
        "right_ankle": pose.RIGHT_ANKLE,
    }
    return {name: np.array([landmarks[index].x, landmarks[index].y]) for name, index in names.items()}


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos_value = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_value)))


def _level_difference(left: np.ndarray, right: np.ndarray) -> float:
    return abs(float(left[1] - right[1]))


def _calculate_metrics(points: dict[str, np.ndarray], hand_count: int) -> dict[str, float]:
    left_knee = _angle(points["left_hip"], points["left_knee"], points["left_ankle"])
    right_knee = _angle(points["right_hip"], points["right_knee"], points["right_ankle"])
    left_elbow = _angle(points["left_shoulder"], points["left_elbow"], points["left_wrist"])
    right_elbow = _angle(points["right_shoulder"], points["right_elbow"], points["right_wrist"])

    shoulder_level = _level_difference(points["left_shoulder"], points["right_shoulder"])
    hip_level = _level_difference(points["left_hip"], points["right_hip"])
    wrist_level = _level_difference(points["left_wrist"], points["right_wrist"])
    knee_balance = abs(left_knee - right_knee)
    elbow_balance = abs(left_elbow - right_elbow)
    hands_near_center = float(np.linalg.norm(points["left_wrist"] - points["right_wrist"]))

    return {
        "left_knee_angle": left_knee,
        "right_knee_angle": right_knee,
        "left_elbow_angle": left_elbow,
        "right_elbow_angle": right_elbow,
        "shoulder_level_delta": shoulder_level,
        "hip_level_delta": hip_level,
        "wrist_level_delta": wrist_level,
        "knee_balance_delta": knee_balance,
        "elbow_balance_delta": elbow_balance,
        "hands_distance": hands_near_center,
        "hand_count": float(hand_count),
    }


def _classify(metrics: dict[str, float]) -> tuple[str, float]:
    model_prediction = _classify_with_model(metrics)
    if model_prediction:
        return model_prediction
    return _classify_baseline(metrics)


def _classify_mudra_from_hands(hands_result: Any) -> tuple[str, float] | None:
    model_bundle = _load_mudra_model()
    if model_bundle is None or pd is None or not hands_result.multi_hand_landmarks:
        return None

    features = _calculate_hand_features(hands_result)
    pipeline = model_bundle["pipeline"]
    feature_columns = model_bundle["feature_columns"]
    row = pd.DataFrame([{column: features[column] for column in feature_columns}])
    raw_label = pipeline.predict(row)[0]
    label = LABEL_DISPLAY.get(str(raw_label), str(raw_label).replace("_", " ").title())

    confidence = 0.0
    if hasattr(pipeline, "predict_proba"):
        confidence = float(np.max(pipeline.predict_proba(row)[0]))
    return label, confidence


def _calculate_hand_features(hands_result: Any) -> dict[str, float]:
    features = {column: 0.0 for column in HAND_FEATURE_COLUMNS}
    hand_landmarks = hands_result.multi_hand_landmarks or []
    handedness = hands_result.multi_handedness or []
    features["hand_count"] = float(len(hand_landmarks))

    by_side: dict[str, Any] = {}
    for landmarks, hand_info in zip(hand_landmarks, handedness):
        side = hand_info.classification[0].label.lower()
        if side in {"left", "right"} and side not in by_side:
            by_side[side] = landmarks

    if len(hand_landmarks) == 1 and not by_side:
        by_side["right"] = hand_landmarks[0]

    wrists = []
    for side in ("left", "right"):
        landmarks = by_side.get(side)
        if not landmarks:
            continue
        side_features, wrist = _single_hand_features(landmarks)
        features[f"{side}_hand_present"] = 1.0
        wrists.append(wrist)
        for key, value in side_features.items():
            features[f"{side}_{key}"] = value

    if len(wrists) == 2:
        features["wrist_distance"] = float(np.linalg.norm(wrists[0] - wrists[1]))
    return features


def _single_hand_features(hand_landmarks: Any) -> tuple[dict[str, float], np.ndarray]:
    points = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks.landmark], dtype=float)
    wrist = points[0]
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    width = max(float(max_xy[0] - min_xy[0]), 1e-6)
    height = max(float(max_xy[1] - min_xy[1]), 1e-6)
    scale = max(width, height, 1e-6)

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(points[a] - points[b]) / scale)

    def extension(tip: int, base: int) -> float:
        return float(np.linalg.norm(points[tip] - wrist) / max(np.linalg.norm(points[base] - wrist), 1e-6))

    features = {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_ratio": width / height,
        "thumb_index_distance": dist(4, 8),
        "index_middle_distance": dist(8, 12),
        "middle_ring_distance": dist(12, 16),
        "ring_pinky_distance": dist(16, 20),
        "palm_spread": dist(5, 17),
        "index_extension": extension(8, 5),
        "middle_extension": extension(12, 9),
        "ring_extension": extension(16, 13),
        "pinky_extension": extension(20, 17),
    }
    return features, wrist


def _classify_with_model(metrics: dict[str, float]) -> tuple[str, float] | None:
    model_bundle = _load_model()
    if model_bundle is None or pd is None:
        return None

    pipeline = model_bundle["pipeline"]
    feature_columns = model_bundle["feature_columns"]
    row = pd.DataFrame([{column: metrics[column] for column in feature_columns}])
    raw_label = pipeline.predict(row)[0]
    label = LABEL_DISPLAY.get(str(raw_label), str(raw_label))

    confidence = 0.0
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(row)[0]
        confidence = float(np.max(probabilities))
    return label, confidence


@lru_cache(maxsize=1)
def _load_model() -> dict[str, Any] | None:
    if not model_available():
        return None
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_mudra_model() -> dict[str, Any] | None:
    if not mudra_model_available():
        return None
    return joblib.load(MUDRA_MODEL_PATH)


def _classify_baseline(metrics: dict[str, float]) -> tuple[str, float]:
    average_knee = (metrics["left_knee_angle"] + metrics["right_knee_angle"]) / 2
    average_elbow = (metrics["left_elbow_angle"] + metrics["right_elbow_angle"]) / 2

    if metrics["hands_distance"] < 0.12 and metrics["hand_count"] >= 1:
        return "Anjali", 0.72
    if 85 <= average_knee <= 135 and metrics["knee_balance_delta"] < 30:
        return "Araimandi", 0.68
    if average_elbow > 130 and metrics["hand_count"] >= 1:
        return "Tripataka", 0.58
    if metrics["wrist_level_delta"] > 0.25 and metrics["knee_balance_delta"] > 35:
        return "Nataraja Pose", 0.55
    return "Pataka", 0.46


def _score_pose(metrics: dict[str, float], label: str | None) -> tuple[float, float, float]:
    shoulder_score = _score_delta(metrics["shoulder_level_delta"], 0.0, 0.08)
    hip_score = _score_delta(metrics["hip_level_delta"], 0.0, 0.08)
    knee_balance_score = _score_delta(metrics["knee_balance_delta"], 0.0, 35.0)
    elbow_balance_score = _score_delta(metrics["elbow_balance_delta"], 0.0, 45.0)
    symmetry_score = np.mean([shoulder_score, hip_score, knee_balance_score, elbow_balance_score])

    if label == "Araimandi":
        knee_target = 105.0
        average_knee = (metrics["left_knee_angle"] + metrics["right_knee_angle"]) / 2
        pose_match = _score_delta(abs(average_knee - knee_target), 0.0, 50.0)
    elif label == "Anjali":
        pose_match = _score_delta(metrics["hands_distance"], 0.0, 0.22)
    else:
        pose_match = np.mean([elbow_balance_score, shoulder_score, _score_delta(metrics["wrist_level_delta"], 0.0, 0.35)])

    posture_score = np.mean([symmetry_score, pose_match, shoulder_score])
    return round(float(posture_score), 1), round(float(symmetry_score), 1), round(float(pose_match), 1)


def _score_delta(value: float, target: float, tolerance: float) -> float:
    del target
    return float(np.clip(100.0 * (1.0 - value / tolerance), 0.0, 100.0))


def _feedback(metrics: dict[str, float], label: str | None) -> list[str]:
    feedback: list[str] = []
    if metrics["shoulder_level_delta"] > 0.06:
        feedback.append("Shoulders appear uneven; lift through the torso and level both shoulders.")
    if metrics["knee_balance_delta"] > 25:
        feedback.append("Knee angles are uneven; balance the bend across both legs.")
    if label == "Araimandi":
        average_knee = (metrics["left_knee_angle"] + metrics["right_knee_angle"]) / 2
        if average_knee > 130:
            feedback.append("The knee bend looks shallow for Araimandi; deepen the half-sitting stance.")
        elif average_knee < 80:
            feedback.append("The knee bend may be too deep; preserve lift through the spine.")
    if label in {"Tripataka", "Pataka"} and metrics["elbow_balance_delta"] > 35:
        feedback.append("Elbow alignment is asymmetric; match the arm extension more evenly.")
    if label == "Anjali" and metrics["hands_distance"] > 0.12:
        feedback.append("Bring both palms closer to the center line for a cleaner Anjali shape.")
    if not feedback:
        feedback.append("Overall alignment looks balanced for this baseline analysis.")
    return feedback
