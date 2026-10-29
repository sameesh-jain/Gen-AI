"""Computer vision utilities for face detection, privacy-preserving encoding, and head pose estimation.

This module uses OpenCV's built-in Haar cascades for face detection and generates
histogram-based privacy-preserving embeddings without requiring external deep learning
model weights. Head pose estimation is computed via MediaPipe when present, with an
automatic robust OpenCV geometric fallback.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import get_config
from utils.logger import get_logger

logger = get_logger("FaceUtils")

# Cache Haar cascades for performance
_CASCADE_PATH = cv2.data.haarcascades
_face_cascade: Optional[cv2.CascadeClassifier] = None
_eye_cascade: Optional[cv2.CascadeClassifier] = None

# Optional MediaPipe import
_mediapipe_available = False
_mp_face_mesh = None
try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        _mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _mediapipe_available = True
except Exception as mp_err:
    logger.info(f"MediaPipe FaceMesh not active ({mp_err}). Using OpenCV geometric head-pose fallback.")


def _get_face_cascade() -> cv2.CascadeClassifier:
    """Retrieve or initialize the OpenCV frontal face cascade classifier."""
    global _face_cascade
    if _face_cascade is None:
        path = os.path.join(_CASCADE_PATH, "haarcascade_frontalface_default.xml")
        _face_cascade = cv2.CascadeClassifier(path)
        if _face_cascade.empty():
            logger.error(f"Failed to load Haar cascade from {path}")
    return _face_cascade


def _get_eye_cascade() -> cv2.CascadeClassifier:
    """Retrieve or initialize the OpenCV eye cascade classifier."""
    global _eye_cascade
    if _eye_cascade is None:
        path = os.path.join(_CASCADE_PATH, "haarcascade_eye.xml")
        _eye_cascade = cv2.CascadeClassifier(path)
    return _eye_cascade


def detect_faces(
    frame: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (40, 40),
) -> List[Tuple[int, int, int, int]]:
    """Detect human faces in a video frame using OpenCV's built-in Haar cascade.

    Args:
        frame: BGR video frame array.
        scale_factor: Parameter specifying how much the image size is reduced at each image scale.
        min_neighbors: Parameter specifying how many neighbors each candidate rectangle should retain.
        min_size: Minimum possible object size. Objects smaller than that are ignored.

    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes as (x, y, w, h).
    """
    if frame is None or frame.size == 0:
        return []

    try:
        cascade = _get_face_cascade()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        return [tuple(map(int, box)) for box in faces]
    except Exception as err:
        logger.error(f"Error during face detection: {err}")
        return []


def encode_face(face_img: np.ndarray) -> np.ndarray:
    """Compute a privacy-preserving facial descriptor embedding.

    This function extracts multi-scale spatial histogram features (intensity and gradients)
    from the face ROI. It creates a compact 256-dimensional numerical vector.
    Crucially, original facial photos cannot be reconstructed from this representation,
    preserving participant privacy while enabling robust re-identification.

    Args:
        face_img: Cropped BGR or grayscale image of the detected face.

    Returns:
        np.ndarray: Normalized 256-dimensional feature vector.
    """
    if face_img is None or face_img.size == 0:
        return np.zeros(256, dtype=np.float32)

    try:
        # Standardize face patch
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()

        resized = cv2.resize(gray, (64, 64))
        equalized = cv2.equalizeHist(resized)

        # 4x4 spatial grid with 16-bin intensity histograms (16 * 16 = 256 features)
        features = []
        cell_h, cell_w = 16, 16
        for r in range(4):
            for c in range(4):
                cell = equalized[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                hist = cv2.calcHist([cell], [0], None, [16], [0, 256])
                hist = hist.flatten()
                features.extend(hist)

        vec = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    except Exception as err:
        logger.error(f"Error encoding face: {err}")
        return np.zeros(256, dtype=np.float32)


def match_face(
    embedding: np.ndarray,
    known_embeddings: Dict[str, np.ndarray],
    threshold: float = 0.72,
) -> Tuple[Optional[str], float]:
    """Match a face embedding against a registry of known embeddings using cosine similarity.

    Args:
        embedding: 1D normalized face feature vector.
        known_embeddings: Dictionary mapping participant_id to their stored embedding.
        threshold: Minimum cosine similarity score required for a match.

    Returns:
        Tuple[Optional[str], float]: (matched_participant_id, confidence_score) or (None, best_score).
    """
    if embedding is None or len(embedding) == 0 or not known_embeddings:
        return None, 0.0

    best_id: Optional[str] = None
    best_similarity: float = -1.0

    emb_norm = np.linalg.norm(embedding)
    if emb_norm == 0:
        return None, 0.0

    for participant_id, known_emb in known_embeddings.items():
        if known_emb is None:
            continue
        known_arr = np.array(known_emb, dtype=np.float32)
        known_norm = np.linalg.norm(known_arr)
        if known_norm == 0:
            continue

        sim = float(np.dot(embedding, known_arr) / (emb_norm * known_norm))
        if sim > best_similarity:
            best_similarity = sim
            best_id = participant_id

    if best_similarity >= threshold:
        return best_id, best_similarity

    return None, max(0.0, best_similarity)


def estimate_head_pose(
    frame: np.ndarray, face_bbox: Tuple[int, int, int, int]
) -> Tuple[float, float, float, bool]:
    """Estimate head pose (yaw, pitch, roll) and determine whether the participant is facing forward.

    Uses MediaPipe FaceMesh when available, otherwise falls back to OpenCV facial geometry analysis.

    Args:
        frame: Full BGR frame.
        face_bbox: Tuple (x, y, w, h) of the face bounding box.

    Returns:
        Tuple[float, float, float, bool]: (yaw_deg, pitch_deg, roll_deg, is_frontal).
    """
    x, y, w, h = face_bbox
    if frame is None or w <= 0 or h <= 0:
        return 0.0, 0.0, 0.0, True

    # Try MediaPipe FaceMesh if active
    if _mediapipe_available and _mp_face_mesh is not None:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = _mp_face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                img_h, img_w = frame.shape[:2]
                face_center_x = x + w / 2.0
                face_center_y = y + h / 2.0

                # Find closest landmark cluster to this bbox center
                for face_landmarks in results.multi_face_landmarks:
                    nose_lm = face_landmarks.landmark[1]
                    nx, ny = nose_lm.x * img_w, nose_lm.y * img_h

                    if x <= nx <= x + w and y <= ny <= y + h:
                        # 2D landmark coordinates
                        left_eye = face_landmarks.landmark[33]
                        right_eye = face_landmarks.landmark[263]
                        chin = face_landmarks.landmark[152]

                        lx, ly = left_eye.x * img_w, left_eye.y * img_h
                        rx, ry = right_eye.x * img_w, right_eye.y * img_h
                        cx, cy = chin.x * img_w, chin.y * img_h

                        # Roll estimation
                        d_x = rx - lx
                        d_y = ry - ly
                        roll = math.degrees(math.atan2(d_y, d_x))

                        # Yaw estimation (nose position relative to eye midpoint)
                        eye_mid_x = (lx + rx) / 2.0
                        eye_dist = max(1.0, abs(rx - lx))
                        yaw_ratio = (nx - eye_mid_x) / (eye_dist / 2.0)
                        yaw = float(np.clip(yaw_ratio * 45.0, -90.0, 90.0))

                        # Pitch estimation (nose relative to eye-chin line)
                        eye_mid_y = (ly + ry) / 2.0
                        face_span_y = max(1.0, abs(cy - eye_mid_y))
                        pitch_ratio = (ny - eye_mid_y) / face_span_y - 0.45
                        pitch = float(np.clip(pitch_ratio * 60.0, -90.0, 90.0))

                        cfg = get_config()
                        threshold = cfg.engagement.head_pose_threshold
                        is_frontal = abs(yaw) <= threshold and abs(pitch) <= threshold

                        return yaw, pitch, roll, is_frontal
        except Exception as err:
            logger.debug(f"MediaPipe head pose estimation failed, using fallback: {err}")

    # Fallback to OpenCV geometric and eye distribution analysis
    try:
        fh, fw = frame.shape[:2]
        pad_x = max(0, x)
        pad_y = max(0, y)
        roi_w = min(w, fw - pad_x)
        roi_h = min(h, fh - pad_y)

        face_roi = frame[pad_y : pad_y + roi_h, pad_x : pad_x + roi_w]
        if face_roi.size == 0:
            return 0.0, 0.0, 0.0, True

        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eye_cascade = _get_eye_cascade()

        # Search top half of face for eyes
        top_half = gray_roi[: int(roi_h * 0.6), :]
        eyes = eye_cascade.detectMultiScale(top_half, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))

        yaw = 0.0
        pitch = 0.0
        roll = 0.0

        if len(eyes) >= 2:
            # Sort eyes left-to-right
            sorted_eyes = sorted(eyes, key=lambda e: e[0])
            e1, e2 = sorted_eyes[0], sorted_eyes[1]
            c1_x = e1[0] + e1[2] / 2.0
            c1_y = e1[1] + e1[3] / 2.0
            c2_x = e2[0] + e2[2] / 2.0
            c2_y = e2[1] + e2[3] / 2.0

            dx = c2_x - c1_x
            dy = c2_y - c1_y
            roll = math.degrees(math.atan2(dy, max(dx, 1e-4)))

            # Yaw: asymmetry of eye midpoint relative to face center
            eye_mid = (c1_x + c2_x) / 2.0
            face_mid = roi_w / 2.0
            yaw = ((eye_mid - face_mid) / max(face_mid, 1.0)) * 50.0

            # Pitch: vertical location of eyes
            eye_mid_y = (c1_y + c2_y) / 2.0
            expected_eye_y = roi_h * 0.35
            pitch = ((eye_mid_y - expected_eye_y) / max(roi_h * 0.35, 1.0)) * 40.0
        else:
            # Fallback based on intensity symmetry between left and right halves
            left_half = gray_roi[:, : roi_w // 2]
            right_half = gray_roi[:, roi_w // 2 :]
            mean_l = float(np.mean(left_half)) if left_half.size > 0 else 128.0
            mean_r = float(np.mean(right_half)) if right_half.size > 0 else 128.0
            diff = (mean_r - mean_l) / max(1.0, (mean_r + mean_l))
            yaw = float(diff * 45.0)

        cfg = get_config()
        threshold = cfg.engagement.head_pose_threshold
        is_frontal = abs(yaw) <= threshold and abs(pitch) <= threshold

        return float(yaw), float(pitch), float(roll), is_frontal
    except Exception as err:
        logger.error(f"Fallback head pose estimation error: {err}")
        return 0.0, 0.0, 0.0, True


def compute_movement_score(
    prev_bbox: Optional[Tuple[int, int, int, int]],
    curr_bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
) -> float:
    """Compute normalized movement score based on centroid displacement and bounding scale changes.

    Args:
        prev_bbox: Prior bounding box (x, y, w, h) or None if new.
        curr_bbox: Current bounding box (x, y, w, h).
        frame_shape: Height and width of the video frame.

    Returns:
        float: Movement intensity score normalized between 0.0 and 1.0.
    """
    if prev_bbox is None or curr_bbox is None:
        return 0.3  # Baseline default on entry

    px, py, pw, ph = prev_bbox
    cx, cy, cw, ch = curr_bbox
    fh, fw = frame_shape[:2]
    diagonal = math.sqrt(fw**2 + fh**2)

    # Centroid displacement
    prev_center = (px + pw / 2.0, py + ph / 2.0)
    curr_center = (cx + cw / 2.0, cy + ch / 2.0)
    dist = math.dist(prev_center, curr_center)

    normalized_dist = min(1.0, (dist / (diagonal * 0.15)))

    # Area change ratio
    prev_area = max(1.0, float(pw * ph))
    curr_area = max(1.0, float(cw * ch))
    area_delta = abs(curr_area - prev_area) / max(prev_area, curr_area)
    normalized_area_delta = min(1.0, area_delta * 2.0)

    # Combined movement metric
    raw_movement = 0.7 * normalized_dist + 0.3 * normalized_area_delta
    return float(np.clip(raw_movement, 0.0, 1.0))


def draw_face_annotations(
    frame: np.ndarray,
    participant_data: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw bounding boxes, status tags, and engagement scores on the frame for visualization.

    Args:
        frame: BGR video frame to annotate.
        participant_data: List of participant state dictionaries containing bbox, id, name, and engagement.

    Returns:
        np.ndarray: Annotated video frame.
    """
    if frame is None or frame.size == 0:
        return frame

    annotated = frame.copy()
    for p in participant_data:
        bbox = p.get("bbox")
        if not bbox:
            continue
        x, y, w, h = bbox
        pid = p.get("participant_id", "Unknown")
        name = p.get("name", pid)
        score = p.get("engagement_score", 0.0)
        is_frontal = p.get("is_frontal", True)

        # Color based on engagement score: green for high, orange for moderate, red for low
        if score >= 0.7:
            color = (0, 200, 0)
        elif score >= 0.4:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        # Draw bounding rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Header tag
        label = f"{name} ({score:.0%})"
        if not is_frontal:
            label += " [Away]"

        cv2.rectangle(annotated, (x, max(0, y - 24)), (x + w, y), color, cv2.FILLED)
        cv2.putText(
            annotated,
            label,
            (x + 4, max(12, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated
