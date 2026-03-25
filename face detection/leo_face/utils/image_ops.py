"""
Image quality filters and face alignment helpers.
"""
import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def crop_face(frame: np.ndarray, bbox: tuple[int, int, int, int],
              margin: float = 0.2) -> np.ndarray | None:
    """
    Crop a face region from frame with margin.
    bbox = (x1, y1, x2, y2) in pixel coords.
    Returns BGR crop or None if invalid.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    # Add margin
    mx, my = int(bw * margin), int(bh * margin)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def is_face_too_small(bbox: tuple[int, int, int, int],
                      min_size: int = None) -> bool:
    min_size = min_size or config.MIN_FACE_SIZE
    x1, y1, x2, y2 = bbox
    return (x2 - x1) < min_size or (y2 - y1) < min_size


def is_blurry(crop: np.ndarray, threshold: float = None) -> bool:
    threshold = threshold or config.BLUR_THRESHOLD
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def is_too_dark(crop: np.ndarray, threshold: float = None) -> bool:
    threshold = threshold or config.DARK_THRESHOLD
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < threshold


def passes_quality(frame: np.ndarray,
                   bbox: tuple[int, int, int, int]) -> tuple[bool, str]:
    """
    Run all quality checks.
    Returns (passed: bool, reason: str).
    """
    if is_face_too_small(bbox):
        return False, "too_small"
    crop = crop_face(frame, bbox, margin=0.0)
    if crop is None:
        return False, "bad_crop"
    if is_blurry(crop):
        return False, "blurry"
    if is_too_dark(crop):
        return False, "too_dark"
    return True, "ok"


def align_face_simple(crop: np.ndarray,
                      target_size: tuple[int, int] = None) -> np.ndarray:
    """
    Simple alignment: just resize to ArcFace input size.
    For production, use 5-point landmark alignment.
    """
    target_size = target_size or config.ARCFACE_INPUT_SIZE
    return cv2.resize(crop, target_size)


def draw_bbox(frame: np.ndarray,
              bbox: tuple[int, int, int, int],
              label: str = "",
              color: tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2):
    """
    Draw a professional bounding box with corner brackets + label pill.
    Gives a modern security/dashboard aesthetic.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w = x2 - x1
    h = y2 - y1
    corner_len = max(15, min(w, h) // 4)  # adaptive corner length

    # ── Draw corner brackets instead of full rectangle ──
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness + 1, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness + 1, cv2.LINE_AA)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness + 1, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness + 1, cv2.LINE_AA)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness + 1, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness + 1, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness + 1, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness + 1, cv2.LINE_AA)

    # ── Thin connecting lines between corners ──
    faint = tuple(max(0, c // 3) for c in color)
    cv2.rectangle(frame, (x1, y1), (x2, y2), faint, 1, cv2.LINE_AA)

    # ── Label pill with semi-transparent background ──
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, font_thickness)
        pad_x, pad_y = 8, 6
        pill_x1 = x1
        pill_y1 = y1 - th - 2 * pad_y - 2
        pill_x2 = x1 + tw + 2 * pad_x
        pill_y2 = y1 - 2

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), (15, 15, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Colored top accent line on pill
        cv2.line(frame, (pill_x1, pill_y1), (pill_x2, pill_y1), color, 2, cv2.LINE_AA)

        # Text
        cv2.putText(frame, label,
                    (pill_x1 + pad_x, pill_y2 - pad_y),
                    font, scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
