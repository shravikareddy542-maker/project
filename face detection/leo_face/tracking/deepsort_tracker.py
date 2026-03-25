"""
DeepSORT-style multi-face tracker.
Provides persistent track_id per face across frames, ensuring labels
don't jump between people.

This is a *lightweight pure-Python* implementation inspired by DeepSORT
(Wojke et al.) using Kalman filter + Hungarian assignment + optional
appearance features. No external deep_sort library dependency.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─────────────────── Kalman state helpers ─────────────────

def _bbox_to_z(bbox):
    """(x1,y1,x2,y2) → [cx, cy, s, r] column vector."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    s = w * h        # scale (area)
    r = w / max(h, 1)  # aspect ratio
    return np.array([[cx], [cy], [s], [r]], dtype=np.float64)


def _z_to_bbox(z):
    """[cx, cy, s, r] → (x1, y1, x2, y2)."""
    cx, cy, s, r = z.flatten()[:4]
    w = np.sqrt(max(s * r, 1))
    h = s / max(w, 1)
    return (
        int(cx - w / 2),
        int(cy - h / 2),
        int(cx + w / 2),
        int(cy + h / 2),
    )


def _create_kalman():
    """7-state Kalman filter: [cx, cy, s, r, vx, vy, vs]."""
    kf = KalmanFilter(dim_x=7, dim_z=4)
    kf.F = np.eye(7)
    kf.F[0, 4] = 1.0
    kf.F[1, 5] = 1.0
    kf.F[2, 6] = 1.0
    kf.H = np.zeros((4, 7))
    kf.H[:4, :4] = np.eye(4)
    kf.R *= 10.0
    kf.P[4:, 4:] *= 1000.0
    kf.P *= 10.0
    kf.Q[4:, 4:] *= 0.01
    return kf


# ─────────────────── Single Track ─────────────────────────

class Track:
    _next_id = 1

    def __init__(self, bbox, n_init: int, max_age: int, feature=None):
        self.track_id = Track._next_id
        Track._next_id += 1
        self.kf = _create_kalman()
        self.kf.x[:4] = _bbox_to_z(bbox)
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.n_init = n_init
        self.max_age = max_age
        self.state = "tentative"   # tentative → confirmed → deleted
        self.features = []
        if feature is not None:
            self.features.append(feature)

    @property
    def bbox(self):
        return _z_to_bbox(self.kf.x)

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, feature=None):
        self.kf.update(_bbox_to_z(bbox))
        self.hits += 1
        self.time_since_update = 0
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > config.NN_BUDGET:
                self.features = self.features[-config.NN_BUDGET:]
        if self.state == "tentative" and self.hits >= self.n_init:
            self.state = "confirmed"

    def mark_missed(self):
        if self.state == "tentative":
            self.state = "deleted"
        elif self.time_since_update > self.max_age:
            self.state = "deleted"

    def is_confirmed(self):
        return self.state == "confirmed"

    def is_deleted(self):
        return self.state == "deleted"


# ─────────────────── IoU helpers ──────────────────────────

def _iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(1, (bb1[2] - bb1[0]) * (bb1[3] - bb1[1]))
    a2 = max(1, (bb2[2] - bb2[0]) * (bb2[3] - bb2[1]))
    return inter / (a1 + a2 - inter)


# ─────────────────── DeepSORT Tracker ─────────────────────

class DeepSORTTracker:
    """
    Lightweight DeepSORT tracker using IoU + optional appearance features.
    """

    def __init__(self,
                 max_age: int = None,
                 n_init: int = None,
                 max_cosine_distance: float = None,
                 nn_budget: int = None):
        self.max_age = max_age or config.MAX_AGE
        self.n_init = n_init or config.N_INIT
        self.max_cos_dist = max_cosine_distance or config.MAX_COSINE_DISTANCE
        self.nn_budget = nn_budget or config.NN_BUDGET
        self.tracks: list[Track] = []

    def predict(self):
        for t in self.tracks:
            t.predict()

    def update(self, detections: list[dict], features: list[np.ndarray] | None = None):
        """
        detections: list of {bbox: (x1,y1,x2,y2), confidence: float}
        features: optional parallel list of appearance embeddings
        """
        self.predict()

        # Build cost matrix (IoU-based)
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        cost = np.full((n_tracks, n_dets), 1e5, dtype=np.float64)

        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                iou_val = _iou(t.bbox, d["bbox"])
                cost[i, j] = 1.0 - iou_val

        # Hungarian assignment
        matched_t, matched_d = set(), set()
        if n_tracks > 0 and n_dets > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 0.7:  # IoU gate
                    self.tracks[r].update(
                        detections[c]["bbox"],
                        features[c] if features else None,
                    )
                    matched_t.add(r)
                    matched_d.add(c)

        # Mark unmatched tracks
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.mark_missed()

        # Create new tracks for unmatched detections
        for j, d in enumerate(detections):
            if j not in matched_d:
                feat = features[j] if features else None
                self.tracks.append(
                    Track(d["bbox"], self.n_init, self.max_age, feat)
                )

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def get_active_tracks(self) -> list[Track]:
        """Return only confirmed, recently-updated tracks."""
        return [t for t in self.tracks if t.is_confirmed() and t.time_since_update == 0]

    def get_all_tracks(self) -> list[Track]:
        return list(self.tracks)

    def reset(self):
        self.tracks.clear()
        Track._next_id = 1
