"""
MediaPipe-based multi-face detector.
Uses the new MediaPipe Tasks API (0.10.14+).
Returns list of bounding boxes in pixel coords for each frame.
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class MediaPipeFaceDetector:
    """
    Wraps MediaPipe Face Detection (Tasks API) for multi-face detection.
    Falls back to the legacy mp.solutions API if available.
    """

    def __init__(self,
                 min_confidence: float = None,
                 model_selection: int = None,
                 max_faces: int = None):
        self.min_confidence = min_confidence or config.DETECTION_CONFIDENCE
        self.model_selection = model_selection if model_selection is not None else config.MODEL_SELECTION
        self.max_faces = max_faces or config.MAX_NUM_FACES
        self._use_tasks_api = True
        self._detector = None

        # Try new Tasks API first
        try:
            self._init_tasks_api()
        except Exception:
            # Fallback to legacy solutions API
            try:
                self._init_legacy_api()
                self._use_tasks_api = False
            except Exception as e:
                raise RuntimeError(
                    f"Cannot initialise MediaPipe face detector: {e}\n"
                    "Install mediapipe: pip install mediapipe"
                )

    def _init_tasks_api(self):
        """Initialise using MediaPipe Tasks API (v0.10.14+)."""
        # Download the model automatically via MediaPipe
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "blaze_face_short_range.tflite"
        )

        # If model file doesn't exist, use the built-in model downloader
        if not os.path.isfile(model_path):
            # MediaPipe can work with a model asset buffer from the package
            import urllib.request
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception:
                # If download fails, try using OpenCV as pure fallback
                raise RuntimeError("Cannot download MediaPipe face detection model")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.min_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def _init_legacy_api(self):
        """Initialise using legacy mp.solutions API (pre-0.10.14)."""
        self._detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=self.min_confidence,
            model_selection=self.model_selection,
        )

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect faces in a BGR frame.
        Returns list of dicts: {bbox: (x1,y1,x2,y2), confidence: float}
        """
        if self._use_tasks_api:
            return self._detect_tasks_api(frame)
        else:
            return self._detect_legacy(frame)

    def _detect_tasks_api(self, frame: np.ndarray) -> list[dict]:
        """Detection using the new Tasks API."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._detector.detect(mp_image)

        detections = []
        if result.detections:
            for det in result.detections[:self.max_faces]:
                bb = det.bounding_box
                x1 = max(0, bb.origin_x)
                y1 = max(0, bb.origin_y)
                x2 = min(w, bb.origin_x + bb.width)
                y2 = min(h, bb.origin_y + bb.height)
                conf = det.categories[0].score if det.categories else 0.0
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                })
        return detections

    def _detect_legacy(self, frame: np.ndarray) -> list[dict]:
        """Detection using the legacy mp.solutions API."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        detections = []
        if results.detections:
            for det in results.detections[:self.max_faces]:
                bb = det.location_data.relative_bounding_box
                x1 = max(0, int(bb.xmin * w))
                y1 = max(0, int(bb.ymin * h))
                x2 = min(w, int((bb.xmin + bb.width) * w))
                y2 = min(h, int((bb.ymin + bb.height) * h))
                conf = det.score[0] if det.score else 0.0
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                })
        return detections

    def close(self):
        if self._detector is not None:
            try:
                if self._use_tasks_api:
                    self._detector.close()
                else:
                    self._detector.close()
            except Exception:
                pass
            self._detector = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
