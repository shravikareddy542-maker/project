"""
main_pipeline.py – Core computer-vision loop for Leo.

Orchestrates:
  MediaPipe detection → DeepSORT tracking → ArcFace recognition → FAISS matching

Updated:
  • fallback removed
  • recognition made more stable
  • avoids poisoning votes with repeated Unknown casts
  • keeps last strong identity per track for smoother recognition
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from detection.mediapipe_detector import MediaPipeFaceDetector
from tracking.deepsort_tracker import DeepSORTTracker
from recognition.arcface_onnx import ArcFaceONNX
from recognition.matcher_faiss import FAISSMatcher
from db.guest_db import get_connection, init_db, get_all_embeddings
from utils.image_ops import crop_face, passes_quality, align_face_simple, draw_bbox
from utils.voting import TrackVoter, GreetCooldown
from utils.greeter import Greeter


@dataclass
class TrackResult:
    track_id: int
    bbox: tuple[int, int, int, int]
    name: Optional[str] = None
    score: float = 0.0
    guest_id: Optional[int] = None
    confirmed: bool = False
    greeted: bool = False


@dataclass
class FrameResult:
    frame: np.ndarray
    tracks: list[TrackResult] = field(default_factory=list)
    fps: float = 0.0
    frame_idx: int = 0


class LeoPipeline:
    def __init__(self, use_arcface: bool = True):
        self.detector = MediaPipeFaceDetector()
        self.tracker = DeepSORTTracker()
        self.voter = TrackVoter()
        self.cooldown = GreetCooldown()
        self.matcher = FAISSMatcher()
        self.greeter = Greeter()

        self.arcface: Optional[ArcFaceONNX] = None
        if use_arcface:
            try:
                self.arcface = ArcFaceONNX()
            except FileNotFoundError as e:
                print(f"[WARN] ArcFace model not loaded: {e}")

        if not self.matcher.load():
            self._rebuild_faiss()

        self._frame_idx = 0
        self._last_detections: list[dict] = []
        self._track_frames: dict[int, int] = {}
        self._track_identity: dict[int, dict] = {}
        self._track_last_recog_frame: dict[int, int] = {}

        self._fps_t0 = time.time()
        self._fps_counter = 0
        self._current_fps = 0.0

        self.event_log: list[str] = []
        self._last_recognized_guest_id: int | None = None

    def _rebuild_faiss(self):
        conn = get_connection()
        init_db(conn)
        records = get_all_embeddings(conn)
        conn.close()
        self.matcher.build_index(records)
        self.matcher.save()

    def rebuild_index(self):
        self._rebuild_faiss()
        self._log("FAISS index rebuilt.")

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.event_log.append(entry)
        if len(self.event_log) > config.LOG_MAX_LINES:
            self.event_log = self.event_log[-config.LOG_MAX_LINES:]

    def _should_run_recognition(self, track_id: int, frames_seen: int) -> bool:
        if frames_seen < config.STABLE_FRAMES_BEFORE_RECOG:
            return False
        if self.arcface is None or self.matcher.total == 0:
            return False

        last_frame = self._track_last_recog_frame.get(track_id, -99999)
        if (self._frame_idx - last_frame) < config.RECOGNIZE_EVERY_N_FRAMES:
            return False

        return True

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        self._frame_idx += 1
        annotated = frame.copy()

        if self._frame_idx % config.DETECT_EVERY_N_FRAMES == 0 or self._frame_idx == 1:
            self._last_detections = self.detector.detect(frame)

        self.tracker.update(self._last_detections)
        active_tracks = self.tracker.get_active_tracks()

        active_ids = set()
        for t in active_tracks:
            active_ids.add(t.track_id)
            self._track_frames[t.track_id] = self._track_frames.get(t.track_id, 0) + 1

        self.voter.cleanup_stale(active_ids)

        stale_track_frames = [tid for tid in self._track_frames if tid not in active_ids]
        for tid in stale_track_frames:
            del self._track_frames[tid]

        stale_identity = [tid for tid in self._track_identity if tid not in active_ids]
        for tid in stale_identity:
            del self._track_identity[tid]

        stale_last_recog = [tid for tid in self._track_last_recog_frame if tid not in active_ids]
        for tid in stale_last_recog:
            del self._track_last_recog_frame[tid]

        track_results: list[TrackResult] = []

        for t in active_tracks:
            tr = TrackResult(track_id=t.track_id, bbox=t.bbox)
            frames_seen = self._track_frames.get(t.track_id, 0)

            if self._should_run_recognition(t.track_id, frames_seen):
                self._track_last_recog_frame[t.track_id] = self._frame_idx

                ok, _reason = passes_quality(frame, t.bbox)
                if ok:
                    crop = crop_face(frame, t.bbox)
                    if crop is not None:
                        aligned = align_face_simple(crop)
                        emb = self.arcface.get_embedding(aligned)
                        name, score, guest_id = self.matcher.match(emb)

                        if guest_id is not None and name and score >= config.RECOGNITION_THRESHOLD:
                            self.voter.cast_vote(t.track_id, name, score)

                            prev = self._track_identity.get(t.track_id)
                            best_score = score
                            if prev and prev.get("name") == name:
                                best_score = max(prev.get("score", 0.0), score)

                            self._track_identity[t.track_id] = {
                                "guest_id": guest_id,
                                "name": name,
                                "score": best_score,
                                "last_seen_frame": self._frame_idx,
                            }

            confirmed = self.voter.get_confirmed(t.track_id)
            identity = self._track_identity.get(t.track_id)

            if confirmed:
                cname, cscore = confirmed
                tr.name = cname
                tr.score = cscore
                tr.confirmed = True

                if identity:
                    tr.guest_id = identity.get("guest_id")
                    identity["last_seen_frame"] = self._frame_idx
                    identity["score"] = max(identity.get("score", 0.0), cscore)

            elif identity:
                age = self._frame_idx - identity.get("last_seen_frame", 0)

                if age <= config.IDENTITY_HOLD_FRAMES:
                    tr.name = identity["name"]
                    tr.score = identity["score"]
                    tr.confirmed = True
                    tr.guest_id = identity["guest_id"]
                else:
                    self._track_identity.pop(t.track_id, None)

            if tr.guest_id is not None and tr.name:
                # Store latest recognized guest for on-demand introduction
                self._last_recognized_guest_id = tr.guest_id

                # Check greeting BEFORE updating presence so the
                # absence gap is computed against the previous last_seen.
                if self.cooldown.should_greet(tr.guest_id):
                    self.cooldown.mark_greeted(tr.guest_id)
                    tr.greeted = True
                    greeting_text = self.greeter.greet_recognized(tr.name)
                    self._log(f"🎉 {greeting_text} (track {t.track_id}, score {tr.score:.2f})")

                # Update presence tracking every frame so cooldown
                # can detect when a guest leaves and comes back.
                self.cooldown.mark_seen(tr.guest_id)

            track_results.append(tr)

        for tr in track_results:
            if tr.confirmed and tr.name:
                label = f"{tr.name} ({tr.score:.2f}) T{tr.track_id}"
                color = (0, 255, 0)
            else:
                frames_seen = self._track_frames.get(tr.track_id, 0)
                if frames_seen < config.STABLE_FRAMES_BEFORE_RECOG:
                    label = f"Stabilizing... T{tr.track_id}"
                else:
                    label = f"Scanning... T{tr.track_id}"
                color = (0, 165, 255)

            draw_bbox(annotated, tr.bbox, label, color)

        self._fps_counter += 1
        elapsed = time.time() - self._fps_t0
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_t0 = time.time()

        return FrameResult(
            frame=annotated,
            tracks=track_results,
            fps=self._current_fps,
            frame_idx=self._frame_idx,
        )

    def get_latest_guest_id(self) -> int | None:
        """Return the most recently recognized guest_id, or None."""
        return self._last_recognized_guest_id

    def reset(self):
        self.tracker.reset()
        self.voter = TrackVoter()
        self.cooldown.reset()
        self._frame_idx = 0
        self._track_frames.clear()
        self._track_identity.clear()
        self._track_last_recog_frame.clear()
        self._last_recognized_guest_id = None
        self._log("Pipeline reset.")


def run_cli():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    pipeline = LeoPipeline()
    print("[INFO] Pipeline started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame)
        cv2.putText(
            result.frame,
            f"FPS: {result.fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        cv2.imshow("Leo Face Recognition", result.frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_cli()