"""
Leo Unified GUI – Real-time Face Detection & Voice Assistant
A single PyQt6 desktop application merging the face detection / recognition
pipeline with the voice assistant (NLU + TTS + YouTube playback).
"""

from __future__ import annotations

import os
import sys
import time
import re
import platform
from dataclasses import dataclass
from typing import Optional

# ── Resolve paths so we can import from the existing leo_face package ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
_LEO_FACE_DIR = os.path.join(_PROJECT_DIR, "face detection", "leo_face")

if _LEO_FACE_DIR not in sys.path:
    sys.path.insert(0, _LEO_FACE_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# ── PRE-LOAD ONNXRuntime DLLs ──
# Import onnxruntime BEFORE PyQt6 to prevent DLL initialization conflicts on Windows.
try:
    import onnxruntime
except ImportError:
    pass

import numpy as np
import cv2
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QFrame, QScrollArea,
    QGroupBox, QSplitter, QSizePolicy, QFormLayout, QMessageBox,
    QStackedWidget, QGraphicsDropShadowEffect, QComboBox
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QSize, QPropertyAnimation,
    QEasingCurve
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QIcon

# ── Face pipeline imports ──
import config as leo_config
from detection.mediapipe_detector import MediaPipeFaceDetector
from tracking.deepsort_tracker import DeepSORTTracker
from db.guest_db import (
    get_connection, init_db, get_all_embeddings,
    add_guest, add_embedding, list_guests, delete_guest,
    count_embeddings_for_guest
)
from utils.image_ops import crop_face, passes_quality, align_face_simple, draw_bbox
from utils.voting import TrackVoter, GreetCooldown
from utils.greeter import Greeter
from utils.welcome_generator import WelcomeGenerator, GuestProfile, load_guest_profile

# ── Lazy imports for onnxruntime-dependent modules ──
ArcFaceONNX = None
FAISSMatcher = None

def _lazy_load_recognition():
    """Import recognition modules on demand to avoid crash if onnxruntime DLL fails."""
    global ArcFaceONNX, FAISSMatcher
    if ArcFaceONNX is None:
        try:
            from recognition.arcface_onnx import ArcFaceONNX as _Arc
            ArcFaceONNX = _Arc
        except Exception as e:
            print(f"[WARN] ArcFace not available: {e}")
    if FAISSMatcher is None:
        try:
            from recognition.matcher_faiss import FAISSMatcher as _FM
            FAISSMatcher = _FM
        except Exception as e:
            print(f"[WARN] FAISS not available: {e}")

# ── Voice assistant imports ──
try:
    import speech_recognition as sr
    HAS_SPEECH = True
except ImportError:
    HAS_SPEECH = False

try:
    import pywhatkit
    HAS_PYWHATKIT = True
except ImportError:
    HAS_PYWHATKIT = False

try:
    import nltk
    from nltk import word_tokenize, pos_tag
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False


# ═══════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE & STYLESHEET
# ═══════════════════════════════════════════════════════════════════════

COLORS = {
    "bg_primary": "#0a0e1a",
    "bg_secondary": "#111827",
    "bg_card": "#1a2035",
    "bg_card_hover": "#1e2640",
    "bg_input": "#0d1321",
    "border": "#2a3450",
    "border_light": "#3a4a6a",
    "accent": "#00d4ff",
    "accent_dim": "#0090b0",
    "success": "#00e676",
    "warning": "#ffab00",
    "danger": "#ff5252",
    "text_primary": "#e8eaf6",
    "text_secondary": "#8892b0",
    "text_muted": "#5a6480",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLORS['bg_primary']};
    color: {COLORS['text_primary']};
    font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
    font-size: 13px;
}}

QGroupBox {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    margin-top: 14px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
    font-size: 13px;
    color: {COLORS['accent']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 14px;
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    color: {COLORS['accent']};
    font-size: 12px;
    letter-spacing: 1px;
}}

QPushButton {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 12px;
    min-height: 32px;
}}
QPushButton:hover {{
    background-color: {COLORS['bg_card_hover']};
    border-color: {COLORS['accent']};
    color: {COLORS['accent']};
}}
QPushButton:pressed {{
    background-color: {COLORS['accent_dim']};
}}
QPushButton#btnPrimary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS['accent']}, stop:1 #007cf0);
    color: #ffffff;
    border: none;
    font-weight: 700;
}}
QPushButton#btnPrimary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #33ddff, stop:1 #3399ff);
}}
QPushButton#btnDanger {{
    background-color: {COLORS['danger']};
    color: #ffffff;
    border: none;
}}
QPushButton#btnDanger:hover {{
    background-color: #ff7777;
}}
QPushButton#btnSuccess {{
    background-color: {COLORS['success']};
    color: #0a0e1a;
    border: none;
    font-weight: 700;
}}

QLineEdit {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: {COLORS['accent_dim']};
}}
QLineEdit:focus {{
    border-color: {COLORS['accent']};
}}

QTextEdit {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_secondary']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 11px;
    selection-background-color: {COLORS['accent_dim']};
}}

QLabel {{
    color: {COLORS['text_primary']};
}}
QLabel#labelMuted {{
    color: {COLORS['text_muted']};
    font-size: 11px;
}}
QLabel#labelAccent {{
    color: {COLORS['accent']};
    font-weight: 700;
}}
QLabel#labelSuccess {{
    color: {COLORS['success']};
    font-weight: 700;
}}

QSplitter::handle {{
    background-color: {COLORS['border']};
    width: 2px;
}}
QSplitter::handle:hover {{
    background-color: {COLORS['accent']};
}}

QScrollBar:vertical {{
    background: {COLORS['bg_secondary']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['border_light']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['accent']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""


# ═══════════════════════════════════════════════════════════════════════
#  NLU (from merge module) – re-implemented inline to avoid import issues
# ═══════════════════════════════════════════════════════════════════════

def _ensure_nltk_data():
    if not HAS_NLTK:
        return
    packages = [
        "punkt", "stopwords", "wordnet", "omw-1.4",
        "averaged_perceptron_tagger", "punkt_tab",
        "averaged_perceptron_tagger_eng",
    ]
    for p in packages:
        try:
            nltk.download(p, quiet=True)
        except Exception:
            pass


def _penn_to_wn(tag: str):
    if not HAS_NLTK:
        return None
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN


class LeoNLU:
    def __init__(self, wake_word: str = "leo"):
        self.wake = wake_word.lower()
        if not HAS_NLTK:
            return
        self.lemmatizer = WordNetLemmatizer()
        base = set(stopwords.words("english"))
        extra = {
            "please", "kindly", "hey", "hi", "hello",
            "can", "could", "would", "will", "you", "me",
            "my", "your", "now", "just", "then", "ok", "okay",
        }
        self.stops = (base | extra) - {"play"}
        self.generic_music = {"song", "music", "video", "track", "youtube", "yt"}
        self.play_words = {"play", "start", "put"}
        self.stop_intent = {"stop", "exit", "quit", "close"}

    def normalize(self, text: str) -> list[str]:
        if not HAS_NLTK:
            return text.lower().split()
        text = text.lower().strip()
        text = re.sub(r"[^\w\s']+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        toks = word_tokenize(text)
        tagged = pos_tag(toks)
        lemmas = [self.lemmatizer.lemmatize(w, _penn_to_wn(t)) for w, t in tagged]
        lemmas = [w for w in lemmas if w != self.wake]
        return [w for w in lemmas if w not in self.stops]

    def parse_play_query(self, raw: str) -> Optional[str]:
        if not HAS_NLTK:
            return None
        raw_l = raw.lower()
        tokens = self.normalize(raw)
        if not tokens:
            return ""
        if any(w in self.stop_intent for w in tokens):
            return "__STOP__"
        play_idx = None
        for i, w in enumerate(tokens):
            if w in self.play_words:
                play_idx = i
                break
        if play_idx is None and "play" in tokens:
            play_idx = tokens.index("play")
        if self.wake not in raw_l:
            if play_idx is None or play_idx != 0:
                return None
        if play_idx is None:
            return ""
        query_tokens = tokens[play_idx + 1:]
        if len(query_tokens) >= 2:
            query_tokens = [w for w in query_tokens if w not in self.generic_music]
        return " ".join(query_tokens).strip()


def is_intro_intent(text: str) -> bool:
    t = text.lower().strip()
    name_intent = ("name" in t) and any(k in t for k in ["your", "you", "what", "tell", "who"])
    about_intent = any(
        p in t for p in [
            "who are you", "tell me about yourself", "tell about yourself",
            "introduce yourself", "about yourself", "self intro", "self-intro",
        ]
    )
    return name_intent or about_intent


def intro_response(text: str, name: str = "Leo") -> str:
    t = text.lower().strip()
    if ("name" in t) and any(k in t for k in ["your", "you", "what", "tell", "who"]):
        return f"Hello! I'm {name}, your intelligent face recognition assistant."
    return (
        f"Hello! I'm {name}, an AI-powered face recognition and voice assistant. "
        "I can detect and recognize faces in real-time, greet guests, "
        "and play your favorite songs on YouTube. Just say my name!"
    )


# ═══════════════════════════════════════════════════════════════════════
#  FACE PIPELINE (reuses existing LeoPipeline logic inline)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrackResult:
    track_id: int
    bbox: tuple
    name: Optional[str] = None
    score: float = 0.0
    guest_id: Optional[int] = None
    confirmed: bool = False
    greeted: bool = False


class FacePipeline:
    """Wraps the full face detection → tracking → recognition pipeline."""

    def __init__(self):
        _lazy_load_recognition()

        self.detector = MediaPipeFaceDetector()
        self.tracker = DeepSORTTracker()
        self.voter = TrackVoter()
        self.cooldown = GreetCooldown()
        self.matcher = FAISSMatcher() if FAISSMatcher is not None else None
        self.greeter = Greeter()

        self.arcface: Optional[ArcFaceONNX] = None
        if ArcFaceONNX is not None:
            try:
                self.arcface = ArcFaceONNX()
            except (FileNotFoundError, Exception) as e:
                print(f"[WARN] ArcFace model not loaded: {e}")

        if self.matcher is not None and not self.matcher.load():
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
        if self.matcher is None:
            return
        conn = get_connection()
        init_db(conn)
        records = get_all_embeddings(conn)
        conn.close()
        self.matcher.build_index(records)
        self.matcher.save()

    def rebuild_index(self):
        self._rebuild_faiss()

    def _should_run_recognition(self, track_id: int, frames_seen: int) -> bool:
        if frames_seen < leo_config.STABLE_FRAMES_BEFORE_RECOG:
            return False
        if self.arcface is None or self.matcher is None or self.matcher.total == 0:
            return False
        last_frame = self._track_last_recog_frame.get(track_id, -99999)
        if (self._frame_idx - last_frame) < leo_config.RECOGNIZE_EVERY_N_FRAMES:
            return False
        return True

    def process_frame(self, frame: np.ndarray):
        self._frame_idx += 1
        annotated = frame.copy()

        if self._frame_idx % leo_config.DETECT_EVERY_N_FRAMES == 0 or self._frame_idx == 1:
            self._last_detections = self.detector.detect(frame)

        self.tracker.update(self._last_detections)
        active_tracks = self.tracker.get_active_tracks()

        active_ids = set()
        for t in active_tracks:
            active_ids.add(t.track_id)
            self._track_frames[t.track_id] = self._track_frames.get(t.track_id, 0) + 1

        self.voter.cleanup_stale(active_ids)

        for tid in [t for t in self._track_frames if t not in active_ids]:
            del self._track_frames[tid]
        for tid in [t for t in self._track_identity if t not in active_ids]:
            del self._track_identity[tid]
        for tid in [t for t in self._track_last_recog_frame if t not in active_ids]:
            del self._track_last_recog_frame[tid]

        track_results: list[TrackResult] = []
        events: list[str] = []

        for t in active_tracks:
            tr = TrackResult(track_id=t.track_id, bbox=t.bbox)
            frames_seen = self._track_frames.get(t.track_id, 0)

            if self._should_run_recognition(t.track_id, frames_seen):
                self._track_last_recog_frame[t.track_id] = self._frame_idx
                ok, _ = passes_quality(frame, t.bbox)
                if ok:
                    crop = crop_face(frame, t.bbox)
                    if crop is not None:
                        aligned = align_face_simple(crop)
                        emb = self.arcface.get_embedding(aligned)
                        name, score, guest_id = self.matcher.match(emb)
                        if guest_id is not None and name and score >= leo_config.RECOGNITION_THRESHOLD:
                            self.voter.cast_vote(t.track_id, name, score)
                            prev = self._track_identity.get(t.track_id)
                            best_score = score
                            if prev and prev.get("name") == name:
                                best_score = max(prev.get("score", 0.0), score)
                            self._track_identity[t.track_id] = {
                                "guest_id": guest_id, "name": name,
                                "score": best_score, "last_seen_frame": self._frame_idx,
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
                if age <= leo_config.IDENTITY_HOLD_FRAMES:
                    tr.name = identity["name"]
                    tr.score = identity["score"]
                    tr.confirmed = True
                    tr.guest_id = identity["guest_id"]
                else:
                    self._track_identity.pop(t.track_id, None)

            if tr.guest_id is not None and tr.name:
                self._last_recognized_guest_id = tr.guest_id
                if self.cooldown.should_greet(tr.guest_id):
                    self.cooldown.mark_greeted(tr.guest_id)
                    tr.greeted = True
                    greeting_text = self.greeter.greet_recognized(tr.name)
                    events.append(f"🎉 {greeting_text}")
                self.cooldown.mark_seen(tr.guest_id)

            track_results.append(tr)

        # Draw bounding boxes
        for tr in track_results:
            if tr.confirmed and tr.name:
                label = f"{tr.name} ({tr.score:.2f}) T{tr.track_id}"
                color = (0, 255, 0)
            else:
                fs = self._track_frames.get(tr.track_id, 0)
                if fs < leo_config.STABLE_FRAMES_BEFORE_RECOG:
                    label = f"Stabilizing... T{tr.track_id}"
                else:
                    label = f"Scanning... T{tr.track_id}"
                color = (0, 165, 255)
            draw_bbox(annotated, tr.bbox, label, color)

        # FPS
        self._fps_counter += 1
        elapsed = time.time() - self._fps_t0
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_t0 = time.time()

        return annotated, track_results, self._current_fps, events

    def get_enrollment_embedding(self, frame: np.ndarray):
        """Capture a face embedding from the current frame for enrollment."""
        detections = self.detector.detect(frame)
        if not detections:
            return None, "No face detected"
        det = detections[0]
        ok, reason = passes_quality(frame, det["bbox"])
        if not ok:
            return None, f"Quality check failed: {reason}"
        if self.arcface is None:
            return None, "ArcFace model not loaded (onnxruntime may not be installed)"
        if self.matcher is None:
            return None, "FAISS matcher not available (faiss-cpu may not be installed)"
        crop = crop_face(frame, det["bbox"])
        if crop is None:
            return None, "Failed to crop face"
        aligned = align_face_simple(crop)
        emb = self.arcface.get_embedding(aligned)
        return emb, "ok"

    def reset(self):
        self.tracker.reset()
        self.voter = TrackVoter()
        self.cooldown.reset()
        self._frame_idx = 0
        self._track_frames.clear()
        self._track_identity.clear()
        self._track_last_recog_frame.clear()
        self._last_recognized_guest_id = None


# ═══════════════════════════════════════════════════════════════════════
#  QThread Workers
# ═══════════════════════════════════════════════════════════════════════

class CameraWorker(QThread):
    """Captures frames from camera, runs face pipeline, emits results."""
    frame_ready = pyqtSignal(np.ndarray, list, float, list)  # annotated, tracks, fps, events
    raw_frame_ready = pyqtSignal(np.ndarray)  # raw frame for enrollment snapshots
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.pipeline: Optional[FacePipeline] = None
        self._last_raw_frame: Optional[np.ndarray] = None

    def run(self):
        self._running = True
        try:
            self.pipeline = FacePipeline()
        except Exception as e:
            self.error.emit(f"Pipeline init failed: {e}")
            return

        cap = cv2.VideoCapture(leo_config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, leo_config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, leo_config.FRAME_HEIGHT)

        if not cap.isOpened():
            self.error.emit("Cannot open camera")
            return

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue
            self._last_raw_frame = frame.copy()
            try:
                annotated, tracks, fps, events = self.pipeline.process_frame(frame)
                self.frame_ready.emit(annotated, tracks, fps, events)
            except Exception as e:
                self.error.emit(f"Frame error: {e}")

        cap.release()

    def stop(self):
        self._running = False
        self.wait(3000)

    def get_raw_frame(self):
        return self._last_raw_frame


class VoiceWorker(QThread):
    """Listens to microphone and emits recognized speech."""
    heard = pyqtSignal(str)
    status = pyqtSignal(str)  # "listening", "processing", "idle", "error"
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = False

    def run(self):
        if not HAS_SPEECH:
            self.error.emit("SpeechRecognition not installed")
            return
        self._running = True
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True

        try:
            mic = sr.Microphone()
        except Exception as e:
            self.error.emit(f"Microphone error: {e}")
            return

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)

        while self._running:
            self.status.emit("listening")
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=6, phrase_time_limit=8)
                self.status.emit("processing")
                text = recognizer.recognize_google(audio).strip()
                if text:
                    self.heard.emit(text)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                self.error.emit(f"Speech API error: {e}")
            except Exception:
                pass

        self.status.emit("idle")

    def stop(self):
        self._running = False
        self.wait(3000)


class TTSWorker(QThread):
    """Speaks text using pyttsx3 in a background thread."""
    finished_speaking = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._queue: list[str] = []
        self._running = False

    def enqueue(self, text: str):
        self._queue.append(text)

    def run(self):
        self._running = True
        while self._running:
            if self._queue:
                text = self._queue.pop(0)
                try:
                    if HAS_TTS:
                        engine = pyttsx3.init()
                        engine.setProperty("rate", 165)
                        engine.setProperty("volume", 1.0)
                        engine.say(text)
                        engine.runAndWait()
                        engine.stop()
                        del engine
                    self.finished_speaking.emit(text)
                except Exception:
                    pass
            else:
                self.msleep(100)

    def stop(self):
        self._running = False
        self._queue.clear()
        self.wait(3000)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════

class LeoMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Leo  ·  Face Recognition & Voice Assistant")
        self.setMinimumSize(1280, 720)
        self.resize(1440, 820)
        self.setStyleSheet(STYLESHEET)

        # Workers
        self.camera_worker: Optional[CameraWorker] = None
        self.voice_worker: Optional[VoiceWorker] = None
        self.tts_worker = TTSWorker()
        self.tts_worker.start()

        # NLU
        _ensure_nltk_data()
        self.nlu = LeoNLU("leo") if HAS_NLTK else None

        # State
        self._camera_running = False
        self._voice_running = False
        self._current_fps = 0.0
        self._active_tracks = 0
        self._recognized_count = 0
        self._total_events = 0

        self._build_ui()
        self._connect_signals()

        # Auto-start camera
        QTimer.singleShot(500, self._start_camera)

    # ────────────────── UI Construction ──────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ── Left: Camera Feed ──
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Title bar
        title_bar = QHBoxLayout()
        logo_label = QLabel("◉  LEO")
        logo_label.setStyleSheet(f"""
            font-size: 22px; font-weight: 800;
            color: {COLORS['accent']};
            letter-spacing: 3px;
        """)
        title_bar.addWidget(logo_label)
        title_bar.addStretch()

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet(f"""
            color: {COLORS['warning']};
            font-family: 'Cascadia Code', 'Consolas', monospace;
            font-size: 13px; font-weight: 600;
            background: {COLORS['bg_card']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px; padding: 4px 12px;
        """)
        title_bar.addWidget(self.fps_label)
        left_layout.addLayout(title_bar)

        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.camera_label.setStyleSheet(f"""
            background: {COLORS['bg_secondary']};
            border: 2px solid {COLORS['border']};
            border-radius: 12px;
        """)
        self.camera_label.setText("📷  Camera feed will appear here...")
        self.camera_label.setFont(QFont("Segoe UI", 14))
        left_layout.addWidget(self.camera_label, stretch=1)

        # Bottom status bar
        status_bar = QHBoxLayout()
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 18px;")
        status_bar.addWidget(self.status_indicator)
        self.status_text = QLabel("Idle")
        self.status_text.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        status_bar.addWidget(self.status_text)
        status_bar.addStretch()

        self.btn_camera = QPushButton("▶  Start Camera")
        self.btn_camera.setObjectName("btnPrimary")
        self.btn_camera.setFixedWidth(160)
        status_bar.addWidget(self.btn_camera)

        left_layout.addLayout(status_bar)

        # ── Right: Control Panels ──
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        right_scroll_widget = QWidget()
        right_scroll_layout = QVBoxLayout(right_scroll_widget)
        right_scroll_layout.setContentsMargins(4, 0, 4, 0)
        right_scroll_layout.setSpacing(8)

        # ── 1. Stats Dashboard ──
        stats_group = QGroupBox("  DASHBOARD")
        stats_layout = QHBoxLayout(stats_group)

        self.stat_fps = self._make_stat_card("FPS", "--", COLORS['accent'])
        self.stat_tracks = self._make_stat_card("TRACKS", "0", COLORS['warning'])
        self.stat_known = self._make_stat_card("KNOWN", "0", COLORS['success'])
        self.stat_events = self._make_stat_card("EVENTS", "0", COLORS['text_secondary'])

        stats_layout.addWidget(self.stat_fps)
        stats_layout.addWidget(self.stat_tracks)
        stats_layout.addWidget(self.stat_known)
        stats_layout.addWidget(self.stat_events)
        right_scroll_layout.addWidget(stats_group)

        # ── 2. Guest Enrollment Panel (PROMINENT) ──
        enroll_group = QGroupBox("  📸  GUEST ENROLLMENT")
        enroll_group.setStyleSheet(enroll_group.styleSheet() + f"""
            QGroupBox {{
                border: 2px solid {COLORS['success']};
            }}
            QGroupBox::title {{
                color: {COLORS['success']};
                border: 1px solid {COLORS['success']};
                font-size: 13px;
                font-weight: 700;
            }}
        """)
        enroll_layout = QVBoxLayout(enroll_group)
        enroll_layout.setSpacing(10)

        # Helper text
        enroll_hint = QLabel("Face the camera, fill in guest details, then click Enroll.")
        enroll_hint.setWordWrap(True)
        enroll_hint.setStyleSheet(f"""
            color: {COLORS['text_secondary']}; font-size: 11px;
            padding: 6px 8px;
            background: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
        """)
        enroll_layout.addWidget(enroll_hint)

        form = QFormLayout()
        form.setSpacing(8)
        self.enroll_name = QLineEdit()
        self.enroll_name.setPlaceholderText("Guest name (required)")
        self.enroll_name.setMinimumHeight(34)
        form.addRow("Name:", self.enroll_name)

        self.enroll_designation = QLineEdit()
        self.enroll_designation.setPlaceholderText("e.g. Professor, CEO, Director")
        self.enroll_designation.setMinimumHeight(34)
        form.addRow("Title:", self.enroll_designation)

        self.enroll_achievements = QLineEdit()
        self.enroll_achievements.setPlaceholderText("Achievements (semicolon separated)")
        self.enroll_achievements.setMinimumHeight(34)
        form.addRow("Info:", self.enroll_achievements)
        enroll_layout.addLayout(form)

        self.btn_enroll = QPushButton("📸   CAPTURE  &  ENROLL  GUEST")
        self.btn_enroll.setObjectName("btnSuccess")
        self.btn_enroll.setMinimumHeight(48)
        self.btn_enroll.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['success']}, stop:1 #00c853);
                color: #0a0e1a;
                border: none;
                border-radius: 10px;
                font-weight: 800;
                font-size: 14px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #33ff99, stop:1 #00e676);
            }}
            QPushButton:pressed {{
                background: #00b248;
            }}
        """)
        enroll_layout.addWidget(self.btn_enroll)

        self.enroll_status = QLabel("")
        self.enroll_status.setWordWrap(True)
        self.enroll_status.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; padding: 4px;")
        enroll_layout.addWidget(self.enroll_status)
        right_scroll_layout.addWidget(enroll_group)

        # ── 3. Welcome Speech Panel (NEW, PROMINENT) ──
        speech_group = QGroupBox("  🎙️  WELCOME SPEECH")
        speech_group.setStyleSheet(speech_group.styleSheet() + f"""
            QGroupBox {{
                border: 2px solid {COLORS['accent']};
            }}
            QGroupBox::title {{
                color: {COLORS['accent']};
                border: 1px solid {COLORS['accent']};
                font-size: 13px;
                font-weight: 700;
            }}
        """)
        speech_layout = QVBoxLayout(speech_group)
        speech_layout.setSpacing(10)

        speech_hint = QLabel("Select a registered guest, then generate and speak their formal welcome introduction.")
        speech_hint.setWordWrap(True)
        speech_hint.setStyleSheet(f"""
            color: {COLORS['text_secondary']}; font-size: 11px;
            padding: 6px 8px;
            background: {COLORS['bg_secondary']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
        """)
        speech_layout.addWidget(speech_hint)

        # Guest selector dropdown
        guest_select_row = QHBoxLayout()
        guest_select_label = QLabel("Guest:")
        guest_select_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 600; font-size: 12px;")
        guest_select_row.addWidget(guest_select_label)

        self.speech_guest_combo = QComboBox()
        self.speech_guest_combo.setMinimumHeight(34)
        self.speech_guest_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
            }}
            QComboBox:focus {{
                border-color: {COLORS['accent']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {COLORS['accent']};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                selection-background-color: {COLORS['accent_dim']};
            }}
        """)
        self.speech_guest_combo.addItem("-- Select a guest --", None)
        guest_select_row.addWidget(self.speech_guest_combo, stretch=1)
        speech_layout.addLayout(guest_select_row)

        self.btn_generate_speech = QPushButton("🎙️   GENERATE  &  SPEAK  WELCOME")
        self.btn_generate_speech.setMinimumHeight(48)
        self.btn_generate_speech.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['accent']}, stop:1 #007cf0);
                color: #ffffff;
                border: none;
                border-radius: 10px;
                font-weight: 800;
                font-size: 14px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #33ddff, stop:1 #3399ff);
            }}
            QPushButton:pressed {{
                background: {COLORS['accent_dim']};
            }}
        """)
        speech_layout.addWidget(self.btn_generate_speech)

        self.speech_preview = QTextEdit()
        self.speech_preview.setReadOnly(True)
        self.speech_preview.setFixedHeight(120)
        self.speech_preview.setPlaceholderText("Welcome speech will appear here after generation...")
        self.speech_preview.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['accent_dim']};
                border-radius: 8px;
                padding: 8px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
                line-height: 1.4;
            }}
        """)
        speech_layout.addWidget(self.speech_preview)

        self.speech_status = QLabel("")
        self.speech_status.setWordWrap(True)
        self.speech_status.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px; padding: 2px;")
        speech_layout.addWidget(self.speech_status)
        right_scroll_layout.addWidget(speech_group)

        # ── 4. Registered Guests ──
        guests_group = QGroupBox("  REGISTERED GUESTS")
        guests_layout = QVBoxLayout(guests_group)
        self.guests_list_widget = QVBoxLayout()
        guests_layout.addLayout(self.guests_list_widget)

        btn_row = QHBoxLayout()
        self.btn_refresh_guests = QPushButton("↻  Refresh")
        self.btn_refresh_guests.setFixedWidth(100)
        btn_row.addWidget(self.btn_refresh_guests)
        btn_row.addStretch()
        guests_layout.addLayout(btn_row)
        right_scroll_layout.addWidget(guests_group)

        # ── 5. Voice Assistant Panel ──
        voice_group = QGroupBox("  VOICE ASSISTANT")
        voice_layout = QVBoxLayout(voice_group)

        voice_status_row = QHBoxLayout()
        self.voice_indicator = QLabel("●")
        self.voice_indicator.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 16px;")
        voice_status_row.addWidget(self.voice_indicator)
        self.voice_status_label = QLabel("Voice: Inactive")
        self.voice_status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        voice_status_row.addWidget(self.voice_status_label)
        voice_status_row.addStretch()
        self.btn_voice = QPushButton("🎤  Start Voice")
        self.btn_voice.setObjectName("btnPrimary")
        self.btn_voice.setFixedWidth(140)
        voice_status_row.addWidget(self.btn_voice)
        voice_layout.addLayout(voice_status_row)

        self.voice_heard_label = QLabel("Last heard: —")
        self.voice_heard_label.setWordWrap(True)
        self.voice_heard_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']}; font-size: 12px;
            background: {COLORS['bg_input']}; border: 1px solid {COLORS['border']};
            border-radius: 6px; padding: 8px;
        """)
        voice_layout.addWidget(self.voice_heard_label)

        self.voice_response_label = QLabel("Response: —")
        self.voice_response_label.setWordWrap(True)
        self.voice_response_label.setStyleSheet(f"""
            color: {COLORS['accent']}; font-size: 12px;
            background: {COLORS['bg_input']}; border: 1px solid {COLORS['accent_dim']};
            border-radius: 6px; padding: 8px;
        """)
        voice_layout.addWidget(self.voice_response_label)
        right_scroll_layout.addWidget(voice_group)

        # ── 6. Event Log ──
        log_group = QGroupBox("  EVENT LOG")
        log_layout = QVBoxLayout(log_group)
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setFixedHeight(120)
        log_layout.addWidget(self.event_log)

        log_btn_row = QHBoxLayout()
        self.btn_clear_log = QPushButton("Clear")
        self.btn_clear_log.setFixedWidth(80)
        log_btn_row.addWidget(self.btn_clear_log)
        log_btn_row.addStretch()
        log_layout.addLayout(log_btn_row)
        right_scroll_layout.addWidget(log_group)

        # ── 7. Controls ──
        controls_group = QGroupBox("  CONTROLS")
        controls_layout = QHBoxLayout(controls_group)
        self.btn_reset_pipeline = QPushButton("🔄  Reset Pipeline")
        self.btn_reset_pipeline.setFixedWidth(140)
        controls_layout.addWidget(self.btn_reset_pipeline)
        self.btn_rebuild_index = QPushButton("⚡  Rebuild Index")
        self.btn_rebuild_index.setFixedWidth(140)
        controls_layout.addWidget(self.btn_rebuild_index)
        controls_layout.addStretch()
        right_scroll_layout.addWidget(controls_group)

        right_scroll_layout.addStretch()
        right_scroll.setWidget(right_scroll_widget)
        right_layout.addWidget(right_scroll)

        # Add to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([800, 420])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # Load guests
        QTimer.singleShot(1000, self._refresh_guest_list)

    def _make_stat_card(self, title: str, value: str, color: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
                padding: 8px;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        val_label = QLabel(value)
        val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val_label.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: 800; border: none;")
        val_label.setObjectName(f"stat_val_{title}")
        layout.addWidget(val_label)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 9px; font-weight: 600; letter-spacing: 2px; border: none;")
        layout.addWidget(title_label)

        card._val_label = val_label
        return card

    def _update_stat(self, card: QFrame, value: str):
        card._val_label.setText(value)

    # ────────────────── Signal Connections ───────────────────

    def _connect_signals(self):
        self.btn_camera.clicked.connect(self._toggle_camera)
        self.btn_voice.clicked.connect(self._toggle_voice)
        self.btn_enroll.clicked.connect(self._enroll_guest)
        self.btn_generate_speech.clicked.connect(self._generate_welcome_speech)
        self.btn_refresh_guests.clicked.connect(self._refresh_guest_list)
        self.btn_clear_log.clicked.connect(lambda: self.event_log.clear())
        self.btn_reset_pipeline.clicked.connect(self._reset_pipeline)
        self.btn_rebuild_index.clicked.connect(self._rebuild_index)

    # ────────────────── Camera Control ──────────────────────

    def _toggle_camera(self):
        if self._camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        if self._camera_running:
            return
        self.camera_worker = CameraWorker()
        self.camera_worker.frame_ready.connect(self._on_frame)
        self.camera_worker.error.connect(self._on_camera_error)
        self.camera_worker.start()
        self._camera_running = True
        self.btn_camera.setText("■  Stop Camera")
        self.btn_camera.setObjectName("btnDanger")
        self.btn_camera.setStyle(self.btn_camera.style())  # force stylesheet refresh
        self.status_indicator.setStyleSheet(f"color: {COLORS['success']}; font-size: 18px;")
        self.status_text.setText("Camera running")
        self._log("Camera started")

    def _stop_camera(self):
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker = None
        self._camera_running = False
        self.btn_camera.setText("▶  Start Camera")
        self.btn_camera.setObjectName("btnPrimary")
        self.btn_camera.setStyle(self.btn_camera.style())
        self.status_indicator.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 18px;")
        self.status_text.setText("Idle")
        self.camera_label.setText("📷  Camera feed will appear here...")
        self._log("Camera stopped")

    @pyqtSlot(np.ndarray, list, float, list)
    def _on_frame(self, annotated, tracks, fps, events):
        # Update camera display
        h, w, ch = annotated.shape
        # Draw FPS on frame
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale to label size maintaining aspect ratio
        lw = self.camera_label.width()
        lh = self.camera_label.height()
        pixmap = QPixmap.fromImage(qt_image).scaled(
            lw, lh, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

        # Update stats
        self._current_fps = fps
        self._active_tracks = len(tracks)
        recognized = [t for t in tracks if t.confirmed and t.name]
        self._recognized_count = len(recognized)

        self._update_stat(self.stat_fps, f"{fps:.1f}")
        self._update_stat(self.stat_tracks, str(self._active_tracks))
        self._update_stat(self.stat_known, str(self._recognized_count))
        self.fps_label.setText(f"FPS: {fps:.1f}")

        # Process events
        for evt in events:
            self._log(evt)
            self._total_events += 1
            self._update_stat(self.stat_events, str(self._total_events))

    @pyqtSlot(str)
    def _on_camera_error(self, msg):
        self._log(f"⚠️ {msg}")

    # ────────────────── Voice Control ───────────────────────

    def _toggle_voice(self):
        if self._voice_running:
            self._stop_voice()
        else:
            self._start_voice()

    def _start_voice(self):
        if self._voice_running:
            return
        self.voice_worker = VoiceWorker()
        self.voice_worker.heard.connect(self._on_voice_heard)
        self.voice_worker.status.connect(self._on_voice_status)
        self.voice_worker.error.connect(self._on_voice_error)
        self.voice_worker.start()
        self._voice_running = True
        self.btn_voice.setText("🔇  Stop Voice")
        self.btn_voice.setObjectName("btnDanger")
        self.btn_voice.setStyle(self.btn_voice.style())
        self.voice_indicator.setStyleSheet(f"color: {COLORS['success']}; font-size: 16px;")
        self.voice_status_label.setText("Voice: Active")
        self._log("Voice assistant started")

    def _stop_voice(self):
        if self.voice_worker:
            self.voice_worker.stop()
            self.voice_worker = None
        self._voice_running = False
        self.btn_voice.setText("🎤  Start Voice")
        self.btn_voice.setObjectName("btnPrimary")
        self.btn_voice.setStyle(self.btn_voice.style())
        self.voice_indicator.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 16px;")
        self.voice_status_label.setText("Voice: Inactive")
        self._log("Voice assistant stopped")

    @pyqtSlot(str)
    def _on_voice_heard(self, text):
        self.voice_heard_label.setText(f"Last heard: \"{text}\"")
        self._log(f"🎤 Heard: {text}")

        # Check exit
        t = text.lower()
        if any(w in t for w in ["stop listening", "stop voice"]):
            self._stop_voice()
            return

        # Check intro intent
        if is_intro_intent(text):
            response = intro_response(text)
            self.voice_response_label.setText(f"Response: {response}")
            self.tts_worker.enqueue(response)
            self._log(f"🤖 {response}")
            return

        # Check play intent
        if self.nlu:
            play_q = self.nlu.parse_play_query(text)
            if play_q == "__STOP__":
                self.voice_response_label.setText("Response: Goodbye!")
                self.tts_worker.enqueue("Goodbye!")
                self._stop_voice()
                return
            if play_q is not None and play_q:
                response = f"Playing {play_q}"
                self.voice_response_label.setText(f"Response: {response}")
                self.tts_worker.enqueue(response)
                self._log(f"🎵 {response}")
                if HAS_PYWHATKIT:
                    try:
                        pywhatkit.playonyt(play_q)
                    except Exception as e:
                        self._log(f"⚠️ YouTube error: {e}")
                return

        self.voice_response_label.setText("Response: (no matching command)")

    @pyqtSlot(str)
    def _on_voice_status(self, status):
        status_map = {
            "listening": ("● Listening...", COLORS['success']),
            "processing": ("● Processing...", COLORS['warning']),
            "idle": ("● Idle", COLORS['text_muted']),
        }
        text, color = status_map.get(status, ("● Unknown", COLORS['text_muted']))
        self.voice_indicator.setStyleSheet(f"color: {color}; font-size: 16px;")
        self.voice_status_label.setText(f"Voice: {text}")

    @pyqtSlot(str)
    def _on_voice_error(self, msg):
        self._log(f"⚠️ Voice: {msg}")

    # ────────────────── Guest Enrollment ────────────────────

    def _enroll_guest(self):
        name = self.enroll_name.text().strip()
        if not name:
            self.enroll_status.setText("❌ Please enter a guest name.")
            self.enroll_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px; padding: 4px;")
            return

        if not self._camera_running or not self.camera_worker or not self.camera_worker.pipeline:
            self.enroll_status.setText("❌ Camera must be running to capture a face.")
            self.enroll_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px; padding: 4px;")
            return

        raw_frame = self.camera_worker.get_raw_frame()
        if raw_frame is None:
            self.enroll_status.setText("❌ No frame available yet.")
            self.enroll_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px; padding: 4px;")
            return

        emb, msg = self.camera_worker.pipeline.get_enrollment_embedding(raw_frame)
        if emb is None:
            self.enroll_status.setText(f"❌ {msg}")
            self.enroll_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px; padding: 4px;")
            return

        # Save to DB
        try:
            conn = get_connection()
            init_db(conn)
            designation = self.enroll_designation.text().strip()
            achievements = self.enroll_achievements.text().strip()
            guest_id = add_guest(conn, name, designation, achievements)
            add_embedding(conn, guest_id, emb)
            conn.close()

            # Rebuild FAISS index
            self.camera_worker.pipeline.rebuild_index()

            self.enroll_status.setText(f"✅ Enrolled \"{name}\" successfully!")
            self.enroll_status.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px; padding: 4px;")
            self._log(f"✅ Enrolled guest: {name} (ID: {guest_id})")
            self.enroll_name.clear()
            self.enroll_designation.clear()
            self.enroll_achievements.clear()
            self._refresh_guest_list()
        except Exception as e:
            self.enroll_status.setText(f"❌ DB error: {e}")
            self.enroll_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px; padding: 4px;")

    def _generate_welcome_speech(self):
        guest_id = self.speech_guest_combo.currentData()
        if guest_id is None:
            self.speech_status.setText("❌ Please select a guest first.")
            self.speech_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px; padding: 2px;")
            return
            
        self.speech_status.setText("⏳ Generating speech...")
        self.speech_status.setStyleSheet(f"color: {COLORS['warning']}; font-size: 11px; padding: 2px;")
        QApplication.processEvents()
        
        try:
            conn = get_connection()
            profile = load_guest_profile(conn, guest_id)
            conn.close()
            
            if not profile:
                self.speech_status.setText("❌ Guest profile not found.")
                self.speech_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px; padding: 2px;")
                return
                
            generator = WelcomeGenerator()
            msg = generator.generate(profile)
            
            self.speech_preview.setPlainText(msg.on_screen_text)
            self.tts_worker.enqueue(msg.spoken_text)
            
            self._log(f"🎙️ Generated welcome for {profile.name}")
            self.speech_status.setText("✅ Speech generated and sent to TTS!")
            self.speech_status.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px; padding: 2px;")
            
        except Exception as e:
            self.speech_status.setText(f"❌ Error generating speech: {str(e)}")
            self.speech_status.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px; padding: 2px;")

    def _refresh_guest_list(self):
        # Clear existing
        while self.guests_list_widget.count():
            item = self.guests_list_widget.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.speech_guest_combo.clear()
        self.speech_guest_combo.addItem("-- Select a guest --", None)

        try:
            conn = get_connection()
            init_db(conn)
            guests = list_guests(conn)
            conn.close()
        except Exception:
            guests = []

        if not guests:
            empty_label = QLabel("No registered guests yet.")
            empty_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; padding: 8px;")
            self.guests_list_widget.addWidget(empty_label)
            return

        for g in guests:
            self.speech_guest_combo.addItem(g["name"], g["guest_id"])
            
            row = QFrame()
            row.setStyleSheet(f"""
                QFrame {{
                    background: {COLORS['bg_secondary']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 8px;
                    padding: 4px;
                }}
                QFrame:hover {{
                    border-color: {COLORS['accent']};
                }}
            """)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 6, 10, 6)
            row_layout.setSpacing(8)

            name_label = QLabel(f"👤  {g['name']}")
            name_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 600; font-size: 12px; border: none;")
            row_layout.addWidget(name_label)

            emb_count = g.get("num_embeddings", 0)
            count_label = QLabel(f"{emb_count} img{'s' if emb_count != 1 else ''}")
            count_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px; border: none;")
            row_layout.addWidget(count_label)

            row_layout.addStretch()

            del_btn = QPushButton("✕")
            del_btn.setFixedSize(26, 26)
            del_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; color: {COLORS['text_muted']};
                    border: none; font-size: 14px; border-radius: 13px;
                }}
                QPushButton:hover {{
                    background: {COLORS['danger']}; color: white;
                }}
            """)
            guest_id = g["guest_id"]
            del_btn.clicked.connect(lambda checked, gid=guest_id, gname=g['name']: self._delete_guest(gid, gname))
            row_layout.addWidget(del_btn)

            self.guests_list_widget.addWidget(row)

    def _delete_guest(self, guest_id: int, name: str):
        reply = QMessageBox.question(
            self, "Delete Guest",
            f"Delete \"{name}\" and all their face data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            conn = get_connection()
            delete_guest(conn, guest_id)
            conn.close()
            if self.camera_worker and self.camera_worker.pipeline:
                self.camera_worker.pipeline.rebuild_index()
            self._log(f"🗑️ Deleted guest: {name}")
            self._refresh_guest_list()
        except Exception as e:
            self._log(f"⚠️ Delete error: {e}")

    # ────────────────── Pipeline Controls ───────────────────

    def _reset_pipeline(self):
        if self.camera_worker and self.camera_worker.pipeline:
            self.camera_worker.pipeline.reset()
            self._log("🔄 Pipeline reset")

    def _rebuild_index(self):
        if self.camera_worker and self.camera_worker.pipeline:
            self.camera_worker.pipeline.rebuild_index()
            self._log("⚡ FAISS index rebuilt")

    # ────────────────── Logging ─────────────────────────────

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.event_log.append(f"<span style='color:{COLORS['text_muted']}'>[{ts}]</span> {msg}")
        # Auto-scroll
        scrollbar = self.event_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ────────────────── Cleanup ─────────────────────────────

    def closeEvent(self, event):
        if self.camera_worker:
            self.camera_worker.stop()
        if self.voice_worker:
            self.voice_worker.stop()
        self.tts_worker.stop()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set app-wide dark palette for Fusion
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS["bg_primary"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLORS["bg_secondary"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS["bg_card"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS["bg_card"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    window = LeoMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
