"""
Leo Face Recognition System – Central Configuration
All tunable knobs live here.
"""

import os

# ───────────────────────── Paths ──────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ARCFACE_MODEL_PATH = os.path.join(MODELS_DIR, "w600k_r50.onnx")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "leo_guests.db")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "faiss_index.bin")
FAISS_MAP_PATH = os.path.join(PROJECT_ROOT, "data", "faiss_map.pkl")

# ───────────────────────── Camera ─────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# ───────────────────── Detection (MediaPipe) ──────────────
DETECTION_CONFIDENCE = 0.45
DETECT_EVERY_N_FRAMES = 2
MAX_NUM_FACES = 10
MODEL_SELECTION = 0

# ───────────────────── Tracking (DeepSORT) ────────────────
MAX_AGE = 40
N_INIT = 2
MAX_COSINE_DISTANCE = 0.45
NN_BUDGET = 100

# ─────────────── Recognition (ArcFace / FAISS) ────────────
ARCFACE_INPUT_SIZE = (112, 112)
EMBEDDING_DIM = 512
RECOGNITION_THRESHOLD = 0.38
TOP_K = 3
RECOGNIZE_EVERY_N_FRAMES = 2
IDENTITY_HOLD_FRAMES = 45

# ──────────────────── Voting / Cooldown ───────────────────
STABLE_FRAMES_BEFORE_RECOG = 3
VOTE_WINDOW = 7
VOTE_REQUIRED = 3
GREET_COOLDOWN_SECONDS = 600          # 10 min – re-greet after this long
GUEST_ABSENCE_THRESHOLD_SEC = 5       # guest must be gone this long to count as "left"

# ──────────────────── Quality Filters ─────────────────────
MIN_FACE_SIZE = 35
BLUR_THRESHOLD = 30.0
DARK_THRESHOLD = 25

# ──────────────────── Enrollment ──────────────────────────
MIN_ENROLL_IMAGES = 1
MAX_ENROLL_IMAGES = 10

# ──────────────────── Streamlit ───────────────────────────
STREAMLIT_PAGE_TITLE = "Leo – Face Recognition Dashboard"
STREAMLIT_PAGE_ICON = "🤖"
LOG_MAX_LINES = 200

# ──────────────────── Greeting ────────────────────────────
ENABLE_GREETING = True
ENABLE_VOICE_GREETING = True

GREETING_TEMPLATE = "Namaste {name} garu, it is a pleasure to meet you. I am Leo."

TTS_RATE = 165
TTS_VOLUME = 1.0
TTS_DEDUPE_WINDOW_SECONDS = 3.0