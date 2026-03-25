# 🤖 Leo – Real-Time Multi-Face Recognition System

A complete **offline, CPU-only** face recognition system for your robot Leo.  
Detects, tracks, and recognises multiple guests in real time using your laptop webcam.

## Tech Stack

| Component | Library |
|---|---|
| Webcam + Overlays | OpenCV |
| Face Detection | MediaPipe (multi-face) |
| Face Embeddings | InsightFace ArcFace via ONNX Runtime CPU |
| Multi-Face Tracking | DeepSORT (pure-Python, Kalman + Hungarian) |
| Similarity Search | FAISS (cosine via inner-product) |
| Guest Database | SQLite |
| Web Dashboard | Streamlit |

## Project Structure

```
leo_face/
├── app.py                          # Streamlit frontend
├── main_pipeline.py                # Core CV loop (detection → tracking → recognition)
├── config.py                       # All tunable parameters
├── requirements.txt
├── README.md
├── models/                         # Place ArcFace ONNX model here
│   └── w600k_r50.onnx
├── data/                           # Auto-created: SQLite DB + FAISS index
├── db/
│   └── guest_db.py                 # SQLite schema + CRUD
├── detection/
│   └── mediapipe_detector.py       # MediaPipe face detector
├── tracking/
│   └── deepsort_tracker.py         # DeepSORT multi-face tracker
├── recognition/
│   ├── arcface_onnx.py             # ArcFace embedding generator
│   └── matcher_faiss.py            # FAISS similarity search
├── utils/
│   ├── image_ops.py                # Crop, quality filters, alignment, drawing
│   └── voting.py                   # Frame voting + greeting cooldown
└── scripts/
    ├── rebuild_index.py            # CLI: rebuild FAISS index
    └── cli_enroll.py               # CLI: enroll guest from images
```

## Setup (Windows)

### 1. Create Virtual Environment

```powershell
cd leo_face
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Download ArcFace ONNX Model

Download the **w600k_r50.onnx** model from the InsightFace model zoo:

1. Go to: https://github.com/deepinsight/insightface/tree/master/model_zoo
2. Download the `buffalo_l` or `w600k_r50` ONNX model
3. Place the `.onnx` file in `leo_face/models/w600k_r50.onnx`

> **Alternative direct link:**  
> https://drive.google.com/file/d/1-1Y1EJFx0gYIHKC1y3pCHCrghmNQSqf0/view  
> (Search "insightface w600k_r50.onnx" if the link changes)

### 3. Enroll Guests

**Via Streamlit UI (recommended):**
1. Run the app (see below)
2. Go to "Enroll Guest" tab
3. Upload 3–10 clear face photos, enter name/designation/achievements
4. Click "Enroll"

**Via CLI:**
```powershell
python scripts/cli_enroll.py --name "John Doe" --designation "Professor" --images photo1.jpg photo2.jpg photo3.jpg
```

### 4. Run the Dashboard

```powershell
streamlit run app.py
```

Opens at `http://localhost:8501`.

### 5. Run CLI Mode (no Streamlit)

```powershell
python main_pipeline.py
```

Press `q` to quit the OpenCV window.

## Configuration

All parameters are in `config.py`. Key knobs:

| Parameter | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | 0 | Webcam device index |
| `DETECT_EVERY_N_FRAMES` | 3 | Run MediaPipe detection every N frames |
| `RECOGNITION_THRESHOLD` | 0.45 | Cosine similarity threshold |
| `STABLE_FRAMES_BEFORE_RECOG` | 5 | Frames before recognition starts |
| `VOTE_REQUIRED` | 5 | Consecutive same-name votes to confirm |
| `GREET_COOLDOWN_SECONDS` | 60 | Seconds before re-greeting |
| `MIN_FACE_SIZE` | 40 | Skip faces smaller than this (px) |
| `BLUR_THRESHOLD` | 50.0 | Laplacian variance for blur detection |

## Troubleshooting

### Low FPS
- Increase `DETECT_EVERY_N_FRAMES` to 5 or 10
- Reduce `FRAME_WIDTH` / `FRAME_HEIGHT`
- Reduce `MAX_NUM_FACES`
- Ensure no other apps are using the camera

### Threshold Tuning
- If too many false positives → **increase** `RECOGNITION_THRESHOLD` (e.g., 0.55)
- If known guests not recognised → **decrease** `RECOGNITION_THRESHOLD` (e.g., 0.35)
- Enroll more images per guest (3–10 with varied lighting/angles)

### Camera Errors
- Check `CAMERA_INDEX` — try 0, 1, or 2
- Close other apps using the camera (Zoom, Teams, etc.)
- Run `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"` to test

### Import Errors
- Ensure you are in the `leo_face` directory
- Ensure venv is activated
- Reinstall: `pip install -r requirements.txt`

### ArcFace Model Not Found
- Ensure `models/w600k_r50.onnx` exists
- Check the exact filename matches `config.py → ARCFACE_MODEL_PATH`

## License

Internal project for Robot Leo. Not for redistribution.
