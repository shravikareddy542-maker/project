"""
app.py – Streamlit Web UI for Leo Face Recognition Dashboard.

Always-on live feed version:
- camera stays live after Start
- uses st.fragment for safe timed refresh
- no bottom st.rerun loop
- no fallback flow
- sidebar summary from session state
"""

import os
import sys
import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from main_pipeline import LeoPipeline
from recognition.arcface_onnx import ArcFaceONNX
from db.guest_db import (
    get_connection,
    init_db,
    add_guest,
    add_embedding,
    list_guests,
    delete_guest,
)
from utils.image_ops import crop_face, align_face_simple
from utils.welcome_generator import WelcomeGenerator, load_guest_profile

st.set_page_config(
    page_title=config.STREAMLIT_PAGE_TITLE,
    page_icon=config.STREAMLIT_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d1321 40%, #1a1a2e 100%); }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1729 0%, #131b2e 50%, #0f1729 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }

    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.55rem 1.2rem;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(99, 102, 241, 0.12);
        margin-bottom: 0.6rem;
    }

    .metric-card .label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 4px;
    }

    .metric-card .value {
        color: #e2e8f0;
        font-size: 1.6rem;
        font-weight: 800;
    }

    .metric-card .value.accent { color: #818cf8; }
    .metric-card .value.green { color: #34d399; }
    .metric-card .value.amber { color: #fbbf24; }

    .event-log {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 12px;
        padding: 0.9rem;
        max-height: 280px;
        overflow-y: auto;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.73rem;
        color: #94a3b8;
        line-height: 1.7;
    }

    .log-greet { color: #34d399; }

    .guest-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #c7d2fe;
        padding: 6px 14px;
        border-radius: 24px;
        margin: 3px 4px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .guest-chip .dot {
        width: 7px;
        height: 7px;
        background: #34d399;
        border-radius: 50%;
    }

    .page-title {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .page-subtitle {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    .status-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 10px;
        border: 1px solid rgba(99, 102, 241, 0.1);
        margin-bottom: 1rem;
    }

    .status-dot { width: 8px; height: 8px; border-radius: 50%; }
    .status-dot.live { background: #34d399; }
    .status-dot.off { background: #475569; }
    .status-text { font-size: 0.8rem; color: #94a3b8; }

    .guest-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
    }

    .guest-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 4px;
    }

    .guest-meta {
        font-size: 0.78rem;
        color: #64748b;
        margin-bottom: 2px;
    }

    .guest-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def _init_state():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "live_detected_names" not in st.session_state:
        st.session_state.live_detected_names = []
    if "live_fps" not in st.session_state:
        st.session_state.live_fps = 0.0
    if "camera_error" not in st.session_state:
        st.session_state.camera_error = ""
    if "frame_fail_count" not in st.session_state:
        st.session_state.frame_fail_count = 0
    if "last_frame_rgb" not in st.session_state:
        st.session_state.last_frame_rgb = None
    # ── Guest Introduction panel state ──
    if "intro_profile" not in st.session_state:
        st.session_state.intro_profile = None
    if "intro_message" not in st.session_state:
        st.session_state.intro_message = None
    if "intro_status" not in st.session_state:
        st.session_state.intro_status = ""
    if "intro_spoken" not in st.session_state:
        st.session_state.intro_spoken = False


_init_state()


def get_pipeline() -> LeoPipeline:
    if st.session_state.pipeline is None:
        st.session_state.pipeline = LeoPipeline()
    return st.session_state.pipeline


def open_camera():
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def start_camera():
    cap = st.session_state.cap
    if cap is not None and cap.isOpened():
        st.session_state.camera_on = True
        st.session_state.camera_error = ""
        return

    cap = open_camera()
    if cap is not None and cap.isOpened():
        st.session_state.cap = cap
        st.session_state.camera_on = True
        st.session_state.camera_error = ""
        st.session_state.frame_fail_count = 0
    else:
        st.session_state.cap = None
        st.session_state.camera_on = False
        st.session_state.camera_error = "❌ Cannot open camera. Check CAMERA_INDEX in config.py"


def stop_camera():
    cap = st.session_state.cap
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass

    st.session_state.cap = None
    st.session_state.camera_on = False
    st.session_state.live_detected_names = []
    st.session_state.live_fps = 0.0
    st.session_state.camera_error = ""
    st.session_state.frame_fail_count = 0
    st.session_state.last_frame_rgb = None


def process_one_frame():
    if not st.session_state.camera_on:
        return st.session_state.get("last_frame_rgb", None)

    cap = st.session_state.cap
    if cap is None:
        st.session_state.camera_error = "⚠️ Camera object is missing."
        st.session_state.camera_on = False
        return st.session_state.get("last_frame_rgb", None)

    if not cap.isOpened():
        st.session_state.camera_error = "⚠️ Camera is not opened."
        st.session_state.camera_on = False
        return st.session_state.get("last_frame_rgb", None)

    ret, frame = cap.read()
    if not ret or frame is None:
        st.session_state.frame_fail_count += 1
        st.session_state.camera_error = "⚠️ Camera frame read failed."

        if st.session_state.frame_fail_count >= 20:
            stop_camera()
            st.session_state.camera_error = "⚠️ Camera stopped after repeated frame read failures."

        return st.session_state.get("last_frame_rgb", None)

    st.session_state.frame_fail_count = 0
    st.session_state.camera_error = ""

    pipeline = get_pipeline()
    result = pipeline.process_frame(frame)

    detected_names = set()
    for tr in result.tracks:
        if tr.confirmed and tr.name:
            detected_names.add(tr.name)

    st.session_state.live_detected_names = sorted(detected_names)
    st.session_state.live_fps = float(result.fps)

    rgb = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
    st.session_state.last_frame_rgb = rgb
    return rgb


with st.sidebar:
    st.markdown("# 🤖 LEO")
    st.caption("Real-Time Face Recognition")
    st.markdown("---")

    st.markdown("### ⚙️ Camera")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start", use_container_width=True, key="btn_start"):
            start_camera()
    with col2:
        if st.button("⏹ Stop", use_container_width=True, key="btn_stop"):
            stop_camera()

    @st.fragment(run_every="500ms")
    def sidebar_live_status():
        if st.session_state.camera_on:
            st.markdown(
                '<div class="status-bar"><div class="status-dot live"></div>'
                '<span class="status-text">Camera Active</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-bar"><div class="status-dot off"></div>'
                '<span class="status-text">Camera Off</span></div>',
                unsafe_allow_html=True,
            )

        if st.session_state.camera_error:
            st.error(st.session_state.camera_error)

        st.markdown("---")
        st.markdown("### ⚡ Performance")

        fps_val = st.session_state.live_fps
        fps_color = "green" if fps_val >= 15 else ("amber" if fps_val >= 8 else "accent")
        st.markdown(
            f'<div class="metric-card"><div class="label">FPS</div>'
            f'<div class="value {fps_color}">{fps_val:.1f}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### 👁️ Live Detections")

        if st.session_state.live_detected_names:
            chips = " ".join(
                f'<span class="guest-chip"><span class="dot"></span>{n}</span>'
                for n in st.session_state.live_detected_names
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#475569; font-size:0.82rem; padding:8px 0;">No confirmed guests yet</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 📋 Event Log")

        pipeline_for_log = st.session_state.pipeline
        if pipeline_for_log and pipeline_for_log.event_log:
            lines = []
            for entry in pipeline_for_log.event_log[-15:]:
                if "🎉" in entry:
                    lines.append(f'<div class="log-greet">{entry}</div>')
                else:
                    lines.append(f"<div>{entry}</div>")
            st.markdown(f'<div class="event-log">{"".join(lines)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="event-log"><div>No events yet</div></div>', unsafe_allow_html=True)

    sidebar_live_status()

    st.markdown("---")
    st.markdown("### 🗄️ Index")
    if st.button("🔄 Rebuild FAISS Index", use_container_width=True, key="btn_rebuild"):
        pipeline = get_pipeline()
        pipeline.rebuild_index()
        st.success(f"Index rebuilt • {pipeline.matcher.total} embeddings")

    conn = get_connection()
    init_db(conn)
    _guests = list_guests(conn)
    conn.close()

    total_guests = len(_guests)
    total_embs = sum(g.get("num_embeddings", 0) for g in _guests)

    st.markdown("---")
    st.markdown("### 📊 Stats")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f'<div class="metric-card"><div class="label">Guests</div>'
            f'<div class="value accent">{total_guests}</div></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div class="metric-card"><div class="label">Embeddings</div>'
            f'<div class="value accent">{total_embs}</div></div>',
            unsafe_allow_html=True,
        )


tab_live, tab_enroll, tab_guests, tab_intro = st.tabs([
    "📹 Live Feed",
    "➕ Enroll Guest",
    "📋 Guest List",
    "🎤 Guest Introduction",
])

with tab_live:
    st.markdown('<div class="page-title">Live Face Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Always-on live feed after camera start</div>',
        unsafe_allow_html=True,
    )

    if st.button("⏹ Stop Live Feed", use_container_width=False, key="stop_live_feed"):
        stop_camera()
        st.rerun()

    @st.fragment(run_every="180ms")
    def live_feed_fragment():
        if not st.session_state.camera_on:
            st.markdown(
                """
                <div style="
                    display:flex;
                    flex-direction:column;
                    align-items:center;
                    justify-content:center;
                    height:350px;
                    background: rgba(15, 23, 42, 0.4);
                    border: 2px dashed rgba(99, 102, 241, 0.2);
                    border-radius: 16px;
                    margin-top: 1rem;
                ">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
                    <div style="color: #94a3b8; font-size: 1rem; font-weight: 500;">Camera is offline</div>
                    <div style="color: #475569; font-size: 0.85rem; margin-top: 6px;">
                        Click <b>▶ Start</b> in the sidebar to begin live recognition
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        frame_rgb = process_one_frame()
        if frame_rgb is not None:
            st.image(frame_rgb, channels="RGB", use_container_width=True)
        else:
            st.markdown(
                """
                <div style="
                    display:flex;
                    flex-direction:column;
                    align-items:center;
                    justify-content:center;
                    height:350px;
                    background: rgba(15, 23, 42, 0.4);
                    border: 2px dashed rgba(99, 102, 241, 0.2);
                    border-radius: 16px;
                    margin-top: 1rem;
                ">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
                    <div style="color: #94a3b8; font-size: 1rem; font-weight: 500;">Waiting for camera…</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    live_feed_fragment()


with tab_enroll:
    st.markdown('<div class="page-title">Enroll New Guest</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Upload clear face photos to register a new guest for Leo to recognise</div>',
        unsafe_allow_html=True,
    )

    with st.form("enroll_form", clear_on_submit=True):
        col_left, col_right = st.columns([1, 1])
        with col_left:
            name = st.text_input("Guest Name", placeholder="e.g. Shravika")
            designation = st.text_input("Designation", placeholder="e.g. Student, Professor")
        with col_right:
            achievements = st.text_area(
                "Achievements / Notes",
                placeholder="e.g. Won robotics competition 2025",
                height=120,
            )

        uploaded = st.file_uploader(
            "Upload Face Images (1–10 clear photos)",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("💾 Enroll Guest", use_container_width=True)

    if submitted:
        if not name.strip():
            st.error("❌ Name is required.")
        elif not uploaded:
            st.error("❌ Please upload at least 1 face image.")
        elif len(uploaded) > config.MAX_ENROLL_IMAGES:
            st.error(f"❌ Maximum {config.MAX_ENROLL_IMAGES} images allowed.")
        else:
            try:
                arcface = ArcFaceONNX()
            except FileNotFoundError as e:
                st.error(f"❌ ArcFace model not found. {e}")
                st.stop()

            conn = get_connection()
            init_db(conn)
            guest_id = add_guest(conn, name.strip(), designation.strip(), achievements.strip())

            progress = st.progress(0, text="Processing images…")
            embeddings_added = 0

            from detection.mediapipe_detector import MediaPipeFaceDetector

            for idx, f in enumerate(uploaded):
                img = Image.open(f).convert("RGB")
                img_np = np.array(img)
                bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                det = MediaPipeFaceDetector()
                faces = det.detect(bgr)
                det.close()

                if faces:
                    best = max(faces, key=lambda d: d["confidence"])
                    crop = crop_face(bgr, best["bbox"])
                    if crop is not None:
                        aligned = align_face_simple(crop)
                        emb = arcface.get_embedding(aligned)
                        add_embedding(conn, guest_id, emb)
                        embeddings_added += 1

                progress.progress((idx + 1) / len(uploaded), text=f"Processing {idx + 1} / {len(uploaded)}…")

            conn.close()

            if embeddings_added > 0:
                pipeline = get_pipeline()
                pipeline.rebuild_index()
                st.success(f"✅ {name} enrolled successfully with {embeddings_added} embedding(s).")
            else:
                st.error("❌ No valid face embeddings could be extracted.")


with tab_guests:
    st.markdown('<div class="page-title">Registered Guests</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Manage enrolled guests and their face embeddings</div>',
        unsafe_allow_html=True,
    )

    conn = get_connection()
    init_db(conn)
    guests = list_guests(conn)
    conn.close()

    if not guests:
        st.info("No guests enrolled yet.")
    else:
        for g in guests:
            emb_count = g.get("num_embeddings", 0)
            st.markdown(
                f"""
                <div class="guest-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div class="guest-name">👤 {g['name']}</div>
                            <div class="guest-meta">
                                {g['designation'] or 'No designation'} &nbsp;•&nbsp;
                                <span class="guest-badge">{emb_count} embedding{'s' if emb_count != 1 else ''}</span>
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div class="guest-meta">ID #{g['guest_id']}</div>
                            <div class="guest-meta">{g['created_at'][:10] if g['created_at'] else '—'}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"🗑️ Delete {g['name']}", key=f"del_{g['guest_id']}"):
                conn = get_connection()
                delete_guest(conn, g["guest_id"])
                conn.close()
                pipeline = get_pipeline()
                pipeline.rebuild_index()
                st.success(f"✅ Deleted {g['name']} and rebuilt index.")
                st.rerun()


with tab_intro:
    st.markdown('<div class="page-title">Guest Introduction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">On-demand introduction for the currently recognized guest</div>',
        unsafe_allow_html=True,
    )

    # ── action buttons ──────────────────────────────────────
    has_guest = st.session_state.intro_profile is not None
    has_intro = st.session_state.intro_message is not None

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        tell_clicked = st.button(
            "📋 Tell About Guest",
            use_container_width=True,
            key="btn_tell",
        )
    with col_b2:
        speak_clicked = st.button(
            "🔊 Speak Intro",
            use_container_width=True,
            key="btn_speak",
            disabled=not has_intro or st.session_state.intro_spoken,
        )
    with col_b3:
        clear_clicked = st.button(
            "🗑️ Clear",
            use_container_width=True,
            key="btn_clear",
            disabled=not has_guest and not has_intro,
        )

    # ── button logic ────────────────────────────────────────
    if tell_clicked:
        pipeline = get_pipeline()
        guest_id = pipeline.get_latest_guest_id()
        if guest_id is None:
            st.session_state.intro_profile = None
            st.session_state.intro_message = None
            st.session_state.intro_status = "⚠️ No recognized guest available."
            st.session_state.intro_spoken = False
        else:
            conn = get_connection()
            init_db(conn)
            profile = load_guest_profile(conn, guest_id)
            conn.close()
            if profile is None:
                st.session_state.intro_profile = None
                st.session_state.intro_message = None
                st.session_state.intro_status = "⚠️ Guest not found in database."
                st.session_state.intro_spoken = False
            else:
                gen = WelcomeGenerator()
                msg = gen.generate(profile)
                st.session_state.intro_profile = profile
                st.session_state.intro_message = msg
                st.session_state.intro_status = "✅ Introduction generated."
                st.session_state.intro_spoken = False
        st.rerun()

    if speak_clicked and st.session_state.intro_message:
        pipeline = get_pipeline()
        pipeline.greeter.speak_generated(st.session_state.intro_message.spoken_text)
        st.session_state.intro_status = "🔊 Introduction spoken."
        st.session_state.intro_spoken = True
        st.rerun()

    if clear_clicked:
        st.session_state.intro_profile = None
        st.session_state.intro_message = None
        st.session_state.intro_status = ""
        st.session_state.intro_spoken = False
        st.rerun()

    # ── status label ────────────────────────────────────────
    if st.session_state.intro_status:
        st.markdown(
            f'<div style="padding:8px 16px; border-radius:10px; '
            f'background:rgba(15,23,42,0.5); border:1px solid rgba(99,102,241,0.15); '
            f'color:#94a3b8; font-size:0.88rem; margin-bottom:1rem;">'
            f'{st.session_state.intro_status}</div>',
            unsafe_allow_html=True,
        )

    # ── guest info card ─────────────────────────────────────
    profile = st.session_state.intro_profile
    if profile:
        ach_display = WelcomeGenerator.sanitize(profile.achievements) or "—"
        st.markdown(
            f"""
            <div class="guest-card">
                <div class="guest-name">👤 {WelcomeGenerator.sanitize(profile.name)}</div>
                <div class="guest-meta" style="margin-top:6px;">
                    <strong>Designation:</strong> {WelcomeGenerator.sanitize(profile.designation) or '—'}
                </div>
                <div class="guest-meta" style="margin-top:4px;">
                    <strong>Achievements:</strong> {ach_display}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                display:flex; flex-direction:column; align-items:center; justify-content:center;
                height:160px; background:rgba(15,23,42,0.4);
                border:2px dashed rgba(99,102,241,0.2); border-radius:16px; margin-top:1rem;
            ">
                <div style="font-size:2.5rem; margin-bottom:0.8rem;">👤</div>
                <div style="color:#94a3b8; font-size:0.95rem; font-weight:500;">No guest selected</div>
                <div style="color:#475569; font-size:0.82rem; margin-top:4px;">
                    Click <b>📋 Tell About Guest</b> after Leo recognizes someone
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── introduction preview ────────────────────────────────
    msg = st.session_state.intro_message
    if msg:
        st.markdown("---")
        st.markdown(
            '<div style="color:#64748b; font-size:0.75rem; font-weight:600; '
            'text-transform:uppercase; letter-spacing:1.5px; margin-bottom:6px;">'
            'Introduction Preview</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="guest-card" style="color:#e2e8f0; font-size:0.92rem; line-height:1.6;">'
            f'{msg.on_screen_text}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="color:#64748b; font-size:0.75rem; font-weight:600; '
            'text-transform:uppercase; letter-spacing:1.5px; margin-bottom:6px; margin-top:12px;">'
            'Spoken Text (TTS)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="guest-card" style="color:#a5b4fc; font-size:0.92rem; line-height:1.6; '
            f'font-style:italic;">🔊 "{msg.spoken_text}"</div>',
            unsafe_allow_html=True,
        )