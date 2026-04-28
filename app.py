"""
app.py
------
TrackVision AI — Professional Sports Multi-Object Detection & Tracking Platform
Built with YOLOv8 + SORT Tracker + Streamlit

Run with:
    streamlit run app.py
"""

import streamlit as st
import tempfile
import os
import time

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrackVision AI | Sports Object Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }

    .company-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #e94560;
        box-shadow: 0 4px 20px rgba(233,69,96,0.15);
    }
    .company-name { font-size:2rem; font-weight:800; color:#e94560; letter-spacing:2px; margin:0; }
    .company-tagline { color:#8892b0; font-size:0.9rem; margin:0; letter-spacing:1px; }
    .product-name { color:#ccd6f6; font-size:1.1rem; font-weight:600; margin:0; }

    .metric-card {
        background:#1a1f2e; border:1px solid #2d3561;
        border-radius:10px; padding:1rem; text-align:center;
    }
    .metric-value { font-size:1.8rem; font-weight:700; color:#e94560; }
    .metric-label { color:#8892b0; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; }

    .badge {
        display:inline-block; background:#0f3460; color:#64ffda;
        padding:0.2rem 0.7rem; border-radius:20px; font-size:0.75rem;
        margin:0.2rem; border:1px solid #64ffda33;
    }
    .section-header {
        color:#ccd6f6; font-size:1.1rem; font-weight:600;
        border-left:3px solid #e94560; padding-left:0.7rem; margin-bottom:1rem;
    }
    .info-box {
        background:#1a1f2e; border:1px solid #2d3561; border-radius:8px;
        padding:1rem; margin:0.5rem 0; color:#8892b0; font-size:0.85rem;
    }
    .footer {
        text-align:center; color:#4a5568; font-size:0.78rem;
        margin-top:2rem; padding-top:1rem; border-top:1px solid #1a1f2e;
    }

    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}

    [data-testid="stSidebar"] { background-color:#0d1117; border-right:1px solid #1a1f2e; }

    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c62a47);
        color:white; border:none; border-radius:8px;
        font-weight:600; letter-spacing:0.5px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff6b6b, #e94560);
        box-shadow: 0 4px 15px rgba(233,69,96,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ── Company Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="company-header">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
        <div>
            <p class="company-name">⚡ TRACKVISION AI</p>
            <p class="company-tagline">ENTERPRISE COMPUTER VISION PLATFORM</p>
        </div>
        <div style="text-align:right;">
            <p class="product-name">🎯 Sports Object Detection & Tracking</p>
            <p class="company-tagline">Powered by YOLOv8 + SORT Algorithm</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom:1.5rem;">
    <span class="badge">🔍 YOLOv8 Detection</span>
    <span class="badge">🎯 SORT Tracking</span>
    <span class="badge">🆔 Persistent IDs</span>
    <span class="badge">🎨 Real-time Annotation</span>
    <span class="badge">📊 Frame Analytics</span>
    <span class="badge">⬇️ Video Export</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛠️ Configuration Panel")
    st.divider()

    st.markdown("**🤖 Detection Model**")
    model_choice = st.selectbox(
        "YOLOv8 Variant",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0,
        help="n=Nano (fastest) | s=Small | m=Medium (most accurate)"
    )
    model_info = {
        "yolov8n.pt": "⚡ Nano — Best for real-time, lower accuracy",
        "yolov8s.pt": "⚖️  Small — Balanced speed & accuracy",
        "yolov8m.pt": "🎯 Medium — Best accuracy, slower"
    }
    st.caption(model_info[model_choice])

    st.divider()
    st.markdown("**🎯 Detection Settings**")
    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.90, 0.35, 0.05)
    iou_threshold  = st.slider("NMS IoU Threshold",    0.10, 0.90, 0.45, 0.05)

    st.divider()
    st.markdown("**📡 Tracking Settings**")
    max_age   = st.slider("Max Track Age (frames)", 5,  60, 30, 5)
    min_hits  = st.slider("Min Hits to Confirm",    1,  10,  3, 1)
    track_iou = st.slider("Tracker IoU Threshold",  0.10, 0.70, 0.30, 0.05)

    st.divider()
    st.markdown("**⚙️ Hardware**")
    device = st.selectbox("Inference Device", ["cpu", "cuda", "mps"])

    st.divider()
    st.markdown("""
    <div style="color:#4a5568; font-size:0.75rem; line-height:1.6;">
        <b style="color:#8892b0;">TrackVision AI v1.0</b><br>
        Computer Vision Platform<br>
        YOLOv8 + SORT Algorithm<br>
        © 2026 All rights reserved
    </div>
    """, unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────────────────────────
col_upload, col_output = st.columns(2, gap="large")

with col_upload:
    st.markdown('<p class="section-header">📤 INPUT VIDEO</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your sports video here",
        type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_file:
        st.video(uploaded_file)
        st.markdown(f"""
        <div class="info-box">
            📁 <b style="color:#ccd6f6;">{uploaded_file.name}</b><br>
            💾 Size: {uploaded_file.size/1_000_000:.1f} MB &nbsp;|&nbsp; Type: {uploaded_file.type}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box" style="text-align:center; padding:2rem;">
            <div style="font-size:2.5rem;">🎬</div>
            <div style="color:#ccd6f6; font-weight:600; margin:0.5rem 0;">Upload a Video</div>
            <div>Supports: MP4, AVI, MOV, MKV</div>
            <div style="margin-top:0.5rem; color:#4a5568;">Recommended: Sports footage with multiple moving subjects</div>
        </div>
        """, unsafe_allow_html=True)

with col_output:
    st.markdown('<p class="section-header">📥 TRACKED OUTPUT</p>', unsafe_allow_html=True)
    output_area = st.empty()
    output_area.markdown("""
    <div class="info-box" style="text-align:center; padding:2rem;">
        <div style="font-size:2.5rem;">🎯</div>
        <div style="color:#ccd6f6; font-weight:600; margin:0.5rem 0;">Awaiting Processing</div>
        <div>Annotated video with bounding boxes and persistent IDs will appear here</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Run Button ────────────────────────────────────────────────────────────────
if uploaded_file:
    btn_col, _ = st.columns([1, 2])
    with btn_col:
        run = st.button("🚀  RUN DETECTION & TRACKING", type="primary", use_container_width=True)

    if run:
        try:
            from detector import Detector
            from tracker import Tracker
            from visualizer import Visualizer
            from utils.video_utils import VideoLoader, VideoWriter
        except ImportError as e:
            st.error(f"❌ Import error: {e}\n\nRun `streamlit run app.py` from inside the `sports_tracker` folder.")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_file.getvalue())
            input_path = tmp_in.name
        output_path = input_path.replace(".mp4", "_tracked.mp4")

        st.markdown("---")
        st.markdown('<p class="section-header">⚙️ PROCESSING PIPELINE</p>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        m1, m2, m3, m4 = st.columns(4)
        frames_m = m1.empty()
        fps_m    = m2.empty()
        tracks_m = m3.empty()
        eta_m    = m4.empty()
        status   = st.empty()

        start_time = time.time()
        total_tracks_seen = set()

        try:
            status.info("🤖 Loading YOLOv8 model...")
            detector   = Detector(model=model_choice, conf_threshold=conf_threshold,
                                  iou_threshold=iou_threshold, device=device)
            tracker    = Tracker(max_age=max_age, min_hits=min_hits, iou_threshold=track_iou)
            visualizer = Visualizer(show_confidence=True)

            status.info("🎬 Opening video stream...")
            video_loader = VideoLoader(input_path)
            video_writer = VideoWriter(output_path, fps=video_loader.fps)
            total_frames = len(video_loader)

            status.info(f"⚡ Processing {total_frames} frames...")

            for i, frame in enumerate(video_loader):
                detections      = detector.detect(frame)
                tracked_objects = tracker.update(detections)
                for obj in tracked_objects:
                    total_tracks_seen.add(obj['id'])
                annotated = visualizer.draw(frame, tracked_objects)
                video_writer.write(annotated)

                if i % 5 == 0:
                    pct     = int((i / max(total_frames, 1)) * 100)
                    elapsed = time.time() - start_time
                    fps_proc = (i + 1) / elapsed if elapsed > 0 else 0
                    eta      = (total_frames - i) / fps_proc if fps_proc > 0 else 0
                    progress_bar.progress(pct)

                    frames_m.markdown(f'<div class="metric-card"><div class="metric-value">{i+1}</div><div class="metric-label">Frames Done</div></div>', unsafe_allow_html=True)
                    fps_m.markdown(   f'<div class="metric-card"><div class="metric-value">{fps_proc:.1f}</div><div class="metric-label">Proc FPS</div></div>', unsafe_allow_html=True)
                    tracks_m.markdown(f'<div class="metric-card"><div class="metric-value">{len(total_tracks_seen)}</div><div class="metric-label">Unique IDs</div></div>', unsafe_allow_html=True)
                    eta_m.markdown(   f'<div class="metric-card"><div class="metric-value">{eta:.0f}s</div><div class="metric-label">ETA</div></div>', unsafe_allow_html=True)

            video_writer.release()
            progress_bar.progress(100)
            elapsed = time.time() - start_time
            status.success(f"✅ Processing complete in {elapsed:.1f} seconds!")

            with col_output:
                output_area.empty()
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                col_output.video(video_bytes)

            st.markdown("---")
            dl_col, stat_col = st.columns([1, 2])
            with dl_col:
                st.download_button(
                    label="⬇️ Download Tracked Video",
                    data=video_bytes,
                    file_name=f"trackvision_{uploaded_file.name}",
                    mime="video/mp4",
                    use_container_width=True
                )
            with stat_col:
                st.markdown(f"""
                <div class="info-box">
                    <b style="color:#64ffda;">📊 Processing Summary</b><br><br>
                    🎬 Frames: <b style="color:#ccd6f6;">{total_frames}</b> &nbsp;|&nbsp;
                    ⏱️ Time: <b style="color:#ccd6f6;">{elapsed:.1f}s</b> &nbsp;|&nbsp;
                    ⚡ Speed: <b style="color:#ccd6f6;">{total_frames/elapsed:.1f} fps</b><br>
                    🆔 Unique Tracks: <b style="color:#ccd6f6;">{len(total_tracks_seen)}</b> &nbsp;|&nbsp;
                    🤖 Model: <b style="color:#ccd6f6;">{model_choice}</b> &nbsp;|&nbsp;
                    🎯 Conf: <b style="color:#ccd6f6;">{conf_threshold}</b>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Processing error: {e}")
            raise e
        finally:
            try:
                os.unlink(input_path)
            except Exception:
                pass
else:
    st.markdown("""
    <div class="info-box" style="text-align:center; padding:1rem;">
        👆 Upload a video above to enable the processing pipeline
    </div>
    """, unsafe_allow_html=True)

# ── How It Works ──────────────────────────────────────────────────────────────
with st.expander("📖 How It Works — Technical Overview"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **🔍 Detection (YOLOv8)**
        - Pre-trained on COCO dataset (80 classes)
        - Detects persons & sports equipment
        - Configurable confidence threshold
        - Non-Max Suppression for clean boxes
        """)
    with c2:
        st.markdown("""
        **🎯 Tracking (SORT)**
        - Kalman Filter for motion prediction
        - IoU-based detection association
        - Persistent unique IDs per subject
        - Handles brief occlusions gracefully
        """)
    with c3:
        st.markdown("""
        **🎨 Visualization**
        - Per-ID consistent color coding
        - Bounding boxes + confidence scores
        - Active track counter per frame
        - Clean readable label backgrounds
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    TrackVision AI &nbsp;|&nbsp; Enterprise Computer Vision Platform &nbsp;|&nbsp;
    YOLOv8 + SORT Algorithm &nbsp;|&nbsp; Built with Python & Streamlit &nbsp;|&nbsp;
    © 2026 TrackVision AI. All rights reserved.
</div>
""", unsafe_allow_html=True)
