import streamlit as st
import cv2

from backend import (
    load_model_and_labels,
    load_hands,
    predict_frame,
    get_stable_prediction,
    apply_prediction,
    new_buffer,
    speak,
    save_to_file,
)

# ======================================================
# Page config
# ======================================================

st.set_page_config(
    page_title="ASL Alphabet Recognition System",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================
# Aurora theme styling
# ======================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 20%, rgba(99, 102, 241, 0.20), transparent 45%),
                radial-gradient(circle at 85% 15%, rgba(45, 212, 191, 0.18), transparent 45%),
                radial-gradient(circle at 50% 90%, rgba(168, 85, 247, 0.15), transparent 50%),
                #0a0e1a;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1424 0%, #0a0e1a 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.1);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

.aurora-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #2dd4bf, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}
.aurora-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 2rem;
}

h1, h2, h3 { font-weight: 700 !important; }

.stButton>button {
    background: linear-gradient(135deg, #6366f1, #2dd4bf);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1rem;
    width: 100%;
    transition: opacity 0.15s ease;
}
.stButton>button:hover {
    opacity: 0.85;
    color: white;
}

.panel-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(6px);
}
.panel-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 0.25rem;
}
.panel-value {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a5b4fc, #5eead4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.video-wrapper {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.15);
    box-shadow: 0 0 40px rgba(99, 102, 241, 0.08);
}

/* Right-side sentence panel */
.sentence-panel {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    min-height: 260px;
    backdrop-filter: blur(6px);
}
.sentence-panel .sentence-text {
    color: #f1f5f9;
    font-size: 1.15rem;
    line-height: 1.7;
    white-space: pre-wrap;   /* preserves spaces + wraps like a paragraph */
    word-wrap: break-word;
    margin-top: 0.4rem;
}

/* History panel */
.history-panel {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    max-height: 220px;
    overflow-y: auto;
}
.history-item {
    border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    padding: 0.5rem 0;
    font-size: 0.9rem;
    color: #cbd5e1;
}
.history-item:last-child { border-bottom: none; }
.history-time {
    color: #94a3b8;
    font-size: 0.72rem;
    display: block;
    margin-bottom: 0.15rem;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# Cached backend resources
# ======================================================

@st.cache_resource
def get_model():
    return load_model_and_labels()

@st.cache_resource
def get_hands():
    return load_hands()

model, labels = get_model()
mp_hands, hands, mp_draw = get_hands()

# ======================================================
# Session state
# ======================================================

if "sentence" not in st.session_state:
    st.session_state.sentence = []
if "buffer" not in st.session_state:
    st.session_state.buffer = new_buffer()
if "last_added_label" not in st.session_state:
    st.session_state.last_added_label = None
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "history" not in st.session_state:
    st.session_state.history = []  # list of (timestamp, text)

# ======================================================
# Sidebar — branding + controls
# ======================================================

with st.sidebar:
    st.markdown("## ASL Alphabet")
    st.caption("Recognition System")
    st.markdown("<div style='color:#94a3b8; font-size:0.85rem; margin-top:-0.5rem;'>Real-time sign language translation</div>", unsafe_allow_html=True)

    st.markdown("---")

    col_start, col_stop = st.columns(2)
    with col_start:
        start_clicked = st.button("▶ Start Camera")
    with col_stop:
        stop_clicked = st.button("■ Stop Camera")

    if start_clicked:
        st.session_state.run_camera = True
    if stop_clicked:
        st.session_state.run_camera = False

    status_text = "🟢 Camera Live" if st.session_state.run_camera else "⚪ Camera Off"
    st.markdown(f"<div style='margin-top:0.5rem; font-size:0.85rem; color:#94a3b8;'>{status_text}</div>", unsafe_allow_html=True)

    st.markdown("### Live Status")
    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()
    confidence_bar_placeholder = st.empty()

    st.markdown("---")
    st.markdown("### Actions")
    col_a, col_b = st.columns(2)
    with col_a:
        clear_clicked = st.button("Clear")
        speak_clicked = st.button("🔊 Speak")
    with col_b:
        delete_clicked = st.button("Delete")
        save_clicked = st.button("💾 Save")

if clear_clicked:
    st.session_state.sentence.clear()
    st.session_state.buffer.clear()
    st.session_state.last_added_label = None

if delete_clicked:
    if st.session_state.sentence:
        st.session_state.sentence.pop()

if speak_clicked:
    speak("".join(st.session_state.sentence))

if save_clicked:
    text = "".join(st.session_state.sentence)
    timestamp = save_to_file(text)
    if timestamp:
        st.session_state.history.insert(0, (timestamp, text))
        st.sidebar.success("Saved to downloads/translation.txt")
    else:
        st.sidebar.warning("Nothing to save yet.")

# ======================================================
# Main area — centered title
# ======================================================

st.markdown('<div class="aurora-title">ASL Alphabet Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="aurora-subtitle">Real-time hand sign detection powered by MediaPipe & TensorFlow</div>', unsafe_allow_html=True)

# Two columns: video (left) + sentence & history (right)
video_col, sentence_col = st.columns([2, 1])

with video_col:
    st.markdown('<div class="video-wrapper">', unsafe_allow_html=True)
    video_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with sentence_col:
    sentence_placeholder = st.empty()
    history_placeholder = st.empty()


def render_sentence_panel(text):
    sentence_placeholder.markdown(
        f"""<div class="sentence-panel">
            <div class="panel-label">Sentence</div>
            <div class="sentence-text">{text if text else "…"}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_history_panel():
    if not st.session_state.history:
        items_html = "<div class='history-item'>No saved sentences yet.</div>"
    else:
        items_html = "".join(
            f"""<div class="history-item">
                <span class="history-time">{ts}</span>{text}
            </div>"""
            for ts, text in st.session_state.history
        )
    history_placeholder.markdown(
        f"""<div class="history-panel">
            <div class="panel-label">History</div>
            {items_html}
        </div>""",
        unsafe_allow_html=True,
    )


render_sentence_panel("".join(st.session_state.sentence))
render_history_panel()

# ======================================================
# Camera loop
# ======================================================

if st.session_state.run_camera:
    camera = cv2.VideoCapture(0)

    while st.session_state.run_camera:
        success, frame = camera.read()
        if not success:
            st.error("Could not access webcam.")
            break

        frame, prediction, confidence, results = predict_frame(
            frame, hands, model, labels, mp_draw, mp_hands
        )

        st.session_state.buffer.append(prediction)
        stable_prediction = get_stable_prediction(st.session_state.buffer)

        if stable_prediction and stable_prediction != st.session_state.last_added_label:
            st.session_state.sentence = apply_prediction(
                st.session_state.sentence, stable_prediction
            )
            st.session_state.last_added_label = stable_prediction
            st.session_state.buffer.clear()

        if prediction == "-":
            st.session_state.last_added_label = None

        current_sentence = "".join(st.session_state.sentence)

        # ---- Sidebar status ----
        prediction_placeholder.markdown(
            f"""<div class="panel-card">
                <div class="panel-label">Prediction</div>
                <div class="panel-value">{prediction}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        confidence_placeholder.markdown(
            f"""<div class="panel-card">
                <div class="panel-label">Confidence</div>
                <div class="panel-value">{confidence:.1%}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        confidence_bar_placeholder.progress(min(confidence, 1.0))

        # ---- Right-side sentence panel ----
        render_sentence_panel(current_sentence)

        # ---- Video frame ----
        rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_display)

    camera.release()
else:
    video_placeholder.markdown(
        """<div style="
            min-height: 480px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(30, 41, 59, 0.4);
            border-radius: 16px;
            color: #94a3b8;
            font-size: 1rem;
            text-align: center;
            padding: 2rem;
        ">
            Click <b style="color:#a5b4fc; margin: 0 4px;">▶ Start Camera</b> in the sidebar to begin translating.
        </div>""",
        unsafe_allow_html=True,
    )