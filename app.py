"""NeuroLens — Brain Tumor Detection Console"""

import time
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from predict import predict_image

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="NeuroLens AI — Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_META = {
    "glioma": {
        "label": "Glioma",
        "color": "#F87171",
        "tone": "alert",
        "about": (
            "Glioma is a tumor that develops from glial cells, the "
            "supportive tissue surrounding neurons in the brain and "
            "spinal cord. It ranges from slow-growing (low-grade) to "
            "aggressive (high-grade, such as glioblastoma)."
        ),
        "symptoms": [
            "Persistent headaches",
            "Seizures",
            "Blurred or double vision",
            "Memory or personality changes",
            "Weakness in limbs",
        ],
        "nature": "Can be benign or malignant, depending on grade.",
        "treatment": ["Surgical resection", "Radiation therapy", "Chemotherapy", "Targeted therapy"],
    },
    "meningioma": {
        "label": "Meningioma",
        "color": "#FBBF24",
        "tone": "caution",
        "about": (
            "Meningioma arises from the meninges, the layered membranes "
            "that cover the brain and spinal cord. Most grow slowly and "
            "sit outside the brain tissue itself."
        ),
        "symptoms": [
            "Gradual headaches",
            "Vision changes",
            "Hearing loss or ringing in ears",
            "Weakness in arms or legs",
            "Seizures",
        ],
        "nature": "Usually benign, though a small share can be atypical or malignant.",
        "treatment": ["Active monitoring", "Surgical removal", "Stereotactic radiosurgery"],
    },
    "pituitary": {
        "label": "Pituitary Tumor",
        "color": "#818CF8",
        "tone": "caution",
        "about": (
            "Pituitary tumors form in the pituitary gland at the base of "
            "the brain, which regulates hormone production. Most are "
            "adenomas and grow slowly."
        ),
        "symptoms": [
            "Hormonal imbalances",
            "Unexplained fatigue",
            "Vision disturbances",
            "Irregular menstrual cycles",
            "Headaches",
        ],
        "nature": "Predominantly benign (adenomas).",
        "treatment": ["Hormone therapy", "Surgical removal", "Radiation therapy"],
    },
    "notumor": {
        "label": "No Tumor",
        "color": "#34D399",
        "tone": "positive",
        "about": (
            "The model did not detect structural patterns associated with "
            "glioma, meningioma, or pituitary tumors in this scan."
        ),
        "symptoms": [],
        "nature": "No tumor indicators identified by the model.",
        "treatment": [],
    },
}

MODEL_NAME = "MobileNetV2 — Fine-tuned CNN Classifier"
NUM_CLASSES = 4
STATIC_ACCURACY = "98.2%"

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-deep: #0A0E16;
    --bg-panel: #10151F;
    --glass-border: rgba(148, 178, 255, 0.14);
    --text-primary: #E7ECF7;
    --text-muted: #7C8AA5;
    --accent-cyan: #34D9C9;
    --accent-violet: #8B7CF6;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(139,124,246,0.08), transparent 45%),
        radial-gradient(circle at 85% 10%, rgba(52,217,201,0.07), transparent 45%),
        var(--bg-deep);
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}

/* Hero */
.hero-wrap {
    padding: 2.4rem 2.6rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(139,124,246,0.10), rgba(52,217,201,0.06));
    border: 1px solid var(--glass-border);
    margin-bottom: 1.6rem;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    background: linear-gradient(90deg, #F4F6FC, #B9C2E8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 1.02rem;
    max-width: 640px;
    line-height: 1.55;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    letter-spacing: 0.06em;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    border: 1px solid rgba(52,217,201,0.35);
    background: rgba(52,217,201,0.08);
    color: var(--accent-cyan);
    margin-bottom: 1rem;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent-cyan);
    box-shadow: 0 0 8px var(--accent-cyan);
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}
.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

/* Verdict card */
.verdict-card {
    border-radius: 16px;
    padding: 1.6rem 1.6rem;
    border: 1px solid var(--verdict-border, var(--glass-border));
    background: var(--verdict-bg, rgba(255,255,255,0.02));
    margin-bottom: 1rem;
}
.verdict-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--verdict-color, var(--text-primary));
    margin-bottom: 0.2rem;
}

/* Summary card */
.summary-card {
    border-left: 3px solid var(--summary-color, var(--accent-cyan));
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-top: 0.6rem;
}
.summary-heading {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--summary-color, var(--accent-cyan));
    margin: 0.9rem 0 0.35rem 0;
}
.disclaimer-box {
    margin-top: 1rem;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    background: rgba(248,113,113,0.06);
    border: 1px solid rgba(248,113,113,0.25);
    color: #F3B4B4;
    font-size: 0.86rem;
    line-height: 1.5;
}

section[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--glass-border);
}

div[data-testid="stFileUploader"] {
    border-radius: 14px;
}

.stButton > button {
    background: linear-gradient(90deg, var(--accent-violet), var(--accent-cyan));
    color: #0A0E16;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
}
.stButton > button:hover {
    filter: brightness(1.08);
}

hr { border-color: var(--glass-border); }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🧠 NeuroLens AI")
    st.caption("Clinical-style deep learning inference console.")
    st.divider()

    with st.expander("📘 About This Project", expanded=True):
        st.write(
            "NeuroLens AI classifies brain MRI scans into four categories "
            "using a fine-tuned convolutional neural network. Built as an "
            "end-to-end deep learning application, from preprocessing to "
            "deployment."
        )

    with st.expander("🗂️ Dataset"):
        st.write(
            "Trained on a curated brain MRI dataset spanning glioma, "
            "meningioma, pituitary tumor, and healthy (no tumor) scans, "
            "with train/validation/test splits and class balancing."
        )

    with st.expander("🏗️ Model Architecture"):
        st.write(
            "MobileNetV2 backbone with transfer learning, a custom "
            "classification head, dropout regularization, and fine-tuned "
            "top layers for the four-class task."
        )

    with st.expander("🔄 Workflow"):
        st.markdown(
            "1. Upload MRI scan\n"
            "2. Preprocessing & normalization\n"
            "3. Model inference\n"
            "4. Confidence scoring\n"
            "5. AI medical summary generation"
        )

    with st.expander("🧰 Technologies Used"):
        st.markdown(
            "- Python, TensorFlow / Keras\n"
            "- Streamlit\n"
            "- Plotly\n"
            "- NumPy, Pillow"
        )

    with st.expander("👤 Developer"):
        st.write("Built and maintained as an independent deep learning project.")

    st.divider()
    st.caption("For educational purposes only — not a substitute for professional medical diagnosis.")

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------

st.markdown(
    """
<div class="hero-wrap">
    <div class="status-pill"><span class="status-dot"></span>MODEL ONLINE</div>
    <div class="hero-eyebrow">Neural Imaging · AI Diagnostics</div>
    <div class="hero-title">NeuroLens — Brain Tumor Detection Console</div>
    <div class="hero-sub">
        Upload an MRI scan to classify it across four categories, review the
        model's confidence, and generate an educational AI medical summary.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Model", "MobileNetV2")
metric_cols[1].metric("Classes", str(NUM_CLASSES))
metric_cols[2].metric("Reported Accuracy", STATIC_ACCURACY)
metric_cols[3].metric("Status", "Ready")

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------

upload_col, result_col = st.columns([1, 1.3], gap="large")

with upload_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Step 01 · Upload</div>', unsafe_allow_html=True)
    st.markdown("#### MRI Scan Upload")

    uploaded_file = st.file_uploader(
        "Drag and drop an MRI scan, or browse files",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded scan", use_container_width=True)

        with st.expander("📄 File Details", expanded=False):
            st.write(f"**Name:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**Dimensions:** {image.size[0]} × {image.size[1]} px")

    st.markdown("</div>", unsafe_allow_html=True)

    predict_clicked = st.button(
        "🧠  Analyze MRI Scan", use_container_width=True, disabled=image is None
    )

with result_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Step 02 · AI Analytics & Verdict</div>', unsafe_allow_html=True)
    st.markdown("#### Prediction Result")

    if "result" not in st.session_state:
        st.session_state.result = None
        st.session_state.infer_ms = None

    if predict_clicked and image is not None:
        with st.status("Running inference...", expanded=True) as status:
            st.write("Preprocessing image...")
            start = time.time()
            time.sleep(0.15)
            st.write("Running model forward pass...")
            result = predict_image(image)
            elapsed_ms = (time.time() - start) * 1000
            st.write("Scoring confidence...")
            status.update(label="Inference complete", state="complete", expanded=False)

        st.session_state.result = result
        st.session_state.infer_ms = elapsed_ms
        st.toast(f"Prediction: {CLASS_META[result['class']]['label']}", icon="🧠")

    result = st.session_state.result

    if result is None:
        st.info("Upload a scan and click **Analyze MRI Scan** to run inference.")
    else:
        meta = CLASS_META[result["class"]]
        confidence_pct = result["confidence"] * 100

        verdict_style = f"--verdict-color:{meta['color']}; --verdict-border:{meta['color']}44; --verdict-bg:{meta['color']}14;"
        icon = "✅" if meta["tone"] == "positive" else "⚠️"
        st.markdown(
            f"""
            <div class="verdict-card" style="{verdict_style}">
                <div class="verdict-title">{icon} {meta['label']}</div>
                <div style="color: var(--text-muted); font-size: 0.9rem;">
                    Detected with {confidence_pct:.1f}% confidence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        gauge_col, chart_col = st.columns(2)

        with gauge_col:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence_pct,
                    number={"suffix": "%", "font": {"color": meta["color"], "size": 34}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#7C8AA5"},
                        "bar": {"color": meta["color"]},
                        "bgcolor": "rgba(255,255,255,0.03)",
                        "borderwidth": 0,
                    },
                )
            )
            gauge.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E7ECF7",
            )
            st.plotly_chart(gauge, use_container_width=True)

        with chart_col:
            probs = result["probabilities"]
            labels = [CLASS_META[c]["label"] for c in probs]
            values = [v * 100 for v in probs.values()]
            colors = [CLASS_META[c]["color"] for c in probs]

            donut = go.Figure(
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.62,
                    marker=dict(colors=colors, line=dict(color="#0A0E16", width=2)),
                    textinfo="none",
                )
            )
            donut.update_layout(
                height=220,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(font=dict(size=10, color="#B9C2E8"), orientation="h", y=-0.15),
            )
            st.plotly_chart(donut, use_container_width=True)

        st.markdown("##### Class-wise Probabilities")
        for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
            cls_meta = CLASS_META[cls]
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; "
                f"font-size:0.85rem; margin-bottom:2px;'>"
                f"<span>{cls_meta['label']}</span><span>{prob * 100:.1f}%</span></div>",
                unsafe_allow_html=True,
            )
            st.progress(prob)

        if st.session_state.infer_ms:
            st.caption(f"Inference time: {st.session_state.infer_ms:.0f} ms · {datetime.now().strftime('%H:%M:%S')}")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# AI Medical Summary
# --------------------------------------------------------------------------

if st.session_state.get("result") is not None:
    result = st.session_state.result
    meta = CLASS_META[result["class"]]
    confidence_pct = result["confidence"] * 100

    with st.expander("🩺 AI Medical Summary", expanded=True):
        st.markdown(
            f"""
            <div class="summary-card" style="--summary-color:{meta['color']};">
                <div style="font-size:1.05rem; font-weight:600;">{meta['label']}</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="summary-heading">About</div>', unsafe_allow_html=True)
        st.write(meta["about"])

        if meta["symptoms"]:
            st.markdown('<div class="summary-heading">Common Symptoms</div>', unsafe_allow_html=True)
            st.markdown("\n".join([f"- {s}" for s in meta["symptoms"]]))

        st.markdown('<div class="summary-heading">Nature</div>', unsafe_allow_html=True)
        st.write(meta["nature"])

        if meta["treatment"]:
            st.markdown('<div class="summary-heading">Possible Treatment Options</div>', unsafe_allow_html=True)
            st.markdown("\n".join([f"- {t}" for t in meta["treatment"]]))

        st.markdown('<div class="summary-heading">Recommendation</div>', unsafe_allow_html=True)
        if meta["tone"] == "positive":
            st.write(
                "No tumor indicators were detected by the model. If symptoms "
                "persist, please consult a physician for further evaluation."
            )
        else:
            st.write(
                "Please consult a qualified neurologist or neurosurgeon for "
                "proper clinical evaluation and confirmation."
            )

        st.markdown(
            f'<div class="summary-heading">Confidence Note</div>'
            f'<div>The model predicts this class with {confidence_pct:.1f}% confidence.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="disclaimer-box">
                ⚠️ <strong>Disclaimer:</strong> This prediction is generated by an
                AI model for educational purposes only and must not be
                considered a medical diagnosis. Always consult a qualified
                healthcare professional for diagnosis and treatment decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )
