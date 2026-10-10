"""
Streamlit demo app for the v2 models (runs_v2/<model>/seed<k>/model.pt).

Upload a dermoscopy image + enter clinical context →
    • Predicted malignancy probability
    • Grad-CAM attention map overlay
    • Counterfactual age sensitivity plot
    • Natural language explanation

Run:
    streamlit run app/streamlit_app.py

⚠️ DISCLAIMER: Research prototype only. Not a medical device. Not for diagnostic use.
"""

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import fit_encoder
from src.data.metadata_encoder import SITE_CATEGORIES, SEX_CATEGORIES
from src.data.transforms import get_inference_transforms
from src.models.factory import build_model
from src.xai.counterfactual_engine import run_full_counterfactual, sweep_age
from src.xai.gradcam import GradCAM, get_default_target_layer, overlay_cam_on_image
from src.xai.visualization import (
    plot_age_sensitivity,
    plot_cam_overlay,
    plot_site_sensitivity,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dermatology AI — Explainable Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
    .main-header p { font-size: 1rem; opacity: 0.8; margin-top: 0.5rem; }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
    }
    .metric-value { font-size: 2.5rem; font-weight: 700; }
    .metric-label { font-size: 0.9rem; opacity: 0.9; }

    .risk-high {
        background: linear-gradient(135deg, #e05c5c 0%, #c62828 100%);
    }
    .risk-low {
        background: linear-gradient(135deg, #4CAF50 0%, #2e7d32 100%);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #FF9800 0%, #e65100 100%);
    }

    .disclaimer-box {
        background: #fff3e0;
        border: 2px solid #FF9800;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #5d4037;
    }

    .explanation-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        border-radius: 4px;
        padding: 1rem;
        font-style: italic;
        color: #333;
        margin-top: 1rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🔬 Explainable Dermatology AI</h1>
        <p>Context-Aware Skin Lesion Classification via Cross-Attention & Counterfactual Metadata Probing</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Disclaimer ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="disclaimer-box">
        ⚠️ <strong>RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE.</strong>
        This tool is for academic demonstration only and must NOT be used for clinical diagnosis,
        patient management, or any medical decision-making. Always consult a qualified dermatologist.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# ─── Model Loading (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_model_and_encoder(checkpoint_path: str):
    """Load a v2 run (model.pt holds weights + config) and refit the train-split metadata encoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = {**ckpt["config"], "pretrained": False}
    model_type = cfg.get("model_type", "cross_attention")
    backbone_name = cfg.get("backbone", "efficientnet_b2")

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    st.success(f"✓ Loaded {model_type} (epoch {ckpt.get('epoch', '?')})")

    encoder = fit_encoder(cfg)  # same encoder as training: fitted on splits/v2/train.csv
    return model, encoder, device, cfg, model_type, backbone_name


# ─── Sidebar — Config ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")

    CHECKPOINT_PATH = st.text_input(
        "Checkpoint (model.pt from runs_v2/)",
        value="runs_v2/cross_attention/seed42/model.pt",
        key="ckpt_path",
    )

    st.markdown("---")
    st.markdown("### 🩺 Patient Information")

    age = st.slider("Patient Age (years)", min_value=10, max_value=90, value=50, step=1)
    sex = st.selectbox("Biological Sex", options=["male", "female", "unknown"])
    site = st.selectbox(
        "Anatomical Site",
        options=SITE_CATEGORIES,
        index=SITE_CATEGORIES.index("torso"),
    )

    st.markdown("---")
    st.markdown("### 🔧 Analysis Options")
    show_gradcam = st.checkbox("Show Grad-CAM overlay", value=True)
    show_age_sweep = st.checkbox("Age sensitivity curve", value=True)
    show_site_sweep = st.checkbox("Site sensitivity chart", value=True)

    st.markdown("---")
    st.markdown(
        "**Project:** Explainable Multi-Modal Dermatology AI  \n"
        "**Dataset:** ISIC 2020  \n"
        "**Architecture:** EfficientNet-B2 + Cross-Attention"
    )


# ─── Main Upload Area ─────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📤 Upload Dermoscopy Image")
    uploaded_file = st.file_uploader(
        "Upload a dermoscopy image (JPEG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image_pil)
        st.image(image_pil, caption="Uploaded Image", use_container_width=True)

    analyze_btn = st.button("🔬 Analyze Lesion", key="analyze_btn", disabled=(uploaded_file is None))


# ─── Analysis ────────────────────────────────────────────────────────────────
with col_result:
    st.markdown("### 📊 Results")

    if uploaded_file and analyze_btn:
        with st.spinner("Loading model …"):
            try:
                model, encoder, device, cfg, model_type, backbone_name = load_model_and_encoder(CHECKPOINT_PATH)
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                st.stop()

        with st.spinner("Analyzing image …"):
            # Prepare image tensor
            transform = get_inference_transforms(cfg.get("img_size", 224))
            image_tensor = transform(image_pil).unsqueeze(0).to(device)  # (1, 3, H, W), PIL as in training

            # Prepare metadata
            import pandas as pd
            meta_df = pd.DataFrame([{
                "age_approx": age,
                "sex": sex,
                "anatom_site_general_challenge": site,
                "age_missing": 0,
            }])
            cont_arr, site_arr = encoder.transform(meta_df)
            cont_tensor = torch.tensor(cont_arr, dtype=torch.float32).to(device)
            site_tensor = torch.tensor(site_arr, dtype=torch.long).to(device)

            # Prediction
            model.eval()
            with torch.no_grad():
                if model_type == "baseline":
                    logit = model(image_tensor)
                else:
                    logit = model(image_tensor, cont_tensor, site_tensor)
                prob = float(torch.sigmoid(logit).item())

        # ─── Risk Card ──────────────────────────────────────────────────────
        if prob >= 0.5:
            risk_class = "risk-high"
            risk_label = "🔴 HIGH RISK"
        elif prob >= 0.2:
            risk_class = "risk-moderate"
            risk_label = "🟡 MODERATE RISK"
        else:
            risk_class = "risk-low"
            risk_label = "🟢 LOW RISK"

        st.markdown(
            f"""
            <div class="metric-card {risk_class}">
                <div class="metric-value">{prob:.1%}</div>
                <div class="metric-label">Predicted Malignancy Probability<br>{risk_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        # ─── Grad-CAM ───────────────────────────────────────────────────────
        if show_gradcam:
            with st.spinner("Generating Grad-CAM …"):
                try:
                    target_layer = get_default_target_layer(model_type, backbone_name)
                    gradcam = GradCAM(model, target_layer)
                    cam = gradcam.generate(
                        image=image_tensor,
                        metadata_cont=cont_tensor if model_type != "baseline" else None,
                        site_idx=site_tensor if model_type != "baseline" else None,
                        model_type=model_type,
                    )
                    gradcam.remove_hooks()
                    cam_overlay = overlay_cam_on_image(image_np, cam)

                    fig_cam = plot_cam_overlay(
                        image_np, cam_overlay,
                        image_name=uploaded_file.name,
                        pred_prob=prob,
                    )
                    st.pyplot(fig_cam)
                    import matplotlib.pyplot as plt
                    plt.close(fig_cam)
                except Exception as e:
                    st.warning(f"Grad-CAM unavailable: {e}")

        # ─── Counterfactual Sweeps ──────────────────────────────────────────
        if show_age_sweep or show_site_sweep:
            with st.spinner("Computing counterfactual sensitivity …"):
                cf_results = run_full_counterfactual(
                    model=model,
                    image=image_tensor,
                    base_cont=cont_arr[0],
                    base_site_idx=int(site_arr[0]),
                    encoder=encoder,
                    device=device,
                    model_type=model_type,
                )

            if show_age_sweep:
                st.markdown("#### Age Sensitivity")
                fig_age = plot_age_sensitivity(
                    cf_results["age_sweep"],
                    base_prob=prob,
                    image_name=uploaded_file.name,
                )
                st.pyplot(fig_age)
                import matplotlib.pyplot as plt
                plt.close(fig_age)

            if show_site_sweep:
                st.markdown("#### Anatomical Site Sensitivity")
                fig_site = plot_site_sensitivity(
                    cf_results["site_sweep"],
                    base_prob=prob,
                    image_name=uploaded_file.name,
                )
                st.pyplot(fig_site)
                import matplotlib.pyplot as plt
                plt.close(fig_site)

            # ─── Natural Language Summary ────────────────────────────────
            age_sweep = cf_results["age_sweep"]
            site_sweep = cf_results["site_sweep"]
            ages, probs_age = age_sweep["ages"], age_sweep["probs"]
            sites, probs_site = site_sweep["sites"], site_sweep["probs"]

            nl = (
                f"For this lesion (uploaded image), the model predicts "
                f"**P(malignant) = {prob:.1%}** given age {age}, sex {sex}, "
                f"site {site}. "
                f"Age sensitivity: risk ranges from {min(probs_age):.1%} (age {ages[int(np.argmin(probs_age))]}) "
                f"to {max(probs_age):.1%} (age {ages[int(np.argmax(probs_age))]}). "
                f"Highest site risk: **{sites[int(np.argmax(probs_site))]}** ({max(probs_site):.1%})."
            )
            st.markdown(
                f'<div class="explanation-box">{nl}</div>',
                unsafe_allow_html=True,
            )

    elif not uploaded_file:
        st.info("⬅️ Upload a dermoscopy image and enter patient information to begin analysis.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.8rem;'>"
    "Explainable Multi-Modal Dermatology AI · Final Year Project · "
    "Dataset: ISIC 2020 (CC BY-NC 4.0) · "
    "⚠️ Not for clinical use"
    "</div>",
    unsafe_allow_html=True,
)
