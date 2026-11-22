
import os, tempfile
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict import predict, META

st.set_page_config(page_title="Paddy Thermal Disease Detector",
                   page_icon="\U0001F33E", layout="wide")

st.title("\U0001F33E Paddy Leaf Disease Detection (Thermal)")
st.caption("Upload a thermal image of a paddy leaf. "
           "Model classifies it as **Healthy** or **Diseased**.")

with st.sidebar:
    st.header("Settings")
    show_gradcam = st.checkbox("Show Grad-CAM", value=True)
    show_thermal = st.checkbox("Show thermal stats", value=True)
    st.markdown("---")
    st.markdown(f"**Version:** `{META.get('version','?')}`")
    st.markdown(f"**Threshold:** `{META['decision_threshold']:.3f}`")
    st.markdown(f"**Params:** `{META.get('params','?')}`")
    st.markdown(f"**Latency:** `{META.get('latency_ms','?'):.1f} ms`"
                if isinstance(META.get('latency_ms'), (int, float)) else "")

uploaded = st.file_uploader("Upload thermal image",
                            type=["jpg","jpeg","png","tif","tiff","bmp"])
if uploaded is None:
    st.info("Awaiting an image."); st.stop()

suffix = os.path.splitext(uploaded.name)[1] or ".png"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read()); tmp_path = tmp.name

with st.spinner("Analyzing..."):
    res = predict(tmp_path, return_gradcam=show_gradcam,
                  return_thermal_stats=show_thermal)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Original")
    st.image(Image.open(tmp_path), use_container_width=True)
with c2:
    st.subheader("Preprocessed leaf")
    st.image(res["preprocessed_u8"], use_container_width=True, clamp=True)

st.markdown("---")
lbl, conf = res["label"], res["confidence"]*100
if lbl == "Diseased":
    st.error(f"### \u26A0\uFE0F {lbl}  —  Confidence: **{conf:.1f}%**")
else:
    st.success(f"### \u2705 {lbl}  —  Confidence: **{conf:.1f}%**")
st.progress(min(max(res["prob_diseased"], 0.), 1.))
st.caption(f"P(diseased) = {res['prob_diseased']:.4f} | "
           f"threshold = {res['threshold']:.3f}")

if show_gradcam and "gradcam" in res:
    st.markdown("### Grad-CAM")
    heat = res["gradcam"]
    heat_r = cv2.resize(heat, (res["preprocessed_u8"].shape[1],
                               res["preprocessed_u8"].shape[0]))
    overlay = cv2.applyColorMap((heat_r*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(res["preprocessed_u8"], cmap="gray"); axes[0].set_title("Leaf")
    axes[1].imshow(heat_r, cmap="jet"); axes[1].set_title("Heatmap")
    axes[2].imshow(res["preprocessed_u8"], cmap="gray")
    axes[2].imshow(overlay, alpha=0.5); axes[2].set_title("Overlay")
    for ax in axes: ax.axis("off")
    st.pyplot(fig, clear_figure=True)

if show_thermal and "thermal_stats" in res:
    st.markdown("### Thermal stats (not validated CWSI)")
    st.json({k: round(v, 4) for k, v in res["thermal_stats"].items()})

st.markdown("---")
st.caption("\u26A0\uFE0F Research prototype. Not for agronomic decisions.")
