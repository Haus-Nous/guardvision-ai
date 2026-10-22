"""
GuardVision Lab: Interactive Digital Image Processing (DIP) & Computer Vision (CV) Platform
========================================================================================
Designed for Live College Professor Presentations and Laboratory Demonstrations.
Featuring real-time laptop webcam capture, interactive filter parameter tuning,
side-by-side comparative views, mathematical formulations, and CV applications.
"""

import time
import math
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GuardVision Lab | DIP & CV Presentation",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact responsive layout
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 98% !important;
    }
    .main-header {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #F8FAFC;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 8px 12px;
        text-align: center;
    }
    .theory-box {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }
    .badge {
        background-color: #1E293B;
        color: #38BDF8;
        padding: 3px 6px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Haar Cascades for Face & Eye Detection (Offline & Built-in)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_cascades():
    try:
        if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            face_cascade = cv2.CascadeClassifier(face_path)
            eye_cascade = cv2.CascadeClassifier(eye_path)
            return face_cascade, eye_cascade
    except Exception as e:
        pass
    return None, None

face_cascade, eye_cascade = load_cascades()

# -----------------------------------------------------------------------------
# Compact Header Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem; padding-bottom:0.4rem; border-bottom:1px solid #1E293B;">
    <div>
        <span class="main-header">👁️ GuardVision Lab: Side-by-Side DIP & CV Studio</span>
        <span style="font-size:0.85rem; color:#94A3B8; margin-left:14px;">Live Laptop Webcam Signal Processing Workbench</span>
    </div>
    <div style="display:flex; gap:6px;">
        <span class="badge">OpenCV 4.14</span>
        <span class="badge">30 FPS Live Stream</span>
        <span class="badge">Side-by-Side View</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Controls & Parameters
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ Camera & Stream Controls")
cam_source = st.sidebar.radio(
    "Camera Ingestion Mode",
    [
        "🎥 Live Laptop Webcam (Continuous Stream)",
        "📸 Browser Camera Snapshot (Direct Face Preview & Multi-Filter)",
        "🧪 Synthetic Demo Generator (Offline Fallback)"
    ],
    index=0
)
cam_index = 0 if "Live Laptop Webcam" in cam_source else -1

stream_active = st.sidebar.toggle("▶️ Activate Video Feed", value=True)
mirror_view = st.sidebar.checkbox("🪞 Mirror Webcam Horizontally", value=True)
view_layout = st.sidebar.radio("Display Layout", ["Side-by-Side (Original vs Processed)", "Processed Only"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Laboratory Modules")

module = st.sidebar.selectbox(
    "Select DIP / CV Technique",
    [
        "1. Color Space Transformations",
        "2. Spatial Filtering & Noise Reduction",
        "3. Edge Detection & Gradients",
        "4. Image Thresholding & Segmentation",
        "5. Morphological Operations",
        "6. Histogram Equalization & Contrast",
        "7. Real-Time Motion Detection (CV)",
        "8. Face & Feature Localization (CV)",
        "9. Contour & Geometric Shape Analysis (CV)"
    ],
    index=7  # Default to Face & Feature Localization so users immediately see their face tracked!
)

st.sidebar.markdown("#### Parameter Tuning")

# Dynamic parameter sliders based on module
params = {}

if module == "1. Color Space Transformations":
    color_space = st.sidebar.selectbox("Target Color Space", ["Grayscale", "HSV", "CIELAB", "YCrCb"], index=0)
    channel_view = "All / Merged"
    if color_space in ["HSV", "CIELAB", "YCrCb"]:
        channel_view = st.sidebar.radio("Channel Split View", ["All / Merged", "Channel 1", "Channel 2", "Channel 3"])
    params = {"color_space": color_space, "channel_view": channel_view}

elif module == "2. Spatial Filtering & Noise Reduction":
    filter_type = st.sidebar.selectbox("Filter Type", ["Gaussian Blur", "Median Blur", "Bilateral Filter (Edge-Preserving)", "Sharpening Kernel"])
    if filter_type == "Gaussian Blur":
        ksize = st.sidebar.slider("Kernel Size (Odd NxN)", min_value=3, max_value=35, value=11, step=2)
        sigma = st.sidebar.slider("Sigma (Standard Deviation)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        params = {"filter": "gaussian", "ksize": ksize, "sigma": sigma}
    elif filter_type == "Median Blur":
        ksize = st.sidebar.slider("Kernel Size (Odd)", min_value=3, max_value=31, value=9, step=2)
        params = {"filter": "median", "ksize": ksize}
    elif filter_type == "Bilateral Filter (Edge-Preserving)":
        d = st.sidebar.slider("Diameter of Neighborhood", min_value=3, max_value=25, value=9, step=2)
        sigma_color = st.sidebar.slider("Sigma Color", min_value=10, max_value=200, value=75, step=5)
        sigma_space = st.sidebar.slider("Sigma Space", min_value=10, max_value=200, value=75, step=5)
        params = {"filter": "bilateral", "d": d, "sigma_color": sigma_color, "sigma_space": sigma_space}
    else:
        strength = st.sidebar.slider("Sharpen Strength", min_value=1.0, max_value=4.0, value=1.5, step=0.1)
        params = {"filter": "sharpen", "strength": strength}

elif module == "3. Edge Detection & Gradients":
    edge_operator = st.sidebar.selectbox("Operator", ["Canny Edge Detector", "Sobel Gradient Operator", "Laplacian 2nd Derivative"])
    if edge_operator == "Canny Edge Detector":
        low_thresh = st.sidebar.slider("Lower Hysteresis Threshold", min_value=1, max_value=255, value=50, step=1)
        high_thresh = st.sidebar.slider("Upper Hysteresis Threshold", min_value=1, max_value=255, value=150, step=1)
        invert_edges = st.sidebar.checkbox("Invert Edges (Black on White)", value=False)
        params = {"operator": "canny", "low": low_thresh, "high": high_thresh, "invert": invert_edges}
    elif edge_operator == "Sobel Gradient Operator":
        sobel_mode = st.sidebar.radio("Sobel Direction", ["Combined Gradient Magnitude", "Sobel X (Vertical Edges)", "Sobel Y (Horizontal Edges)"])
        ksize = st.sidebar.select_slider("Kernel Size", options=[3, 5, 7], value=3)
        params = {"operator": "sobel", "mode": sobel_mode, "ksize": ksize}
    else:
        ksize = st.sidebar.select_slider("Laplacian Kernel Size", options=[1, 3, 5, 7], value=3)
        params = {"operator": "laplacian", "ksize": ksize}

elif module == "4. Image Thresholding & Segmentation":
    thresh_type = st.sidebar.selectbox("Segmentation Method", ["Global Binary Threshold", "Otsu's Bimodal Thresholding", "Adaptive Gaussian Thresholding"])
    if thresh_type == "Global Binary Threshold":
        thresh_val = st.sidebar.slider("Threshold Value", min_value=0, max_value=255, value=127, step=1)
        invert = st.sidebar.checkbox("Invert Binary Mask (THRESH_BINARY_INV)", value=False)
        params = {"type": "binary", "thresh": thresh_val, "invert": invert}
    elif thresh_type == "Otsu's Bimodal Thresholding":
        blur_first = st.sidebar.checkbox("Pre-apply Gaussian Blur (Recommended for Noise)", value=True)
        params = {"type": "otsu", "blur": blur_first}
    else:
        block_size = st.sidebar.slider("Neighborhood Block Size (Odd)", min_value=3, max_value=41, value=15, step=2)
        c_val = st.sidebar.slider("Constant C Subtracted", min_value=-10, max_value=20, value=2, step=1)
        params = {"type": "adaptive", "block_size": block_size, "c": c_val}

elif module == "5. Morphological Operations":
    morph_op = st.sidebar.selectbox("Operation", ["Dilation", "Erosion", "Opening (Noise Removal)", "Closing (Hole Filling)", "Morphological Gradient (Edges)"])
    struct_shape = st.sidebar.selectbox("Structuring Element Shape", ["Rectangle", "Ellipse", "Cross"])
    kernel_size = st.sidebar.slider("Kernel Size", min_value=3, max_value=25, value=5, step=2)
    iterations = st.sidebar.slider("Iterations", min_value=1, max_value=5, value=1)
    params = {"op": morph_op, "shape": struct_shape, "ksize": kernel_size, "iterations": iterations}

elif module == "6. Histogram Equalization & Contrast":
    hist_mode = st.sidebar.selectbox("Method", ["CLAHE (Contrast Limited Adaptive)", "Standard Global Equalization"])
    if hist_mode.startswith("CLAHE"):
        clip_limit = st.sidebar.slider("Clip Limit", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
        grid_size = st.sidebar.slider("Tile Grid Size", min_value=2, max_value=16, value=8, step=1)
        params = {"method": "clahe", "clip": clip_limit, "grid": grid_size}
    else:
        params = {"method": "global"}

elif module == "7. Real-Time Motion Detection (CV)":
    motion_thresh = st.sidebar.slider("Difference Sensitivity Threshold", min_value=10, max_value=80, value=25, step=1)
    min_area = st.sidebar.slider("Min Detection Contour Area (px)", min_value=200, max_value=8000, value=1000, step=100)
    accumulate_bg = st.sidebar.checkbox("Running Background Average (Smooth BG Model)", value=True)
    params = {"thresh": motion_thresh, "min_area": min_area, "smooth": accumulate_bg}

elif module == "8. Face & Feature Localization (CV)":
    scale_factor = st.sidebar.slider("Scale Factor (Pyramid Downscale)", min_value=1.05, max_value=1.40, value=1.15, step=0.05)
    min_neighbors = st.sidebar.slider("Min Neighbors", min_value=2, max_value=10, value=5, step=1)
    detect_eyes = st.sidebar.checkbox("Also Detect Eyes within Faces", value=True)
    params = {"scale": scale_factor, "neighbors": min_neighbors, "eyes": detect_eyes}

elif module == "9. Contour & Geometric Shape Analysis (CV)":
    min_contour_area = st.sidebar.slider("Min Contour Area (px)", min_value=100, max_value=10000, value=800, step=100)
    draw_bounding = st.sidebar.checkbox("Draw Bounding Rectangles & Area Labels", value=True)
    draw_convex_hull = st.sidebar.checkbox("Draw Convex Hulls", value=False)
    params = {"min_area": min_contour_area, "bounding": draw_bounding, "hull": draw_convex_hull}

# -----------------------------------------------------------------------------
# DIP Mathematical Foundations Helper
# -----------------------------------------------------------------------------
def render_academic_theory(active_module):
    with st.expander("📘 DIP & CV Mathematical Foundations & Algorithm Theory (Click to Expand for Professor Q&A)", expanded=False):
        if active_module.startswith("1."):
            st.markdown(r"""
            **Color Space Transformation Theory:**
            - **Grayscale conversion (Luma weighting / Rec. 601):**
              $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
              Human photoreceptors have peak spectral sensitivity in the green spectrum, hence the $58.7\%$ weight.
            - **HSV (Hue, Saturation, Value):** Decouples chromaticity (Hue/Saturation) from intensity ($V = \max(R,G,B)$), making it invariant to lighting fluctuations.
            - **CIELAB:** Designed to be perceptually uniform; $L^*$ is perceptual lightness, $a^*$ represents green-red opponent axis, $b^*$ represents blue-yellow opponent axis.
            """)
        elif active_module.startswith("2."):
            st.markdown(r"""
            **Spatial Filtering & 2D Convolution Theory:**
            - Linear filtering applies a spatial convolution mask $h(x,y)$ over image $f(x,y)$:
              $$g(x,y) = \sum_{s=-a}^a \sum_{t=-b}^b h(s,t) \cdot f(x-s, y-t)$$
            - **Gaussian 2D Kernel:**
              $$G(x,y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$$
            - **Bilateral Filter (Edge-Preserving Smoothing):** Combines spatial Gaussian distance weighting $g_s$ with radiometric pixel intensity similarity $g_r$:
              $$I^{\text{filtered}}(p) = \frac{1}{W_p} \sum_{q \in \Omega} I(q) \cdot g_s(\|p-q\|) \cdot g_r(\|I(p)-I(q)\|)$$
            """)
        elif active_module.startswith("3."):
            st.markdown(r"""
            **Edge Detection & Gradient Operators:**
            - **Canny Edge Detection Multi-Stage Pipeline:**
              1. *Gaussian Smoothing:* Suppress high-frequency noise.
              2. *Gradient Magnitude & Orientation:* Compute $G = \sqrt{G_x^2 + G_y^2}$ and $\theta = \arctan(G_y / G_x)$ via Sobel operators.
              3. *Non-Maximum Suppression (NMS):* Thin edges down to 1-pixel ridges by checking along gradient normal direction.
              4. *Hysteresis Thresholding:* Pixels above $T_{\text{high}}$ are strong edges; pixels below $T_{\text{low}}$ are suppressed; pixels between are retained only if connected to a strong edge.
            """)
        elif active_module.startswith("4."):
            st.markdown(r"""
            **Image Segmentation & Thresholding:**
            - **Global Thresholding:**
              $$g(x,y) = \begin{cases} 255 & \text{if } f(x,y) \ge T \\ 0 & \text{otherwise} \end{cases}$$
            - **Otsu's Thresholding Criterion:** Automatically selects optimal threshold $T^*$ by maximizing between-class variance $\sigma_B^2(T)$:
              $$\sigma_B^2(T) = \omega_0(T) \omega_1(T) [\mu_0(T) - \mu_1(T)]^2$$
              where $\omega_0, \omega_1$ are probabilities of foreground and background, and $\mu_0, \mu_1$ are class mean intensities.
            """)
        elif active_module.startswith("5."):
            st.markdown(r"""
            **Mathematical Morphology (Minkowski Operations):**
            - Let $A$ be the binary image and $B$ the structuring element:
              - **Dilation ($A \oplus B$):** Expands foreground boundaries: $A \oplus B = \bigcup_{b \in B} A_b$.
              - **Erosion ($A \ominus B$):** Shrinks foreground, removes isolated noise: $A \ominus B = \{z \mid (B)_z \subseteq A\}$.
              - **Opening ($A \circ B$):** Erosion followed by dilation; eliminates small protrusions without altering object size.
              - **Closing ($A \bullet B$):** Dilation followed by erosion; closes narrow breaks and fills small holes.
            """)
        elif active_module.startswith("6."):
            st.markdown(r"""
            **Histogram Equalization Theory:**
            - Stretches contrast across dynamic range via Cumulative Distribution Function (CDF):
              $$s_k = T(r_k) = (L-1) \sum_{j=0}^k p_r(r_j)$$
            - **CLAHE:** Computes equalization on local $8 \times 8$ grid tiles and clips histograms above a threshold to prevent over-amplification of background noise.
            """)
        elif active_module.startswith("7."):
            st.markdown(r"""
            **Temporal Frame Differencing (Computer Vision Motion Detection):**
            - Measures temporal gradient between consecutive frames $I_t(x,y)$ and $I_{t-1}(x,y)$:
              $$\Delta I_t(x,y) = |I_t(x,y) - I_{t-1}(x,y)|$$
            - Segmented motion mask:
              $$M_t(x,y) = \begin{cases} 1 & \text{if } \Delta I_t(x,y) > \theta \\ 0 & \text{otherwise} \end{cases}$$
            - Connected component / contour analysis groups active pixels into bounded physical motion zones.
            """)
        elif active_module.startswith("8."):
            st.markdown(r"""
            **Object Localization via Viola-Jones (Haar Cascades):**
            - Uses rectangular Haar-like digital filters computed in constant time $O(1)$ via the **Integral Image (Summed-Area Table)**.
            - AdaBoost selects a subset of critical discriminative features out of $160,000+$ candidates.
            - Attentional cascade organizes classifiers in stages: early stages quickly reject non-face subwindows with negligible computational cost.
            """)
        else:
            st.markdown(r"""
            **Contour Analysis & Boundary Traversal:**
            - Suzuki's border following algorithm identifies topological boundaries of connected components.
            - Computes geometric spatial moments $m_{pq} = \sum_x \sum_y x^p y^q I(x,y)$ to extract centroid $(\bar{x}, \bar{y})$ and orientation.
            """)

# -----------------------------------------------------------------------------
# Image Processing Functions
# -----------------------------------------------------------------------------
def process_color_space(frame, p):
    space = p["color_space"]
    channel = p["channel_view"]

    if space == "Grayscale":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), "Grayscale Luma Matrix"

    if space == "HSV":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if channel == "Channel 1":
            return cv2.applyColorMap(h, cv2.COLORMAP_HSV), "HSV: Hue (0-179)"
        elif channel == "Channel 2":
            return cv2.applyColorMap(s, cv2.COLORMAP_BONE), "HSV: Saturation (0-255)"
        elif channel == "Channel 3":
            return cv2.applyColorMap(v, cv2.COLORMAP_HOT), "HSV: Value / Luminance"
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), "HSV Full Spectrum"

    if space == "CIELAB":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        if channel == "Channel 1":
            return cv2.cvtColor(l, cv2.COLOR_GRAY2RGB), "CIELAB: L* (Lightness)"
        elif channel == "Channel 2":
            return cv2.applyColorMap(a, cv2.COLORMAP_COOL), "CIELAB: a* (Green-Red)"
        elif channel == "Channel 3":
            return cv2.applyColorMap(b, cv2.COLORMAP_PARULA), "CIELAB: b* (Blue-Yellow)"
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), "CIELAB Perceptual Space"

    # YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    if channel == "Channel 1":
        return cv2.cvtColor(y, cv2.COLOR_GRAY2RGB), "YCrCb: Y (Luma)"
    elif channel == "Channel 2":
        return cv2.applyColorMap(cr, cv2.COLORMAP_MAGMA), "YCrCb: Cr (Red-difference)"
    elif channel == "Channel 3":
        return cv2.applyColorMap(cb, cv2.COLORMAP_VIRIDIS), "YCrCb: Cb (Blue-difference)"
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB), "YCrCb Digital Video Space"


def process_filtering(frame, p):
    ftype = p["filter"]
    if ftype == "gaussian":
        k = p["ksize"]
        sig = p["sigma"]
        blurred = cv2.GaussianBlur(frame, (k, k), sig)
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB), f"Gaussian Blur ({k}x{k}, σ={sig})"

    if ftype == "median":
        k = p["ksize"]
        med = cv2.medianBlur(frame, k)
        return cv2.cvtColor(med, cv2.COLOR_BGR2RGB), f"Median Filter ({k}x{k}) - Salt & Pepper Noise Removal"

    if ftype == "bilateral":
        d = p["d"]
        sc = p["sigma_color"]
        ss = p["sigma_space"]
        bil = cv2.bilateralFilter(frame, d, sc, ss)
        return cv2.cvtColor(bil, cv2.COLOR_BGR2RGB), f"Bilateral Filter (d={d}, σ_c={sc}, σ_s={ss})"

    # Sharpening
    strg = p["strength"]
    kernel = np.array([[0, -1, 0], [-1, 4 + strg, -1], [0, -1, 0]])
    sharp = cv2.filter2D(frame, -1, kernel)
    return cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB), f"Spatial Sharpening (Strength={strg})"


def process_edges(frame, p):
    op = p["operator"]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if op == "canny":
        edges = cv2.Canny(gray, p["low"], p["high"])
        if p["invert"]:
            edges = cv2.bitwise_not(edges)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), f"Canny Edges (Low={p['low']}, High={p['high']})"

    if op == "sobel":
        mode = p["mode"]
        k = p["ksize"]
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)

        if "X" in mode:
            abs_x = cv2.convertScaleAbs(gx)
            return cv2.applyColorMap(abs_x, cv2.COLORMAP_JET), f"Sobel-X Vertical Gradients (k={k})"
        elif "Y" in mode:
            abs_y = cv2.convertScaleAbs(gy)
            return cv2.applyColorMap(abs_y, cv2.COLORMAP_JET), f"Sobel-Y Horizontal Gradients (k={k})"
        else:
            mag = cv2.magnitude(gx, gy)
            mag = cv2.convertScaleAbs(mag)
            return cv2.applyColorMap(mag, cv2.COLORMAP_MAGMA), f"Sobel Gradient Magnitude sqrt(Gx² + Gy²)"

    # Laplacian
    k = p["ksize"]
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    abs_lap = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(abs_lap, cv2.COLOR_GRAY2RGB), f"Laplacian 2nd Derivative Operator (k={k})"


def process_thresholding(frame, p):
    ttype = p["type"]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ttype == "binary":
        th = p["thresh"]
        flag = cv2.THRESH_BINARY_INV if p["invert"] else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, th, 255, flag)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB), f"Global Binary Threshold (T={th})"

    if ttype == "otsu":
        if p["blur"]:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB), f"Otsu's Optimal Threshold (Calculated T* = {int(ret)})"

    # Adaptive
    bs = p["block_size"]
    c = p["c"]
    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, c)
    return cv2.cvtColor(adap, cv2.COLOR_GRAY2RGB), f"Adaptive Gaussian (BlockSize={bs}, C={c})"


def process_morphology(frame, p):
    op = p["op"]
    ksize = p["ksize"]
    shape_str = p["shape"]
    iters = p["iterations"]

    shape_map = {
        "Rectangle": cv2.MORPH_RECT,
        "Ellipse": cv2.MORPH_ELLIPSE,
        "Cross": cv2.MORPH_CROSS
    }
    kernel = cv2.getStructuringElement(shape_map[shape_str], (ksize, ksize))

    # Pre-threshold to binary for crisp morphological demonstration
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    if op == "Dilation":
        res = cv2.dilate(binary, kernel, iterations=iters)
    elif op == "Erosion":
        res = cv2.erode(binary, kernel, iterations=iters)
    elif op.startswith("Opening"):
        res = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iters)
    elif op.startswith("Closing"):
        res = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iters)
    else:
        res = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    return cv2.cvtColor(res, cv2.COLOR_GRAY2RGB), f"{op} ({shape_str} Kernel: {ksize}x{ksize}, Iters={iters})"


def process_histogram(frame, p):
    method = p["method"]
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))

    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=p["clip"], tileGridSize=(p["grid"], p["grid"]))
        channels[0] = clahe.apply(channels[0])
        label = f"CLAHE Adaptive Equalization (Clip={p['clip']}, Grid={p['grid']}x{p['grid']})"
    else:
        channels[0] = cv2.equalizeHist(channels[0])
        label = "Standard Global Histogram Equalization"

    merged = cv2.merge(channels)
    res = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)
    return res, label


# Frame difference persistent state
bg_avg = None
prev_gray = None

def process_motion(frame, p):
    global bg_avg, prev_gray
    thresh_val = p["thresh"]
    min_area = p["min_area"]
    smooth = p["smooth"]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

    display = frame.copy()
    h, w, _ = frame.shape
    total_pixels = h * w
    motion_pixels = 0

    if smooth:
        if bg_avg is None:
            bg_avg = gray_blur.astype("float")
            return cv2.cvtColor(display, cv2.COLOR_BGR2RGB), 0, "Initializing Background Model..."

        cv2.accumulateWeighted(gray_blur, bg_avg, 0.1)
        frame_delta = cv2.absdiff(gray_blur, cv2.convertScaleAbs(bg_avg))
    else:
        if prev_gray is None:
            prev_gray = gray_blur
            return cv2.cvtColor(display, cv2.COLOR_BGR2RGB), 0, "Initializing Frame History..."
        frame_delta = cv2.absdiff(prev_gray, gray_blur)
        prev_gray = gray_blur

    _, thresh = cv2.threshold(frame_delta, thresh_val, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        motion_detected = True
        motion_pixels += area
        x, y, cw, ch = cv2.boundingRect(c)
        cv2.rectangle(display, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
        cv2.putText(display, f"MOTION: {int(area)}px", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    motion_pct = min(100.0, (motion_pixels / total_pixels) * 100.0 * 2.5)
    hud_color = (0, 0, 255) if motion_detected else (0, 255, 0)
    status_text = f"MOTION DETECTED ({motion_pct:.1f}%)" if motion_detected else "SCENE IDLE (0.0%)"
    cv2.putText(display, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2, cv2.LINE_AA)

    return cv2.cvtColor(display, cv2.COLOR_BGR2RGB), motion_pct, status_text


def process_face_localization(frame, p):
    display = frame.copy()
    if face_cascade is None:
        cv2.putText(display, "Haar Cascade Not Available in this OpenCV build", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)
        return cv2.cvtColor(display, cv2.COLOR_BGR2RGB), 0, "Haar Cascade Unavailable"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=p["scale"],
        minNeighbors=p["neighbors"],
        minSize=(40, 40)
    )

    eye_count = 0
    primary_center = "N/A"
    for (x, y, w, h) in faces:
        # Tactical face bounding box
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Center crosshairs
        cx, cy = x + w // 2, y + h // 2
        primary_center = f"[{cx}, {cy}]"
        cv2.drawMarker(display, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(display, f"FACE [{cx},{cy}]", (x, max(24, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        if p["eyes"] and eye_cascade is not None:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = display[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
            for (ex, ey, ew, eh) in eyes:
                eye_count += 1
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 120, 0), 2)
                cv2.putText(roi_color, "EYE", (ex, max(14, ey - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 120, 0), 1, cv2.LINE_AA)

    if len(faces) > 0:
        label = f"Viola-Jones: {len(faces)} Face(s) Locked | {eye_count} Eye(s) Tracked"
        cv2.putText(display, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)
    else:
        label = "Viola-Jones: Searching for Face (Look directly at camera)"
        cv2.putText(display, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        
    extra = {"faces": len(faces), "eye_count": eye_count, "center": primary_center}
    return cv2.cvtColor(display, cv2.COLOR_BGR2RGB), len(faces), label, extra


def process_contours(frame, p):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    display = frame.copy()
    valid_count = 0
    total_perimeter = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < p["min_area"]:
            continue
        valid_count += 1
        perim = cv2.arcLength(c, True)
        total_perimeter += perim

        # Draw contour boundary in cyan
        cv2.drawContours(display, [c], -1, (255, 255, 0), 2)

        if p["bounding"]:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(display, f"A:{int(area)}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        if p["hull"]:
            hull = cv2.convexHull(c)
            cv2.drawContours(display, [hull], -1, (0, 0, 255), 1)

    label = f"Contours: {valid_count} Objects | Total Perimeter: {int(total_perimeter)}px"
    cv2.putText(display, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)
    return cv2.cvtColor(display, cv2.COLOR_BGR2RGB), valid_count, label


# -----------------------------------------------------------------------------
# Unified Algorithm Dispatcher
# -----------------------------------------------------------------------------
def dispatch_filter(raw_frame, selected_module, p):
    metric_val = 0
    tag = ""
    extra = {}
    if selected_module.startswith("1."):
        proc_img, tag = process_color_space(raw_frame, p)
    elif selected_module.startswith("2."):
        proc_img, tag = process_filtering(raw_frame, p)
    elif selected_module.startswith("3."):
        proc_img, tag = process_edges(raw_frame, p)
    elif selected_module.startswith("4."):
        proc_img, tag = process_thresholding(raw_frame, p)
    elif selected_module.startswith("5."):
        proc_img, tag = process_morphology(raw_frame, p)
    elif selected_module.startswith("6."):
        proc_img, tag = process_histogram(raw_frame, p)
    elif selected_module.startswith("7."):
        proc_img, metric_val, tag = process_motion(raw_frame, p)
        extra = {"motion_pct": metric_val}
    elif selected_module.startswith("8."):
        proc_img, metric_val, tag, extra = process_face_localization(raw_frame, p)
    elif selected_module.startswith("9."):
        proc_img, metric_val, tag = process_contours(raw_frame, p)
        extra = {"contours": metric_val}
    else:
        proc_img, tag = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB), "Raw Pass-through"
    return proc_img, metric_val, tag, extra


# Offline Synthetic Generator Helper
def get_synthetic_frame(t):
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (30, 35, 45)

    # Perspective laboratory grid
    for y in range(0, h, 40):
        cv2.line(frame, (0, y), (w, y), (45, 55, 70), 1)
    for x in range(0, w, 40):
        cv2.line(frame, (x, 0), (x, h), (45, 55, 70), 1)

    # Dynamic animated circle (demonstrates motion, edges, filters)
    cx = int(w/2 + math.sin(t * 2.0) * 180)
    cy = int(h/2 + math.cos(t * 1.5) * 120)
    cv2.circle(frame, (cx, cy), 55, (0, 200, 255), -1)
    cv2.circle(frame, (cx, cy), 55, (255, 255, 255), 3)

    # Secondary moving rectangle
    rx = int(w/2 + math.cos(t * 1.0) * 140)
    ry = int(h/2 + math.sin(t * 2.5) * 80)
    cv2.rectangle(frame, (rx - 40, ry - 30), (rx + 40, ry + 30), (255, 50, 100), -1)

    # Synthetic Face Landmark
    fx, fy = int(w * 0.75), int(h * 0.35)
    cv2.circle(frame, (fx, fy), 45, (220, 200, 180), -1) # Head
    cv2.circle(frame, (fx - 15, fy - 10), 6, (40, 40, 40), -1) # Eye 1
    cv2.circle(frame, (fx + 15, fy - 10), 6, (40, 40, 40), -1) # Eye 2
    cv2.ellipse(frame, (fx, fy + 15), (15, 8), 0, 0, 180, (40, 40, 40), 2) # Mouth

    cv2.putText(frame, "SYNTHETIC DEMO STREAM (OFFLINE FALLBACK)", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)
    return frame


# -----------------------------------------------------------------------------
# MODE A: Browser Camera Snapshot & Multi-Filter Inspection
# -----------------------------------------------------------------------------
if cam_source.startswith("📸 Browser Camera Snapshot"):
    st.markdown("### 📸 Live Browser Camera Snapshot & Multi-Filter Inspection")
    st.info("Your browser will prompt for camera access. Look into the camera and click **'Take Photo'** below to immediately test and compare DIP & Computer Vision algorithms on your own face!")

    img_file = st.camera_input("Take a photo of your face:")

    if img_file is not None:
        bytes_data = img_file.getvalue()
        raw_frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        if mirror_view:
            raw_frame = cv2.flip(raw_frame, 1)
        raw_frame = cv2.resize(raw_frame, (640, 480))
        rgb_raw = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

        proc_img, metric_val, tag, extra = dispatch_filter(raw_frame, module, params)

        # Side-by-side display
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("##### 📹 Original Snapshot (Your Face)")
            st.image(rgb_raw, width="stretch")
            st.markdown(f"""
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:6px 12px; font-size:0.8rem; color:#94A3B8; display:flex; justify-content:space-between; margin-top:-4px;">
                <span>🟢 <b>Raw Ingestion Photo</b></span>
                <span>📐 <b>{raw_frame.shape[1]}x{raw_frame.shape[0]}</b></span>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"##### ⚡ Processed Output: {module.split('.')[1].strip()}")
            st.image(proc_img, caption=tag, width="stretch")
            if module.startswith("8."):
                status_color = "#38BDF8" if metric_val > 0 else "#F59E0B"
                status_text = "🟢 FACE LOCKED" if metric_val > 0 else "🟡 SEARCHING FACE"
                st.markdown(f"""
                <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:6px 12px; font-size:0.8rem; color:#E2E8F0; display:flex; justify-content:space-between; margin-top:-4px;">
                    <span style="color:{status_color}; font-weight:700;">{status_text}</span>
                    <span>👤 Faces: <b style="color:#38BDF8;">{metric_val}</b></span>
                    <span>👁️ Eyes: <b style="color:#38BDF8;">{extra.get('eye_count', 0)}</b></span>
                    <span>📍 Center: <b>{extra.get('center', 'N/A')}</b></span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:6px 12px; font-size:0.8rem; color:#E2E8F0; display:flex; justify-content:space-between; margin-top:-4px;">
                    <span style="color:#38BDF8; font-weight:700;">⚡ {module.split('.')[1].strip()}</span>
                    <span>🔬 <b>{tag}</b></span>
                </div>
                """, unsafe_allow_html=True)

        # Multi-filter comparison gallery!
        with st.expander("🔬 **Multi-Filter Gallery: See 6 Core DIP & CV Algorithms Applied to Your Face Simultaneously**", expanded=True):
            st.markdown("Here is how 6 distinct mathematical computer vision transformations alter your exact facial portrait:")
            gcol1, gcol2, gcol3 = st.columns(3)
            with gcol1:
                # 1. Viola-Jones Face & Eye Localization
                face_img, _, _, _ = process_face_localization(raw_frame, {"scale": 1.15, "neighbors": 5, "eyes": True})
                st.markdown("**1. Face & Eye Localization**")
                st.image(face_img, caption="Viola-Jones Bounding Boxes & Crosshairs", width="stretch")

                # 4. Canny Edges
                canny_img, _ = process_edges(raw_frame, {"operator": "canny", "low": 50, "high": 150, "invert": False})
                st.markdown("**4. Canny Edge Detection**")
                st.image(canny_img, caption="Multi-Stage Hysteresis Wireframe", width="stretch")

            with gcol2:
                # 2. Bilateral Skin Smoothing
                bil_img, _ = process_filtering(raw_frame, {"filter": "bilateral", "d": 9, "sigma_color": 75, "sigma_space": 75})
                st.markdown("**2. Bilateral Filter**")
                st.image(bil_img, caption="Edge-Preserving Smoothing (Skin Smooth)", width="stretch")

                # 5. Otsu Thresholding
                otsu_img, _ = process_thresholding(raw_frame, {"type": "otsu", "blur": True})
                st.markdown("**5. Otsu's Bimodal Thresholding**")
                st.image(otsu_img, caption="Optimal Class Variance Binary Mask", width="stretch")

            with gcol3:
                # 3. CLAHE Contrast Equalization
                clahe_img, _ = process_histogram(raw_frame, {"method": "clahe", "clip": 3.0, "grid": 8})
                st.markdown("**3. CLAHE Equalization**")
                st.image(clahe_img, caption="Adaptive Local Contrast Enhancement", width="stretch")

                # 6. Grayscale Rec. 601
                gray_img, _ = process_color_space(raw_frame, {"color_space": "Grayscale", "channel_view": "All / Merged"})
                st.markdown("**6. Grayscale Conversion**")
                st.image(gray_img, caption="Rec. 601 Luma Weighting (0.299R+0.587G+0.114B)", width="stretch")

    else:
        st.write("👉 Look into your camera and press **'Take Photo'** above to begin analysis.")


# -----------------------------------------------------------------------------
# MODE B: Real-Time Continuous Video Stream (OpenCV / Synthetic)
# -----------------------------------------------------------------------------
else:
    cap = None
    if stream_active and cam_index >= 0:
        try:
            cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                st.warning("⚠️ Laptop webcam could not be accessed directly. Using high-fidelity synthetic demo generator.")
                cap = None
            else:
                # Warm up camera sensor so initial frames have auto-exposure locked
                for _ in range(8):
                    cap.read()
        except Exception as e:
            st.warning(f"⚠️ Camera capture exception: {e}. Switching to synthetic demo mode.")
            cap = None

    # Side-by-side single screen layout without scrolling
    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        st.markdown("##### 📹 Live Camera Feed (Your Raw Image)")
        st_frame_left = st.empty()
        st_left_telemetry = st.empty()
    with col_right:
        st.markdown(f"##### ⚡ Live Processed Vision: {module.split('.')[1].strip()}")
        st_frame_right = st.empty()
        st_right_telemetry = st.empty()

    frame_count = 0
    start_time = time.time()

    if stream_active:
        # Run continuous live stream for up to 3,000 frames (~2.5 minutes)
        for step in range(3000):
            t = time.time() - start_time
            frame_count += 1

            # Capture or synthesize
            if cap and cap.isOpened():
                ret, raw_frame = cap.read()
                if not ret or raw_frame is None:
                    raw_frame = get_synthetic_frame(t)
                elif mirror_view:
                    raw_frame = cv2.flip(raw_frame, 1)
            else:
                raw_frame = get_synthetic_frame(t)

            # Standardize resolution for smooth browser throughput
            raw_frame = cv2.resize(raw_frame, (640, 480))
            rgb_raw = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

            proc_img, metric_val, tag, extra = dispatch_filter(raw_frame, module, params)

            # Render side-by-side video frames
            st_frame_left.image(rgb_raw, width="stretch")
            st_frame_right.image(proc_img, width="stretch")

            # Telemetry under left camera
            st_left_telemetry.markdown(f"""
            <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:6px 12px; font-size:0.8rem; color:#94A3B8; display:flex; justify-content:space-between; margin-top:-4px;">
                <span>🟢 <b>Live Ingestion</b> (FaceTime HD Device 0)</span>
                <span>📐 <b>640x480</b></span>
                <span>🪞 <b>Mirrored View</b></span>
            </div>
            """, unsafe_allow_html=True)

            # Telemetry under right processed frame
            fps = frame_count / max(0.01, (time.time() - start_time))
            if module.startswith("8."):
                status_color = "#38BDF8" if metric_val > 0 else "#F59E0B"
                status_text = "🟢 FACE LOCKED" if metric_val > 0 else "🟡 SEARCHING FACE"
                st_right_telemetry.markdown(f"""
                <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:6px 12px; font-size:0.8rem; color:#E2E8F0; display:flex; justify-content:space-between; margin-top:-4px;">
                    <span style="color:{status_color}; font-weight:700;">{status_text}</span>
                    <span>👤 Faces: <b style="color:#38BDF8;">{metric_val}</b></span>
                    <span>👁️ Eyes: <b style="color:#38BDF8;">{extra.get('eye_count', 0)}</b></span>
                    <span>📍 Center: <b>{extra.get('center', 'N/A')}</b></span>
                    <span style="color:#A78BFA; font-family:monospace;">⚡ {fps:.1f} FPS</span>
                </div>
                """, unsafe_allow_html=True)
            elif module.startswith("7."):
                mot_color = "#EF4444" if metric_val > 5 else "#10B981"
                mot_text = "🚨 MOTION DETECTED" if metric_val > 5 else "🟢 SCENE IDLE"
                st_right_telemetry.markdown(f"""
                <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:6px 12px; font-size:0.8rem; color:#E2E8F0; display:flex; justify-content:space-between; margin-top:-4px;">
                    <span style="color:{mot_color}; font-weight:700;">{mot_text}</span>
                    <span>🏃 Motion Index: <b style="color:#EF4444;">{metric_val:.1f}%</b></span>
                    <span>📐 Sensitivity: <b>{params.get('thresh', 25)}</b></span>
                    <span style="color:#A78BFA; font-family:monospace;">⚡ {fps:.1f} FPS</span>
                </div>
                """, unsafe_allow_html=True)
            elif module.startswith("9."):
                st_right_telemetry.markdown(f"""
                <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:6px 12px; font-size:0.8rem; color:#E2E8F0; display:flex; justify-content:space-between; margin-top:-4px;">
                    <span style="color:#38BDF8; font-weight:700;">📐 CONTOURS</span>
                    <span>Objects Extracted: <b style="color:#38BDF8;">{metric_val}</b></span>
                    <span>Min Area: <b>{params.get('min_area', 800)}px</b></span>
                    <span style="color:#A78BFA; font-family:monospace;">⚡ {fps:.1f} FPS</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st_right_telemetry.markdown(f"""
                <div style="background:#0F172A; border:1px solid #1E293B; border-radius:6px; padding:6px 12px; font-size:0.8rem; color:#E2E8F0; display:flex; justify-content:space-between; margin-top:-4px;">
                    <span style="color:#38BDF8; font-weight:700;">⚡ {module.split('.')[1].strip()}</span>
                    <span>🔬 Pipeline: <b>{tag}</b></span>
                    <span style="color:#A78BFA; font-family:monospace;">⚡ {fps:.1f} FPS</span>
                </div>
                """, unsafe_allow_html=True)

            time.sleep(0.015)

        if cap:
            cap.release()
    else:
        st.info("Stream is currently paused. Toggle **'Activate Video Feed'** in the sidebar to resume.")


# -----------------------------------------------------------------------------
# Collapsible Educational & Mathematical Foundations
# -----------------------------------------------------------------------------
render_academic_theory(module)

with st.expander("💡 **How to Demonstrate Facial & Video Processing to Your Professor**", expanded=False):
    st.markdown("""
    - **Face Tracking (Module 8)**: As you move your head, notice the green bounding box and yellow crosshair following your facial centroid with exact coordinates.
    - **Eye Tracking**: Notice the blue boxes over your eyes. Blink or cover an eye to show the eye counter change from 2 to 1.
    - **Motion Detection (Module 7)**: Wave your hand in front of the camera to see the red motion bounding boxes and motion percentage meter react to temporal frame differencing.
    - **Edge Detection (Module 3)**: Tune the lower and upper hysteresis sliders in the left sidebar to show how Canny suppresses noise while highlighting facial contours.
    """)
