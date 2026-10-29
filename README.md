# GuardVision Lab: Interactive Digital Image Processing (DIP) & Computer Vision (CV) Studio

[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit: 1.40+](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV: 4.x](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

An interactive, real-time Computer Vision and Digital Image Processing engineering studio built with **Python, Streamlit, and OpenCV**. Featuring live webcam ingestion, dynamic algorithm parameter tuning, side-by-side comparative visualization, mathematical theory, and real-time computer vision analysis.

---

## System Architecture & Pipeline Flowchart

Use this diagram to trace the complete end-to-end dataflow from raw physical sensor input through mathematical transformations to real-time visualization:

```mermaid
flowchart TD
    %% STAGE 1: INGESTION
    subgraph S1 ["Stage 1: Hardware Ingestion & Normalization"]
        direction TB
        CAM["Live Hardware Camera (cv2.VideoCapture)"] --> FRAME["Raw Sensor Frame (H x W x 3)"]
        SNAP["Browser Snapshot (HTML5 Camera Input)"] --> FRAME
        SYNTH["Synthetic Moving Fallback Generator"] -.->|Auto-Fallback| FRAME
        FRAME --> RESIZE["Standardize Resolution to 640 x 480"]
        RESIZE --> FLIP["Mirror View (Horizontal Flip: cv2.flip)"]
        FLIP --> SWAP["Color Transposition: BGR to RGB (cv2.cvtColor)"]
    end

    %% STAGE 2: CONTROL & DISPATCH
    subgraph S2 ["Stage 2: Dynamic Control & Dispatch Engine"]
        direction TB
        SIDEBAR["Sidebar Controls (Kernel, Sigma, Threshold Sliders)"] --> PARAMS["Runtime Parameters Dict (p)"]
        MODELS["Offline XML Trees (models/haarcascades/)"] --> CACHE["Cached Resource (@st.cache_resource)"]
        SWAP --> DISPATCH["Central Dispatcher: dispatch_filter(frame, module, params)"]
        PARAMS --> DISPATCH
        CACHE --> DISPATCH
    end

    %% STAGE 3: MATHEMATICAL PIPELINE BRANCHING
    subgraph S3 ["Stage 3: Algorithmic Transformation Pipelines"]
        direction TB
        DISPATCH --> DIP["Branch A: DIP Mathematical Foundations"]
        DISPATCH --> CV["Branch B: Real-Time Computer Vision"]

        %% DIP Modules
        subgraph S3_DIP ["DIP Modules (Modules 1 - 6)"]
            direction TB
            DIP --> M1["Color Spaces: Rec. 601 Grayscale, HSV Decoupling, CIELAB"]
            DIP --> M2["Spatial 2D Filters: Separable Gaussian O(2K), Median, Bilateral"]
            DIP --> M3["Edge Operators: Canny 5-Stage Hysteresis, Sobel, Laplacian"]
            DIP --> M4["Segmentation: Global Cutoff, Otsu Variance σ_B², Adaptive Gaussian"]
            DIP --> M5["Morphology: Erosion ⊖, Dilation ⊕, Opening, Closing, Gradient"]
            DIP --> M6["Contrast: Global HE (CDF Transfer), CLAHE (Tile Clipping)"]
        end

        %% CV Modules
        subgraph S3_CV ["CV Applications (Modules 7 - 9)"]
            direction TB
            CV --> M7["Motion Detection: Running Background Model B_t = (1-α)B_t-1 + αI_t"]
            CV --> M8["Multi-Face & Eye: Viola-Jones + EqualizeHist + Ensemble + NMS (IoU=0.35)"]
            CV --> M9["Contour Geometry: Suzuki-Abe Border Following + Moments (x̄, ȳ) + Convex Hull"]
        end
    end

    %% STAGE 4: PRESENTATION & TELEMETRY
    subgraph S4 ["Stage 4: Synchronous Presentation & Telemetry HUD"]
        direction TB
        S3_DIP --> RENDER["Dual Viewport Synchronous Renderer (30 FPS)"]
        S3_CV --> RENDER
        RENDER --> RAW_FEED["Left Viewport: Raw Ingested Camera Input"]
        RENDER --> PROC_FEED["Right Viewport: Live Processed Algorithm Output"]
        RENDER --> HUD["Telemetry HUD: Real-Time FPS, Centroids, Face Count, Motion Index %"]
        RENDER --> GALLERY["Multi-Filter Gallery: 6 Core Algorithms Applied Simultaneously"]
    end
```

### How to Explain This Flowchart to Your Professor (60-Second Walkthrough)

1. **Point to Stage 1 (Top):** *"Professor, our pipeline starts with frame ingestion. Regardless of whether frames come from a live hardware camera or browser snapshot, we normalize the resolution to $640 \times 480$, apply a mirror flip for natural user interaction, and transpose the memory layout from OpenCV's native BGR to sRGB."*
2. **Point to Stage 2 (Middle-Left):** *"Next, the frame enters our central `dispatch_filter()` engine, which couples the incoming image with runtime parameters dynamically tuned via the sidebar sliders (like Gaussian kernel size or Canny thresholds) and pre-cached Haar feature trees."*
3. **Point to Stage 3 (Middle):** *"The engine routes the frame into one of two algorithmic branches:  
   - **Branch A (DIP Foundations):** Evaluates mathematical transformations (Rec. 601 Luma, separable Gaussian convolution, Canny 5-stage hysteresis, Otsu variance maximization, morphology, and CLAHE).  
   - **Branch B (Computer Vision):** Executes higher-level perception—temporal motion detection using a running background model, multi-subject face and eye localization using Viola-Jones with NMS deduplication, and contour boundary extraction with spatial image moments."*
4. **Point to Stage 4 (Bottom):** *"Finally, both the raw camera feed and the processed output are rendered side-by-side at 30 FPS alongside live telemetry showing FPS, centroids, and detection metrics."*

---

## Key Features & Demonstration Modules

### 1. Digital Image Processing (DIP) Fundamentals
- **Color Space Conversions**:
  - RGB to Grayscale (Rec. 601 luma weighting: $Y = 0.299R + 0.587G + 0.114B$).
  - HSV Color Space with individual Hue, Saturation, and Value channel inspection.
  - CIELAB Perceptual Color Space ($L^*$, $a^*$, $b^*$).
  - YCrCb Digital Video Encoding.
- **Spatial 2D Filtering & Smoothing**:
  - **Gaussian Blur**: 2D Gaussian kernel with interactive kernel size ($3 \dots 35$) and standard deviation ($\sigma$).
  - **Median Blur**: Non-linear ranking filter for salt-and-pepper impulse noise removal.
  - **Bilateral Filter**: Non-linear, edge-preserving smoothing combining spatial closeness with radiometric photometric similarity.
  - **Spatial Sharpening**: High-pass filter kernel boosting edge gradients.
- **Edge Detection & Gradient Operators**:
  - **Canny Edge Detector**: Full 4-stage pipeline (Gaussian smooth, gradient magnitude/angle, Non-Maximum Suppression, hysteresis thresholding with dual sliders).
  - **Sobel Operator**: First-order derivative filters ($G_x$ vertical edges, $G_y$ horizontal edges, combined gradient magnitude $\sqrt{G_x^2 + G_y^2}$).
  - **Laplacian Operator**: Second-order isotropic derivative filter highlighting rapid intensity discontinuities.
- **Image Segmentation & Thresholding**:
  - **Global Binary Thresholding**: Interactive threshold cutoff ($T \in [0, 255]$) with inverted binary masking.
  - **Otsu’s Optimal Thresholding**: Automatic threshold calculation via between-class variance maximization ($\sigma_B^2$).
  - **Adaptive Gaussian Thresholding**: Local neighborhood thresholding for uneven illumination.
- **Mathematical Morphology**:
  - Dilation, Erosion, Morphological Opening, Closing, and Morphological Gradient with customizable structuring elements (Rectangle, Ellipse, Cross) and iterations.
- **Contrast & Histograms**:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) preventing noise over-amplification.
  - Global Histogram Equalization.

### 2. Computer Vision (CV) Applications
- **Real-Time Temporal Motion Detection**:
  - Consecutive frame differencing ($\Delta I_t = |I_t - I_{t-1}| > \theta$) with thresholding, morphological noise suppression, contour grouping, and dynamic green/red bounding boxes.
  - Real-time **"Motion Activity Index (%)"** progress bar and metric.
- **Multi-Subject Face & Eye Localization**:
  - Real-time simultaneous multi-face detection using ensembled Haar Cascades with NMS, histogram equalization, center coordinates, and eye tracking.
- **Contour Analysis & Geometric Shape Extraction**:
  - Border traversal, area/perimeter computation, bounding rectangle fitting, and convex hull generation.

### 3. Core Capabilities
- **Side-by-Side Comparative View**: Displays the live raw webcam feed on the left and the processed DIP filter on the right.
- **Mathematical Theory Foundations**: Collapsible panel displaying KaTeX formulas and theoretical explanations for each technique.
- **Dual Stream Ingestion**: Live laptop webcam (`cv2.VideoCapture(0)`) with automatic fallback to an offline synthetic moving generator if the webcam is blocked or unavailable.

---

## Quickstart Guide

### 1. Installation

Clone or open the project folder in your terminal:

```bash
# 1. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application Locally

Run this command:

```bash
streamlit run app.py
```

The application will immediately launch in your browser at:
👉 **[http://localhost:8501](http://localhost:8501)**
