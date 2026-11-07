"""
Streamlit Web Showcase & Interactive Forensic Dashboard.
Document Copy-Move Forgery Detection (CMFD).

Run locally:
    streamlit run app.py
"""

import io
import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

# Internal CV pipeline imports
from preprocess import load_and_preprocess_image, _generate_synthetic_document_if_empty
from feature_extraction import extract_sift_features, visualize_keypoints
from flann_matching import match_features_to_self, visualize_matches
from geometric_verification import cluster_and_verify_matches, visualize_forgery_regions
from evaluation_metrics import (
    generate_predicted_mask,
    calculate_cmfd_metrics,
    _generate_synthetic_ground_truth_mask,
    visualize_masks_side_by_side
)

# ==============================================================================
# Streamlit Page Setup & Styling
# ==============================================================================
st.set_page_config(
    page_title="Document Forgery Detector | Classical CMFD",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Dark Aesthetic
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background: #1e2130;
        border: 1px solid #2e344e;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        text-align: center;
    }
    .verdict-tampered {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        color: #ef4444;
        font-weight: 800;
        font-size: 1.4rem;
    }
    .verdict-authentic {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        color: #10b981;
        font-weight: 800;
        font-size: 1.4rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
        border-radius: 6px;
        padding: 0 16px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Sidebar & Parameter Controls
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/search-in-list.png", width=70)
st.sidebar.title("CMFD Inspector")
st.sidebar.caption("Classical Computer Vision Pipeline (SIFT + FLANN + DBSCAN + RANSAC)")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Pipeline Configuration")

preset = st.sidebar.selectbox(
    "Configuration Preset",
    [
        "Standard Balanced (Recommended)",
        "High Sensitivity (Faint Stamps & Signatures)",
        "Aggressive Filtering (Dense Text Documents)",
        "Dense SIFT Mode (Texture-less Erasure Detection)",
        "Custom Parameters"
    ]
)

# Preset mapping
if preset == "Standard Balanced (Recommended)":
    p_max_dim, p_ratio, p_spatial_dist, p_eps, p_min_samples, p_reproj, p_min_density = (
        1500, 0.72, 45.0, 12.0, 4, 5.0, 500
    )
elif preset == "High Sensitivity (Faint Stamps & Signatures)":
    p_max_dim, p_ratio, p_spatial_dist, p_eps, p_min_samples, p_reproj, p_min_density = (
        1800, 0.78, 30.0, 15.0, 3, 6.0, 400
    )
elif preset == "Aggressive Filtering (Dense Text Documents)":
    p_max_dim, p_ratio, p_spatial_dist, p_eps, p_min_samples, p_reproj, p_min_density = (
        1400, 0.65, 60.0, 10.0, 6, 4.0, 600
    )
elif preset == "Dense SIFT Mode (Texture-less Erasure Detection)":
    p_max_dim, p_ratio, p_spatial_dist, p_eps, p_min_samples, p_reproj, p_min_density = (
        1200, 0.75, 40.0, 14.0, 4, 5.0, 10000  # forces Dense SIFT
    )
else:
    p_max_dim, p_ratio, p_spatial_dist, p_eps, p_min_samples, p_reproj, p_min_density = (
        1500, 0.72, 45.0, 12.0, 4, 5.0, 500
    )

with st.sidebar.expander("🛠️ Advanced Hyperparameters", expanded=(preset == "Custom Parameters")):
    max_dim = st.slider("Max Image Dimension (px)", 600, 3000, p_max_dim, 100)
    ratio_thresh = st.slider("Lowe's Ratio Threshold", 0.50, 0.85, p_ratio, 0.01)
    spatial_dist = st.slider("Min Spatial Distance (px)", 10.0, 150.0, p_spatial_dist, 5.0)
    eps = st.slider("DBSCAN Epsilon (px)", 3.0, 30.0, p_eps, 1.0)
    min_samples = st.slider("DBSCAN Min Samples", 3, 15, p_min_samples, 1)
    reproj_thresh = st.slider("RANSAC Reprojection Error (px)", 1.0, 15.0, p_reproj, 0.5)
    close_kernel = st.slider("Morphological Close Kernel", 9, 45, 25, 2)
    min_density = st.number_input("Dense SIFT Fallback Threshold", 100, 50000, p_min_density, 100)

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Synthetic Benchmark")
load_sample_btn = st.sidebar.button("📄 Load Tampered Contract Demo", use_container_width=True)

st.sidebar.markdown(
    """
    <div style='font-size: 0.8rem; color: #888; margin-top: 20px;'>
    <b>Document Forgery Detector v2.0</b><br>
    Zero-GPU Explainable Computer Vision<br>
    License: MIT
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# Helper Functions for File Handling
# ==============================================================================
def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Converts uploaded file bytes into a BGR OpenCV image."""
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bytes_to_gray(image_bytes: bytes) -> np.ndarray:
    """Converts uploaded mask bytes into single-channel grayscale."""
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


# ==============================================================================
# Main User Interface
# ==============================================================================
st.title("Document Copy-Move Forgery Detector")
st.markdown(
    "Automated forensic detection and pixel-level localization of duplicated stamps, signatures, "
    "and clauses using classical scale-invariant geometry (**SIFT + FLANN + DBSCAN + RANSAC**)."
)

# File Upload Section
col_upload_1, col_upload_2 = st.columns(2)

with col_upload_1:
    uploaded_image = st.file_uploader(
        "Upload Document Scan (*.png, *.jpg, *.tif, *.bmp)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        help="Upload a scanned document to analyze for copy-move tampering."
    )

with col_upload_2:
    uploaded_mask = st.file_uploader(
        "Upload Ground-Truth Mask (Optional for Metric Benchmark)",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Optional binary mask to calculate IoU, Precision, Recall, and F1-Score."
    )

# Handle Sample Demo Button
if load_sample_btn:
    st.session_state["use_demo"] = True

# Resolve input data source
active_image_bgr = None
active_mask_gray = None
doc_name = "Uploaded_Document"

if st.session_state.get("use_demo", False) and uploaded_image is None:
    demo_img_path = Path("data/original/synthetic_tampered_doc.png")
    demo_mask_path = Path("data/masks/synthetic_tampered_doc_mask.png")
    _generate_synthetic_document_if_empty(demo_img_path)
    _generate_synthetic_ground_truth_mask(demo_mask_path)

    active_image_bgr = cv2.imread(str(demo_img_path), cv2.IMREAD_COLOR)
    active_mask_gray = cv2.imread(str(demo_mask_path), cv2.IMREAD_GRAYSCALE)
    doc_name = "Synthetic_Tampered_Contract.png"
    st.info("ℹ️ Loaded Synthetic Tampered Document with Ground-Truth Mask.")
elif uploaded_image is not None:
    st.session_state["use_demo"] = False
    active_image_bgr = bytes_to_cv2(uploaded_image.read())
    doc_name = uploaded_image.name
    if uploaded_mask is not None:
        active_mask_gray = bytes_to_gray(uploaded_mask.read())


# ==============================================================================
# Pipeline Execution
# ==============================================================================
if active_image_bgr is not None:
    analyze_btn = st.button("🚀 Analyze Document Integrity", type="primary", use_container_width=True)

    if analyze_btn or st.session_state.get("use_demo", False):
        with st.spinner("Executing 5-Stage Computer Vision Forensic Pipeline..."):
            start_t = time.perf_counter()

            # Stage 1: Ingestion & Megapixel Decimation
            orig_h, orig_w = active_image_bgr.shape[:2]
            longest = max(orig_h, orig_w)
            if longest > max_dim:
                scale_factor = float(max_dim) / float(longest)
                new_w = int(round(orig_w * scale_factor))
                new_h = int(round(orig_h * scale_factor))
                scaled_bgr = cv2.resize(active_image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                scale_factor = 1.0
                scaled_bgr = active_image_bgr.copy()

            rgb_img = cv2.cvtColor(scaled_bgr, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(scaled_bgr, cv2.COLOR_BGR2GRAY)
            preprocessed_gray = cv2.GaussianBlur(gray_img, (3, 3), 0.5)
            h, w = gray_img.shape

            # Stage 2: Feature Extraction (with Dense SIFT Fallback)
            keypoints, descriptors = extract_sift_features(
                gray_image=preprocessed_gray,
                contrast_threshold=0.03,
                edge_threshold=10.0,
                sigma=1.6,
                n_features=0,
                min_density_threshold=min_density,
                dense_grid_step=10,
                dense_keypoint_size=16.0
            )
            total_kp = len(keypoints)

            # Stage 3: FLANN Self-Matching
            valid_matches = match_features_to_self(
                keypoints=keypoints,
                descriptors=descriptors,
                ratio_threshold=ratio_thresh,
                spatial_distance_threshold=spatial_dist
            )
            total_matches = len(valid_matches)

            # Stage 4: DBSCAN & RANSAC Verification
            verified_matches, homography, (src_pts, dst_pts) = cluster_and_verify_matches(
                keypoints=keypoints,
                valid_matches=valid_matches,
                eps=eps,
                min_samples=min_samples,
                ransac_reproj_threshold=reproj_thresh
            )
            total_inliers = len(verified_matches)
            is_forged = (total_inliers >= min_samples) and (homography is not None)

            # Stage 5: Predicted Mask Generation (Morphological Closing)
            pred_mask = generate_predicted_mask(
                image_shape=(h, w),
                src_points=src_pts,
                dst_points=dst_pts,
                point_radius=8,
                close_kernel_size=close_kernel
            )

            # Metrics (if ground truth available)
            metrics = None
            if active_mask_gray is not None:
                gt_resized = active_mask_gray
                if active_mask_gray.shape != (h, w):
                    gt_resized = cv2.resize(active_mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
                metrics = calculate_cmfd_metrics(pred_mask, gt_resized)

            total_elapsed = time.perf_counter() - start_t

        # ----------------------------------------------------------------------
        # Top-Level Forensic Verdict Banner
        # ----------------------------------------------------------------------
        st.markdown("### 📋 Forensic Analysis Verdict")
        if is_forged:
            tx, ty = homography[0, 2], homography[1, 2]
            shift_dist = np.sqrt(tx**2 + ty**2)
            st.markdown(
                f"""
                <div class="verdict-tampered">
                    ⚠️ COPY-MOVE FORGERY DETECTED<br>
                    <span style="font-size: 1rem; font-weight: normal; color: #fca5a5;">
                    Planar duplication identified with {total_inliers} verified keypoint inliers.
                    Estimated translation shift: dx={tx:+.1f}px, dy={ty:+.1f}px (Distance: {shift_dist:.1f}px).
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="verdict-authentic">
                    ✅ AUTHENTIC DOCUMENT VERIFIED<br>
                    <span style="font-size: 1rem; font-weight: normal; color: #6ee7b7;">
                    Zero geometrically consistent copy-move operations detected across all scale-space octaves.
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # ----------------------------------------------------------------------
        # Forensic Key Performance Indicators (Cards)
        # ----------------------------------------------------------------------
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total SIFT Keypoints", f"{total_kp:,}")
        with c2:
            st.metric("Candidate Pairs (FLANN)", f"{total_matches:,}")
        with c3:
            st.metric("Verified Inliers (RANSAC)", f"{total_inliers:,}")
        with c4:
            reduction = ((total_kp - total_inliers) / total_kp * 100) if total_kp > 0 else 0
            st.metric("Noise Rejection", f"{reduction:.1f}%")
        with c5:
            st.metric("Analysis Time", f"{total_elapsed:.2f}s")

        st.markdown("---")

        # ----------------------------------------------------------------------
        # Multi-Tab Diagnostic Showcase
        # ----------------------------------------------------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔬 Forensic Localization",
            "🔍 CV Pipeline Stages",
            "📊 Metric Evaluation",
            "📐 Mathematical Theory"
        ])

        # TAB 1: Forensic Localization
        with tab1:
            st.subheader("Forensic Tampering Localization")
            annotated_forensic = visualize_forgery_regions(
                rgb_image=rgb_img,
                src_points=src_pts,
                dst_points=dst_pts,
                verified_matches=verified_matches,
                keypoints=keypoints,
                title=f"Forensic Copy-Move Localization: {doc_name}",
                show_plot=False
            )

            col_view_1, col_view_2 = st.columns(2)
            with col_view_1:
                st.image(rgb_img, caption=f"Original Document ({w}x{h} px)", use_container_width=True)
            with col_view_2:
                st.image(annotated_forensic, caption="Forensic Detection Overlay (Bounding Boxes & Vectors)", use_container_width=True)

            # Export Button
            buf = io.BytesIO()
            Image.fromarray(annotated_forensic).save(buf, format="PNG")
            st.download_button(
                label="💾 Download Forensic Report Image",
                data=buf.getvalue(),
                file_name=f"forensic_verdict_{Path(doc_name).stem}.png",
                mime="image/png"
            )

        # TAB 2: CV Pipeline Stages
        with tab2:
            st.subheader("Step-by-Step Classical Vision Pipeline Inspection")

            # Stage 2: Rich Keypoints
            annotated_kp = visualize_keypoints(rgb_img, keypoints[:1500], show_plot=False)
            # Stage 3: FLANN Vectors
            annotated_flann = visualize_matches(rgb_img, keypoints, valid_matches, show_plot=False)

            st.markdown("#### 1. SIFT Scale-Space Extrema vs. 2. FLANN Self-Matching")
            col_st_1, col_st_2 = st.columns(2)
            with col_st_1:
                st.image(annotated_kp, caption=f"Scale-Space Extrema (Showing Top Keypoints)", use_container_width=True)
            with col_st_2:
                st.image(annotated_flann, caption=f"FLANN Matches before Geometric Clustering ({total_matches:,} pairs)", use_container_width=True)

            st.markdown("#### 3. Morphological Closing Concave Mask")
            st.image(pred_mask, caption="Predicted Binary Mask via Keypoint Splatting + Morphological Closing", width=600)

        # TAB 3: Metric Evaluation
        with tab3:
            st.subheader("Pixel-Level Segmentation & Anomaly Benchmark")
            if metrics is not None:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("IoU (Jaccard Index)", f"{metrics['iou'] * 100:.2f}%")
                with m2:
                    st.metric("Precision (Evidence Purity)", f"{metrics['precision'] * 100:.2f}%")
                with m3:
                    st.metric("Recall (Completeness)", f"{metrics['recall'] * 100:.2f}%")
                with m4:
                    st.metric("F1-Score (Dice Coeff)", f"{metrics['f1'] * 100:.2f}%")

                st.markdown("#### Visual Mask Comparison")
                fig_eval, axes_eval = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
                axes_eval[0].imshow(rgb_img)
                axes_eval[0].set_title("Input Document Image", fontweight="bold")
                axes_eval[0].axis("off")

                axes_eval[1].imshow(pred_mask, cmap="gray", vmin=0, vmax=255)
                axes_eval[1].set_title(f"Predicted Mask\n(IoU: {metrics['iou']*100:.2f}%)", fontweight="bold")
                axes_eval[1].axis("off")

                axes_eval[2].imshow(gt_resized, cmap="gray", vmin=0, vmax=255)
                axes_eval[2].set_title("Ground Truth Mask", fontweight="bold")
                axes_eval[2].axis("off")
                plt.tight_layout()
                st.pyplot(fig_eval)

                st.markdown("#### Pixel Confusion Matrix")
                st.table({
                    "Component": ["True Positives (TP)", "False Positives (FP)", "False Negatives (FN)", "True Negatives (TN)"],
                    "Pixel Count": [f"{metrics['tp']:,}", f"{metrics['fp']:,}", f"{metrics['fn']:,}", f"{metrics['tn']:,}"],
                    "Significance": [
                        "Correctly identified forged pixels",
                        "Authentic document background mistakenly flagged",
                        "Forged pixels missed by the detector",
                        "Authentic background correctly preserved"
                    ]
                })
            else:
                st.info("ℹ️ Upload a Ground-Truth Binary Mask in the top section to compute IoU, Precision, Recall, and Confusion Matrix.")

        # TAB 4: Mathematical Theory
        with tab4:
            st.subheader("Classical Computer Vision Foundations")
            st.markdown(r"""
            ### 1. Scale-Space Extrema (Difference of Gaussians)
            To detect keypoints invariant to scaling, the image is convolved with Gaussian kernels across multiple octaves:
            $$D(x, y, \sigma) = (G(x, y, k\sigma) - G(x, y, \sigma)) * I(x, y)$$
            Keypoints are localized at 3D extrema across scale-space $(x, y, \sigma)$.

            ### 2. Lowe's Ratio Test in Feature Space ($\mathbb{R}^{128}$)
            When matching an image against itself with $k=3$ nearest neighbors ($k_1$ is identity, $k_2$ is candidate, $k_3$ is background):
            $$\frac{\|\mathbf{f}_{\text{query}} - \mathbf{f}_{k_2}\|_2}{\|\mathbf{f}_{\text{query}} - \mathbf{f}_{k_3}\|_2} < \tau \quad (\tau \approx 0.72)$$

            ### 3. Canonical Half-Plane Normalization
            Antipodal vectors between cloned regions ($A \to B$ vs $B \to A$) are normalized into the positive half-plane:
            $$\text{if } \Delta x < 0 \text{ or } (\Delta x = 0 \land \Delta y < 0): \quad (\Delta x, \Delta y) \leftarrow (-\Delta x, -\Delta y)$$

            ### 4. RANSAC Planar Homography Consensus
            Projective consistency is verified via minimal 4-point sample consensus:
            $$s \begin{bmatrix} x_{\text{dst}} \\ y_{\text{dst}} \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x_{\text{src}} \\ y_{\text{src}} \\ 1 \end{bmatrix}$$
            """)
else:
    st.info("👆 Please upload a document image or click 'Load Tampered Contract Demo' in the sidebar to begin.")
