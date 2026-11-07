"""
Production Document Copy-Move Forgery Detection (CMFD) Pipeline - CLI Orchestrator.
Main entry point for unified execution: Preprocessing, Feature Extraction (Sparse/Dense),
FLANN Self-Matching, DBSCAN & RANSAC Verification, and Pixel-Level Metric Evaluation.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cross-module pipeline imports
from preprocess import load_and_preprocess_image
from feature_extraction import extract_sift_features, visualize_keypoints
from flann_matching import match_features_to_self, visualize_matches
from geometric_verification import cluster_and_verify_matches, visualize_forgery_regions
from evaluation_metrics import (
    generate_predicted_mask,
    calculate_cmfd_metrics,
    visualize_masks_side_by_side
)


# ==============================================================================
# CLI Argument Parser Setup
# ==============================================================================
def parse_arguments() -> argparse.Namespace:
    """Configures command-line options for production document analysis."""
    parser = argparse.ArgumentParser(
        prog="CMFD-Detector",
        description="Classical Computer Vision Pipeline for Document Copy-Move Forgery Detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core I/O
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input document image file (*.png, *.jpg, *.tif, *.bmp)."
    )
    parser.add_argument(
        "-m", "--mask",
        type=str,
        default=None,
        help="Optional path to binary ground-truth forgery mask for evaluation."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional output image path to save forensic localization results."
    )
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        help="Flag to display interactive Matplotlib visualization windows."
    )

    # Architectural Tuning Parameters
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1500,
        help="Maximum longest-edge pixel dimension before INTER_AREA decimation."
    )
    parser.add_argument(
        "--min-density",
        type=int,
        default=500,
        help="Keypoint count threshold below which Dense SIFT fallback is triggered."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.72,
        help="Lowe's ratio test threshold (k=2 vs k=3) in 128-D descriptor space."
    )
    parser.add_argument(
        "--spatial-dist",
        type=float,
        default=45.0,
        help="Minimum Euclidean pixel separation required between matched keypoints."
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=12.0,
        help="DBSCAN displacement vector epsilon neighborhood (in pixels)."
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=4,
        help="DBSCAN minimum samples required to establish a core cluster."
    )
    parser.add_argument(
        "--reproj-thresh",
        type=float,
        default=5.0,
        help="RANSAC maximum reprojection error in pixels for planar homography."
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=25,
        help="Kernel size for morphological closing during mask rasterization."
    )

    return parser.parse_args()


# ==============================================================================
# Pipeline Execution Engine
# ==============================================================================
def run_cmfd_pipeline(args: argparse.Namespace) -> int:
    """
    Executes the end-to-end 5-stage CMFD pipeline:
    1. Preprocessing & Megapixel Decimation
    2. Scale-Space SIFT Extraction with Dense SIFT Fallback
    3. FLANN Self-Matching (k=3) & Distance Filtering
    4. Displacement Vector Normalization, DBSCAN & RANSAC Homography
    5. Morphological Closing Mask Generation & Metric Evaluation
    """
    start_time = time.perf_counter()

    print("=" * 80)
    print(" DOCUMENT COPY-MOVE FORGERY DETECTION (CMFD) - PRODUCTION RUNNER")
    print("=" * 80)

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"[!] Error: Target document not found at: {input_path}")
        return 1

    print(f"[*] Processing Document: {input_path.name}")
    print(f"[*] Full System Path   : {input_path}")

    # -------------------------------------------------------------------------
    # STAGE 1: Image Ingestion & Megapixel Decimation
    # -------------------------------------------------------------------------
    print("\n--- [Stage 1/5: Ingestion & Preprocessing] ---")
    t0 = time.perf_counter()
    rgb_img, gray_img, scale_factor = load_and_preprocess_image(
        image_path=input_path,
        max_dim=args.max_dim,
        gaussian_kernel=(3, 3),
        gaussian_sigma=0.5
    )
    h, w, c = rgb_img.shape
    print(
        f"[+] Loaded image ({w}x{h} px, {c} channels) "
        f"[Scale factor: {scale_factor:.4f}] in {(time.perf_counter() - t0):.3f}s"
    )

    # -------------------------------------------------------------------------
    # STAGE 2: Keypoint Detection & Feature Extraction (with Dense Fallback)
    # -------------------------------------------------------------------------
    print("\n--- [Stage 2/5: SIFT Feature Extraction] ---")
    t0 = time.perf_counter()
    keypoints, descriptors = extract_sift_features(
        gray_image=gray_img,
        contrast_threshold=0.03,
        edge_threshold=10.0,
        sigma=1.6,
        n_features=0,
        min_density_threshold=args.min_density,
        dense_grid_step=10,
        dense_keypoint_size=16.0
    )
    total_kp = len(keypoints)
    print(f"[+] Extracted {total_kp:,} keypoints in {(time.perf_counter() - t0):.3f}s")

    if total_kp < 4 or descriptors is None:
        print("[!] Insufficient keypoints extracted for self-matching. Exiting cleanly.")
        return 0

    # -------------------------------------------------------------------------
    # STAGE 3: FLANN Self-Matching (k=3) & Distance Filtering
    # -------------------------------------------------------------------------
    print("\n--- [Stage 3/5: FLANN Self-Matching & Distance Filtering] ---")
    t0 = time.perf_counter()
    valid_matches = match_features_to_self(
        keypoints=keypoints,
        descriptors=descriptors,
        ratio_threshold=args.ratio,
        spatial_distance_threshold=args.spatial_dist
    )
    total_matches = len(valid_matches)
    print(
        f"[+] FLANN Matcher retained {total_matches:,} candidate clone pairs "
        f"(Ratio < {args.ratio}, Dist > {args.spatial_dist}px) in {(time.perf_counter() - t0):.3f}s"
    )

    # -------------------------------------------------------------------------
    # STAGE 4: Displacement Clustering & Geometric Homography Verification
    # -------------------------------------------------------------------------
    print("\n--- [Stage 4/5: DBSCAN Clustering & RANSAC Geometric Verification] ---")
    t0 = time.perf_counter()
    verified_matches, homography, (src_pts, dst_pts) = cluster_and_verify_matches(
        keypoints=keypoints,
        valid_matches=valid_matches,
        eps=args.eps,
        min_samples=args.min_samples,
        ransac_reproj_threshold=args.reproj_thresh
    )
    total_inliers = len(verified_matches)
    print(f"[+] Verification completed in {(time.perf_counter() - t0):.3f}s")

    is_forged = total_inliers >= args.min_samples and homography is not None

    # -------------------------------------------------------------------------
    # STAGE 5: Forensic Mask Generation & Metric Evaluation
    # -------------------------------------------------------------------------
    print("\n--- [Stage 5/5: Forensic Localization & Metric Evaluation] ---")
    pred_mask = generate_predicted_mask(
        image_shape=(h, w),
        src_points=src_pts,
        dst_points=dst_pts,
        point_radius=8,
        close_kernel_size=args.close_kernel
    )

    # Optional Ground Truth Evaluation
    metrics = None
    gt_mask_raw = None
    if args.mask is not None:
        mask_path = Path(args.mask).resolve()
        if mask_path.is_file():
            gt_mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask_raw is not None:
                if gt_mask_raw.shape != (h, w):
                    gt_mask_raw = cv2.resize(gt_mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
                metrics = calculate_cmfd_metrics(pred_mask, gt_mask_raw)
        else:
            print(f"[!] Warning: Specified ground truth mask was not found: {mask_path}")

    # -------------------------------------------------------------------------
    # FORENSIC REPORT SUMMARY
    # -------------------------------------------------------------------------
    elapsed_total = time.perf_counter() - start_time
    print("\n" + "=" * 65)
    print(" FORENSIC VERDICT & ANALYSIS REPORT")
    print("=" * 65)
    print(f"  Target File             : {input_path.name}")
    print(f"  Execution Time          : {elapsed_total:.3f} seconds")
    print(f"  Decimated Resolution    : {w}x{h} px (s={scale_factor:.4f})")
    print(f"  Extracted Keypoints     : {total_kp:,}")
    print(f"  FLANN Candidate Pairs   : {total_matches:,}")
    print(f"  RANSAC Confirmed Inliers: {total_inliers:,}")
    print("  -------------------------------------------------------------")

    if is_forged:
        print("  >> FORENSIC VERDICT     : [TAMPERED / COPY-MOVE DETECTED] <<")
        tx, ty = homography[0, 2], homography[1, 2]
        shift_norm = np.sqrt(tx**2 + ty**2)
        print(f"     Estimated Translation Shift : dx={tx:+.1f}px, dy={ty:+.1f}px")
        print(f"     Physical Shift Distance     : {shift_norm:.1f} pixels")
    else:
        print("  >> FORENSIC VERDICT     : [AUTHENTIC / NO DUPLICATION DETECTED] <<")

    if metrics is not None:
        print("  -------------------------------------------------------------")
        print("  Ground Truth Evaluation Metrics:")
        print(f"    - Intersection over Union (IoU) : {metrics['iou'] * 100:.2f}%")
        print(f"    - Precision (Evidence Purity)   : {metrics['precision'] * 100:.2f}%")
        print(f"    - Recall (Forgery Completeness) : {metrics['recall'] * 100:.2f}%")
        print(f"    - F1-Score (Dice Coefficient)   : {metrics['f1'] * 100:.2f}%")
    print("=" * 65 + "\n")

    # -------------------------------------------------------------------------
    # Visual Output Rendering & File Saving
    # -------------------------------------------------------------------------
    annotated_output = visualize_forgery_regions(
        rgb_image=rgb_img,
        src_points=src_pts,
        dst_points=dst_pts,
        verified_matches=verified_matches,
        keypoints=keypoints,
        title=f"Forensic Copy-Move Localization: {input_path.name}",
        show_plot=False
    )

    # Save to disk if requested
    if args.output is not None:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert RGB -> BGR for native OpenCV disk write
        cv2.imwrite(str(out_path), cv2.cvtColor(annotated_output, cv2.COLOR_RGB2BGR))
        print(f"[+] Saved forensic result image to: {out_path}")

    # Display interactive GUI plots if flag is present
    if args.visualize:
        if metrics is not None and gt_mask_raw is not None:
            visualize_masks_side_by_side(
                rgb_image=rgb_img,
                predicted_mask=pred_mask,
                ground_truth_mask=gt_mask_raw,
                metrics=metrics,
                title=f"CMFD Evaluation: {input_path.name}",
                show_plot=True
            )
        else:
            plt.figure(figsize=(14, 10), dpi=120)
            plt.imshow(annotated_output)
            plt.title(f"Forensic CMFD Output: {input_path.name}", fontsize=13, fontweight="bold")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

    return 0


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
if __name__ == "__main__":
    cli_args = parse_arguments()
    sys.exit(run_cmfd_pipeline(cli_args))

