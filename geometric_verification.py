"""
Clustering & Geometric Verification Module.
Document Copy-Move Forgery Detection (CMFD) Pipeline - Step 4.

This module applies spatial translation vector clustering (DBSCAN) and projective
geometry verification (RANSAC Homography) to eliminate false-positive feature matches
and isolate duplicated document regions with bounding boxes and semi-transparent overlays.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

# Cross-module imports within the project workspace
try:
    from preprocess import load_and_preprocess_image, _generate_synthetic_document_if_empty
    from feature_extraction import extract_sift_features
    from flann_matching import match_features_to_self
except ImportError:
    from .preprocess import load_and_preprocess_image, _generate_synthetic_document_if_empty
    from .feature_extraction import extract_sift_features
    from .flann_matching import match_features_to_self


# ==============================================================================
# DBSCAN Clustering & RANSAC Geometric Verification
# ==============================================================================
def cluster_and_verify_matches(
    keypoints: Sequence[cv2.KeyPoint],
    valid_matches: Sequence[cv2.DMatch],
    eps: float = 10.0,
    min_samples: int = 4,
    ransac_reproj_threshold: float = 5.0
) -> Tuple[List[cv2.DMatch], Optional[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Groups candidate matches into coherent spatial displacement clusters and verifies
    planar projective geometry using RANSAC Homography.

    ============================================================================
    MATHEMATICAL FOUNDATION: DISPLACEMENT SPACE & VECTOR NORMALIZATION
    ============================================================================
    A copy-move forgery is governed by a rigid or affine spatial displacement:
        p_target = H * p_source

    For translation-dominated operations (common in document stamp/signature copies),
    every keypoint in the cloned region shifts by the SAME constant vector:
        v = (dx, dy) = (x_target - x_source, y_target - y_source)

    THE CRITICAL CMFD DIRECTIONALITY TRICK:
    In self-matching, keypoints A (source) and B (target) can be matched as:
        - Query=A, Train=B  =>  dx = x_B - x_A,  dy = y_B - y_A  (vector v)
        - Query=B, Train=A  =>  dx = x_A - x_B,  dy = y_A - y_B  (vector -v)

    Although both represent the EXACT same physical clone relationship, their
    vectors point in opposite directions (+v vs -v) and lie on opposite sides
    of displacement space R^2. Without normalization, DBSCAN would split them into
    two separate clusters or classify both as sparse noise!

    Canonical Half-Plane Normalization:
    We enforce all displacement vectors to lie in the right half-plane:
        if dx < 0 or (dx == 0 and dy < 0):
            (dx, dy) = (-dx, -dy)
            swap(source, target)  # Maintain consistent source -> target mapping

    DBSCAN IN DISPLACEMENT SPACE:
    - Genuine copy-move matches form a dense hyper-sphere in (dx, dy) space because
      they all share the exact same spatial translation within epsilon tolerance.
    - Random spurious matches across repetitive text (letters 'e', 't', punctuation)
      have wildly dispersed shift vectors and are assigned label -1 (Noise).

    RANSAC HOMOGRAPHY VERIFICATION:
    For the dominant cluster, we solve for the 3x3 homography matrix H:
        s * [x_target, y_target, 1]^T = H * [x_source, y_source, 1]^T
    RANSAC iteratively selects minimal subsets (4 points), estimates H, and counts
    inliers whose Euclidean reprojection error ||p_target - H(p_source)||_2 <= threshold.
    Only points passing RANSAC consensus are retained as confirmed forgery inliers.

    Args:
        keypoints: Sequence of cv2.KeyPoint objects from the image.
        valid_matches: Filtered candidate matches from Step 3 (FLANN).
        eps: DBSCAN neighborhood radius in pixels of displacement space (dx, dy).
        min_samples: Minimum number of samples in a neighborhood to form a core point.
        ransac_reproj_threshold: Maximum allowed reprojection error (in pixels) for RANSAC.

    Returns:
        Tuple:
            - verified_matches: List of cv2.DMatch objects passing both DBSCAN & RANSAC.
            - homography_matrix: 3x3 numpy array (float64) transformation matrix, or None.
            - (src_points, dst_points): Numpy arrays of shape [M, 2] with matched coordinates.
    """
    empty_result = ([], None, (np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)))

    # Guard clause: Homography requires at least 4 non-collinear point correspondences
    if len(valid_matches) < min_samples or len(valid_matches) < 4:
        print(f"[!] Warning: Insufficient matches ({len(valid_matches)}) for verification. Minimum required: 4.")
        return empty_result

    # -------------------------------------------------------------------------
    # 1. Compute & Canonically Normalize Displacement Vectors
    # -------------------------------------------------------------------------
    displacement_vectors = []
    canonical_pairs = []  # Stores (src_pt, dst_pt, match_obj)

    for match in valid_matches:
        p_query = np.array(keypoints[match.queryIdx].pt, dtype=np.float32)
        p_train = np.array(keypoints[match.trainIdx].pt, dtype=np.float32)

        dx = p_train[0] - p_query[0]
        dy = p_train[1] - p_query[1]

        # Vector Normalization: Force vector into canonical right half-plane
        if dx < 0 or (dx == 0 and dy < 0):
            norm_dx = -dx
            norm_dy = -dy
            src_pt = p_train
            dst_pt = p_query
        else:
            norm_dx = dx
            norm_dy = dy
            src_pt = p_query
            dst_pt = p_train

        displacement_vectors.append([norm_dx, norm_dy])
        canonical_pairs.append((src_pt, dst_pt, match))

    vectors_array = np.array(displacement_vectors, dtype=np.float32)

    # -------------------------------------------------------------------------
    # 2. DBSCAN Spatial Clustering
    # -------------------------------------------------------------------------
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    cluster_labels = dbscan.fit_predict(vectors_array)

    # Filter out noise (label == -1) and find all valid clusters
    unique_labels = set(cluster_labels)
    valid_cluster_labels = [lbl for lbl in unique_labels if lbl != -1]

    if not valid_cluster_labels:
        print("[!] DBSCAN: No coherent displacement clusters found. All matches classified as noise (-1).")
        return empty_result

    # Identify the largest cluster (the dominant copy-move operation)
    largest_cluster_label = max(
        valid_cluster_labels,
        key=lambda lbl: np.sum(cluster_labels == lbl)
    )
    cluster_inlier_indices = np.where(cluster_labels == largest_cluster_label)[0]
    print(f"[*] DBSCAN: Identified largest cluster #{largest_cluster_label} with {len(cluster_inlier_indices)} matches.")

    # Extract coordinates and matches for the largest cluster
    clustered_src = np.array([canonical_pairs[i][0] for i in cluster_inlier_indices], dtype=np.float32)
    clustered_dst = np.array([canonical_pairs[i][1] for i in cluster_inlier_indices], dtype=np.float32)
    clustered_matches = [canonical_pairs[i][2] for i in cluster_inlier_indices]

    if len(clustered_src) < 4:
        print("[!] Warning: Largest cluster has fewer than 4 points. Cannot compute RANSAC homography.")
        return empty_result

    # -------------------------------------------------------------------------
    # 3. RANSAC Geometric Homography Verification
    # -------------------------------------------------------------------------
    # cv2.findHomography computes H such that: dst ~ H * src
    homography_matrix, inlier_mask = cv2.findHomography(
        srcPoints=clustered_src,
        dstPoints=clustered_dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold
    )

    if homography_matrix is None or inlier_mask is None:
        print("[!] RANSAC: Homography computation failed (singular matrix or degenerate configuration).")
        return empty_result

    # Flatten inlier mask (1 = inlier, 0 = outlier)
    inlier_mask_flat = inlier_mask.ravel().astype(bool)
    verified_matches = [m for m, is_inlier in zip(clustered_matches, inlier_mask_flat) if is_inlier]
    verified_src = clustered_src[inlier_mask_flat]
    verified_dst = clustered_dst[inlier_mask_flat]

    print(f"[+] RANSAC: Verified {len(verified_matches)} inliers from {len(clustered_matches)} clustered candidates.")
    return verified_matches, homography_matrix, (verified_src, verified_dst)


# ==============================================================================
# Visualization & Bounding Box Extraction Module
# ==============================================================================
def visualize_forgery_regions(
    rgb_image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    verified_matches: Sequence[cv2.DMatch],
    keypoints: Optional[Sequence[cv2.KeyPoint]] = None,
    padding: int = 12,
    alpha: float = 0.35,
    title: str = "Document Copy-Move Forgery: Forensic Localization",
    show_plot: bool = True
) -> np.ndarray:
    """
    Renders bounding boxes and semi-transparent highlight overlays around the detected
    source and duplicated target regions, connected by verified displacement vectors.

    Args:
        rgb_image: 3-channel original RGB image (uint8, shape [H, W, 3]).
        src_points: Nx2 array of verified source coordinates (float32).
        dst_points: Nx2 array of verified target coordinates (float32).
        verified_matches: Sequence of RANSAC-verified cv2.DMatch objects.
        keypoints: Optional list of all keypoints (for vector line rendering).
        padding: Pixel padding applied around the calculated bounding boxes.
        alpha: Opacity factor for the semi-transparent red overlay (0.0 to 1.0).
        title: Matplotlib figure title.
        show_plot: Whether to call plt.show() immediately.

    Returns:
        np.ndarray: Annotated RGB image array with bounding boxes and overlays.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"[!] rgb_image must be a 3-channel RGB image. Got shape: {rgb_image.shape}")

    h, w, _ = rgb_image.shape
    base_image = rgb_image.copy()

    if len(src_points) == 0 or len(dst_points) == 0:
        print("[!] No verified inliers to visualize. Rendering raw image with warning banner.")
        plt.figure(figsize=(12, 8), dpi=120)
        plt.imshow(base_image)
        plt.title(f"{title} (NO FORGERY VERIFIED)", fontsize=13, fontweight="bold", color="red")
        plt.axis("off")
        if show_plot:
            plt.show()
        return base_image

    # 1. Compute Axis-Aligned Bounding Boxes with Bounds Clamping
    def get_clamped_bbox(pts: np.ndarray) -> Tuple[int, int, int, int]:
        bx, by, bw, bh = cv2.boundingRect(pts.astype(np.int32))
        x1 = max(0, bx - padding)
        y1 = max(0, by - padding)
        x2 = min(w - 1, bx + bw + padding)
        y2 = min(h - 1, by + bh + padding)
        return x1, y1, x2 - x1, y2 - y1

    src_x, src_y, src_w, src_h = get_clamped_bbox(src_points)
    dst_x, dst_y, dst_w, dst_h = get_clamped_bbox(dst_points)

    # 2. Render Semi-Transparent Red Overlays
    overlay = base_image.copy()
    # Color: Bright Red in RGB (255, 0, 0)
    highlight_color = (255, 20, 20)

    # Draw filled solid rectangles on overlay layer
    cv2.rectangle(overlay, (src_x, src_y), (src_x + src_w, src_y + src_h), highlight_color, thickness=-1)
    cv2.rectangle(overlay, (dst_x, dst_y), (dst_x + dst_w, dst_y + dst_h), highlight_color, thickness=-1)

    # Blend overlay with base image: Output = base * (1 - alpha) + overlay * alpha
    annotated = cv2.addWeighted(overlay, alpha, base_image, 1.0 - alpha, 0)

    # 3. Draw Bold Bounding Box Borders
    border_color = (220, 0, 0)  # Deep red border
    cv2.rectangle(annotated, (src_x, src_y), (src_x + src_w, src_y + src_h), border_color, thickness=3)
    cv2.rectangle(annotated, (dst_x, dst_y), (dst_x + dst_w, dst_y + dst_h), border_color, thickness=3)

    # 4. Add Region Identification Badges
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated, "SOURCE / CLONE REGION A", (src_x, max(25, src_y - 8)), font, 0.65, (255, 255, 255), 3)
    cv2.putText(annotated, "SOURCE / CLONE REGION A", (src_x, max(25, src_y - 8)), font, 0.65, (200, 0, 0), 2)

    cv2.putText(annotated, "TARGET / CLONE REGION B", (dst_x, max(25, dst_y - 8)), font, 0.65, (255, 255, 255), 3)
    cv2.putText(annotated, "TARGET / CLONE REGION B", (dst_x, max(25, dst_y - 8)), font, 0.65, (200, 0, 0), 2)

    # 5. Draw Inlier Vectors Between Matched Coordinates
    for sp, dp in zip(src_points, dst_points):
        p1 = (int(round(sp[0])), int(round(sp[1])))
        p2 = (int(round(dp[0])), int(round(dp[1])))
        # Draw displacement vector
        cv2.line(annotated, p1, p2, (0, 255, 255), thickness=1, lineType=cv2.LINE_AA)  # Yellow vector line
        cv2.circle(annotated, p1, radius=3, color=(0, 255, 0), thickness=-1)            # Green source dot
        cv2.circle(annotated, p2, radius=3, color=(0, 255, 0), thickness=-1)            # Green target dot

    # Render with Matplotlib
    plt.figure(figsize=(14, 10), dpi=120)
    plt.imshow(annotated)
    plt.title(
        f"{title}\n"
        f"RANSAC Verified Inliers: {len(verified_matches):,} | "
        f"Region A Box: [{src_w}x{src_h}] at ({src_x}, {src_y}) | "
        f"Region B Box: [{dst_w}x{dst_h}] at ({dst_x}, {dst_y})",
        fontsize=12,
        fontweight="bold"
    )
    plt.axis("off")
    plt.tight_layout()

    if show_plot:
        plt.show()

    return annotated


# ==============================================================================
# Full Pipeline Testing & Integration Block
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(" CMFD Pipeline - Step 4: Full Pipeline & Geometric Verification")
    print("=" * 80)

    # 1. Resolve Sample Image
    candidate_dirs = [
        Path(r"D:\Code\copy_move_detection\data\original"),
        Path(__file__).resolve().parent / "data" / "original",
        Path(__file__).resolve().parent
    ]

    sample_file: Optional[Path] = None
    for directory in candidate_dirs:
        if directory.exists():
            forged_images = list(directory.glob("*_F*.png")) + list(directory.glob("*_F*.jpg"))
            if forged_images:
                sample_file = forged_images[0]
                break
            all_images = list(directory.glob("*.png")) + list(directory.glob("*.jpg"))
            if all_images:
                sample_file = all_images[0]
                break

    if sample_file is not None and sample_file.is_file():
        print(f"[+] Ingesting dataset image: {sample_file.name}")
    else:
        fallback_path = Path(__file__).resolve().parent / "data" / "original" / "synthetic_tampered_doc.png"
        sample_file = _generate_synthetic_document_if_empty(fallback_path)
        print(f"[+] Ingesting synthetic test document: {sample_file.name}")

    # Stage 1: Preprocessing
    print("\n--- [Stage 1/5: Image Preprocessing] ---")
    preprocess_result = load_and_preprocess_image(sample_file)
    rgb_img, gray_img = preprocess_result[0], preprocess_result[1]
    scale_factor = preprocess_result[2] if len(preprocess_result) > 2 else 1.0
    print(f"[+] Loaded image ({rgb_img.shape[1]}x{rgb_img.shape[0]} px, 3 channels)")

    # Stage 2: SIFT Feature Extraction
    print("\n--- [Stage 2/5: SIFT Feature Extraction] ---")
    keypoints, descriptors = extract_sift_features(
        gray_image=gray_img,
        contrast_threshold=0.03,
        edge_threshold=10.0,
        sigma=1.6
    )
    total_keypoints = len(keypoints)
    print(f"[+] Detected {total_keypoints:,} total keypoints across scale-space octaves.")

    # Stage 3: FLANN Self-Matching & Distance Filtering
    print("\n--- [Stage 3/5: FLANN Self-Matching (k=3) & Distance Filtering] ---")
    valid_matches = match_features_to_self(
        keypoints=keypoints,
        descriptors=descriptors,
        ratio_threshold=0.72,
        spatial_distance_threshold=45.0
    )
    total_flann_matches = len(valid_matches)
    print(f"[+] Retained {total_flann_matches:,} candidate matches after Lowe's ratio & spatial filters.")

    # Stage 4: Displacement Vector Clustering & RANSAC Verification
    print("\n--- [Stage 4/5: DBSCAN Clustering & RANSAC Homography Verification] ---")
    verified_matches, homography_matrix, (src_pts, dst_pts) = cluster_and_verify_matches(
        keypoints=keypoints,
        valid_matches=valid_matches,
        eps=12.0,                  # Radius tolerance for translation shifts in pixels
        min_samples=4,             # Minimum points to constitute a cluster
        ransac_reproj_threshold=5.0 # Max reprojection error in pixels
    )

    # Stage 5: Summary & Forensic Localization
    print("\n" + "=" * 50)
    print(" CMFD FORENSIC PIPELINE SUMMARY REPORT")
    print("=" * 50)
    print(f"  1. Original Keypoints Detected  : {total_keypoints:,}")
    print(f"  2. FLANN Initial Matches        : {total_flann_matches:,}")
    print(f"  3. Final Verified Inliers       : {len(verified_matches):,}")

    if homography_matrix is not None:
        print("\n[+] Estimated Transformation Matrix (Homography H):")
        for row in homography_matrix:
            print(f"    [ {row[0]:+10.4f}  {row[1]:+10.4f}  {row[2]:+10.4f} ]")

        # Derive approximate translation vector
        tx, ty = homography_matrix[0, 2], homography_matrix[1, 2]
        print(f"\n[+] Estimated Forgery Translation Shift: tx={tx:+.1f}px, ty={ty:+.1f}px")
        print(f"[+] Total Translation Distance         : {np.sqrt(tx**2 + ty**2):.1f} pixels")
    else:
        print("\n[-] Forgery Status: Clean document (or insufficient geometric consistency).")

    print("=" * 50 + "\n")

    # Stage 6: Visualization
    print("[*] Launching forensic localization viewer with bounding box overlay...")
    visualize_forgery_regions(
        rgb_image=rgb_img,
        src_points=src_pts,
        dst_points=dst_pts,
        verified_matches=verified_matches,
        keypoints=keypoints,
        title=f"Forensic Copy-Move Localization: {sample_file.name}"
    )

