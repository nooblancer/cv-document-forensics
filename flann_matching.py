"""
FLANN-Based Self-Matching & Spatial Filtering Module.
Document Copy-Move Forgery Detection (CMFD) Pipeline - Step 3.

This module implements Fast Library for Approximate Nearest Neighbors (FLANN)
self-matching on 128-dimensional SIFT descriptors. It applies a dual-stage filter:
1. Lowe's Ratio Test (k=2 vs. k=3) in high-dimensional descriptor space.
2. Euclidean Spatial Distance Filtering in physical 2D image coordinate space.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cross-module imports within the project workspace
try:
    from preprocess import load_and_preprocess_image, _generate_synthetic_document_if_empty
    from feature_extraction import extract_sift_features
except ImportError:
    from .preprocess import load_and_preprocess_image, _generate_synthetic_document_if_empty
    from .feature_extraction import extract_sift_features


# ==============================================================================
# FLANN Self-Matching Engine
# ==============================================================================
def match_features_to_self(
    keypoints: Sequence[cv2.KeyPoint],
    descriptors: Optional[np.ndarray],
    ratio_threshold: float = 0.7,
    spatial_distance_threshold: float = 50.0
) -> List[cv2.DMatch]:
    """
    Performs feature self-matching using FLANN and filters candidate forgery pairs
    via Lowe's ratio test and physical Euclidean distance constraints.

    ============================================================================
    MATHEMATICAL & ALGORITHMIC FOUNDATION:
    ============================================================================
    In traditional two-image matching (Image A -> Image B), we query descriptors of
    Image A against an index of Image B with k=2 to find the best match (k=1) and
    second-best match (k=2).

    In Copy-Move Forgery Detection (CMFD), the image is matched AGAINST ITSELF:
        Query Set: Q = {f_1, f_2, ..., f_N}
        Index Set: T = {f_1, f_2, ..., f_N}  (where T == Q)

    1. Why k=3 in Self-Matching?
       When querying descriptor f_i against the database containing f_i:
       - k=1 (d_1 = 0.0): The descriptor matches itself trivially (queryIdx == trainIdx).
       - k=2 (d_2 > 0.0): The nearest DISTINCT neighbor in 128-D descriptor space.
                          This is the candidate copy-move clone pair!
       - k=3 (d_3 >= d_2): The second-nearest distinct neighbor, representing the
                           closest competing background feature hypothesis.

    2. Lowe's Ratio Test in Self-Matching:
       David Lowe demonstrated that the probability density of distance ratios for
       false matches peaks near 1.0, while for genuine matches it is significantly lower.
       Therefore, the candidate pair is accepted only if:
           d_2 < ratio_threshold * d_3   (where ratio_threshold in [0.6, 0.8])
       This rejects ambiguous matches caused by repetitive typography (e.g., repeated
       letters 'e', 'a', dots, or line grid markers).

    3. Physical Euclidean Distance Constraint (Spatial Filter):
       Natural textures, circular stamps, and dense text characters exhibit high local
       self-similarity. Without spatial filtering, adjacent keypoints within the SAME
       stamp or character stroke will match each other.
       
       Let p_i = (x_i, y_i) and p_j = (x_j, y_j) be the coordinates of keypoints i and j:
           D_spatial(p_i, p_j) = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
       
       A forged clone must have been copied and pasted to a spatially separated region:
           D_spatial(p_i, p_j) > spatial_distance_threshold
       Any match violating this constraint is discarded as an intra-patch artifact.

    4. Bidirectional Deduplication:
       If keypoint i matches keypoint j, keypoint j will often match keypoint i.
       We enforce queryIdx < trainIdx to preserve unique undirected edge connections.

    Args:
        keypoints: Sequence of cv2.KeyPoint objects extracted from the image.
        descriptors: 2D numpy array of shape [N, 128] with dtype float32.
        ratio_threshold: Lowe's ratio test threshold (typically between 0.6 and 0.8).
        spatial_distance_threshold: Minimum physical Euclidean pixel distance required
            between two keypoints to be considered a valid copy-move pair.

    Returns:
        List[cv2.DMatch]: Filtered list of unique cv2.DMatch objects representing
        confirmed candidate clone pairs.
    """
    # -------------------------------------------------------------------------
    # Guard Clauses & Validation
    # -------------------------------------------------------------------------
    if descriptors is None:
        print("[!] Warning: Descriptors array is None. Returning empty match list.")
        return []

    num_descriptors = descriptors.shape[0]
    if num_descriptors < 3:
        print(
            f"[!] Warning: Insufficient descriptors ({num_descriptors}) for k=3 matching. "
            f"At least 3 keypoints are required. Returning empty match list."
        )
        return []

    if len(keypoints) != num_descriptors:
        raise ValueError(
            f"[!] Mismatch: {len(keypoints)} keypoints provided for {num_descriptors} descriptors."
        )

    # Enforce float32 dtype required by FLANN KD-Tree indexer
    if descriptors.dtype != np.float32:
        descriptors = descriptors.astype(np.float32)

    # -------------------------------------------------------------------------
    # FLANN Index Construction
    # -------------------------------------------------------------------------
    # Algorithm 1 = FLANN_INDEX_KDTREE
    # trees=5 provides a balanced trade-off between build time and search precision
    index_params = dict(algorithm=1, trees=5)
    
    # checks=50 specifies the number of times trees in the index are traversed.
    # Higher values give higher precision at the expense of search time.
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Query the descriptor set against itself with k=3
    # raw_matches is a list of lists of cv2.DMatch objects
    raw_matches = flann.knnMatch(queryDescriptors=descriptors, trainDescriptors=descriptors, k=3)

    # -------------------------------------------------------------------------
    # Dual-Stage Filtering Pipeline
    # -------------------------------------------------------------------------
    valid_matches: List[cv2.DMatch] = []
    seen_pairs = set()

    for match_group in raw_matches:
        # Verify that all 3 nearest neighbors were successfully returned
        if len(match_group) < 3:
            continue

        m1, m2, m3 = match_group
        # m1: query descriptor matching itself (m1.queryIdx == m1.trainIdx, distance ~ 0)
        # m2: best distinct descriptor match (candidate clone)
        # m3: second best distinct descriptor match (Lowe's comparator)

        # Sanity check: Ensure m2 is not matching itself
        if m2.queryIdx == m2.trainIdx:
            # Fallback if m1 wasn't self due to approximation: use m3 as candidate if valid
            continue

        # Stage 1: Lowe's Ratio Test in 128-D Feature Space
        # d(f_query, f_best) < ratio_thresh * d(f_query, f_second_best)
        if m2.distance >= ratio_threshold * m3.distance:
            continue

        # Stage 2: Physical Spatial Euclidean Distance Filter
        pt_source = keypoints[m2.queryIdx].pt
        pt_target = keypoints[m2.trainIdx].pt

        dx = pt_source[0] - pt_target[0]
        dy = pt_source[1] - pt_target[1]
        spatial_dist = np.sqrt(dx * dx + dy * dy)

        if spatial_dist <= spatial_distance_threshold:
            # Discard intra-patch self-similarity (keypoints too close physically)
            continue

        # Stage 3: Undirected Pair Deduplication
        # Avoid storing both (i -> j) and (j -> i)
        idx_pair = (min(m2.queryIdx, m2.trainIdx), max(m2.queryIdx, m2.trainIdx))
        if idx_pair in seen_pairs:
            continue
        seen_pairs.add(idx_pair)

        valid_matches.append(m2)

    return valid_matches


# ==============================================================================
# Visualization Module (Single-Canvas Overlay)
# ==============================================================================
def visualize_matches(
    rgb_image: np.ndarray,
    keypoints: Sequence[cv2.KeyPoint],
    valid_matches: Sequence[cv2.DMatch],
    title: str = "FLANN Copy-Move Candidate Matches",
    line_color: Tuple[int, int, int] = (255, 69, 0),    # Red-Orange vector lines
    point_color: Tuple[int, int, int] = (0, 255, 255),  # Cyan endpoint markers
    line_thickness: int = 2,
    point_radius: int = 4,
    show_plot: bool = True
) -> np.ndarray:
    """
    Renders copy-move clone pairs directly on a single copy of the original image.

    Note: We do NOT use cv2.drawMatches because that function places two separate images
    side-by-side (designed for stereo vision or object recognition across different scenes).
    In CMFD, we must draw vector connections between source and target regions on the SAME canvas.

    Args:
        rgb_image: 3-channel RGB image (uint8, shape [H, W, 3]).
        keypoints: Sequence of cv2.KeyPoint objects.
        valid_matches: Filtered list of cv2.DMatch objects.
        title: Descriptive title for the Matplotlib visualization.
        line_color: RGB tuple for connecting lines between matched keypoints.
        point_color: RGB tuple for keypoint circles.
        line_thickness: Pixel thickness of connecting lines.
        point_radius: Radius of endpoint circles in pixels.
        show_plot: Whether to immediately render via plt.show().

    Returns:
        np.ndarray: Annotated RGB image array.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"[!] rgb_image must be a 3-channel RGB image. Got shape: {rgb_image.shape}")

    # Create a deep copy to preserve the original image unmodified
    annotated_image = rgb_image.copy()

    # Draw each verified clone vector
    for match in valid_matches:
        pt_query = keypoints[match.queryIdx].pt
        pt_train = keypoints[match.trainIdx].pt

        # Convert floating-point coordinates to integer tuples for OpenCV rendering
        p1 = (int(round(pt_query[0])), int(round(pt_query[1])))
        p2 = (int(round(pt_train[0])), int(round(pt_train[1])))

        # 1. Draw connection line representing the copy-move displacement vector
        cv2.line(
            annotated_image,
            pt1=p1,
            pt2=p2,
            color=line_color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA
        )

        # 2. Draw source and target keypoint markers
        cv2.circle(annotated_image, center=p1, radius=point_radius, color=point_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(annotated_image, center=p2, radius=point_radius, color=point_color, thickness=-1, lineType=cv2.LINE_AA)

    # Render with Matplotlib
    plt.figure(figsize=(12, 10), dpi=120)
    plt.imshow(annotated_image)
    plt.title(f"{title}\nFiltered Clone Matches: {len(valid_matches):,}", fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    if show_plot:
        plt.show()

    return annotated_image


# ==============================================================================
# Testing & Verification Entrypoint
# ==============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print(" CMFD Pipeline - Step 3: FLANN Self-Matching & Distance Filtering")
    print("=" * 75)

    # 1. Resolve candidate paths for sample forged document
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
        print(f"[+] Located test image: {sample_file}")
    else:
        # Fallback to reproducible synthetic forged document
        fallback_path = Path(__file__).resolve().parent / "data" / "original" / "synthetic_tampered_doc.png"
        sample_file = _generate_synthetic_document_if_empty(fallback_path)
        print(f"[+] Using synthetic forged test image at: {sample_file}")

    # 2. Preprocess & Extract Features
    print("\n[*] Ingesting and preprocessing image...")
    preprocess_result = load_and_preprocess_image(sample_file)
    rgb_img, gray_img = preprocess_result[0], preprocess_result[1]
    scale_factor = preprocess_result[2] if len(preprocess_result) > 2 else 1.0

    print("[*] Extracting SIFT keypoints and descriptors...")
    keypoints, descriptors = extract_sift_features(
        gray_image=gray_img,
        contrast_threshold=0.03,
        edge_threshold=10.0,
        sigma=1.6
    )

    total_keypoints = len(keypoints)
    print(f"\n[+] Total Keypoints Detected: {total_keypoints:,}")

    # 3. Execute FLANN Self-Matching & Filtering
    print("[*] Executing FLANN Self-Matching (k=3) with Lowe's Ratio & Spatial Distance Filters...")
    ratio_thresh = 0.70
    spatial_dist_thresh = 50.0  # Minimum 50 pixels physical separation

    valid_matches = match_features_to_self(
        keypoints=keypoints,
        descriptors=descriptors,
        ratio_threshold=ratio_thresh,
        spatial_distance_threshold=spatial_dist_thresh
    )

    total_matches = len(valid_matches)
    print(f"[+] Filtering Summary:")
    print(f"    - Input Keypoints             : {total_keypoints:,}")
    print(f"    - Lowe's Ratio Threshold (tau): {ratio_thresh}")
    print(f"    - Spatial Distance Threshold  : {spatial_dist_thresh} px")
    print(f"    - Final Valid Matches Retained: {total_matches:,}")

    # Diagnostic inspection of first 5 matches
    if total_matches > 0:
        print("\n--- Sample Matched Clone Pairs ---")
        for i, m in enumerate(valid_matches[:5]):
            p1 = keypoints[m.queryIdx].pt
            p2 = keypoints[m.trainIdx].pt
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            dist = np.sqrt(dx * dx + dy * dy)
            print(
                f"  Pair #{i+1:02d}: "
                f"QueryKeypoint[{m.queryIdx:4d}] at ({p1[0]:.1f}, {p1[1]:.1f}) <---> "
                f"TrainKeypoint[{m.trainIdx:4d}] at ({p2[0]:.1f}, {p2[1]:.1f}) | "
                f"Descriptor Dist={m.distance:.4f} | "
                f"Spatial Separation={dist:.1f}px"
            )
        print("-----------------------------------\n")
    else:
        print("[!] No matches passed the combined Lowe's ratio and spatial distance thresholds.")

    # 4. Single-Canvas Clone Pair Visualization
    print("[*] Visualizing clone vectors on single document canvas...")
    visualize_matches(
        rgb_image=rgb_img,
        keypoints=keypoints,
        valid_matches=valid_matches,
        title=f"CMFD Candidate Clone Vectors: {sample_file.name}"
    )

