"""
SIFT Feature Extraction & Keypoint Visualization Module.
Document Copy-Move Forgery Detection (CMFD) Pipeline - Step 2.

This module extracts Scale-Invariant Feature Transform (SIFT) keypoints and
128-dimensional descriptors from preprocessed document images. It includes a
Dense SIFT fallback mechanism to detect smooth / texture-less forged erasures.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Ensure clean imports whether executed as script or imported as package
try:
    from preprocess import load_and_preprocess_image, _generate_synthetic_document_if_empty
except ImportError:
    from .preprocess import load_and_preprocess_image, _generate_synthetic_document_if_empty


# ==============================================================================
# SIFT Feature Extractor (with Dense SIFT Fallback)
# ==============================================================================
def extract_sift_features(
    gray_image: np.ndarray,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
    n_features: int = 0,
    min_density_threshold: int = 500,
    dense_grid_step: int = 10,
    dense_keypoint_size: float = 16.0
) -> Tuple[Tuple[cv2.KeyPoint, ...], Optional[np.ndarray]]:
    """
    Extracts SIFT keypoints and 128-dimensional local gradient descriptors
    from a single-channel grayscale image.

    Includes an automatic fallback to "Dense SIFT" when standard scale-space extrema
    detection yields insufficient keypoints (< min_density_threshold).

    ============================================================================
    COMPUTER VISION FOUNDATION: THE DENSE SIFT FALLBACK FOR TEXTURE-LESS ERASURE
    ============================================================================
    Standard SIFT is a sparse feature detector: it relies on Difference of Gaussians (DoG)
    to find local scale-space extrema. Extrema require high-frequency gradients (corners,
    edges, text strokes, and stamp contours).

    Forensic Blindspot in Standard Sparse SIFT:
    Attackers frequently perform copy-move forgeries to ERASE authentic information:
    - Example: Copying a blank, texture-less white/cream patch of background paper
      and pasting it directly over a legal clause, signature, or numerical sum.
    - Because the duplicated patch is smooth and devoid of gradients, DoG detects zero
      keypoints inside it. Sparse SIFT is blind to texture-less copy-move erasures!

    Dense SIFT Mechanism:
    When total detected keypoints fall below `min_density_threshold` (e.g., 500), the
    algorithm switches from DoG extrema search to a deterministic spatial lattice:
    1. A regular 2D grid is cast across the document canvas with step size `dense_grid_step`.
    2. Artificial `cv2.KeyPoint` objects with fixed scale `dense_keypoint_size` are populated.
    3. The 128-dimensional gradient orientation histograms are extracted at each grid vertex.
    This guarantees that smooth background erasures, uniform stamps, and watermark erasures
    still produce descriptor clusters for FLANN matching and DBSCAN verification.

    Args:
        gray_image: Single-channel 2D uint8 numpy array of shape [Height, Width].
        contrast_threshold: Threshold to filter out weak, low-contrast candidate extrema.
        edge_threshold: Eigenvalue ratio threshold (r) to filter out edge-like extrema.
        sigma: Standard deviation of Gaussian applied to base scale octave.
        n_features: Number of best features to retain (0 = retain all).
        min_density_threshold: Minimum acceptable keypoint count before triggering Dense SIFT.
        dense_grid_step: Pixel spacing for the Dense SIFT regular grid lattice.
        dense_keypoint_size: Neighborhood diameter for Dense SIFT descriptor sampling.

    Returns:
        Tuple[Tuple[cv2.KeyPoint, ...], Optional[np.ndarray]]:
            - keypoints: Tuple of cv2.KeyPoint objects.
            - descriptors: 2D numpy array [N, 128] with dtype float32 (or None if N=0).
    """
    # 1. Input Integrity Validation
    if not isinstance(gray_image, np.ndarray):
        raise TypeError(f"[!] Expected gray_image as np.ndarray, got {type(gray_image).__name__}")

    if gray_image.ndim != 2:
        raise ValueError(
            f"[!] gray_image must be a 2D single-channel array [H, W]. Received shape: {gray_image.shape}"
        )

    if gray_image.size == 0:
        raise ValueError("[!] gray_image is empty (size is 0).")

    # Enforce uint8 depth as required by OpenCV
    if gray_image.dtype != np.uint8:
        gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)

    # 2. Instantiate SIFT Engine
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        nOctaveLayers=3,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma
    )

    # 3. Sparse SIFT Detection (DoG Extrema)
    keypoints, descriptors = sift.detectAndCompute(gray_image, mask=None)
    kp_count = len(keypoints) if keypoints is not None else 0

    # 4. Density Check & Dense SIFT Fallback for Texture-less Erasures
    if kp_count < min_density_threshold:
        print(
            f"[*] Keypoint Density Alert: Only {kp_count} sparse extrema found "
            f"(threshold: {min_density_threshold})."
        )
        print(
            f"[*] Activating Dense SIFT Fallback (grid_step={dense_grid_step}px, "
            f"scale={dense_keypoint_size}px) to capture potential texture-less erasures..."
        )

        h, w = gray_image.shape
        grid_keypoints: List[cv2.KeyPoint] = []
        margin = max(dense_grid_step, int(dense_keypoint_size // 2))

        # Sample uniform 2D grid
        for y in range(margin, h - margin, dense_grid_step):
            for x in range(margin, w - margin, dense_grid_step):
                # cv2.KeyPoint(x, y, size)
                grid_keypoints.append(
                    cv2.KeyPoint(x=float(x), y=float(y), size=float(dense_keypoint_size))
                )

        # Compute 128-D descriptors over grid keypoints
        keypoints, descriptors = sift.compute(gray_image, grid_keypoints)
        print(f"[+] Dense SIFT Extraction Complete: {len(keypoints):,} grid keypoints populated.")

    if keypoints is None or len(keypoints) == 0:
        print("[!] WARNING: No SIFT keypoints were detected.")
        return (), None

    return tuple(keypoints), descriptors


# ==============================================================================
# Visualization Module
# ==============================================================================
def visualize_keypoints(
    rgb_image: np.ndarray,
    keypoints: Union[List[cv2.KeyPoint], Tuple[cv2.KeyPoint, ...]],
    title: str = "SIFT Keypoint Detection (Rich Keypoints: Scale & Orientation)",
    show_plot: bool = True
) -> np.ndarray:
    """
    Visualizes extracted keypoints over the original RGB image using rich annotations.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"[!] rgb_image must be a 3-channel RGB image. Got shape: {rgb_image.shape}")

    if not keypoints:
        print("[!] Notice: Keypoint list is empty. Displaying raw image without annotations.")
        annotated_rgb = rgb_image.copy()
    else:
        image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Draw rich keypoints with scale diameter and orientation indicator
        annotated_bgr = cv2.drawKeypoints(
            image=image_bgr,
            keypoints=keypoints,
            outImage=None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 10), dpi=120)
    plt.imshow(annotated_rgb)
    plt.title(f"{title}\nTotal Keypoints: {len(keypoints):,}", fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    if show_plot:
        plt.show()

    return annotated_rgb


# ==============================================================================
# Testing & Verification Entrypoint
# ==============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print(" CMFD Pipeline - Step 2: SIFT Feature Extraction & Verification")
    print("=" * 75)

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
        print(f"[+] Located dataset image: {sample_file}")
    else:
        fallback_path = Path(__file__).resolve().parent / "data" / "original" / "synthetic_tampered_doc.png"
        sample_file = _generate_synthetic_document_if_empty(fallback_path)
        print(f"[+] Using synthetic forged test image at: {sample_file}")

    # Stage 1: Load and Preprocess
    preprocess_result = load_and_preprocess_image(sample_file, max_dim=1500)
    rgb_image, gray_image = preprocess_result[0], preprocess_result[1]
    scale_factor = preprocess_result[2] if len(preprocess_result) > 2 else 1.0

    print(f"    - Image Resolution : {rgb_image.shape[1]}x{rgb_image.shape[0]} (WxH)")
    print(f"    - Scale Factor     : {scale_factor:.4f}")

    # Stage 2: Feature Extraction with Dense SIFT check
    keypoints, descriptors = extract_sift_features(
        gray_image=gray_image,
        contrast_threshold=0.03,
        edge_threshold=10.0,
        sigma=1.6,
        min_density_threshold=500
    )

    num_kp = len(keypoints)
    print(f"\n[+] Total Keypoints Detected : {num_kp:,}")

    if descriptors is not None and num_kp > 0:
        print(f"[+] Descriptor Matrix Shape  : {descriptors.shape} (N_keypoints, 128_dimensions)")
        print(f"[+] Descriptor Data Type     : {descriptors.dtype}")

        print("\n--- Sample Keypoint Spatial & Scale Profiles ---")
        for idx in range(min(3, num_kp)):
            kp = keypoints[idx]
            print(
                f"  Keypoint #{idx+1:02d}: "
                f"Coord=(x={kp.pt[0]:.2f}, y={kp.pt[1]:.2f}) | "
                f"Scale={kp.size:.2f}px | "
                f"Angle={kp.angle:.1f}° | "
                f"Response={kp.response:.4f}"
            )
        print("------------------------------------------------\n")

    visualize_keypoints(
        rgb_image=rgb_image,
        keypoints=keypoints,
        title=f"SIFT Rich Keypoints: {sample_file.name}"
    )
