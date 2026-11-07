"""
Evaluation & Metric Calculation Module for Copy-Move Forgery Detection (CMFD).
Document Copy-Move Forgery Detection Pipeline - Step 5.

This module evaluates predicted forgery masks against pixel-level ground truth masks.
It implements a tight Morphological Closing boundary generator (avoiding convex hull
over-segmentation) and computes standard segmentation/forensic metrics:
- Intersection over Union (IoU / Jaccard Index)
- Precision (Positive Predictive Value)
- Recall (True Positive Rate / Sensitivity)
- F1-Score (Dice Similarity Coefficient)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cross-module imports within the project workspace
try:
    from preprocess import load_and_preprocess_image, _generate_synthetic_document_if_empty
    from feature_extraction import extract_sift_features
    from flann_matching import match_features_to_self
    from geometric_verification import cluster_and_verify_matches
except ImportError:
    from .preprocess import load_and_preprocess_image, _generate_synthetic_document_if_empty
    from .feature_extraction import extract_sift_features
    from .flann_matching import match_features_to_self
    from .geometric_verification import cluster_and_verify_matches


# ==============================================================================
# Predicted Mask Generation Module (Morphological Closing Approach)
# ==============================================================================
def generate_predicted_mask(
    image_shape: Tuple[int, int],
    src_points: np.ndarray,
    dst_points: np.ndarray,
    point_radius: int = 8,
    close_kernel_size: int = 25
) -> np.ndarray:
    """
    Generates a tight, concave binary mask indicating detected copy-move forgery regions
    using point splatting followed by Morphological Closing.

    ============================================================================
    WHY MORPHOLOGICAL CLOSING REPLACES CONVEX HULL IN PRODUCTION CMFD:
    ============================================================================
    The Convex Hull approach suffers from severe OVER-SEGMENTATION:
    - By mathematical definition, a convex hull CH(S) connects outer boundary points
      with straight lines:
          Hull(S) = { sum(lambda_i * p_i) | lambda_i >= 0, sum(lambda_i) = 1 }
    - In real-world documents, forged objects are inherently non-convex:
      1. Cursive signatures have loops, empty concave descenders, and whitespace between words.
      2. Official stamps/seals are circular rings with empty paper centers.
      3. Forged sentences or text lines have authentic white margins between characters.
    - Applying a convex hull forces all interior whitespace to be labeled as "Forged",
      generating massive amounts of False Positives (FP) and severely depressing Precision.

    The Morphological Closing Approach (Alpha-Shape / Concave Approximation):
    1. Point Splatting:
       Initialize an all-black canvas (H x W) and render a small filled circle (radius = 8 px)
       at every verified keypoint coordinate (x, y) in both the source and target clouds:
           M_raw = union_{p in S U D} Circle(p, radius=8)

    2. Morphological Closing (Dilation followed by Erosion):
       Apply closing with a large elliptical structuring element B (e.g., 25x25):
           M_closed = (M_raw (+) B) (-) B
       - The dilation step bridges the spatial gaps between adjacent keypoint circles within
         the same physical stroke or seal.
       - The subsequent erosion step restores the original outer boundary contours.
       - Large concave indentations, interior hollows of seals, and un-tampered margins
         remain untouched (True Negatives).
    This produces a tightly bounded, organic mask with superior Precision and IoU.

    Args:
        image_shape: (Height, Width) of the document.
        src_points: Mx2 numpy array of verified source coordinates (float32).
        dst_points: Mx2 numpy array of verified destination coordinates (float32).
        point_radius: Radius of the initial splatted circles at each keypoint coordinate.
        close_kernel_size: Diameter of the elliptical structuring element used for closing.

    Returns:
        np.ndarray: Binary mask [H, W] with dtype uint8 (0=authentic, 255=forged).
    """
    h, w = image_shape[:2]
    predicted_mask = np.zeros((h, w), dtype=np.uint8)

    # Return empty mask if no verified points are present
    if len(src_points) == 0 and len(dst_points) == 0:
        return predicted_mask

    # Step 1: Splat filled circles at every verified coordinate
    def splat_points(pts: np.ndarray, canvas: np.ndarray) -> None:
        if len(pts) == 0:
            return
        for pt in pts:
            px = int(round(pt[0]))
            py = int(round(pt[1]))
            # Clamp to canvas boundaries
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(canvas, (px, py), radius=point_radius, color=255, thickness=-1)

    splat_points(src_points, predicted_mask)
    splat_points(dst_points, predicted_mask)

    # Step 2: Apply Morphological Closing with an Elliptical Structuring Element
    # B = Ellipse of size (close_kernel_size x close_kernel_size)
    if close_kernel_size > 1:
        closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel_size, close_kernel_size)
        )
        predicted_mask = cv2.morphologyEx(
            predicted_mask,
            op=cv2.MORPH_CLOSE,
            kernel=closing_kernel
        )

    return predicted_mask


# ==============================================================================
# Metric Calculation Engine
# ==============================================================================
def calculate_cmfd_metrics(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray
) -> Dict[str, Union[float, int]]:
    """
    Computes rigorous pixel-level confusion matrix and segmentation metrics.

    Args:
        predicted_mask: 2D array [H, W] of predicted binary labels (uint8 or bool).
        ground_truth_mask: 2D array [H, W] of ground truth binary labels (uint8 or bool).

    Returns:
        Dict containing:
            - 'tp': True Positive pixel count
            - 'fp': False Positive pixel count
            - 'fn': False Negative pixel count
            - 'tn': True Negative pixel count
            - 'iou': Intersection over Union [0.0, 1.0]
            - 'precision': Precision score [0.0, 1.0]
            - 'recall': Recall score [0.0, 1.0]
            - 'f1': F1-Score / Dice coefficient [0.0, 1.0]
    """
    if predicted_mask.shape != ground_truth_mask.shape:
        raise ValueError(
            f"[!] Dimension mismatch: predicted_mask {predicted_mask.shape} != "
            f"ground_truth_mask {ground_truth_mask.shape}"
        )

    # Cleanse & Binarize Masks
    pred_bin = (predicted_mask > 127).astype(bool)
    gt_bin = (ground_truth_mask > 127).astype(bool)

    # Pixel Confusion Matrix via Vectorized Boolean Logic
    tp = int(np.count_nonzero(pred_bin & gt_bin))
    fp = int(np.count_nonzero(pred_bin & ~gt_bin))
    fn = int(np.count_nonzero(~pred_bin & gt_bin))
    tn = int(np.count_nonzero(~pred_bin & ~gt_bin))

    total_positives = tp + fp + fn

    if total_positives == 0:
        iou = 1.0
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        iou = float(tp / total_positives)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        if (precision + recall) > 0:
            f1 = float(2.0 * (precision * recall) / (precision + recall))
        else:
            f1 = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ==============================================================================
# Visualization Module
# ==============================================================================
def visualize_masks_side_by_side(
    rgb_image: np.ndarray,
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    metrics: Optional[Dict[str, Any]] = None,
    title: str = "Document CMFD: Forensic Verification & Segmentation Evaluation",
    show_plot: bool = True
) -> None:
    """
    Renders a 1x3 side-by-side diagnostic visualization comparing:
    1. Original Document Image
    2. Algorithm's Predicted Binary Mask (Morphological Closing)
    3. Ground Truth Binary Mask from Dataset
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)

    # 1. Original RGB Image
    axes[0].imshow(rgb_image)
    axes[0].set_title("Input Document Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2. Predicted Mask
    axes[1].imshow(predicted_mask, cmap="gray", vmin=0, vmax=255)
    pred_coverage = (np.count_nonzero(predicted_mask > 127) / predicted_mask.size) * 100.0
    axes[1].set_title(f"Predicted Mask (Morphological Close)\nCoverage: {pred_coverage:.2f}%", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # 3. Ground Truth Mask
    axes[2].imshow(ground_truth_mask, cmap="gray", vmin=0, vmax=255)
    gt_coverage = (np.count_nonzero(ground_truth_mask > 127) / ground_truth_mask.size) * 100.0
    axes[2].set_title(f"Ground Truth Mask\nCoverage: {gt_coverage:.2f}%", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    if metrics is not None:
        metrics_banner = (
            f"Evaluation Metrics:  IoU: {metrics['iou']*100:.2f}%  |  "
            f"Precision: {metrics['precision']*100:.2f}%  |  "
            f"Recall: {metrics['recall']*100:.2f}%  |  "
            f"F1-Score: {metrics['f1']*100:.2f}%"
        )
        fig.suptitle(f"{title}\n{metrics_banner}", fontsize=13, fontweight="bold", y=0.98)
    else:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    plt.tight_layout()
    if show_plot:
        plt.show()


# ==============================================================================
# Synthetic Ground Truth Helper
# ==============================================================================
def _generate_synthetic_ground_truth_mask(mask_path: Path, image_shape: Tuple[int, int] = (512, 512)) -> Path:
    """
    Creates the exact ground-truth binary mask matching the synthetic forged document
    generated in preprocess.py.
    """
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_path.is_file():
        return mask_path

    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.circle(mask, (140, 410), 48, 255, thickness=-1)
    cv2.circle(mask, (370, 410), 48, 255, thickness=-1)

    cv2.imwrite(str(mask_path), mask)
    return mask_path


# ==============================================================================
# Full Pipeline Testing & Evaluation Block
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(" CMFD Pipeline - Step 5: Metric Evaluation against Ground Truth")
    print("=" * 80)

    search_dirs = [
        Path(r"D:\Code\copy_move_detection\data"),
        Path(__file__).resolve().parent / "data",
        Path(__file__).resolve().parent
    ]

    image_file: Optional[Path] = None
    mask_file: Optional[Path] = None

    for base in search_dirs:
        orig_dir = base / "original" if (base / "original").exists() else base
        masks_dir = base / "masks" if (base / "masks").exists() else base

        forged_images = list(orig_dir.glob("*_F*.png")) + list(orig_dir.glob("*_F*.jpg"))
        if forged_images:
            image_file = forged_images[0]
            stem_core = image_file.stem.replace("_F", "").replace("_f", "")
            matching_masks = list(masks_dir.glob(f"*{stem_core}*_B*.png")) + list(masks_dir.glob(f"*{stem_core}*mask*.png"))
            if matching_masks:
                mask_file = matching_masks[0]
                break

    if image_file is None or mask_file is None:
        local_data = Path(__file__).resolve().parent / "data"
        synth_img_path = local_data / "original" / "synthetic_tampered_doc.png"
        synth_mask_path = local_data / "masks" / "synthetic_tampered_doc_mask.png"

        image_file = _generate_synthetic_document_if_empty(synth_img_path)
        mask_file = _generate_synthetic_ground_truth_mask(synth_mask_path, (512, 512))
        print(f"[+] Loaded reproducible synthetic test pair: {image_file.name} & {mask_file.name}")
    else:
        print(f"[+] Located CoMoFoD dataset pair: {image_file.name} & {mask_file.name}")

    # Stage 1: Preprocessing (with max_dim and scale_factor handling)
    preprocess_result = load_and_preprocess_image(image_file, max_dim=1500)
    rgb_img, gray_img = preprocess_result[0], preprocess_result[1]
    scale_factor = preprocess_result[2] if len(preprocess_result) > 2 else 1.0

    # Stage 2: SIFT Feature Extraction (with Dense SIFT check)
    keypoints, descriptors = extract_sift_features(
        gray_image=gray_img,
        contrast_threshold=0.03,
        edge_threshold=10.0,
        sigma=1.6
    )

    # Stage 3: FLANN Self-Matching
    valid_matches = match_features_to_self(
        keypoints=keypoints,
        descriptors=descriptors,
        ratio_threshold=0.72,
        spatial_distance_threshold=45.0
    )

    # Stage 4: DBSCAN & RANSAC
    verified_matches, homography_matrix, (src_pts, dst_pts) = cluster_and_verify_matches(
        keypoints=keypoints,
        valid_matches=valid_matches,
        eps=12.0,
        min_samples=4,
        ransac_reproj_threshold=5.0
    )

    # Stage 5: Predicted Mask Generation (Morphological Closing)
    pred_mask = generate_predicted_mask(
        image_shape=rgb_img.shape[:2],
        src_points=src_pts,
        dst_points=dst_pts,
        point_radius=8,
        close_kernel_size=25
    )

    # Ingest Ground Truth Mask
    gt_mask_raw = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    if gt_mask_raw is None:
        raise ValueError(f"Failed to load ground truth mask at: {mask_file}")

    if gt_mask_raw.shape != rgb_img.shape[:2]:
        gt_mask_raw = cv2.resize(gt_mask_raw, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    metrics = calculate_cmfd_metrics(pred_mask, gt_mask_raw)

    print("\n" + "=" * 55)
    print(" CMFD FORENSIC SEGMENTATION PERFORMANCE EVALUATION")
    print("=" * 55)
    print(f"  Pixel Confusion Matrix:")
    print(f"    - True Positives  (TP) : {metrics['tp']:,} px")
    print(f"    - False Positives (FP) : {metrics['fp']:,} px")
    print(f"    - False Negatives (FN) : {metrics['fn']:,} px")
    print(f"    - True Negatives  (TN) : {metrics['tn']:,} px")
    print("  -------------------------------------------------")
    print(f"  Intersection over Union (IoU) : {metrics['iou'] * 100:.2f}%")
    print(f"  Precision (Evidence Purity)   : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall (Forgery Completeness) : {metrics['recall'] * 100:.2f}%")
    print(f"  F1-Score (Dice Coefficient)   : {metrics['f1'] * 100:.2f}%")
    print("=" * 55 + "\n")

    visualize_masks_side_by_side(
        rgb_image=rgb_img,
        predicted_mask=pred_mask,
        ground_truth_mask=gt_mask_raw,
        metrics=metrics
    )
