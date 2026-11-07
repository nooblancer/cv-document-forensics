"""
Preprocessing Module for Document Copy-Move Forgery Detection (CMFD).

Loads high-resolution documents/images, handles megapixel memory bottlenecks via
aspect-ratio-preserving decimation (cv2.INTER_AREA), normalizes color representations,
and applies optimal low-pass filtering tailored for scale-space feature detectors (SIFT).
"""

from pathlib import Path
from typing import Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_and_preprocess_image(
    image_path: Union[str, Path],
    max_dim: int = 1500,
    gaussian_kernel: Tuple[int, int] = (3, 3),
    gaussian_sigma: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Loads an image from disk, validates its integrity, downscales large scans to prevent
    OOM errors and quadratic matching bottlenecks, and prepares it for SIFT extraction.

    Mathematical Workflow:
    1. Megapixel Decimation & Aspect Ratio Preservation:
       High-resolution 300 DPI A4 scans produce images exceeding 2480x3508 (~8.7 MP).
       Computing SIFT and k-NN matching on >30,000 keypoints incurs O(N^2) memory and time.
       If max(Height, Width) > max_dim:
           scale_factor = max_dim / max(Height, Width)
           new_W = round(W * scale_factor)
           new_H = round(H * scale_factor)
       Decimation uses `cv2.INTER_AREA` (pixel area resampling), which prevents Moiré
       aliasing patterns and preserves fine document line weights without introducing ringing.

    2. Color Correction:
       OpenCV reads images in BGR format. Converts to standard RGB for visualization:
           I_RGB = cv2.cvtColor(I_BGR, cv2.COLOR_BGR2RGB)

    3. Structural Luminance Mapping (Grayscale):
       SIFT keypoints and descriptors rely on spatial gradient magnitudes and orientations.
       I_gray(x, y) = 0.299 * R + 0.587 * G + 0.114 * B

    4. Scanner Grain Suppression (Mild Gaussian Filtering):
       Applies a conservative 3x3 2D Gaussian filter to suppress sensor noise and halftone
       paper texture while preserving high-frequency edges:
           G(x, y, sigma) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))

    Args:
        image_path: File system path (str or pathlib.Path) to target document image.
        max_dim: Maximum allowable pixel length for the longest edge. If image exceeds
            this dimension, it is proportionally downscaled. Default is 1500 px.
        gaussian_kernel: Size of Gaussian blur kernel (width, height). Must be positive odd integers.
        gaussian_sigma: Standard deviation along both X and Y axes. sigma=0.5 provides subtle smoothing.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            - image_rgb: Scaled image in RGB format (uint8, shape [H, W, 3]).
            - preprocessed_gray: Preprocessed grayscale image (uint8, shape [H, W]).
            - scale_factor: Ratio of resized dimension to original dimension (new / original <= 1.0).
              Allows projecting detected coordinates/bounding boxes back to native scan resolution.

    Raises:
        FileNotFoundError: If the provided image path does not exist on disk.
        ValueError: If the file is corrupt or cannot be decoded as an image.
    """
    resolved_path = Path(image_path).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"[!] Specified image file was not found: {resolved_path}")

    # Read image in native BGR format
    image_bgr = cv2.imread(str(resolved_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(
            f"[!] Failed to decode image at '{resolved_path}'. File may be corrupted or format unsupported."
        )

    orig_h, orig_w = image_bgr.shape[:2]
    longest_edge = max(orig_h, orig_w)

    # 1. Check and Apply Aspect-Ratio Preserving Downscaling (INTER_AREA)
    if longest_edge > max_dim:
        scale_factor = float(max_dim) / float(longest_edge)
        new_w = int(round(orig_w * scale_factor))
        new_h = int(round(orig_h * scale_factor))

        # cv2.INTER_AREA is optimal for decimation (prevents high-frequency aliasing)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(
            f"[*] High-Resolution Image Rescaled: ({orig_w}x{orig_h}) -> ({new_w}x{new_h}) "
            f"[Scale Factor: {scale_factor:.4f}]"
        )
    else:
        scale_factor = 1.0

    # 2. Convert BGR -> RGB for display
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 3. Convert BGR -> Grayscale for gradient extraction
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 4. Low-pass filter to attenuate sensor/paper noise without shifting DoG extrema
    preprocessed_gray = cv2.GaussianBlur(
        image_gray,
        ksize=gaussian_kernel,
        sigmaX=gaussian_sigma,
        sigmaY=gaussian_sigma,
        borderType=cv2.BORDER_DEFAULT
    )

    return image_rgb, preprocessed_gray, scale_factor


def _generate_synthetic_document_if_empty(sample_path: Path) -> Path:
    """
    Fallback helper: Generates a realistic synthetic test document with a forged stamp
    if the CoMoFoD dataset has not been downloaded yet. Ensures immediate out-of-the-box testing.
    """
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_path.is_file():
        return sample_path

    print(f"[*] Generating synthetic document image for testing at: {sample_path}")
    doc = np.ones((512, 512, 3), dtype=np.uint8) * 248

    # Scanner noise simulation
    noise = np.random.normal(0, 4, (512, 512, 3)).astype(np.int16)
    doc = np.clip(doc.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Text header
    cv2.putText(doc, "CONFIDENTIAL CONTRACT - 2026", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.line(doc, (40, 75), (470, 75), (50, 50, 50), 1)

    # Text lines
    for y in range(120, 320, 30):
        cv2.line(doc, (45, y), (460, y), (140, 140, 140), 2)

    # Stamp 1 (Source object)
    cv2.circle(doc, (140, 410), 45, (40, 40, 180), 3)
    cv2.putText(doc, "APPROVED", (105, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 180), 1)

    # Copy-Move: Clone Stamp 1 to Target location
    stamp_roi = doc[360:460, 90:190].copy()
    doc[360:460, 320:420] = stamp_roi

    cv2.imwrite(str(sample_path), doc)
    return sample_path


if __name__ == "__main__":
    print("=" * 70)
    print(" CMFD Pipeline - Preprocessing Verification")
    print("=" * 70)

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data" / "original"

    sample_images = list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg"))
    if sample_images:
        test_image_path = sample_images[0]
        print(f"[+] Loaded existing dataset sample: {test_image_path.name}")
    else:
        test_image_path = _generate_synthetic_document_if_empty(
            data_dir / "synthetic_cmfd_sample.png"
        )

    rgb_img, gray_img, s_factor = load_and_preprocess_image(test_image_path, max_dim=1500)

    h, w, c = rgb_img.shape
    print(f"\n[+] Output Resolution      : {w}x{h} (Width x Height)")
    print(f"[+] Scale Factor           : {s_factor:.4f}")
    print(f"[+] Preprocessed Grayscale : shape={gray_img.shape}, dtype={gray_img.dtype}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=120)
    axes[0].imshow(rgb_img)
    axes[0].set_title(f"RGB Input (Rescaled: s={s_factor:.2f})\n({w}x{h})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gray_img, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Preprocessed Grayscale (SIFT Input)\n(3x3 Gaussian Blur, sigma=0.5)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
