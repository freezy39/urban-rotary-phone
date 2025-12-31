# adaptive_thresholding_portfolio_m6.py
# Python 3.13 + OpenCV 4.x
# Compares Otsu, Adaptive Mean, and Adaptive Gaussian thresholding on three images.
# Produces a 3x4 subplot figure and saves individual results.

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# image paths
INDOOR_PATH  = r"C:\Users\jash.farrell\Downloads\indoor.jpg"
OUTDOOR_PATH = r"C:\Users\jash.farrell\Downloads\outdoor.jpg"
OBJECT_PATH  = r"C:\Users\jash.farrell\OneDrive - Exxel Outdoors\Pictures\Screenshots\object.png"

# Can be tweaked if necessary. Block size must be odd and >= 3.
BLOCK_SIZE = 31   # window size for adaptive threshold (try 15, 31, 51 depending on detail)
C_VALUE    = 5    # constant subtracted from the mean or the weighted mean

# Utility Functions
def read_gray_with_clahe(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads image at 'path', converts to grayscale, and returns:
      gray         : standard grayscale
      gray_clahe   : CLAHE-normalized grayscale (better for local thresholding)
    Raises FileNotFoundError if path is invalid or unreadable.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image file: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE for local contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    return gray, gray_clahe

def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Otsu global automatic thresholding (with a light Gaussian blur).
    Returns a binary mask of dtype uint8 with values {0, 255}.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def adaptive_mean(gray: np.ndarray, block_size: int, C: int) -> np.ndarray:
    """
    Adaptive Mean thresholding. Returns uint8 binary {0, 255}.
    """
    return cv2.adaptiveThreshold(
        gray, 255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=C
    )

def adaptive_gaussian(gray: np.ndarray, block_size: int, C: int) -> np.ndarray:
    """
    Adaptive Gaussian thresholding. Returns uint8 binary {0, 255}.
    """
    return cv2.adaptiveThreshold(
        gray, 255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=C
    )

def ensure_odd(k: int) -> int:
    """Ensure kernel/block size is odd and at least 3."""
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    return k

def safe_title(text: str) -> str:
    # Shorten long Windows paths in subplot titles
    base = os.path.basename(text)
    return base if base else text

def save_binary(mask: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, mask)

# Main Processing
def process_one(path: str, block_size: int, C: int):
    """
    For a single image:
      - load grayscale and CLAHE grayscale
      - compute Otsu on normal gray
      - compute Adaptive Mean and Adaptive Gaussian on CLAHE gray
    Returns dict with all outputs.
    """
    gray, gray_clahe = read_gray_with_clahe(path)

    # Methods
    th_otsu = otsu_threshold(gray)
    th_mean = adaptive_mean(gray_clahe, block_size, C)
    th_gaus = adaptive_gaussian(gray_clahe, block_size, C)

    return {
        "gray": gray,
        "gray_clahe": gray_clahe,
        "otsu": th_otsu,
        "amean": th_mean,
        "agauss": th_gaus
    }

def build_grid(results_by_image: list[tuple[str, dict]]):
    """
    Build a 3x4 matplotlib figure:
      Columns: Grayscale, Otsu, Adaptive Mean, Adaptive Gaussian
      Rows   : Indoor, Outdoor, Object (in the order provided)
    """
    nrows = len(results_by_image)
    ncols = 4
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4.8 * nrows))

    col_titles = ["Grayscale (CLAHE shown)", "Otsu (auto global)", 
                  f"Adaptive Mean\n(block={BLOCK_SIZE}, C={C_VALUE})",
                  f"Adaptive Gaussian\n(block={BLOCK_SIZE}, C={C_VALUE})"]

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=12)

    for r, (path, outs) in enumerate(results_by_image):
        # Left column: show CLAHE gray for what the adaptive methods saw
        axes[r, 0].imshow(outs["gray_clahe"], cmap="gray")
        axes[r, 0].set_ylabel(safe_title(path), fontsize=11)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(outs["otsu"], cmap="gray")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(outs["amean"], cmap="gray")
        axes[r, 2].axis("off")

        axes[r, 3].imshow(outs["agauss"], cmap="gray")
        axes[r, 3].axis("off")

    plt.tight_layout()
    return fig

def main():
    # Validate and normalize params
    block = ensure_odd(BLOCK_SIZE)
    C = int(C_VALUE)

    image_paths = [
        INDOOR_PATH,
        OUTDOOR_PATH,
        OBJECT_PATH
    ]

    # Process each image
    results_by_image = []
    for p in image_paths:
        try:
            outs = process_one(p, block, C)
            results_by_image.append((p, outs))
        except FileNotFoundError as e:
            print(e)
            print("Fix the path above and rerun.")
            return

    # Save per-image binary outputs
    out_dir = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(),
                           "adaptive_threshold_outputs")
    os.makedirs(out_dir, exist_ok=True)

    for p, outs in results_by_image:
        stem, ext = os.path.splitext(os.path.basename(p))
        save_binary(outs["otsu"],  os.path.join(out_dir, f"{stem}_otsu.png"))
        save_binary(outs["amean"], os.path.join(out_dir, f"{stem}_adaptive_mean.png"))
        save_binary(outs["agauss"],os.path.join(out_dir, f"{stem}_adaptive_gaussian.png"))

    # Build and save the comparison grid
    fig = build_grid(results_by_image)
    grid_path = os.path.join(out_dir, "comparison_grid.png")
    fig.savefig(grid_path, dpi=180)
    print(f"\nSaved individual masks and the grid to:\n{out_dir}")
    print(f"Grid image: {grid_path}")

    # Show the figure window
    plt.show()

if __name__ == "__main__":
    main()
