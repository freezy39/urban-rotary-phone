import cv2
import numpy as np
import os

#

PLATE_CASCADE_FILE = "haarcascade_russian_plate_number.xml"

# My three images
IMAGE_FILES = [
    "russian1.jpg",      # Russian plate, closer
    "russian2.jpg",      # Russian plate, farther away
    "southafrica1.jpg"   # Non Russian plate
]

# Output directory (same as script directory by default)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR


# Utility functions

def ensure_cascade(path: str) -> cv2.CascadeClassifier:
    """Load the Haar cascade and verify it is valid."""
    full_path = os.path.join(SCRIPT_DIR, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Cascade XML not found at: {full_path}")
    cascade = cv2.CascadeClassifier(full_path)
    if cascade.empty():
        raise IOError(f"Failed to load cascade from: {full_path}")
    print(f"[INFO] Loaded cascade from: {full_path}")
    return cascade


def load_image(filename: str):
    """Load an image from the script directory."""
    full_path = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(full_path):
        print(f"[WARN] Image file not found: {full_path}. Skipping.")
        return None, None
    img = cv2.imread(full_path)
    if img is None:
        print(f"[WARN] Failed to read image: {full_path}. Skipping.")
        return None, None
    return img, full_path


def choose_best_plate(candidates, img_shape):
    """
    Heuristic to choose the best plate candidate from detectMultiScale results.
    Candidates is a list of (x, y, w, h).
    """
    if len(candidates) == 0:
        return None

    img_h, img_w = img_shape[:2]
    img_area = img_w * img_h

    scores = []
    for (x, y, w, h) in candidates:
        area = w * h
        aspect = w / float(h + 1e-5)
        # Rough plate aspect ratio preference
        aspect_score = 1.0 - abs(aspect - 4.0) / 4.0
        area_score = area / float(img_area)
        score = aspect_score + area_score
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return candidates[best_idx]


def preprocess_plate_for_chars(plate_bgr):
    """
    Preprocess extracted plate region for character segmentation:
      - convert to grayscale
      - resize to a standard height
      - adaptive threshold
      - morphological operations
    Returns the preprocessed binary image.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    # Resize to fixed height, keep aspect ratio
    target_height = 80
    h, w = gray.shape[:2]
    scale = target_height / float(h)
    new_w = max(1, int(w * scale))
    gray_resized = cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    # Slight blur to reduce noise
    blurred = cv2.GaussianBlur(gray_resized, (3, 3), 0)

    # Adaptive threshold for better handling of uneven lighting
    bin_img = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        8
    )

    # Morph close to connect strokes
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closed


def segment_character_regions(binary_plate):
    """
    Simple character segmentation by contour detection on a binary plate image.
    Returns list of bounding boxes (x, y, w, h) in plate coordinates.
    """
    contours, _ = cv2.findContours(binary_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary_plate.shape[:2]

    char_boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Filter by relative size and aspect ratio
        if ch < h * 0.3:         # too short
            continue
        if ch > h * 1.1:         # too tall (probably noise)
            continue
        aspect = cw / float(ch + 1e-5)
        if aspect < 0.2 or aspect > 1.2:   # very skinny or too wide
            continue
        char_boxes.append((x, y, cw, ch))

    # Sort from left to right
    char_boxes = sorted(char_boxes, key=lambda b: b[0])
    return char_boxes


def draw_and_save(filename_base, img, suffix):
    """Save an image with a suffix."""
    out_path = os.path.join(OUTPUT_DIR, f"{filename_base}_{suffix}.jpg")
    cv2.imwrite(out_path, img)
    print(f"[INFO] Saved: {out_path}")


# Main processing for each image

def process_image(img_name, plate_cascade):
    img, full_path = load_image(img_name)
    if img is None:
        return

    base_name = os.path.splitext(os.path.basename(full_path))[0]
    print("\n" + "=" * 60)
    print(f"[INFO] Processing image: {img_name}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # Try detection with default parameters
    plates = plate_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(40, 20)
    )

    # If nothing found, try slightly different params
    if len(plates) == 0:
        print("[WARN] No plates found with first pass. Trying alternative parameters...")
        plates = plate_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 15)
        )

    if len(plates) == 0:
        print("[WARN] No plate detected in this image.")
        draw_and_save(base_name, img, "no_plate_detected")
        return

    print(f"[INFO] Detected {len(plates)} candidate plate regions")

    best_plate = choose_best_plate(plates, img.shape)
    if best_plate is None:
        print("[WARN] Could not choose a best plate candidate.")
        draw_and_save(base_name, img, "no_plate_chosen")
        return

    x, y, w, h = best_plate
    print(f"[INFO] Best plate region: x={x}, y={y}, w={w}, h={h}")

    # Draw rectangle on original
    img_with_box = img.copy()
    cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 0, 255), 2)
    draw_and_save(base_name, img_with_box, "plate_detected")

    # Crop plate region
    plate_roi = img[y:y + h, x:x + w]
    draw_and_save(base_name, plate_roi, "plate_cropped")

    # Preprocess for character recognition
    bin_plate = preprocess_plate_for_chars(plate_roi)
    # Save binary plate for inspection
    bin_plate_color = cv2.cvtColor(bin_plate, cv2.COLOR_GRAY2BGR)
    draw_and_save(base_name, bin_plate_color, "plate_preprocessed")

    # Segment character regions
    char_boxes = segment_character_regions(bin_plate)
    print(f"[INFO] Found {len(char_boxes)} candidate character regions")

    # Draw character boxes on preprocessed plate
    char_vis = cv2.cvtColor(bin_plate, cv2.COLOR_GRAY2BGR)
    for (cx, cy, cw, ch) in char_boxes:
        cv2.rectangle(char_vis, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 1)
    draw_and_save(base_name, char_vis, "plate_char_segments")

    # For this project we do not implement full OCR.
    # Instead we log that these are the segmented character regions.
    if len(char_boxes) > 0:
        approx_chars = len(char_boxes)
        print(f"[INFO] Approximate number of characters segmented: {approx_chars}")
    else:
        print("[WARN] No character regions segmented. Plate may be too small, contrast too low, or at an extreme angle.")



# Entry point

def main():
    print("[INFO] Starting license plate detection and character segmentation project...")

    # Load the plate cascade
    plate_cascade = ensure_cascade(PLATE_CASCADE_FILE)

    # Process each image
    for img_name in IMAGE_FILES:
        process_image(img_name, plate_cascade)

    print("\n[INFO] Processing complete.")
    print(f"[INFO] Check the output JPG files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
