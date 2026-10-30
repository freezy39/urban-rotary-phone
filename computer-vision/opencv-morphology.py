import cv2
import numpy as np
import matplotlib.pyplot as plt

# -
IMG_PATH = r"C:\Users\jash.farrell\OneDrive - Exxel Outdoors\Pictures\Screenshots\Screenshot 2025-11-05 122858.png"

def main():
    # Load my handwritten cursive post-it note image
    img_color = cv2.imread(IMG_PATH) 
    if img_color is None:
        print("Could not read input image. Check IMG_PATH.")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # use light Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # Otsu’s method threshold (invert so handwriting is white)
    _, binary = cv2.threshold(
        gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # element structure (3x3 kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphology
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Convert the colors for Matplotlib 
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    # Display the results in 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Sticky Note")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("Grayscale")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(binary, cmap="gray")
    axes[0, 2].set_title("Binary (Otsu Threshold)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(erosion, cmap="gray")
    axes[1, 0].set_title("Erosion")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(dilation, cmap="gray")
    axes[1, 1].set_title("Dilation")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(opening, cmap="gray")
    axes[1, 2].set_title("Opening (followed by Closing overlay)")
    axes[1, 2].axis("off")

    # Overlay the closing result in red color to show filled regions
    closing_rgb = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)
    closing_rgb[:, :, 2] = np.maximum(closing_rgb[:, :, 2], closing)
    axes[1, 2].imshow(closing_rgb)

    fig.suptitle("Morphology for Handwritten Sticky Note", fontsize=14)
    plt.tight_layout()

    # Save output figure file
    out_path = "sticky_morphology_results.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved comparison figure to {out_path}")

    plt.show()

if __name__ == "__main__":
    main()
