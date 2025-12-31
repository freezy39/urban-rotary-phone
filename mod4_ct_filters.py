import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the assignment image
IMAGE_PATH = r"C:\Users\jash.farrell\Downloads\Mod4CT2.jpg"
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image from {IMAGE_PATH}")

# Grayscale Conversion to make edges clear
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the params
kernels = [(3, 3), (5, 5), (7, 7)]
sigma = 1.5  # good balance for Gaussian blur and detail retention

# Create subplot grid
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle("Comparison of Laplacian and Gaussian Filters for Different Kernels", fontsize=14)

for i, k in enumerate(kernels):
    # The Gaussian filter
    gaussian = cv2.GaussianBlur(gray, k, sigma)
    
    # The Laplacian filter
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    
    # The Gaussian + Laplacian (smoothed before edge enhancement)
    gauss_then_lap = cv2.GaussianBlur(gray, k, sigma)
    gauss_then_lap = cv2.Laplacian(gauss_then_lap, cv2.CV_64F)
    gauss_then_lap = cv2.convertScaleAbs(gauss_then_lap)
    
    # Plot row labels
    axes[i, 0].imshow(gaussian, cmap='gray')
    axes[i, 0].set_title(f"Gaussian {k}")
    
    axes[i, 1].imshow(laplacian, cmap='gray')
    axes[i, 1].set_title(f"Laplacian {k}")
    
    axes[i, 2].imshow(gauss_then_lap, cmap='gray')
    axes[i, 2].set_title(f"Gaussian + Laplacian {k}")

# Column labels
for ax, col in zip(axes[0], ["Gaussian", "Laplacian", "Gaussian + Laplacian"]):
    ax.set_title(col, fontsize=12)

# Clean up axis ticks
for ax_row in axes:
    for ax in ax_row:
        ax.axis('off')

plt.tight_layout()
plt.show()
