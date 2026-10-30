import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your synthetic image
img = cv2.imread(r"C:\Users\jash.farrell\OneDrive - Exxel Outdoors\Documents\synthetic_teal_shapes.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ----- 1. CANNY -----
canny = cv2.Canny(gray, 100, 200)

# ----- 2. SOBEL -----
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.magnitude(sobelx, sobely)

# ----- 3. LAPLACIAN -----
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# ----- DISPLAY ALL RESULTS -----
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(canny, cmap="gray")
axes[1].set_title("Canny Edges")
axes[1].axis("off")

axes[2].imshow(sobel_combined, cmap="gray")
axes[2].set_title("Sobel Edges")
axes[2].axis("off")

axes[3].imshow(laplacian, cmap="gray")
axes[3].set_title("Laplacian Edges")
axes[3].axis("off")

plt.tight_layout()
plt.show()
