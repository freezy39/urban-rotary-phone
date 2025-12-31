# CSC515 - Module 2 Discussion
# Geometric transformations on the banknote image

import cv2
import numpy as np

# Step 1: Read the image
img = cv2.imread(r"C:\Users\jash.farrell\Downloads\shutterstock227361781--125.jpg")
if img is None:
    print("Could not read the image. Check the file path.")
    exit()
print("Original shape:", img.shape)

# Step 2: Translation (move image)
tx, ty = 50, 20
T = np.float32([[1, 0, tx],
                [0, 1, ty]])
translated = cv2.warpAffine(img, T, (img.shape[1], img.shape[0]))

# Step 3: Rotation (-90 degrees about the image center)
h, w = img.shape[:2]
center = (w / 2, h / 2)
M = cv2.getRotationMatrix2D(center, -90, 1.0)
rotated = cv2.warpAffine(translated, M, (w, h))

# Step 4: Scaling (resize width to 800 pixels)
scale_factor = 800 / w
scaled = cv2.resize(rotated, (800, int(h * scale_factor)), interpolation=cv2.INTER_AREA)

# Step 5: Save and display the result
output_path = r"C:\Users\jash.farrell\Desktop\banknotes_transformed.jpg"
cv2.imwrite(output_path, scaled)
print(f"Transformed image saved to: {output_path}")

cv2.imshow("Original Banknote", img)
cv2.imshow("Transformed Banknote", scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
