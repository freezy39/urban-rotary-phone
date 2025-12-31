# milestone1_brain.py
# CSC515 – Module 1 Milestone
# This script reads, displays, and writes a copy of the brain image using OpenCV.

import cv2

# Step 1: Read the brain image
img_path = r"C:\Users\jash.farrell\Downloads\brain.jpg"       # change if needed
img = cv2.imread(img_path)

# Verify it loaded
if img is None:
    print("Could not read image. Check the file path.")
    exit()

print("Image loaded successfully!")

# Step 2: Display the image in a window
cv2.imshow("Brain Image", img)
cv2.waitKey(0)                 # waits until any key is pressed
cv2.destroyAllWindows()        # closes the window

# Step 3: Write a copy of the image to Desktop
output_path = r"C:\Users\jash.farrell\Desktop\brain_copy.jpg"
success = cv2.imwrite(output_path, img)

if success:
    print(f"Copy saved successfully at: {output_path}")
else:
    print("Could not save the copy.")
