This project expands an image dataset using basic image augmentation techniques in Python. Instead of collecting new images, the script creates realistic variations of existing images to improve model generalization and reduce overfitting.

The script (image_augmentation.py) automatically processes all image files in the folder and generates augmented versions using horizontal flips, small rotations, and brightness and contrast adjustments. Augmented files are saved with an aug_ prefix so they are easy to distinguish from the original images.

To run the script, place it in the same folder as the images and execute:

python image_augmentation.py

Both the original and augmented datasets are included as part of the submission.