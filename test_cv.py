import cv2
from pathlib import Path

img_path = Path(r"C:\Users\jash.farrell\Downloads\test.jpg")

# read
img = cv2.imread(str(img_path))
print("read ok:", img is not None)
if img is None:
    raise SystemExit("Could not read image. Check the path or file type.")

# show
cv2.imshow("preview", img)
cv2.waitKey(0)        # press any key in the image window to close
cv2.destroyAllWindows()

# write
out_path = img_path.with_name("test_out.jpg")
ok = cv2.imwrite(str(out_path), img)
print("write ok:", ok, "->", out_path)
