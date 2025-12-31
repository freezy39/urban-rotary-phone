# CSC515 – Module 3 Portfolio Milestone (Option 2: Blending Images in OpenCV)
# Face swap by aligning eyes, warping selfie to target, and seamless cloning.

import cv2
import numpy as np
from pathlib import Path

# ====== SET THESE ======
TARGET_PATH = r"C:\Users\jash.farrell\Downloads\ventura.jpg"     # your scene image
SELFIE_PATH = r"C:\Users\jash.farrell\Downloads\selfie.jpg"      # your frontal selfie
OUT_DIR = Path(r"C:\Users\jash.farrell\Downloads")               # where results will be saved
# =======================

def detect_face_and_eyes(img_bgr, scaleFactor=1.1, minNeighbors=5):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    if len(faces) == 0:
        return None, None

    # largest face
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = faces[0]

    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
    eyes = sorted(eyes, key=lambda r: r[2], reverse=True)[:2]

    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda r: r[0])  # left to right
        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[1]
        c1 = (x + ex1 + ew1 // 2, y + ey1 + eh1 // 2)
        c2 = (x + ex2 + ew2 // 2, y + ey2 + eh2 // 2)
        return (x, y, w, h), (c1, c2)

    return (x, y, w, h), None

def similarity_from_eyes(src_c1, src_c2, dst_c1, dst_c2):
    src = np.float32([src_c1, src_c2])
    dst = np.float32([dst_c1, dst_c2])
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    return M

def elliptical_mask(shape, face_rect, scale=1.05):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, fw, fh = face_rect
    cx, cy = x + fw // 2, y + fh // 2
    axes = (int(fw * 0.45 * scale), int(fh * 0.60 * scale))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    return mask

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target = cv2.imread(TARGET_PATH)
    source = cv2.imread(SELFIE_PATH)
    if target is None:
        print("Could not read TARGET_PATH. Set it to a valid image.")
        return
    if source is None:
        print("Could not read SELFIE_PATH. Set it to a valid image.")
        return

    # detect faces and eyes
    tgt_face, tgt_eyes = detect_face_and_eyes(target)
    src_face, src_eyes = detect_face_and_eyes(source)
    if tgt_face is None:
        print("No face found in target image.")
        return
    if src_face is None:
        print("No face found in selfie.")
        return

    # fallback eyes if detector misses
    def fallback_eyes(rect):
        x, y, w, h = rect
        cy = y + int(h * 0.4)
        return (x + int(w * 0.35), cy), (x + int(w * 0.65), cy)

    if tgt_eyes is None:
        tgt_eyes = fallback_eyes(tgt_face)
    if src_eyes is None:
        src_eyes = fallback_eyes(src_face)

    # estimate similarity and warp selfie to target canvas
    M = similarity_from_eyes(src_eyes[0], src_eyes[1], tgt_eyes[0], tgt_eyes[1])
    H, W = target.shape[:2]
    source_warp = cv2.warpAffine(source, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # build blend mask around target face
    mask = elliptical_mask(target.shape, tgt_face, scale=1.05)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)

    # seamless clone
    x, y, w, h = tgt_face
    center = (x + w // 2, y + h // 2)
    blended = cv2.seamlessClone(source_warp, target, mask, center, cv2.NORMAL_CLONE)

    # save final
    out_final = OUT_DIR / "m3_face_blend_result.jpg"
    cv2.imwrite(str(out_final), blended)
    print("Saved:", out_final)

    # debug visuals
    vis_t = target.copy()
    vis_s = source.copy()
    cv2.rectangle(vis_t, (x, y), (x + w, y + h), (0, 0, 255), 2)
    sx, sy, sw, sh = src_face
    cv2.rectangle(vis_s, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    for im, eyes in [(vis_t, tgt_eyes), (vis_s, src_eyes)]:
        cv2.circle(im, eyes[0], 4, (0, 255, 0), -1)
        cv2.circle(im, eyes[1], 4, (0, 255, 0), -1)

    # helper to resize and equalize heights for collage
    def resize_max(im, maxw=600):
        h, w = im.shape[:2]
        if w <= maxw:
            return im
        s = maxw / w
        return cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    def same_height(images, target_h=None):
        if target_h is None:
            target_h = min(im.shape[0] for im in images)
        fixed = []
        for im in images:
            h, w = im.shape[:2]
            s = target_h / h
            fixed.append(cv2.resize(im, (int(w * s), target_h), interpolation=cv2.INTER_AREA))
        return fixed

    a = resize_max(vis_s)
    b = resize_max(vis_t)
    c = resize_max(blended)
    a, b, c = same_height([a, b, c], target_h=300)
    collage = np.hstack([a, b, c])

    out_debug = OUT_DIR / "m3_face_blend_debug.jpg"
    cv2.imwrite(str(out_debug), collage)
    print("Saved:", out_debug)

    cv2.imshow("Selfie detections", vis_s)
    cv2.imshow("Target detections", vis_t)
    cv2.imshow("Blended result", blended)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
