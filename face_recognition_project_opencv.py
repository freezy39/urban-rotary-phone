import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


def ensure_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def detect_faces_yunet(detector, img_bgr: np.ndarray) -> np.ndarray:
    """
    Returns faces as Nx15 array from YuNet:
    [x, y, w, h, lmk1x, lmk1y, ... lmk5x, lmk5y, score]
    """
    h, w = img_bgr.shape[:2]
    detector.setInputSize((w, h))
    faces = detector.detect(img_bgr)
    if faces is None or len(faces) == 0:
        return np.empty((0, 15), dtype=np.float32)
    return faces[1] if isinstance(faces, tuple) else faces


def get_largest_face(faces: np.ndarray) -> np.ndarray:
    if faces.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]


def face_box_from_row(row: np.ndarray) -> np.ndarray:
    # OpenCV expects [x, y, w, h] as float32
    return row[:4].astype(np.float32)


def compute_embedding(recognizer, img_bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
    face_box = face_box_from_row(face_row)
    aligned = recognizer.alignCrop(img_bgr, face_box)
    feat = recognizer.feature(aligned)
    # feat is 1x128 float32
    return feat.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def draw_face(img: np.ndarray, face_row: np.ndarray, text: str) -> None:
    x, y, w, h = face_row[:4].astype(int)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, text, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Face recognition: check if an individual is in a group photo using OpenCV YuNet + SFace.")
    parser.add_argument("--individual", required=True, help="Path to the single-person (query) image")
    parser.add_argument("--group", required=True, help="Path to the group image")
    parser.add_argument("--detector", default="face_detection_yunet_2023mar.onnx", help="Path to YuNet ONNX model")
    parser.add_argument("--recognizer", default="face_recognition_sface_2021dec.onnx", help="Path to SFace ONNX model")
    parser.add_argument("--threshold", type=float, default=0.363, help="Cosine similarity threshold (SFace typical default around 0.363)")
    parser.add_argument("--out", default="group_annotated.png", help="Output path for annotated group image")
    args = parser.parse_args()

    ensure_file(args.individual, "Individual image")
    ensure_file(args.group, "Group image")
    ensure_file(args.detector, "YuNet detector model")
    ensure_file(args.recognizer, "SFace recognizer model")

    # Load images
    img_ind = cv2.imread(args.individual)
    img_grp = cv2.imread(args.group)
    if img_ind is None:
        raise ValueError(f"Could not read individual image: {args.individual}")
    if img_grp is None:
        raise ValueError(f"Could not read group image: {args.group}")

    # Create detector and recognizer
    # YuNet: score_thresh, nms_thresh, top_k
    detector = cv2.FaceDetectorYN_create(args.detector, "", (320, 320), 0.9, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF_create(args.recognizer, "")

    # Detect face in individual image (use largest face)
    faces_ind = detect_faces_yunet(detector, img_ind)
    ind_row = get_largest_face(faces_ind)
    if ind_row.size == 0:
        print("No face detected in the individual image.")
        return

    emb_ind = compute_embedding(recognizer, img_ind, ind_row)

    # Detect faces in group image and compare
    faces_grp = detect_faces_yunet(detector, img_grp)
    if faces_grp.shape[0] == 0:
        print("No faces detected in the group image.")
        return

    best_sim = -1.0
    best_idx = -1

    for i in range(faces_grp.shape[0]):
        emb_i = compute_embedding(recognizer, img_grp, faces_grp[i])
        sim = cosine_similarity(emb_ind, emb_i)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    is_match = best_sim >= args.threshold
    print(f"Best cosine similarity: {best_sim:.4f}")
    print(f"Threshold: {args.threshold:.4f}")
    print(f"Match found: {is_match}")

    # Annotate group image
    annotated = img_grp.copy()
    for i in range(faces_grp.shape[0]):
        sim_text = ""
        if i == best_idx:
            sim_text = f"best={best_sim:.3f} match={is_match}"
        draw_face(annotated, faces_grp[i], sim_text)

    cv2.imwrite(args.out, annotated)
    print(f"Saved annotated group image: {args.out}")


if __name__ == "__main__":
    main()