"""
cli_enroll.py – CLI tool to enroll a guest from local image files.
Usage: python scripts/cli_enroll.py --name "John Doe" --images img1.jpg img2.jpg
"""
import argparse
import sys, os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db.guest_db import get_connection, init_db, add_guest, add_embedding
from detection.mediapipe_detector import MediaPipeFaceDetector
from recognition.arcface_onnx import ArcFaceONNX
from recognition.matcher_faiss import FAISSMatcher
from utils.image_ops import crop_face, align_face_simple
import config


def main():
    parser = argparse.ArgumentParser(description="Enroll a guest via CLI")
    parser.add_argument("--name", required=True, help="Guest name")
    parser.add_argument("--designation", default="", help="Guest designation")
    parser.add_argument("--achievements", default="", help="Guest achievements")
    parser.add_argument("--images", nargs="+", required=True, help="Paths to face images")
    args = parser.parse_args()

    print(f"[INFO] Enrolling guest: {args.name}")

    # Initialise components
    conn = get_connection()
    init_db(conn)
    detector = MediaPipeFaceDetector()
    arcface = ArcFaceONNX()

    # Create guest record
    guest_id = add_guest(conn, args.name, args.designation, args.achievements)
    print(f"[INFO] Created guest_id = {guest_id}")

    embeddings_added = 0
    for img_path in args.images:
        if not os.path.isfile(img_path):
            print(f"[WARN] File not found: {img_path}")
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        faces = detector.detect(bgr)
        if not faces:
            print(f"[WARN] No face detected in {img_path}")
            continue

        best = max(faces, key=lambda d: d["confidence"])
        crop = crop_face(bgr, best["bbox"])
        if crop is None:
            print(f"[WARN] Bad crop in {img_path}")
            continue

        aligned = align_face_simple(crop)
        emb = arcface.get_embedding(aligned)
        add_embedding(conn, guest_id, emb)
        embeddings_added += 1
        print(f"  ✔ {img_path} → embedding added")

    conn.close()
    detector.close()

    if embeddings_added == 0:
        print("[ERROR] No embeddings were generated. Check your images.")
        return

    print(f"[INFO] Added {embeddings_added} embedding(s) for '{args.name}'.")

    # Rebuild FAISS index
    print("[INFO] Rebuilding FAISS index...")
    conn = get_connection()
    from db.guest_db import get_all_embeddings
    records = get_all_embeddings(conn)
    conn.close()

    matcher = FAISSMatcher()
    matcher.build_index(records)
    matcher.save()
    print(f"[INFO] Done. FAISS index has {matcher.total} vectors.")


if __name__ == "__main__":
    main()
