# src/extract_face_landmarks_tasks.py
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


VIDEO_ROOT = Path("data/raw")
OUT_PATH = Path("data/processed/face_landmark_features.csv")

N_FRAMES = 10
VIDEO_GLOB = "02-*.mp4"

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def get_evenly_spaced_indices(total_frames: int, n: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames < n:
        return list(range(total_frames))
    step = (total_frames - 1) / (n - 1)
    return [round(i * step) for i in range(n)]


def parse_actor_from_path(video_path: Path) -> int:
    m = re.search(r"Actor_(\d+)", str(video_path))
    if not m:
        raise ValueError(f"Could not parse actor from path: {video_path}")
    return int(m.group(1))


def parse_emotion_from_filename(video_path: Path) -> tuple[str, str]:
    parts = video_path.stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {video_path.name}")
    emotion_code = parts[2]
    emotion = EMOTION_MAP.get(emotion_code, "unknown")
    return emotion_code, emotion


def sample_frames(video_path: Path, n_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = get_evenly_spaced_indices(total_frames, n_frames)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    return frames


def extract_landmarks_from_frame(landmarker: vision.FaceLandmarker, frame_bgr: np.ndarray) -> np.ndarray | None:
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # MediaPipe Image wrapper
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    # First face, list of landmarks with x,y,z
    lm = result.face_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (468, 3)
    return coords.flatten()  # (1404,)


def summarize_landmarks(landmark_vectors: list[np.ndarray]) -> dict:
    X = np.stack(landmark_vectors, axis=0)  # (n_frames, dims)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)

    feats = {}
    for i in range(mu.shape[0]):
        feats[f"lm_{i:04d}_mean"] = float(mu[i])
        feats[f"lm_{i:04d}_std"] = float(sd[i])
    return feats


def main():
    video_files = list(VIDEO_ROOT.rglob(VIDEO_GLOB))
    if not video_files:
        raise FileNotFoundError(f"No videos found under {VIDEO_ROOT} matching {VIDEO_GLOB}")

    print(f"Found {len(video_files)} videos (so far).")

    # Download the FaceLandmarker model (one-time) to a local file
    model_path = Path("models/face_landmarker.task")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print("Downloading FaceLandmarker model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"Saved model to {model_path}")

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )

    rows = []
    failures = 0
    no_face = 0

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, vp in enumerate(video_files, start=1):
            try:
                actor = parse_actor_from_path(vp)
                emotion_code, emotion = parse_emotion_from_filename(vp)

                frames = sample_frames(vp, N_FRAMES)
                landmark_vectors = []
                for fr in frames:
                    vec = extract_landmarks_from_frame(landmarker, fr)
                    if vec is not None:
                        landmark_vectors.append(vec)

                if not landmark_vectors:
                    no_face += 1
                    continue

                feats = summarize_landmarks(landmark_vectors)
                row = {
                    "path": str(vp),
                    "actor": actor,
                    "emotion_code": emotion_code,
                    "emotion": emotion,
                    "n_frames_sampled": len(frames),
                    "n_frames_with_face": len(landmark_vectors),
                }
                row.update(feats)
                rows.append(row)

            except Exception as e:
                failures += 1
                if failures <= 5:
                    print(f"[WARN] Failed on {vp.name}: {e}")

            if i % 50 == 0:
                print(f"Processed {i}/{len(video_files)} | rows: {len(rows)} | no-face: {no_face} | failures: {failures}")

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("\nDone.")
    print(f"Saved: {OUT_PATH}")
    print(f"Rows saved: {len(df)} / {len(video_files)}")
    print(f"Skipped (no face): {no_face}")
    print(f"Failures: {failures}")
    print("\nEmotion counts:")
    print(df["emotion"].value_counts())


if __name__ == "__main__":
    main()