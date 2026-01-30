from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---- EDIT THIS PATH ----
IMG_PATH = Path("data/processed/frames_preview/Actor_04/02-01-03-02-02-02-04/frame_07_idx_087.jpg")
# -----------------------

MODEL_PATH = Path("models/face_landmarker.task")
OUT_POINTS = IMG_PATH.with_name(IMG_PATH.stem + "_points.jpg")
OUT_MESH = IMG_PATH.with_name(IMG_PATH.stem + "_mesh.jpg")


def main():
    img_bgr = cv2.imread(str(IMG_PATH))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {IMG_PATH}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Load model (Tasks API)
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise RuntimeError("No face detected.")

    landmarks = result.face_landmarks[0]
    h, w, _ = img_bgr.shape

    # -------- Variant 1: points only --------
    img_points = img_bgr.copy()
    pts = []

    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))
        cv2.circle(img_points, (x, y), 1, (0, 255, 0), -1)

    cv2.imwrite(str(OUT_POINTS), img_points)

    # -------- Variant 2: simple mesh (index connections) --------
    img_mesh = img_bgr.copy()

    for i in range(len(pts) - 1):
        cv2.line(img_mesh, pts[i], pts[i + 1], (0, 255, 0), 1)

    cv2.imwrite(str(OUT_MESH), img_mesh)

    print("Saved:")
    print(OUT_POINTS)
    print(OUT_MESH)


if __name__ == "__main__":
    main()