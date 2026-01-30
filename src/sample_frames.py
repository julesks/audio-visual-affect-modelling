# src/sample_frames.py
from pathlib import Path
import cv2


VIDEO_ROOT = Path("data/raw")  # we will search for .mp4 under here
OUT_DIR = Path("data/processed/frames_preview")  # safe to delete anytime

N_FRAMES = 10  # number of frames to sample per video


def find_one_video() -> Path:
    # Keep only audio-visual version to avoid duplicates
    vids = list(VIDEO_ROOT.rglob("02-*.mp4"))
    if not vids:
        raise FileNotFoundError(
            f"No matching .mp4 files found under {VIDEO_ROOT}. "
            "Expected files like '02-*.mp4' in data/raw/."
        )
    return vids[0]


def get_evenly_spaced_indices(total_frames: int, n: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames < n:
        # If very short, just take what exists
        return list(range(total_frames))
    step = (total_frames - 1) / (n - 1)
    return [round(i * step) for i in range(n)]


def sample_frames(video_path: Path, n_frames: int = N_FRAMES) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = get_evenly_spaced_indices(total_frames, n_frames)

    # Output folder per video (so it’s easy to inspect)
    rel = video_path.relative_to(VIDEO_ROOT)
    video_out = OUT_DIR / rel.parent / video_path.stem
    video_out.mkdir(parents=True, exist_ok=True)

    print(f"\nVideo: {video_path}")
    print(f"FPS: {fps:.2f} | total frames: {total_frames} | sampling: {len(indices)} frames")
    print(f"Saving to: {video_out}")

    saved = 0
    for idx in indices:
        # Jump to frame idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[WARN] Could not read frame {idx}")
            continue

        # Save as jpg
        out_path = video_out / f"frame_{saved:02d}_idx_{idx:03d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    print(f"Saved {saved} frames.")


def main():
    # For now, just do ONE video so it’s quick and you can inspect results
    vid = find_one_video()
    sample_frames(vid, N_FRAMES)


if __name__ == "__main__":
    main()