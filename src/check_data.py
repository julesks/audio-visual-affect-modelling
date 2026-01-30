from pathlib import Path

DATA_DIR = Path("data/raw")

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_DIR}. "
            "Download RAVDESS and place it in data/raw/ravdess/"
        )

    # List a few files so you know your path is correct
    all_files = list(DATA_DIR.rglob("*"))
    print(f"Found {len(all_files)} total files/folders under {DATA_DIR}")

    # Show some video files (common extensions)
    video_files = [p for p in all_files if p.suffix.lower() in {".mp4", ".avi", ".mov"}]
    audio_files = [p for p in all_files if p.suffix.lower() in {".wav"}]

    print(f"Video files found: {len(video_files)}")
    print(f"Audio files found: {len(audio_files)}")

    print("Example paths:")
    for p in (video_files[:3] + audio_files[:3]):
        print("  ", p)

if __name__ == "__main__":
    main()