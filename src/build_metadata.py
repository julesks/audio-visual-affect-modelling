from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

# RAVDESS emotion codes (speech/song)
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

def parse_ravdess_filename(path: Path) -> dict:
    """
    Example: 03-01-05-01-02-01-16.wav
    modality-channel: 03
    vocal-channel: 01
    emotion: 05
    intensity: 01
    statement: 02
    repetition: 01
    actor: 16
    """
    stem = path.stem
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected filename format: {path.name}")

    modality = parts[0]
    vocal = parts[1]
    emotion_code = parts[2]
    intensity = parts[3]
    statement = parts[4]
    repetition = parts[5]
    actor = parts[6]

    return {
        "path": str(path),
        "modality": modality,
        "vocal_channel": vocal,
        "emotion_code": emotion_code,
        "emotion": EMOTION_MAP.get(emotion_code, "unknown"),
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor": int(actor),
    }

def main():
    wav_files = list(DATA_DIR.rglob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found under {DATA_DIR}")

    rows = [parse_ravdess_filename(p) for p in wav_files]
    df = pd.DataFrame(rows)

    out_path = Path("data/processed/metadata.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved metadata: {out_path}")
    print(df.head())
    print("\nEmotion counts:")
    print(df["emotion"].value_counts())

if __name__ == "__main__":
    main()