from pathlib import Path
from collections import Counter

VIDEO_ROOT = Path("data/raw")

def main():
    mp4s = list(VIDEO_ROOT.rglob("*.mp4"))
    print("Total mp4:", len(mp4s))

    # Look at first field in filename: e.g. 03-01-...-16.mp4  -> "03"
    first_fields = []
    for p in mp4s[:500]:  # sample first 500
        parts = p.stem.split("-")
        if len(parts) >= 1:
            first_fields.append(parts[0])

    c = Counter(first_fields)
    print("Filename first-field counts (sample of 500):", c)

    # Show a few examples
    print("\nExamples:")
    for p in mp4s[:10]:
        print(" ", p.name)

if __name__ == "__main__":
    main()