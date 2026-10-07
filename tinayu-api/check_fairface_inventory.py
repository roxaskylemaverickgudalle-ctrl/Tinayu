from pathlib import Path
import sys

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}

def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "fairface_evaluation")
    if not root.exists():
        print(f"ERROR: Directory not found: {root}")
        return 1

    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )

    print("=" * 64)
    print("TINAYU FAIRFACE DATASET INVENTORY")
    print("=" * 64)
    print(f"Root: {root}")
    print(f"Image files found recursively: {len(files)}")
    print()

    if files:
        print("Sample files:")
        for p in files[:20]:
            print(f"  {p}")

    if len(files) >= 500:
        print()
        print("READY: At least 500 images are available.")
    else:
        print()
        print(f"NEED: {500 - len(files)} more images to reach 500.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
