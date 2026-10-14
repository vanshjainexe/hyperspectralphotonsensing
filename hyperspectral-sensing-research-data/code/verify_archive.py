"""Check the archive's files against its SHA256 manifest."""

import csv
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "validation" / "archive_checksums.csv"
    failed = []
    checked = 0
    skipped_raw = 0

    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = root / row["path"]
            if not path.exists() and row["path"].startswith("data/raw/"):
                skipped_raw += 1
                continue
            if not path.is_file():
                failed.append(row["path"] + " (missing)")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.stat().st_size != int(row["bytes"]) or digest != row["sha256"]:
                failed.append(row["path"])
            checked += 1

    print(f"Checked {checked} files.")
    if skipped_raw:
        print(f"Raw image files omitted from this copy: {skipped_raw}")
    if failed:
        for name in failed:
            print("Mismatch:", name)
        raise SystemExit(1)
    print("All available files match the manifest.")


if __name__ == "__main__":
    main()
