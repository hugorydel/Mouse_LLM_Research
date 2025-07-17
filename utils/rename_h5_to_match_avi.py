#!/usr/bin/env python3
import re
from pathlib import Path

# ─────── CONFIG ───────
# Change this to wherever your .avi and .h5 files live:
DATA_FOLDER = Path(__file__).parent.parent / "data"
# ──────────────────────


def rename_h5_files(data_folder: Path):
    avi_files = sorted(data_folder.glob("*.avi"))
    if not avi_files:
        print("No .avi files found in", data_folder)
        return

    for avi in avi_files:
        stem = avi.stem
        # extract camera_#_trial_# prefix
        m = re.match(r"(camera_\d+_trial_\d+)", stem)
        if not m:
            print(f"⚠️ couldn’t parse prefix from {avi.name}, skipping")
            continue

        prefix = m.group(1)
        # look for any .h5 starting with that prefix
        candidates = list(data_folder.glob(f"{prefix}*.h5"))
        if not candidates:
            print(f"⚠️ no .h5 found for prefix {prefix}, skipping")
            continue
        if len(candidates) > 1:
            print(f"⚠️ multiple .h5 files for {prefix}: {candidates}, skipping")
            continue

        h5 = candidates[0]
        new_name = stem + ".h5"
        new_path = h5.with_name(new_name)

        if h5 == new_path:
            print(f"✔ already named {new_name}")
        else:
            print(f"Renaming:\n  {h5.name}\n→ {new_name}\n")
            h5.rename(new_path)


if __name__ == "__main__":
    print(f"Renaming .h5 files in {DATA_FOLDER} to match .avi files...")

    if not DATA_FOLDER.exists() or not DATA_FOLDER.is_dir():
        print(
            f"ERROR: DATA_FOLDER does not exist or is not a directory:\n  {DATA_FOLDER}"
        )
    else:
        rename_h5_files(DATA_FOLDER)
