#!/usr/bin/env python
"""
Batch‑convert four‑camera DLC CSVs to a single 3‑D HDF5 per trial.

USAGE
-----
python utils\csv_to_3d_h5.py --data    C:\Users\hugos\Desktop\AmadeusGPT\data \
                       --calib   C:\Users\hugos\Desktop\AmadeusGPT\bodycam_calibrations \
                       --project C:\Users\hugos\Desktop\AmadeusGPT\utils
"""
import argparse
import re
import shutil
from pathlib import Path

import deeplabcut as dlc
import pandas as pd
import yaml

# --------------------------------------------------------------------------
# USER‑EDITABLE CONSTANTS
# --------------------------------------------------------------------------
SCORER = "DLC_resnet50_Common_Network_cvBCOct18shuffle1_100000"
BODYPARTS = [
    "snout",
    "left_ear",
    "right_ear",
    "left_implant",  # will be ignored later
    "right_implant",  # will be ignored later
    "white_cable",  # will be ignored later
    "neck_base",
    "body_midpoint",
    "tail_base",
    "neck_body",
    "body_tail_midpoint",
]
CAMERAS = [1, 2, 3, 4]  # expected camera indices
# --------------------------------------------------------------------------


def csv_to_h5(csv_path: Path) -> Path:
    """Wrap a DLC CSV (3‑row header) into DLC‑style HDF5 ('df_with_missing')."""
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    h5_path = csv_path.with_suffix(".h5")
    df.to_hdf(h5_path, key="df_with_missing", mode="w")
    return h5_path


def write_minimal_config(project_dir: Path, videos):
    """Create the tiniest config.yaml that satisfies DLC‑3‑D."""
    cfg = dict(
        scorer=SCORER,
        bodyparts=BODYPARTS,
        video_sets={v.name: str(v) for v in videos},
        camera_names=[v.stem.split("_")[0] for v in videos],  # 'camera_1', ...
        snapshotindex=-1,
        project_path=str(project_dir),
        triangulation={"csv_dir": "csv", "h5_dir": "h5"},
    )
    with open(project_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)


def dlc_triangulate(project_dir: Path):
    """Run DLC's canonical 2‑D ➜ 3‑D pipeline in this one‑trial project."""
    cfg = str(project_dir / "config.yaml")
    dlc.convertcsv2h5(cfg, userfeedback=False, scorer=SCORER)  # 2‑D h5 per cam
    dlc.triangulate(cfg, save_as_csv=False)  # single 3‑D h5


def main(data_dir, calib_dir, project_root):
    data_dir = Path(data_dir)
    calib_dir = Path(calib_dir)
    project_root = Path(project_root)
    project_root.mkdir(parents=True, exist_ok=True)

    csv_pat = re.compile(r"camera_(\d+)_trial_(\d+)_([0-9\-]+)DLC", re.IGNORECASE)

    # 1) Group the CSVs by trial
    trials = {}
    for csv in data_dir.glob("camera_*_trial_*DLC*.csv"):
        m = csv_pat.match(csv.name)
        if not m:
            continue
        cam, trial_idx, date_stamp = m.groups()
        key = f"trial_{trial_idx}_{date_stamp}"
        trials.setdefault(key, {})[int(cam)] = csv

    for key, cams in trials.items():
        if len(cams) != 4:
            print(f"[{key}] Skipped – found {len(cams)} cameras, need 4.")
            continue

        # ── project skeleton ───────────────────────────────────────────────
        pdir = project_root / key
        csv_dir = pdir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        h5_dir = pdir / "h5"
        h5_dir.mkdir(exist_ok=True)

        # ── 1. CSV ➜ HDF5 per camera ──────────────────────────────────────
        for cam, csv in cams.items():
            h5 = csv_to_h5(csv)
            shutil.copy(csv, csv_dir / csv.name)
            shutil.copy(h5, h5_dir / h5.name)

        # ── 2. collect the four video paths (needed for config.yaml) ─────
        videos = []
        for cam in CAMERAS:
            vname = re.sub(r"DLC.*\.csv$", ".avi", cams[cam].name)
            vpath = data_dir / vname
            if vpath.exists():
                videos.append(vpath)
            else:
                print(f"   ⚠  Video {vpath.name} not found – still OK.")

        # ── 3. minimal config.yaml & calibration pickles ─────────────────
        if not (pdir / "config.yaml").exists():
            write_minimal_config(pdir, videos)
        for cam in CAMERAS:
            shutil.copy(
                calib_dir / f"camera-{cam}.pickle", pdir / f"camera-{cam}.pickle"
            )

        # ── 4. DLC triangulate ───────────────────────────────────────────
        print(f"[{key}] Triangulating …")
        dlc_triangulate(pdir)

        # DLC saves the 3‑D H5 inside <project>/h5/, grab and copy it back
        out3d = next(h5_dir.glob("*_3d.h5"))
        final = data_dir / out3d.name
        shutil.copy(out3d, final)
        print(f"[{key}]  ➜ {final.relative_to(data_dir.parent)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder with CSVs/Videos")
    ap.add_argument("--calib", required=True, help="Folder with camera‑*.pickle")
    ap.add_argument("--project", required=True, help="Where to build DLC projects")
    main(**vars(ap.parse_args()))
