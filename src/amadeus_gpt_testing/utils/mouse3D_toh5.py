#!/usr/bin/env python3
"""
Convert a nested `mouse3D.mat` (cell array: 1 × nTrials, each element shaped
nBodyparts × 3 × nFrames) into DeepLabCut‑style 3‑D keypoint files:
   <PREFIX><trial_index>_DLC3D.h5

Just edit the ALL‑CAPS constants below and run the script — no command‑line flags needed.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import scipy.io as sio
import yaml

# ───────────────────────────────────────────────────────────────────────
# USER‑EDITABLE CONSTANTS
# ───────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent.parent.parent
MAT_FILE = BASE / "data_configs" / "mouse_3D_files" / "mouse3D.mat"  # your .mat
OUT_DIR = BASE / "data"  # where to write .h5
PREFIX = "camera_1_trial_"  # prefix before trial index
CELL_NAME = "mouse3D"  # key in .mat
SCORER_NAME = "DLC3D_fusedCam_2025"  # your chosen scorer label
CONFIG_YML = BASE / "data_configs" / "config.yaml"  # DLC config for bodyparts
# ───────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────
# Extra Notes
# ───────────────────────────────────────────────────
# - The script reads a MATLAB file containing 3D keypoint data for multiple trials.
# - It converts each trial's data into a DeepLabCut-style DataFrame and saves it as an HDF5 file.
# - Empty trials are skipped (e.g. if trial 1 is empty, it will not be converted to a 3d h5 file).


def load_bodyparts(config_yml: Path, fallback_n: int) -> List[str]:
    """Read body‑part names from a DLC config, or create generic labels."""
    if config_yml.is_file():
        with open(config_yml, "r") as fh:
            cfg = yaml.safe_load(fh)
        bp = cfg.get("bodyparts", [])
        if len(bp) == fallback_n:
            return bp
        print(
            f"[WARN] config.yaml has {len(bp)} bodyparts but data says "
            f"{fallback_n}; falling back to auto‑labels."
        )
    return [f"bp{i+1}" for i in range(fallback_n)]


def trial_to_dataframe(arr: np.ndarray, bodyparts: List[str]) -> pd.DataFrame:
    """
    Convert one trial array (nBP × 3 × T) → DLC‑style DataFrame
    (T rows, MultiIndex columns scorer/bodyparts/(x|y|z)).
    """
    # MATLAB shape → (nBP, 3, T); transpose → (T, nBP, 3)
    kp = np.transpose(arr, (2, 0, 1))
    n_frames, n_bp, _ = kp.shape

    cols = pd.MultiIndex.from_product(
        [[SCORER_NAME], bodyparts, ["x", "y", "z"]],
        names=["scorer", "bodyparts", "coords"],  # <--- plural level names
    )
    data = kp.reshape(n_frames, n_bp * 3)
    return pd.DataFrame(data, columns=cols)


def convert_all():
    mat_path = MAT_FILE.expanduser().resolve()
    out_folder = OUT_DIR.expanduser().resolve()
    cfg_path = CONFIG_YML.expanduser().resolve()

    print(f"[INFO] loading MATLAB file: {mat_path}")
    m = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    if CELL_NAME not in m:
        raise KeyError(f"Variable '{CELL_NAME}' not found in {mat_path.name}")
    trials = m[CELL_NAME]
    # handle both (nTrials,) array or list
    n_trials = trials.shape[0] if isinstance(trials, np.ndarray) else len(trials)
    print(f"[INFO] found {n_trials} trial(s)")

    out_folder.mkdir(parents=True, exist_ok=True)

    # pick first nonempty to get number of bodyparts
    template = next((t for t in trials if hasattr(t, "size") and t.size), None)
    if template is None:
        raise ValueError("Every trial appears empty!")
    n_bp = template.shape[0]
    bodyparts = load_bodyparts(cfg_path, n_bp)
    print(f"[INFO] bodyparts: {bodyparts}")

    for idx, trial in enumerate(trials, start=1):
        if not hasattr(trial, "size") or trial.size == 0:
            print(f"[SKIP] trial {idx} is empty")
            continue

        df = trial_to_dataframe(trial, bodyparts)
        out_file = out_folder / f"{PREFIX}{idx}_DLC3D.h5"
        df.to_hdf(out_file, key="df", mode="w")  # <--- key must be "df"
        print(f"  ↳ wrote {out_file.name} ({df.shape[0]} frames, {n_bp} bodyparts)")

    print("[DONE] all trials processed.")


if __name__ == "__main__":
    convert_all()
