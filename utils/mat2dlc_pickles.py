"""
mat2dlc_pickles.py  –  save DLC‑ready calibration pickles to your Desktop
-----------------------------------------------------------------------

• Extracts K, R, t from a 1×4 cell array of 3×4 projection matrices P{1, i}
• Assumes zero lens–distortion (change `dist_coeff` if you know k1–k5)
• Writes:
      Desktop\camera_matrix\
          camera_1.pickle
          camera_2.pickle
          camera_3.pickle
          camera_4.pickle
          stereo_params.pickle
"""

from pathlib import Path

import cv2
import numpy as np
import scipy.io
from deeplabcut.utils.auxiliaryfunctions import write_pickle

# ------------------------------------------------------------------------
# USER SETTINGS
mat_file = r"D:\Hugo\bodycam_calibrations\Pcal_rp.mat"  # <‑‑ path to .mat camera calibration file
out_root = Path.home() / "Desktop" / "camera_matrix"  # <‑‑ desktop
out_root.mkdir(parents=True, exist_ok=True)
# ------------------------------------------------------------------------

# ---- 1 load .mat -------------------------------------------------------
mat = scipy.io.loadmat(mat_file)
P_list = mat["P"].squeeze()  # 1×4 cell ⇒ ndarray length 4

Ks, Rs, Ts = [], [], []

# ---- 2 decompose each 3×4 matrix --------------------------------------
for idx, P in enumerate(P_list, 1):
    P = np.asarray(P, dtype=float)
    K, R, t, *_ = cv2.decomposeProjectionMatrix(P)
    K /= K[2, 2]  # normalise K so K[2,2] = 1
    t = (t / t[3])[:3].ravel()  # hom → euclidean

    calib = {
        "camera_matrix": K,
        "dist_coeff": np.zeros(5),  # <-‑‑ replace if you have real coeffs
        "rvecs": cv2.Rodrigues(R)[0].ravel(),
        "tvecs": t,
    }
    write_pickle(out_root / f"camera_{idx}.pickle", calib)
    Ks.append(K)
    Rs.append(R)
    Ts.append(t)
    print(f"✓  camera_{idx}.pickle written")

# ---- 3 one global stereo_params.pickle -----------------------------_
