# -*- coding: utf-8 -*-
"""behavior_analysis.py

Self‑contained utility for inspecting behavioural bouts.

2025‑07‑22 → **New features**
---------------------------
* Histograms are now embedded in the generated **`gallery.html`** next to the
  sample clips, so you can see bout duration distributions in one place.
* All artefacts (PNG + MP4 + HTML) still stored inside `OUTPUT_DIR / EXAMPLES_SUBDIR`.
* ± 0.5 s buffer on clips, zero‑CLI workflow intact.

Dependencies: `numpy`, `matplotlib`, `tqdm`, `moviepy`, `opencv-python`.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
except ImportError:
    VideoFileClip = None

# -----------------------------------------------------------------------------
# >>>>>>>>>  USER‑CONFIGURABLE CONSTANTS  <<<<<<<<<<<
# -----------------------------------------------------------------------------
DATA_DIR: Path = Path(r"C:\Users\hugos\Desktop\AmadeusGPT\results")  # *.txt files
VIDEO_PATH: Path = Path(
    r"C:\Users\hugos\Desktop\AmadeusGPT\data\camera_1_trial_3_2023-08-30-100947-0000.avi"
)
OUTPUT_DIR: Path = Path(r"C:\Users\hugos\Desktop\AmadeusGPT\results")
EXAMPLES_SUBDIR: str = "example_output"  # sub‑folder for artefacts

# Plotting / sampling
BINS: int = 20
SAMPLES_PER_BEHAVIOUR: int = 5  # 0 = skip sampling
BUFFER_SEC: float = 0.5  # context on each side of bout

# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------

EXAMPLES_DIR = OUTPUT_DIR / EXAMPLES_SUBDIR
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi"}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def parse_tuple(line: str) -> tuple[int, int]:
    """Parse a `(start, end)` pair of ints from a string line."""
    tup = ast.literal_eval(line.strip())
    if (
        isinstance(tup, (list, tuple))
        and len(tup) == 2
        and all(isinstance(x, int) for x in tup)
    ):
        return int(tup[0]), int(tup[1])
    raise ValueError(f"Invalid bout tuple: {line!r}")


def load_bouts(txt_path: Path) -> list[tuple[int, int]]:
    bouts = [parse_tuple(l) for l in txt_path.read_text().splitlines() if l.strip()]
    bouts.sort(key=lambda t: t[0])
    return bouts


def clean_bouts(raw_bouts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove <3‑frame bouts unless a neighbour is <10 frames away and merge."""
    cleaned: list[tuple[int, int]] = []
    i = 0
    while i < len(raw_bouts):
        start, end = raw_bouts[i]
        dur = end - start + 1
        if dur < 3 and i + 1 < len(raw_bouts):
            nxt_start, nxt_end = raw_bouts[i + 1]
            if nxt_start - end < 10:
                cleaned.append((start, nxt_end))
                i += 2
                continue
            i += 1  # drop
            continue
        if dur >= 3:
            cleaned.append((start, end))
        i += 1
    return cleaned


def save_histogram(bouts: list[tuple[int, int]], behaviour: str) -> str:
    """Plot histogram, save PNG, return filename (relative)."""
    durations = [e - s + 1 for s, e in bouts]
    plt.figure()
    plt.hist(durations, bins=BINS)
    plt.title(f"{behaviour} bout durations (frames)")
    plt.xlabel("Duration (frames)")
    plt.ylabel("Count")
    out = EXAMPLES_DIR / f"{behaviour}_histogram.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out.name


def extract_sample_clips(
    video: Path, bouts: list[tuple[int, int]], behaviour: str, fps: float
) -> list[str]:
    if SAMPLES_PER_BEHAVIOUR <= 0:
        return []
    if VideoFileClip is None:
        raise RuntimeError("`moviepy` missing – `pip install moviepy` to export clips.")

    master = VideoFileClip(str(video))
    video_dur = master.duration
    saved: list[str] = []
    use_subclipped = hasattr(master, "subclipped") and not hasattr(master, "subclip")

    for idx, (s_f, e_f) in enumerate(bouts[:SAMPLES_PER_BEHAVIOUR]):
        s_t = max(0.0, s_f / fps - BUFFER_SEC)
        e_t = min(video_dur, e_f / fps + BUFFER_SEC)
        sub = (
            master.subclipped(s_t, e_t) if use_subclipped else master.subclip(s_t, e_t)
        )
        out = EXAMPLES_DIR / f"{behaviour}_{idx}.mp4"
        sub.write_videofile(
            str(out), codec="libx264", audio=False, logger=None, preset="ultrafast"
        )
        saved.append(out.name)
    master.close()
    return saved


def write_html_gallery(behaviour_assets: dict[str, list[str]]):
    html = EXAMPLES_DIR / "gallery.html"
    with html.open("w", encoding="utf-8") as f:
        f.write("<html><head><title>Behaviour Gallery</title></head><body>\n")
        f.write("<h1>Behaviour Histograms & Sample Clips</h1>\n")
        for behaviour, assets in behaviour_assets.items():
            if not assets:
                continue
            f.write(
                f"<h2>{behaviour}</h2>\n<div style='display:flex;flex-wrap:wrap;gap:20px'>\n"
            )
            for asset in assets:
                ext = Path(asset).suffix.lower()
                if ext in IMG_EXTENSIONS:
                    f.write(
                        f"<img src='{asset}' style='max-width:400px;margin-bottom:10px;'>\n"
                    )
                elif ext in VIDEO_EXTENSIONS:
                    f.write(
                        f"<video width='320' controls style='margin-bottom:10px;'>\n"
                        f"  <source src='{asset}' type='video/mp4'>\n"
                        f"</video>\n"
                    )
            f.write("</div>\n")
        f.write("</body></html>")
    return html


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"VIDEO_PATH not found: {VIDEO_PATH}")
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(DATA_DIR.glob("*.txt"))
    if not txt_files:
        raise RuntimeError("No *.txt behaviour files found in DATA_DIR.")

    if VideoFileClip is None:
        raise RuntimeError("moviepy required – `pip install moviepy`.")
    tmp = VideoFileClip(str(VIDEO_PATH))
    fps = tmp.fps
    tmp.close()

    behaviour_assets: dict[str, list[str]] = defaultdict(list)

    for txt in txt_files:
        behaviour = txt.stem
        bouts = clean_bouts(load_bouts(txt))

        # Histogram PNG
        hist_png = save_histogram(bouts, behaviour)
        behaviour_assets[behaviour].append(hist_png)

        # Video clips
        clips = extract_sample_clips(VIDEO_PATH, bouts, behaviour, fps)
        behaviour_assets[behaviour].extend(clips)

    gallery = write_html_gallery(behaviour_assets)
    print("Gallery saved to:", gallery)
    print("All artefacts in:", EXAMPLES_DIR)


if __name__ == "__main__":
    main()
