#!/usr/bin/env python
"""
Plot a histogram of rearing‑bout durations taken from
`rearing_behavior_analysis.txt`.

Assumes frame ranges are listed exactly as:

    Frames <start> to <end>

under the header line:
    Frame Ranges of Rearing Behavior:
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
TXT_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "results"
    / "rearing_behavior_analysis.txt"
)  # adjust if needed
BINS = "auto"  # or an int like 30
FIGSIZE = (9, 4.5)
TITLE = "Histogram of Rearing‑Bout Durations"
XLABEL = "Duration (frames)"
YLABEL = "Count"
FRAMES_PER_SECOND = 30  # adjust if needed


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def extract_ranges(path: Path) -> list[tuple[int, int]]:
    """Return list of (start, end) frame tuples."""
    ranges, collecting = [], False
    rx = re.compile(r"Frames\s+(\d+)\s+to\s+(\d+)", re.I)

    with path.open(encoding="utf‑8") as fh:
        for line in fh:
            line = line.strip()
            if not collecting:
                # Start collecting after header
                if line.startswith("Frame Ranges of Rearing Behavior"):
                    collecting = True
                continue

            # Stop if we hit the next section or a blank line
            if not line or line.startswith("Average Head Height"):
                break

            m = rx.match(line)
            if m:
                start, end = map(int, m.groups())
                ranges.append((start, end))

    return ranges


def frame_durations(ranges: list[tuple[int, int]]) -> list[int]:
    """Convert (start, end) tuples to frame‑count durations."""
    return [end - start + 1 for start, end in ranges]


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    if not TXT_PATH.exists():
        raise SystemExit(f"File not found: {TXT_PATH}")

    ranges = extract_ranges(TXT_PATH)
    durations = frame_durations(ranges)

    print(
        f"Parsed {len(durations)} bouts; "
        f"min={min(durations)}  max={max(durations)}  "
        f"mean={sum(durations)/len(durations):.1f} frames"
    )

    plt.figure(figsize=FIGSIZE)
    plt.hist(durations, bins=BINS, edgecolor="black")
    plt.title(TITLE)
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
