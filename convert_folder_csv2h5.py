from itertools import islice
from pathlib import Path

import pandas as pd
from deeplabcut.auxiliaryfunctions import guarantee_multiindex_rows


def convert_folder_csv2h5(folder, scorer=None):
    """
    For every .csv in `folder`, read it, optionally overwrite `scorer`,
    and write out a same‑named .h5 alongside it.
    """
    folder = Path(folder)
    for csv_path in folder.glob("*.csv"):
        print(f"Converting {csv_path.name} → {csv_path.with_suffix('.h5').name}")
        # --- detect header rows without loading full file ---
        with open(csv_path, "r") as f:
            lines = list(islice(f, 0, 5))
        # multi‑animal vs single
        header = list(range(4)) if "individuals" in lines[1] else list(range(3))
        # index column heuristic
        idx0 = lines[-1].split(",")[0]
        index_col = [0, 1, 2] if idx0 == "labeled-data" else 0

        df = pd.read_csv(csv_path, index_col=index_col, header=header)

        # optionally override scorer
        if scorer:
            df.columns = df.columns.set_levels([scorer], level="scorer")

        guarantee_multiindex_rows(df)
        df.to_hdf(csv_path.with_suffix(".h5"), key="df_with_missing", mode="w")
