#!/usr/bin/env python3
"""
Convert likelihood-scan CSV files to the .npy arrays used for training.

    python conv_csv_npy.py scan1.csv [scan2.csv ...] [--out-dir DIR] [--overwrite]

Each CSV has a header row and one row per scan point:

    <yield columns ...>, nLL_exp_mu0, nLL_exp_mu1, nLL_obs_mu0, nLL_obs_mu1,
                         nLLA_exp_mu0, nLLA_exp_mu1, nLLA_obs_mu0, nLLA_obs_mu1

i.e. the (total) yields of every bin, followed by the 8 negative log-likelihood
columns. The .npy is the same table without the header, as a float64 array of
shape (n_points, n_bins + 8), rows and columns in the CSV's order. Nothing is
filtered or reordered.

By default the .npy is written next to its CSV, with the same name. With
--columns the header is also written to <name>_columns.txt, one name per line.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

NLL_COLUMNS = ["nLL_exp_mu0", "nLL_exp_mu1", "nLL_obs_mu0", "nLL_obs_mu1",
               "nLLA_exp_mu0", "nLLA_exp_mu1", "nLLA_obs_mu0", "nLLA_obs_mu1"]
# value a scan writes when a fit failed
FAILED_FIT = 1e9


def convert(csv_path, out_dir=None, overwrite=False, columns=False):
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = out_dir or os.path.dirname(os.path.abspath(csv_path))
    npy_path = os.path.join(out_dir, stem + ".npy")
    if os.path.exists(npy_path) and not overwrite:
        raise FileExistsError(f"{npy_path} exists (use --overwrite)")

    df = pd.read_csv(csv_path)
    header = [c.strip() for c in df.columns]
    if len(header) <= len(NLL_COLUMNS):
        raise ValueError(f"{csv_path}: {len(header)} columns, expected the bin yields plus {len(NLL_COLUMNS)} nLL columns")
    if header[-len(NLL_COLUMNS):] != NLL_COLUMNS:
        print(f"  warning: last 8 columns are {header[-len(NLL_COLUMNS):]}, expected {NLL_COLUMNS}", file=sys.stderr)
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"{csv_path}: non-numeric columns {non_numeric}")

    data = df.to_numpy(dtype=np.float64)
    n_nonfinite = int((~np.isfinite(data)).any(axis=1).sum())
    n_failed = int((np.abs(data[:, -len(NLL_COLUMNS):]) >= FAILED_FIT).any(axis=1).sum())

    os.makedirs(out_dir, exist_ok=True)
    np.save(npy_path, data)
    if columns:
        with open(os.path.join(out_dir, stem + "_columns.txt"), "w") as f:
            f.write("\n".join(header) + "\n")

    print(f"{npy_path}: {data.shape[0]} points, {data.shape[1] - len(NLL_COLUMNS)} bins")
    if n_nonfinite:
        print(f"  warning: {n_nonfinite} rows contain NaN/inf", file=sys.stderr)
    if n_failed:
        print(f"  warning: {n_failed} rows have an nLL >= {FAILED_FIT:g} (failed fit)", file=sys.stderr)
    return npy_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="CSV files to convert")
    ap.add_argument("--out-dir", help="where to write the .npy files (default: next to each CSV)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing .npy files")
    ap.add_argument("--columns", action="store_true", help="also write <name>_columns.txt with the header")
    args = ap.parse_args()

    failed = 0
    for path in args.csv:
        try:
            convert(path, args.out_dir, args.overwrite, args.columns)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
