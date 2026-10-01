#
# Merge many per-scan result files into one final table + metadata.
#
# Adapted from MLlikelihoods/misc/merge.py, hardened to also work on
# UNFINISHED runs (e.g. a cluster job that was killed mid-scan):
#
#   completed scan : results-<suffix>-<n>.csv + results-<suffix>-<n>.json
#   unfinished scan: table-<suffix>-<n>.csv   + metadata.json (same directory)
#
# table-*.csv is written incrementally (flushed every buffer_size points), so
# it holds real, usable rows even if the process never reached the final
# merge/rename step; metadata.json is written once at the very start of the
# scan (see utils.create_metadata) and only gains x_min/x_max/y_min/y_max and
# nLL_*_max after a successful run (see utils.update_metadata) - so those
# fields are simply absent for an unfinished job. This script recomputes the
# yield/likelihood extrema (x_min/x_max/y_min/y_max) directly from the merged
# data instead of trusting each chunk to have already computed them, so they
# always match whatever actually ended up in the merged table. Every other
# config field is backfilled across chunks - a value missing or None in one
# chunk's metadata.json is taken from another chunk that has it - and
# nLL_*_max, which normally can't be recovered from the per-point data, falls
# back to an upper bound read off the merged table if no chunk recorded a
# real fit.
#
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

#: Mirrors likelihood.NAN_PLACEHOLDER - a row with a likelihood at or beyond
#: this magnitude marks a failed evaluation and is dropped before merging.
#: Kept as a local constant (not imported) so this script stays independent
#: of the sampler's heavy dependencies (tensorflow/jax/spey/pyhf).
NAN_PLACEHOLDER = 1e10
#: Mirrors likelihood.N_LIKELIHOOD_COLUMNS - the last N columns of every row
#: are the likelihood values, everything before that is bin yields.
N_LIKELIHOOD_COLUMNS = 8
#: Metadata keys that are per-chunk summaries, recomputed fresh from the
#: merged data (x_min/x_max/y_min/y_max) or reduced across chunks
#: (nLL_*_max) rather than copied verbatim from any one input file.
_COMPUTED_KEYS = ("x_min", "x_max", "y_min", "y_max",
                   "nLL_exp_max", "nLL_obs_max", "nLLA_exp_max", "nLLA_obs_max")


def find_input_files(input_patterns):
    """Expand the glob patterns into a sorted, de-duplicated file list.

    Args:
        input_patterns (list[str]): Wildcard patterns, e.g. ``'tables/*/table-*.csv'``.

    Returns:
        list[str]: Matching paths, sorted for reproducible merge order.
    """
    all_files = set()
    for pattern in input_patterns:
        matches = glob.glob(pattern)
        if not matches:
            print(f"[WARNING] Pattern matched no files: {pattern!r}")
        all_files.update(matches)
    return sorted(all_files)


def find_metadata_for_csv(csv_path):
    """Locate the metadata sidecar for one result CSV.

    Tries the completed-run convention first (``<basename>.json`` next to
    the CSV), then falls back to the unfinished-run convention (a shared
    ``metadata.json`` in the same directory, written once at scan start).

    Args:
        csv_path (str): Path to a ``results-*.csv`` or ``table-*.csv`` file.

    Returns:
        str or None: Path to the metadata file, or ``None`` if neither
        convention matches anything on disk.
    """
    same_name = os.path.splitext(csv_path)[0] + ".json"
    if os.path.isfile(same_name):
        return same_name
    sibling = os.path.join(os.path.dirname(csv_path), "metadata.json")
    if os.path.isfile(sibling):
        return sibling
    return None


def read_csv_robust(path):
    """Read one result CSV, tolerating a truncated final line.

    A process killed mid-write (the common case for an unfinished cluster
    job) can leave a partial last row - fewer fields than the header, cut
    off mid-number. pandas does not treat this as a parse error (a short
    row is just padded with NaN), so a truncated row would otherwise slip
    through silently; every sampled row is always a complete set of real
    floats, so any row containing NaN here can only be that kind of
    truncation and is dropped. Lines with *too many* fields are skipped
    with a warning by pandas itself (``on_bad_lines='warn'``); a file that
    fails to parse at all, or that has no usable rows left, is skipped
    entirely.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame or None: The parsed rows, or ``None`` if the file
        could not be read or contained no usable rows.
    """
    if os.path.getsize(path) == 0:
        print(f"[WARNING] Skipping empty file: {path}")
        return None
    try:
        df = pd.read_csv(path, on_bad_lines="warn", engine="python")
    except Exception as e:  # noqa: BLE001 - genuinely want to catch anything and continue
        print(f"[WARNING] Skipping unreadable file {path}: {e!r}")
        return None

    truncated = df.isna().any(axis=1)
    if truncated.any():
        print(f"[WARNING] Dropping {int(truncated.sum())} truncated/short row(s) from {path} "
              f"(likely a process killed mid-write).")
        df = df.loc[~truncated].reset_index(drop=True)

    if df.shape[0] == 0:
        print(f"[WARNING] Skipping file with no usable data rows: {path}")
        return None
    return df


def drop_placeholder_rows(df):
    """Remove rows whose likelihood columns carry the NaN/inf placeholder.

    Mirrors ``utils.find_placeholder_rows`` - a value written as
    ``+/-NAN_PLACEHOLDER`` means that point's likelihood evaluation failed
    and carries no usable information.

    Args:
        df (pandas.DataFrame): Merged result rows.

    Returns:
        tuple: ``(clean_df, n_dropped)``.
    """
    if df.shape[1] < N_LIKELIHOOD_COLUMNS:
        return df, 0
    likelihoods = df.iloc[:, -N_LIKELIHOOD_COLUMNS:].to_numpy(dtype=float)
    bad = np.any(np.abs(likelihoods) >= NAN_PLACEHOLDER * (1.0 - 1e-9), axis=1)
    n_dropped = int(bad.sum())
    return df.loc[~bad].reset_index(drop=True), n_dropped


def merge_csv_files(input_patterns, output_file):
    """Collect, validate, clean and concatenate all matching result CSVs.

    Files with a column layout that does not match the first successfully
    read file are excluded (and reported), rather than silently unioned
    with NaN-filled columns - mixing schemas from different analyses is
    almost always a mistake, not something to paper over.

    Args:
        input_patterns (list[str]): Wildcard patterns locating CSV files.
        output_file (str): Path the merged CSV will be written to.

    Returns:
        tuple: ``(merged_df, used_files)`` - the cleaned, concatenated data
        and the list of CSV files actually included in it. ``merged_df`` is
        ``None`` if nothing usable was found.
    """
    all_files = find_input_files(input_patterns)
    if not all_files:
        print("No CSV files found matching the given patterns.")
        return None, []

    frames = []
    used_files = []
    reference_columns = None
    for file in all_files:
        df = read_csv_robust(file)
        if df is None:
            continue
        if reference_columns is None:
            reference_columns = list(df.columns)
        elif list(df.columns) != reference_columns:
            print(f"[WARNING] Skipping {file}: column layout does not match "
                  f"the other files ({df.shape[1]} columns vs "
                  f"{len(reference_columns)}). Merging different analyses "
                  f"or patchsets together is not supported.")
            continue
        frames.append(df)
        used_files.append(file)

    if not frames:
        print("No usable rows found in any matched file.")
        return None, []

    merged_df = pd.concat(frames, ignore_index=True)
    n_total = len(merged_df)

    merged_df, n_placeholder = drop_placeholder_rows(merged_df)
    if n_placeholder:
        print(f"Dropped {n_placeholder} row(s) carrying the NaN/inf likelihood placeholder.")

    n_before_dedup = len(merged_df)
    merged_df = merged_df.drop_duplicates().reset_index(drop=True)
    n_duplicates = n_before_dedup - len(merged_df)
    if n_duplicates:
        print(f"Removed {n_duplicates} duplicate row(s)")
    else:
        print("No duplicates found")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved to {output_file} "
          f"({len(merged_df)} of {n_total} rows kept, from {len(used_files)} of {len(all_files)} files)")

    return merged_df, used_files


def compute_extrema(df):
    """Compute per-column min/max directly from the merged data.

    This is the key difference from relying on each chunk's own recorded
    x_min/x_max/y_min/y_max: those fields only exist once a scan has
    finished (see utils.update_metadata), so an unfinished job's
    metadata.json never has them. Computing straight from the final,
    cleaned, merged table works regardless of whether every input was
    complete, and is always consistent with what actually got merged.

    Args:
        df (pandas.DataFrame): Cleaned, merged result rows.

    Returns:
        dict: ``{'x_min': [...], 'x_max': [...], 'y_min': [...], 'y_max': [...]}``,
        as plain Python lists (JSON-serialisable), or empty lists for a
        column group that doesn't exist (e.g. fewer than 8 columns total).
    """
    if df.shape[1] <= N_LIKELIHOOD_COLUMNS:
        return {"x_min": [], "x_max": [], "y_min": [], "y_max": []}
    values = df.to_numpy(dtype=float)
    data_min = values.min(axis=0)
    data_max = values.max(axis=0)
    return {
        "x_min": data_min[:-N_LIKELIHOOD_COLUMNS].tolist(),
        "x_max": data_max[:-N_LIKELIHOOD_COLUMNS].tolist(),
        "y_min": data_min[-N_LIKELIHOOD_COLUMNS:].tolist(),
        "y_max": data_max[-N_LIKELIHOOD_COLUMNS:].tolist(),
    }


def merge_json_metadata(csv_files, computed_extrema, output_json):
    """Merge the metadata sidecars for a set of result CSVs.

    Locates each CSV's metadata via :func:`find_metadata_for_csv` (handling
    both completed- and unfinished-run naming conventions), deduplicates
    metadata files shared by several CSVs in the same directory (multiple
    ``scans`` chunks share one ``metadata.json``), and combines them:

    * run configuration is merged field-by-field across all chunks: a key
      missing or recorded as ``None`` in one chunk is backfilled from
      another chunk that has it, with a warning only when two chunks both
      have a real value and it disagrees (``seed`` excepted, since every
      chunk legitimately has its own);
    * ``x_min``/``x_max``/``y_min``/``y_max`` come from ``computed_extrema``
      (derived from the actual merged data), not from the chunks;
    * ``nLL_*_max`` (from the one-time maximum-likelihood fit, which cannot
      be recovered from the per-point data) is taken as the best value
      across whichever chunks actually have it. If no chunk has it, it
      falls back to an upper bound read off the merged table (the best
      mu=0/mu=1 value found for that quantity); only left ``None`` - never
      a placeholder ``inf``, which is not valid JSON - if the table itself
      lacks those columns.

    Args:
        csv_files (list[str]): Result CSVs that were actually merged.
        computed_extrema (dict): Output of :func:`compute_extrema`.
        output_json (str): Path the merged metadata will be written to.

    Returns:
        dict or None: The merged metadata, or ``None`` if no metadata file
        could be found for any input CSV (the merged CSV is still valid;
        there is simply no configuration to report).
    """
    metadata_paths = []
    seen = set()
    missing_for = []
    for csv_file in csv_files:
        meta_path = find_metadata_for_csv(csv_file)
        if meta_path is None:
            missing_for.append(csv_file)
            continue
        resolved = os.path.abspath(meta_path)
        if resolved not in seen:
            seen.add(resolved)
            metadata_paths.append(meta_path)

    if missing_for:
        print(f"[WARNING] No metadata found for {len(missing_for)} file(s), e.g. {missing_for[0]}")

    if not metadata_paths:
        print("[WARNING] No metadata files found at all - writing the merged CSV without metadata.")
        return None

    nll_max = {key: None for key in
               ("nLL_exp_max", "nLL_obs_max", "nLLA_exp_max", "nLLA_obs_max")}
    loaded_metadata = []
    any_unfinished = False

    for meta_path in metadata_paths:
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"[WARNING] Skipping unreadable metadata file {meta_path}: {e!r}")
            continue
        loaded_metadata.append((meta_path, metadata))

        if not any(key in metadata for key in ("x_min", "nLL_exp_max")):
            # written by create_metadata but never updated: the run this
            # chunk came from was killed/still running when we read it
            any_unfinished = True

        for key in nll_max:
            entry = metadata.get(key)
            if isinstance(entry, list) and len(entry) == 2 and entry[1] is not None:
                if nll_max[key] is None or entry[1] < nll_max[key][1]:
                    nll_max[key] = entry

    if not loaded_metadata:
        print("[WARNING] No metadata file could be read - writing the merged CSV without metadata.")
        return None

    # Generic config fields: a value missing or recorded as None in one
    # chunk's metadata.json (e.g. a killed run can be missing keys that are
    # only filled in by a later step) is backfilled from whichever other
    # chunk has it, instead of only ever looking at the first file found.
    # Two chunks that both have a real, differing value are a genuine
    # inconsistency and are reported, not silently picked between.
    merged_metadata = {}
    for meta_path, metadata in loaded_metadata:
        for key, value in metadata.items():
            if key in _COMPUTED_KEYS or key in ("seed", "merged"):
                continue
            if key not in merged_metadata or merged_metadata[key] is None:
                merged_metadata[key] = value
            elif value is not None and merged_metadata[key] != value:
                print(f"[WARNING] Inconsistent value for {key!r} across chunks: "
                      f"{merged_metadata[key]!r} vs {value!r} ({meta_path})")

    # nLL_*_max (the one-time maximum-likelihood fit) cannot be recovered
    # from the per-point data in general - but if no chunk has it, the best
    # (lowest) nLL actually observed among that quantity's mu=0/mu=1
    # evaluations in the merged table is a valid upper bound on it (the free
    # fit is at least as good as either fixed-mu point), and a better-than-
    # nothing stand-in. mu_hat is left None since it isn't known this way.
    y_min = computed_extrema.get("y_min") or []
    if len(y_min) == N_LIKELIHOOD_COLUMNS:
        # column order is [nLL_exp_mu0, nLL_exp_mu1, nLL_obs_mu0, nLL_obs_mu1,
        # nLLA_exp_mu0, nLLA_exp_mu1, nLLA_obs_mu0, nLLA_obs_mu1] (see
        # likelihood.py's likelihoods_to_save) - each nLL_*_max pairs with
        # its own mu0/mu1 columns two apart.
        fallback_columns = {
            "nLL_exp_max": (0, 1), "nLL_obs_max": (2, 3),
            "nLLA_exp_max": (4, 5), "nLLA_obs_max": (6, 7),
        }
        for key, (i_mu0, i_mu1) in fallback_columns.items():
            if nll_max[key] is None:
                best = min(y_min[i_mu0], y_min[i_mu1])
                nll_max[key] = [None, best]
                print(f"[WARNING] {key} missing from all metadata; approximated as the best "
                      f"value found in the merged table ({best:.6g}). This is an upper bound "
                      "on the true maximum-likelihood fit, not the fit itself - re-run "
                      "calculate_Lmax for an exact value.")

    merged_metadata.update(computed_extrema)
    merged_metadata.update(nll_max)
    merged_metadata["merged"] = True
    merged_metadata["partial_merge"] = any_unfinished
    merged_metadata["n_source_files"] = len(csv_files)

    with open(output_json, "w") as f:
        json.dump(merged_metadata, f, indent=4)
    tag = "PARTIAL (source run(s) had not finished)" if any_unfinished else "complete"
    print(f"Merged metadata saved to {output_json} [{tag}]")
    return merged_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple sampling result CSVs (and their metadata) into one "
                    "final table. Works on completed runs (results-*.csv/.json) and on "
                    "unfinished/killed runs (table-*.csv + metadata.json) alike."
    )
    parser.add_argument("input_patterns", nargs="+",
                        help="Wildcard patterns to locate CSV files, "
                             "e.g. 'tables/1911.12606-*/results-*.csv' or "
                             "'tables/1911.12606-*/table-*.csv' for unfinished runs.")
    parser.add_argument("-o", "--output", required=True,
                        help="Base path for the output files (.csv and .json are appended).")

    args = parser.parse_args()
    merged_csv_file = args.output + ".csv"
    merged_json_file = args.output + ".json"

    merged_df, used_csv_files = merge_csv_files(args.input_patterns, merged_csv_file)
    if merged_df is not None:
        extrema = compute_extrema(merged_df)
        merge_json_metadata(used_csv_files, extrema, merged_json_file)
