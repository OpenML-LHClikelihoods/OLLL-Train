"""
Training-data facts an OLLL metadata block needs, printed as JSON.

Runs where the data is (`site run <site> likelihoods -- python
tools/olll_training_stats.py ...`) and prints, for one dataset, what
tools/olll_publish.py cannot derive from the ONNX file alone:

  - the mu=0 baselines (median of the even nLL columns) and their spread,
  - x/y bounds over the TRAINING block, reproducing experiment.init_data's
    split (seed-1234 shuffle, or the on-disk train|val|test presplit),
  - rows carrying a failed-fit sentinel (|nLL| >= SENTINEL), which are
    excluded from the bounds and counted,
  - a sample of held-out rows (test block) for the numerical closure test.

Output is one line `OLLL_STATS <json>` so it survives the site tool's framing.
"""

import argparse
import json
import os

import numpy as np

SENTINEL = 1e9


def load_ordered(data_dir, dataset):
    """Rows in experiment.init_data's train|val|test order, and n_train rule."""
    path = os.path.join(data_dir, f"{dataset}.npy")
    val = os.path.join(data_dir, f"{dataset}_val.npy")
    test = os.path.join(data_dir, f"{dataset}_test.npy")
    if os.path.exists(val):
        parts = [np.load(p, allow_pickle=True) for p in (path, val, test)]
        data = np.concatenate(parts, axis=0)
        presplit = tuple(len(p) for p in parts)
        rng = np.random.default_rng(1234)
        data[: presplit[0]] = data[rng.permutation(presplit[0])]
        return data, presplit
    data = np.load(path, allow_pickle=True)
    np.random.seed(1234)
    np.random.shuffle(data)
    return data, None


def training_stats(data_dir, dataset, train_frac, subsample=None, n_sample=2000):
    """The facts above, as a dict (what the OLLL_STATS line carries)."""
    data, presplit = load_ordered(data_dir, dataset)
    data = data.astype(np.float64)
    n = len(data)
    if presplit is not None:
        n_train = presplit[0]
        test_start = presplit[0] + presplit[1]
    else:
        n_train = int(n * train_frac)
        if subsample is not None:
            n_train = min(subsample, n_train)
        test_start = n_train

    nll = data[:, -8:]
    mu0 = np.median(nll[:, 0::2], axis=0)
    mu0_spread = (nll[:, 0::2].max(axis=0) - nll[:, 0::2].min(axis=0))

    train = data[:n_train]
    bad = np.any(np.abs(train[:, -8:]) >= SENTINEL, axis=1)
    good = train[~bad]
    x = good[:, :-8]
    t = good[:, -8:]
    delta = t[:, 1::2] - t[:, 0::2]

    rng = np.random.default_rng(0)
    test = data[test_start:]
    test = test[~np.any(np.abs(test[:, -8:]) >= SENTINEL, axis=1)]
    pick = rng.choice(len(test), size=min(n_sample, len(test)), replace=False)

    return {
        "dataset": dataset,
        "n_rows": n,
        "n_train": n_train,
        "presplit": presplit,
        "n_features": int(data.shape[1] - 8),
        "mu0": mu0.tolist(),
        "mu0_spread": mu0_spread.tolist(),
        "n_sentinel_train_rows": int(bad.sum()),
        "x_min": x.min(axis=0).tolist(),
        "x_max": x.max(axis=0).tolist(),
        "y_min": delta.min(axis=0).tolist(),
        "y_max": delta.max(axis=0).tolist(),
        "sample": test[pick].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train-frac", type=float, required=True)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    ap.add_argument("--n-sample", type=int, default=2000)
    args = ap.parse_args()
    out = training_stats(args.data_dir, args.dataset, args.train_frac, args.subsample, args.n_sample)
    print("OLLL_STATS " + json.dumps(out))


if __name__ == "__main__":
    main()
