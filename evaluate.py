"""Evaluate acquisition-date predictions on the AerialStamp benchmark.

A prediction is a probability distribution over the 365 day bins of the submitted
year for every orthophoto (rows as in metadata.csv), saved as a .npy array of shape
(N, 365) or as one array inside an .npz. Rows are normalised to sum to 1. Only rows
with evaluate == True are scored (full dates that are not on placeholder days).

    python evaluate.py baselines.npz --key s2_model
    python evaluate.py my_predictions.npy

A method that returns a single date can be scored on the error metrics by putting
all probability on that day; coverage and NLL are then not meaningful.

Metrics (identical to the paper):
  MAE, median   absolute circular error in days between the predicted date (circular
                median of the distribution) and the submitted date
  MAE_bw        MAE with every spatial block weighted equally
  within30      share of orthophotos dated within 30 days
  cov50/80/95   share whose submitted date lies in the 50/80/95 % prediction interval
                (smallest set of days holding that probability)
  width80       mean size of the 80 % interval in days
  NLL           mean negative log-probability of the submitted day
"""
import argparse
import json
from os.path import dirname, join, abspath

import numpy as np
import pandas as pd

NBINS = 365
DAY_PER_BIN = 365.25 / NBINS
DAYS = np.arange(NBINS)


def circ_diff(a, b):
    d = (np.asarray(a, float) - np.asarray(b, float)) % NBINS
    return np.where(d > NBINS / 2, d - NBINS, d)


_D = np.abs(circ_diff(DAYS[:, None], DAYS[None, :]))


def predicted_bin(p):
    """Circular median: the day that minimises the expected absolute error."""
    return (p @ _D).argmin(1)


def interval_mask(p, mass):
    """(N, 365) mask of the smallest set of days holding `mass` (may be several periods)."""
    order = np.argsort(-p, axis=1)
    cs = np.cumsum(np.take_along_axis(p, order, 1), axis=1)
    k = (cs < mass).sum(1) + 1
    mask = np.zeros_like(p, bool)
    mask[np.arange(len(p))[:, None], order] = np.arange(NBINS)[None, :] < k[:, None]
    return mask


def evaluate(p, meta):
    """p: (N, 365) for all rows of metadata.csv -> dict of metrics on the evaluation rows."""
    assert p.shape == (len(meta), NBINS), f"expected {(len(meta), NBINS)}, got {p.shape}"
    sel = meta["evaluate"].values.astype(bool)
    p = np.asarray(p[sel], np.float64)
    p = p / p.sum(1, keepdims=True)
    true = meta["day_bin"].values[sel]
    tb = np.floor(true).astype(int) % NBINS
    err = np.abs(circ_diff(predicted_bin(p) + 0.5, true)) * DAY_PER_BIN
    block = meta["block"].values[sel]
    w = 1.0 / pd.Series(block).map(pd.Series(block).value_counts()).values
    out = dict(n=int(sel.sum()), MAE=err.mean(), MAE_bw=(err * w).sum() / w.sum(),
               median=float(np.median(err)), within30=(err <= 30).mean())
    rows = np.arange(len(p))
    for mass in (0.5, 0.8, 0.95):
        m = interval_mask(p, mass)
        out[f"cov{int(mass * 100)}"] = m[rows, tb].mean()
        if mass == 0.8:
            out["width80"] = m.sum(1).mean() * DAY_PER_BIN
    out["NLL"] = -np.log(np.clip(p[rows, tb], 1e-12, None)).mean()
    return {k: (round(float(v), 3) if k != "n" else v) for k, v in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("predictions")
    ap.add_argument("--key", help="array name when the file is an .npz")
    ap.add_argument("--metadata", default=join(dirname(abspath(__file__)), "metadata.csv"))
    ap.add_argument("--by", help="also report per group of this metadata column, e.g. biome")
    a = ap.parse_args()
    meta = pd.read_csv(a.metadata)
    z = np.load(a.predictions)
    p = z[a.key] if a.key else (z if isinstance(z, np.ndarray) else z[z.files[0]])
    p = np.asarray(p, np.float64)
    print(json.dumps(evaluate(p, meta), indent=1))
    if a.by:
        for k, g in meta.groupby(a.by):
            if g["evaluate"].sum() >= 20:
                m = meta.copy(); m["evaluate"] &= (m[a.by] == k)
                print(k, json.dumps(evaluate(p, m)))
