"""Adapter between the benchmark files and the paper's training code (model.py).

Set AERIALSTAMP_BENCHMARK to the folder that holds features.npz (default: the
repository root); runs are written to AERIALSTAMP_RUNS (default: ./runs).
"""
import os
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd

ROOT = abspath(join(dirname(__file__), ".."))
DATA = os.environ.get("AERIALSTAMP_BENCHMARK", ROOT)
RUNS_DIR = os.environ.get("AERIALSTAMP_RUNS", join(ROOT, "runs"))
OUT = DATA
NBINS, N_FOLDS, SEED = 365, 5, 0
DAYS = np.arange(NBINS)


def circ_diff(a, b):
    d = (np.asarray(a, float) - np.asarray(b, float)) % NBINS
    return np.where(d > NBINS / 2, d - NBINS, d)


def wrapped_gaussian(center, sigma):
    center = np.atleast_1d(np.asarray(center, float))
    d = circ_diff(DAYS[None, :] + 0.5, center[:, None])
    k = np.exp(-0.5 * (d / sigma)**2)
    return k / k.sum(1, keepdims=True)


def metadata():
    return pd.read_csv(join(ROOT, "metadata.csv"))


def load():
    """(lab, ds) in the layout the research pipeline uses."""
    m = metadata()
    f = np.load(join(DATA, "features.npz"))
    assert (f["id"] == m["id"].values).all()
    n = len(m)
    lab = pd.DataFrame(dict(id=m["id"], fold=m["fold"], group=m["block"], year=m["year"],
                            bin=m["day_bin"],
                            tier=np.where(m["label"] == "train_month_only", "ym", "full")))
    sf = f["site_facts"]
    static = np.zeros((n, 10), np.float32)  # columns 7, 8 (Sentinel-2 availability flags)
    static[:, :7], static[:, 9] = sf[:, :7], sf[:, 7]  # are unused in the paper: zero
    ds = dict(static=static, curves=np.zeros((n, NBINS), np.float32),
              target=f["target"].astype(np.float32),
              s2w=f["week_tokens"], s2pos=f["week_pos"], s2m=f["week_mask"],
              has_label=m["train"].values.astype(bool), has_match=np.ones(n, bool),
              p10_emb=f["patch_emb"], p10_col=np.zeros((n, 16, 1), np.float32),
              p10_valid=f["patch_valid"])
    return lab, ds
