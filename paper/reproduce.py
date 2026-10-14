"""Reproduce the paper's main results from the benchmark files.

    python paper/reproduce.py main          # Table 3: S2 model and image-only model, 5 seeds each
    python paper/reproduce.py ablation      # Table 5 rows that the benchmark supports, 3 seeds each
    python paper/reproduce.py plausibility  # Table 6 from baselines.npz (no training)

Recipe (paper, Section 3.4): model.py VARIANTS["final"] with epochs=32, no early
stopping, equal sample weights, window truncation 0.3 for the S2 model; five random
initialisations averaged; calibration per fold on the other folds. One run takes a
few minutes on a GPU. Results differ from the paper by seed noise (about 0.5 days in
block-weighted MAE), because the row order differs from the research pipeline.
"""
import json
import os
import sys
from os.path import dirname, abspath, join

sys.path.insert(0, dirname(abspath(__file__)))
sys.path.insert(0, join(dirname(abspath(__file__)), ".."))
import numpy as np

from bench import DATA, NBINS, RUNS_DIR, metadata
from calibrate import calibrate_oof
from evaluate import evaluate, interval_mask

BASE = dict(epochs=32, es=0, block_weight=0.0, s2_only=True)
RECIPES = {"s2": dict(trunc=0.3), "image_only": dict(s2=False)}
ABLATIONS = {  # Table 5; rows needing colour statistics or coarse views are not in the benchmark;
    # cross_entropy and block_weighting are additional variants, not in the paper's table
    "one_patch": dict(trunc=0.3, p10=1), "mean_pooling": dict(trunc=0.3, pool="mean"),
    "no_image_input": dict(trunc=0.3, p10=0), "corr_features_only": dict(trunc=0.3, wk="match"),
    "bands_only": dict(trunc=0.3, wk="raw"), "cross_entropy": dict(trunc=0.3, eps=0.0),
    "block_weighting": dict(trunc=0.3, block_weight=0.5)}


def fit(name, sets, seeds):
    import model
    ps = []
    for s in seeds:
        run = f"{name}_s{s}"
        f = join(RUNS_DIR, run, "oof.npy")
        if not os.path.exists(f):
            cfg = dict(model.DEFAULT); cfg.update(model.VARIANTS["final"]); cfg.update(BASE)
            cfg.update(sets); cfg["seed"] = s
            model.run(run, cfg, "cuda")
        p = np.load(f).astype(np.float64)
        ps.append(p / p.sum(1, keepdims=True))
    return np.mean(ps, 0)


def main(what):
    meta = metadata()
    if what == "main":
        for name, sets in RECIPES.items():
            p, chosen = calibrate_oof(fit(name, sets, range(5)), meta)
            np.save(join(RUNS_DIR, f"{name}_calibrated.npy"), p.astype(np.float32))
            print(name, "calibration (T, s) per fold:", chosen)
            print(name, json.dumps(evaluate(p, meta)))
    elif what == "ablation":
        full = evaluate(fit("s2", RECIPES["s2"], range(3)), meta)["MAE_bw"]
        print(f"S2 model (uncalibrated, 3 seeds): MAE bw {full}")
        for name, sets in ABLATIONS.items():
            v = evaluate(fit(f"abl_{name}", sets, range(3)), meta)["MAE_bw"]
            print(f"{name:22s} MAE bw {v:.1f} ({v - full:+.1f})")
        v = evaluate(fit("image_only", RECIPES["image_only"], range(3)), meta)["MAE_bw"]
        print(f"{'image_only':22s} MAE bw {v:.1f} ({v - full:+.1f})")
    elif what == "plausibility":
        p = np.load(join(DATA, "baselines.npz"))["s2_model"].astype(np.float64)
        p /= p.sum(1, keepdims=True)
        sel = meta["day_bin"].notna().values
        tb = np.floor(meta["day_bin"].values[sel]).astype(int) % NBINS
        pt = p[sel][np.arange(sel.sum()), tb]
        pi = (p[sel] * (p[sel] <= pt[:, None])).sum(1)  # plausibility of the submitted day
        lab = meta["label"].values[sel]
        for g in ("clean", "placeholder_1st_15th", "placeholder_jan1", None):
            m = np.ones(len(pi), bool) if g is None else lab == g
            print(f"{g or 'all':22s} n={m.sum():5d} " + "  ".join(
                f"pi<{a}: {(pi[m] < a).sum()} ({a * m.sum():.1f} expected)" for a in (0.05, 0.01, 0.001)))


if __name__ == "__main__":
    os.makedirs(RUNS_DIR, exist_ok=True)
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
