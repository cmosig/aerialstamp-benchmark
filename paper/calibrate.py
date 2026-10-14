"""Calibration of the paper: p_cal ∝ smooth_s(p^(1/T)), with temperature T and smoothing
width s (days) chosen per fold on the out-of-fold predictions of the other folds."""
import numpy as np

from bench import N_FOLDS, NBINS, wrapped_gaussian


def apply(p, T, s):
    q = np.clip(p.astype(np.float64), 1e-12, None)**(1.0 / T)
    q /= q.sum(1, keepdims=True)
    if s > 0:
        k = wrapped_gaussian([-0.5], s)[0]
        q = np.real(np.fft.ifft(np.fft.fft(q, axis=1) * np.fft.fft(k)[None], axis=1))
        q = np.clip(q, 1e-12, None)
        q /= q.sum(1, keepdims=True)
    return q


def nll(p, tb):
    return -np.log(p[np.arange(len(p)), np.floor(tb).astype(int) % NBINS]).mean()


def calibrate_oof(p, meta):
    """Fitted on the rows with calibrate == True (full dates, placeholder days included)."""
    folds, tb, ok = meta["fold"].values, meta["day_bin"].values, meta["calibrate"].values
    out, chosen = p.copy(), []
    for k in range(N_FOLDS):
        fit, te = np.where(ok & (folds != k))[0], np.where(folds == k)[0]
        _, T, s = min((nll(apply(p[fit], T, s), tb[fit]), T, s)
                      for T in (0.8, 1.0, 1.2, 1.4, 1.7, 2.0) for s in (0, 2, 4, 7, 10))
        out[te] = apply(p[te], T, s)
        chosen.append((T, s))
    return out, chosen
