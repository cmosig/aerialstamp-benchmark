# This file is the training code of the AerialStamp paper (train.py of the research
# pipeline), unchanged except that it reads the benchmark files through bench.load().
# Options that need data outside the benchmark (pheno=1, curve=..., wksrc=modis,
# feats=col/both, p1m, ov) are kept for reference but cannot be run from it.
"""Learned day-of-year model, cross-validated on the blocked folds.

logits(day) = prior_head(day | ortho views, site, phenology curve)
            + s2_head(day | S2 weeks of the flight year, ortho, site)

- ortho views: frozen DINOv2 tokens and/or colour statistics of the stored
  views (10 cm patches, 1 m windows, AOI overview), projected to D and pooled
  (mean per scale, or attention with a learned query over all tokens);
- prior_head: an MLP evaluated per day on [site curve(day), its slope,
  Fourier(day)] + the ortho/site embedding -> one logit per day;
- s2_head: per-week tokens [S2 bands, indices, clear fraction, ortho<->S2
  match features, week position] + the embedding (wk="raw" / "match" keeps
  only the S2 or only the match columns), two masked temporal convs,
  one score per week, spread to days by a Gaussian kernel in time. Weeks exist
  only for sites with a cube; otherwise the term is zero.

Loss: noise-robust circular likelihood
  L = -log( eps/365 + (1 - eps) * sum_d p(d) q(d) )
with q the soft target (wrapped Gaussian around the recorded day, or uniform
over the month for year+month labels). A wrong label can cost at most
-log(eps/365), so a minority of bad dates cannot dominate the gradient.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench import N_FOLDS, NBINS, OUT, RUNS_DIR, SEED, load

VARIANTS = {
    # name: views/pooling/features
    "noortho": dict(p10=0, p1m=False, ov=False),
    "p10x1": dict(p10=1, pool="mean"),
    "p10x4": dict(p10=4, pool="mean"),
    "p10x8": dict(p10=8, pool="mean"),
    "p10x16": dict(p10=16, pool="mean"),
    "p10x16_attn": dict(p10=16, pool="attn"),
    "ov": dict(p10=0, ov=True),
    "p1m": dict(p10=0, p1m=True),
    "ov_p1m": dict(p10=0, p1m=True, ov=True),
    "multiscale": dict(p10=16, p1m=True, ov=True, pool="mean"),
    "multiscale_attn": dict(p10=16, p1m=True, ov=True, pool="attn"),
    "multiscale_colonly": dict(p10=16, p1m=True, ov=True, feats="col"),
    "multiscale_dinoonly": dict(p10=16, p1m=True, ov=True, feats="dino"),
    "multiscale_nos2": dict(p10=16, p1m=True, ov=True, s2=False),
    "multiscale_ce": dict(p10=16, p1m=True, ov=True, eps=0.0),
    "multiscale_nopheno": dict(p10=16, p1m=True, ov=True, pheno=False),
    # component ablations on the v2 winner (16 x 10 cm patches, attention)
    "attn_dinoonly": dict(p10=16, pool="attn", feats="dino"),
    "attn_colonly": dict(p10=16, pool="attn", feats="col"),
    "attn_nos2": dict(p10=16, pool="attn", s2=False),
    "attn_ce": dict(p10=16, pool="attn", eps=0.0),
    "attn_nopheno": dict(p10=16, pool="attn", pheno=False),
    "attn_noblockw": dict(p10=16, pool="attn", block_weight=0.0),
    # v5 final: no Sentinel-1 (dataset), no colour statistics, no MODIS curve;
    # its ablations are --set overrides of this entry (run_v5_final.sh)
    "final": dict(p10=16, pool="attn", feats="dino", pheno=False, s2_only=False),
}
# v2 defaults: v1 (D=192, lr=1e-3, wd=0.02, 60 epochs, no feature dropout,
# no block weighting) overfit within 1-10 epochs
DEFAULT = dict(p10=0, p1m=False, ov=False, pool="mean", feats="both", s2=True,
               eps=0.05, pheno=True, D=128, epochs=40, lr=5e-4, wd=0.05, bs=128,
               token_drop=0.2, feat_drop=0.3, block_weight=0.5, seed=0,
               s2_only=True, wk="all", es=1, noyear=0, tsplit=0,
               trunc=0.0, curve="", wksrc="s2")
# noyear=1 zeroes the flight-year site fact (assemble.static_features col 6);
# tsplit=Y replaces the spatial folds by one forward-in-time split: train on
# flights before year Y, predict Y and later (a deployment on new uploads)
YEAR_COL = 6
# trunc=p: in training, with probability p cut an ortho's S2 window at a
# random point (uniform over bins TRUNC_RANGE, independent of the label), as
# for a cube that ends inside the flight year -- the normal case for new
# uploads, whose cube is downloaded shortly after the flight
TRUNC_RANGE = (60.0, 400.0)
# curve=<ver> (with pheno=1): the site phenology curve from
# OUT/pheno_curve_<ver>.npy (build_pheno_curves.py) instead of the dataset's
# (v9, build_labels.py)
# wksrc=modis: the week tokens come from daily MODIS NBAR at ~5 km
# (modis_tokens.py) instead of the S2 cube; the S2-availability site facts
# then say whether MODIS tokens exist (col 7) and 0 (col 8)
# week-token columns (assemble.week_features): 17 S2 bands/indices, clear
# share + S2 flag, 12 match features + match flag
WK_RAW, WK_FLAGS, WK_MATCH = slice(0, 17), slice(17, 19), slice(19, 32)


class ViewEncoder(nn.Module):

    def __init__(self, emb_dim, col_dim, D, feats, pool, scales, feat_drop=0.0):
        super().__init__()
        self.feats, self.pool, self.scales = feats, pool, scales
        if feats in ("both", "dino"):
            self.emb = nn.Sequential(nn.LayerNorm(emb_dim), nn.Dropout(feat_drop),
                                     nn.Linear(emb_dim, D))
        if feats in ("both", "col"):
            self.col = nn.Sequential(nn.Linear(col_dim, D), nn.GELU(),
                                     nn.Linear(D, D))
        self.scale_emb = nn.Parameter(torch.zeros(len(scales), D))
        if pool == "attn":
            self.q = nn.Parameter(torch.randn(1, 1, D) * 0.02)
            self.att = nn.MultiheadAttention(D, 4, batch_first=True)
            self.out = nn.Linear(D, D)
        else:
            self.out = nn.Linear(D * len(scales), D)

    def forward(self, views):
        toks, masks = [], []
        for si, s in enumerate(self.scales):
            e, c, v = views[s]
            t = 0
            if self.feats in ("both", "dino"):
                t = t + self.emb(e.float())
            if self.feats in ("both", "col"):
                t = t + self.col(c)
            toks.append(t + self.scale_emb[si])
            masks.append(v)
        if self.pool == "attn":
            T = torch.cat(toks, 1)
            M = torch.cat(masks, 1)
            allmiss = ~M.any(1)
            M = M.clone()
            M[allmiss, 0] = True  # avoid NaN; output zeroed below
            q = self.q.expand(T.shape[0], -1, -1)
            o, _ = self.att(q, T, T, key_padding_mask=~M)
            o = o[:, 0] * (~allmiss).float()[:, None]
            return self.out(o)
        pooled = []
        for t, m in zip(toks, masks):
            mf = m.float()[..., None]
            pooled.append((t * mf).sum(1) / mf.sum(1).clamp(min=1))
        return self.out(torch.cat(pooled, 1))


class DOYModel(nn.Module):

    def __init__(self, cfg, emb_dim, col_dim, s2_dim, static_dim):
        super().__init__()
        D = cfg["D"]
        self.cfg = cfg
        self.scales = [s for s, on in (("p10", cfg["p10"] > 0), ("p1m", cfg["p1m"]),
                                       ("ov", cfg["ov"])) if on]
        self.views = ViewEncoder(emb_dim, col_dim, D, cfg["feats"], cfg["pool"],
                                 self.scales, cfg["feat_drop"]) if self.scales else None
        self.static = nn.Sequential(nn.Linear(static_dim, 64), nn.GELU(),
                                    nn.Linear(64, 64))
        skeep = torch.ones(static_dim)
        if cfg.get("noyear"):
            skeep[YEAR_COL] = 0
        self.register_buffer("static_keep", skeep, persistent=False)
        self.trunk = nn.Sequential(nn.Linear(D + 64, 256), nn.GELU(),
                                   nn.Dropout(0.1), nn.Linear(256, 256), nn.GELU())
        ang = 2 * np.pi * (np.arange(NBINS) + 0.5) / NBINS
        four = np.concatenate([np.stack([np.sin(k * ang), np.cos(k * ang)], 1)
                               for k in range(1, 5)], 1)
        self.register_buffer("four", torch.tensor(four, dtype=torch.float32))
        nday = 8 + (2 if cfg["pheno"] else 0)
        self.day_in = nn.Linear(nday, 96)
        self.day_ctx = nn.Linear(256, 96)
        self.day_out = nn.Sequential(nn.GELU(), nn.Linear(96, 96), nn.GELU(),
                                     nn.Linear(96, 1))
        if cfg["s2"]:
            self.wk_in = nn.Linear(s2_dim + 3, 128)
            self.wk_ctx = nn.Linear(256, 128)
            self.conv1 = nn.Conv1d(128, 128, 5, padding=2)
            self.conv2 = nn.Conv1d(128, 128, 5, padding=2)
            self.wk_out = nn.Linear(128, 1)
            self.log_bw = nn.Parameter(torch.tensor(np.log(4.0), dtype=torch.float32))
            keep = torch.ones(s2_dim)
            if cfg["wk"] == "raw":
                keep[WK_MATCH] = 0
            elif cfg["wk"] == "match":
                keep[WK_RAW] = 0
            self.register_buffer("wk_keep", keep)

    def forward(self, b, train=False):
        z_s = self.static(b["static"] * self.static_keep)
        if self.views is not None:
            z_o = self.views(b["views"])
        else:
            z_o = torch.zeros(z_s.shape[0], self.cfg["D"], device=z_s.device)
        h = self.trunk(torch.cat([z_o, z_s], 1))
        B = h.shape[0]
        dayf = [self.four[None].expand(B, -1, -1)]
        if self.cfg["pheno"]:
            c = b["curve"]
            slope = torch.roll(c, -7, 1) - torch.roll(c, 7, 1)
            dayf = [c[..., None], slope[..., None]] + dayf
        x = self.day_in(torch.cat(dayf, -1)) + self.day_ctx(h)[:, None]
        logits = self.day_out(x)[..., 0]
        if self.cfg["s2"]:
            pos, m = b["s2pos"], b["s2m"]
            ang = 2 * np.pi * pos / NBINS
            wk = torch.cat([b["s2w"].float() * self.wk_keep, torch.sin(ang)[..., None],
                            torch.cos(ang)[..., None], (pos / NBINS)[..., None]], -1)
            t = F.gelu(self.wk_in(wk) + self.wk_ctx(h)[:, None])
            t = t * m[..., None]
            t = t.transpose(1, 2)
            t = F.gelu(self.conv1(t)) * m[:, None]
            t = F.gelu(self.conv2(t)) * m[:, None]
            s = self.wk_out(t.transpose(1, 2))[..., 0]  # (B, W)
            days = torch.arange(NBINS, device=pos.device).float() + 0.5
            bw = self.log_bw.exp()
            k = torch.exp(-0.5 * ((days[None, :, None] - pos[:, None, :]) / bw)**2)
            k = k * m[:, None, :].float()
            logits = logits + (k * s[:, None, :]).sum(-1) / (k.sum(-1) + 0.1)
        return logits


def robust_loss(logits, q, eps, w=None):
    """Per-sample loss averaged with weights w (block weighting)."""
    if w is None:
        w = torch.ones(len(logits), device=logits.device)
    w = w / w.sum()
    logp = F.log_softmax(logits, 1)
    lik = torch.logsumexp(logp + torch.log(q.clamp(min=1e-12)), 1)  # log sum p q
    if eps <= 0:
        return (-(q * logp).sum(1) * w).sum()  # plain soft CE (HL-Gauss)
    a = torch.log(torch.tensor(eps / NBINS, device=logits.device))
    b = torch.log(torch.tensor(1 - eps, device=logits.device)) + lik
    return (-torch.logaddexp(a, b) * w).sum()


def build_tensors(ds, cfg, dev):
    T = {"static": torch.tensor(ds["static"], device=dev),
         "curve": torch.tensor(ds["curves"], device=dev),
         "target": torch.tensor(ds["target"], device=dev)}
    if cfg["s2"]:
        T["s2w"] = torch.tensor(ds["s2w"], device=dev)
        T["s2pos"] = torch.tensor(ds["s2pos"], device=dev)
        T["s2m"] = torch.tensor(ds["s2m"], device=dev)
    for s in ("p10", "p1m", "ov"):
        if f"{s}_emb" in ds:
            T[f"{s}_emb"] = torch.tensor(ds[f"{s}_emb"], device=dev)
            T[f"{s}_col"] = torch.tensor(ds[f"{s}_col"], device=dev)
            T[f"{s}_valid"] = torch.tensor(ds[f"{s}_valid"], device=dev)
    return T


def make_batch(T, idx, cfg, col_stats, train, gen):
    b = {k: T[k][idx] for k in ("static", "curve", "target")}
    if cfg["s2"]:
        b["s2w"], b["s2pos"], b["s2m"] = T["s2w"][idx], T["s2pos"][idx], T["s2m"][idx]
        if train and cfg["trunc"] > 0:
            n = len(idx)
            cut = TRUNC_RANGE[0] + (TRUNC_RANGE[1] - TRUNC_RANGE[0]) * torch.rand(
                n, device=idx.device, generator=gen)
            on = torch.rand(n, device=idx.device, generator=gen) < cfg["trunc"]
            cut = torch.where(on, cut, torch.full_like(cut, 1e9))
            b["s2m"] = b["s2m"] & (b["s2pos"] < cut[:, None])
    views = {}
    for s in ("p10", "p1m", "ov"):
        n = cfg["p10"] if s == "p10" else (4 if s == "p1m" else 1)
        on = cfg["p10"] > 0 if s == "p10" else cfg[s]
        if not on:
            continue
        e, c, v = T[f"{s}_emb"][idx], T[f"{s}_col"][idx], T[f"{s}_valid"][idx]
        if s == "p10":
            if train:
                # a fresh random subset of the 16 stored patches each step
                perm = torch.argsort(torch.rand(len(idx), 16, device=e.device,
                                                generator=gen), 1)[:, :n]
            else:
                perm = torch.arange(n, device=e.device)[None].expand(len(idx), -1)
            e = torch.gather(e, 1, perm[..., None].expand(-1, -1, e.shape[-1]))
            c = torch.gather(c, 1, perm[..., None].expand(-1, -1, c.shape[-1]))
            v = torch.gather(v, 1, perm)
        c = (c - col_stats[s][0]) / col_stats[s][1]
        if train and cfg["token_drop"] > 0 and s != "ov":
            drop = torch.rand(v.shape, device=v.device, generator=gen) < cfg["token_drop"]
            keep_one = v & ~drop
            v = torch.where(keep_one.any(1, keepdim=True), keep_one, v)
        views[s] = (e, c, v)
    b["views"] = views
    return b


@torch.no_grad()
def predict(model, T, idx, cfg, col_stats, bs=512):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        out.append(F.softmax(model(make_batch(T, j, cfg, col_stats, False, None)), 1))
    return torch.cat(out).cpu().numpy()


def run(name, cfg, dev, save_all=False, deploy=False):
    """Cross-validated fit (OOF predictions), or with deploy=True one fit on
    every labelled ortho, saved with what inference needs (deploy.pt).

    es=1: early stopping on an inner validation split (10 % of the training
    groups); es=0: no inner split, the model after the last epoch is kept --
    the recipe deploy mode needs, since it has no held-out data to stop on."""
    lab, ds = load()  # the benchmark files (bench.py)
    if cfg["curve"]:
        assert cfg["pheno"], "curve=<ver> needs pheno=1"
        c = np.load(f"{OUT}/pheno_curve_{cfg['curve']}.npy")
        assert c.shape == ds["curves"].shape and np.isfinite(c).all()
        ds["curves"] = c
    if cfg["wksrc"] == "modis":
        z = np.load(f"{OUT}/modis_tokens.npz")
        ds["s2w"], ds["s2pos"], ds["s2m"] = z["mw"], z["mpos"], z["mm"]
        ds["static"] = ds["static"].copy()
        ds["static"][:, 7] = z["has_modis"]
        ds["static"][:, 8] = 0
    T = build_tensors(ds, cfg, dev)
    folds = lab["fold"].values
    has_label = ds["has_label"]
    full = (lab["tier"] == "full").values
    groups = lab["group"].values
    N = len(lab)
    oof = np.zeros((N, NBINS), np.float32)
    allp = np.zeros((N_FOLDS, N, NBINS), np.float16) if save_all else None
    log = {}
    out_dir = f"{RUNS_DIR}/{'deploy_' if deploy else ''}{name}"
    os.makedirs(out_dir, exist_ok=True)
    emb_dim = ds["p10_emb"].shape[-1]
    col_dim = ds["p10_col"].shape[-1]
    s2_dim = ds["s2w"].shape[-1]
    # dataset from the current assemble.py
    assert cfg["wksrc"] != "s2" or s2_dim == WK_MATCH.stop, s2_dim
    assert not deploy or not cfg["es"], "deploy fits need es=0"
    if cfg["tsplit"]:
        assert not deploy
        folds = np.where(lab["year"].values >= cfg["tsplit"], 0, -1)
    for k in ([N_FOLDS] if deploy else [0] if cfg["tsplit"] else range(N_FOLDS)):
        seed = SEED + k + 1000 * cfg["seed"]
        torch.manual_seed(seed)
        # the inner-validation groups depend on the fold only, not the seed
        rng = np.random.default_rng(SEED + k)
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        # from v3 on, only sites with S2 (Clemens: S2 is essentially required)
        site_ok = ds["has_match"] if cfg["s2_only"] else np.ones(N, bool)
        tr_all = np.where(has_label & site_ok & (folds != k))[0]
        ug = np.unique(groups[tr_all])
        val_g = set(rng.choice(ug, max(1, int(0.1 * len(ug))), replace=False)) \
            if cfg["es"] else set()
        is_val = np.array([g in val_g for g in groups[tr_all]])
        tr, va = tr_all[~is_val], tr_all[is_val]
        va = va[full[va]]
        col_stats = {}
        for s in ("p10", "p1m", "ov"):
            if f"{s}_col" in T:
                c = T[f"{s}_col"][torch.tensor(tr, device=dev)]
                v = T[f"{s}_valid"][torch.tensor(tr, device=dev)]
                cc = c[v]
                col_stats[s] = (cc.mean(0), cc.std(0).clamp(min=1e-4))
        model = DOYModel(cfg, emb_dim, col_dim, s2_dim, ds["static"].shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        steps = cfg["epochs"] * int(np.ceil(len(tr) / cfg["bs"]))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps,
                                                    pct_start=0.1)
        best, best_state, hist = np.inf, None, []
        trt = torch.tensor(tr, device=dev)
        vat = torch.tensor(va, device=dev)
        vbin = np.floor(lab["bin"].values[va]).astype(int) % NBINS
        # heavy spatial clusters (one group = 1,080 orthos): weight every
        # geographic group equally in validation, and by 1/sqrt(size) in
        # training (block_weight=0 switches the training weights off)
        vg = groups[va]
        _, vinv, vcnt = np.unique(vg, return_inverse=True, return_counts=True)
        vw = 1.0 / vcnt[vinv]
        vw /= vw.sum()
        gsz = pd.Series(groups[tr]).map(pd.Series(groups[tr]).value_counts()).values
        tw = torch.tensor(gsz.astype(np.float32)**(-cfg["block_weight"]), device=dev)
        tw_all = torch.zeros(N, device=dev)
        tw_all[torch.tensor(tr, device=dev)] = tw
        t0 = time.time()
        for ep in range(cfg["epochs"]):
            model.train()
            perm = trt[torch.randperm(len(trt), device=dev, generator=gen)]
            tl = 0
            for i in range(0, len(perm), cfg["bs"]):
                idx = perm[i:i + cfg["bs"]]
                b = make_batch(T, idx, cfg, col_stats, True, gen)
                loss = robust_loss(model(b, True), b["target"], cfg["eps"], tw_all[idx])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tl += loss.item() * len(idx)
            if cfg["es"]:
                pv = predict(model, T, vat, cfg, col_stats)
                vnll = -(np.log(np.clip(pv[np.arange(len(va)), vbin], 1e-9,
                                        None)) * vw).sum()
            else:
                vnll = -float(ep)  # no inner validation: keep the last epoch
            hist.append((ep, tl / len(tr), float(vnll)))
            if not np.isfinite(vnll):
                print(f"[{name}] fold {k} ep {ep}: non-finite val NLL", flush=True)
            if best_state is None or vnll < best:
                best = vnll
                best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        if deploy:
            torch.save(dict(state=best_state, cfg=cfg, emb_dim=emb_dim, col_dim=col_dim,
                            s2_dim=s2_dim, static_dim=ds["static"].shape[1],
                            col_stats={s: (a.cpu(), b.cpu()) for s, (a, b) in
                                       col_stats.items()},
                            n_train=len(tr)), f"{out_dir}/deploy.pt")
            json.dump(dict(cfg=cfg, n_train=len(tr), secs=time.time() - t0,
                           hist=hist), open(f"{out_dir}/log.json", "w"))
            print(f"[{name}] deploy fit on {len(tr)} orthos "
                  f"({time.time() - t0:.0f}s)", flush=True)
            return
        te = np.where(folds == k)[0]
        oof[te] = predict(model, T, torch.tensor(te, device=dev), cfg, col_stats)
        if save_all:
            allp[k] = predict(model, T, torch.arange(N, device=dev), cfg, col_stats)
            torch.save(best_state, f"{out_dir}/model_fold{k}.pt")
        best_ep = int(np.argmin([h[2] for h in hist]))
        log[k] = dict(best_val_nll=float(best), best_epoch=best_ep, n_train=len(tr),
                      n_val=len(va), secs=time.time() - t0, hist=hist)
        print(f"[{name}] fold {k} best val NLL {best:.3f} @ep{best_ep} "
              f"({time.time() - t0:.0f}s)", flush=True)
    np.save(f"{out_dir}/oof.npy", oof.astype(np.float16))
    if save_all:
        np.save(f"{out_dir}/allfolds.npy", allp)
    json.dump(dict(cfg=cfg, log=log), open(f"{out_dir}/log.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--save-all", action="store_true")
    ap.add_argument("--deploy", action="store_true",
                    help="one fit on all labelled orthos (needs es=0)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--set", nargs="*", default=[],
                    help="config overrides key=value (numbers parsed)")
    args = ap.parse_args()
    dev = "cuda"
    for v in args.variants:
        cfg = dict(DEFAULT)
        cfg.update(VARIANTS[v])
        if args.epochs:
            cfg["epochs"] = args.epochs
        for kv in args.set:
            key, val = kv.split("=")
            cfg[key] = type(DEFAULT[key])(float(val)) if not isinstance(
                DEFAULT[key], str) else val
        run(v + args.suffix, cfg, dev, args.save_all, args.deploy)


if __name__ == "__main__":
    main()
