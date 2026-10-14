"""Minimal example: train a small model on the benchmark features with the paper's
five spatial folds and score it with evaluate.py. It is a starting point, not the
paper's model (that is AerialStamp, https://github.com/cmosig/aerialstamp).

Model: mean of the 16 DINOv2 patch embeddings + site facts -> MLP -> 365 day bins,
trained with cross-entropy against the soft targets. Roughly the paper's image-only
model without attention pooling, ensembling and calibration.

    python example_train.py /path/to/features.npz   ->  example_predictions.npy
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluate import evaluate

meta = pd.read_csv("metadata.csv")
f = np.load(sys.argv[1])
assert (f["id"] == meta["id"].values).all()
emb = f["patch_emb"].astype(np.float32)
valid = f["patch_valid"].astype(np.float32)[..., None]
x = np.concatenate([(emb * valid).sum(1) / np.clip(valid.sum(1), 1, None), f["site_facts"]], 1)
y = f["target"].astype(np.float32)
dev = "cuda" if torch.cuda.is_available() else "cpu"
pred = np.full((len(meta), 365), 1 / 365, np.float32)
for fold in range(5):
    tr = (meta["train"] & (meta["fold"] != fold)).values
    te = (meta["fold"] == fold).values
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    X = torch.tensor((x - mu) / sd, device=dev)
    Y = torch.tensor(y, device=dev)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(x.shape[1], 256), nn.GELU(), nn.Dropout(0.3),
                        nn.Linear(256, 365)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 5e-4, weight_decay=0.05)
    idx = torch.tensor(np.where(tr)[0], device=dev)
    for epoch in range(8):
        net.train()
        for b in idx[torch.randperm(len(idx), device=dev)].split(128):
            loss = -(Y[b] * torch.log_softmax(net(X[b]), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred[te] = torch.softmax(net(X[torch.tensor(np.where(te)[0], device=dev)]), 1).cpu().numpy()
    print("fold", fold, "done")
np.save("example_predictions.npy", pred)
print(evaluate(pred, meta))
