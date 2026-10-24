# AerialStamp benchmark: when was this aerial orthophoto flown?

A benchmark for estimating the **acquisition date (day of year) of a drone or airborne
orthophoto** from the image, its location and submitted year, and the weekly Sentinel-2
time series of the site. It accompanies the paper

> Mosig, C., Montero Loaiza, D., Mahecha, M. D., Vajna-Jehle, J., Kattenborn, T. (2026).
> *AerialStamp: dating aerial orthophotos from visual cues and Sentinel-2 time series.*
> EarthArXiv (preprint). https://doi.org/10.31223/X5SZ4Q

and holds the data of the paper's experiments, its evaluation protocol and the
out-of-fold predictions of its models as baselines. The trained models are in the
[AerialStamp package](https://github.com/cmosig/aerialstamp).

**6,984 orthophotos** from [deadtrees.earth](https://deadtrees.earth) in 129 countries,
flown 2015–2026, all publicly available under CC BY; **6,225** of them form the
evaluation set. This is exactly the data of the paper.

## Task

For every orthophoto, predict a probability for each of the 365 day bins of the
submitted year (bin `b` covers the fraction `[b/365, (b+1)/365)` of the year, so leap
years lose one bin). The year is given; the day within the year is the target.

The labels are the dates that contributors submitted. They are mostly right, but not
all: the paper estimates that at least 1.3 % are wrong. Dates on the 1st or 15th of a
month and on 1 January are over-represented and probably often placeholders; they are
excluded from training and evaluation, but included in the data (column `label`).

## Files

Repository (this folder):

| file | content |
|---|---|
| `metadata.csv` | one row per orthophoto: id, link, location, biome, date, split, licence, authors |
| `columns.json` | names of the week-token and site-fact columns |
| `evaluate.py` | the evaluation protocol (metrics of the paper) |
| `example_train.py` | a minimal model on the features, as a starting point |
| `paper/` | the paper's training and calibration code, reading the benchmark files (see "Reproducing the paper") |

Data archive [`aerialstamp-benchmark-v1.zip`](https://data2.deadtrees.earth/assets/v1/aerialstampbench/aerialstamp-benchmark-v1.zip) (341 MB; see "Download"), rows aligned with `metadata.csv`:

| file | array | shape | content |
|---|---|---|---|
| `features.npz` | `patch_emb` | (6984, 16, 1536) | DINOv2 ViT-B/14-reg4 embedding (class token + mean patch token) of 16 random 10 cm patches (224 × 224 px, 22.4 m) inside the area of interest |
| | `patch_valid` | (6984, 16) | patch exists |
| | `week_tokens` | (6984, 64, 32) | one row per weekly Sentinel-2 composite from 35 days before to 35 days after the submitted year: area mean of 12 bands (square root of reflectance), NDVI, NDMI, NGRDI (column `grvi`), chromaticity, clear-sky share, and 12 spatial correlation features between the orthophoto (averaged to 10 m) and that week's Sentinel-2 image (`columns.json`) |
| | `week_pos`, `week_mask` | (6984, 64) | position of the week in day bins (outside 0–365 for the padding weeks); week exists |
| | `site_facts` | (6984, 8) | sine/cosine of latitude and longitude, latitude, hemisphere, scaled year, log ground sampling distance |
| | `target` | (6984, 365) | training target: wrapped Gaussian (σ = 3 days) on the submitted day, or uniform over the month for month-only labels |
| `baselines.npz` | `s2_model`, `image_only_model`, `climatology`, `phenology`, `most_common_date` | (6984, 365) | out-of-fold predicted distributions of the paper's models and baselines |
| `patches.npy` (separate, 17 GB) | | (6984, 16, 224, 224, 3) uint8 | the raw 10 cm RGB patches, for models that do not start from DINOv2 embeddings |

`metadata.csv` columns: `index` (row in the arrays), `id` and `url` (deadtrees.earth
dataset), `lat`, `lon`, `country`, `biome`, `platform`, `gsd_cm`, `year`, `month`, `day`,
`day_bin` (target position, NaN for month-only labels), `label` (`clean`,
`placeholder_1st_15th`, `placeholder_jan1`, `train_month_only`), `train`, `evaluate`,
`calibrate` (rows the paper's calibration is fitted on: all full dates),
`fold` (0–4), `block` (spatial block), `has_aoi`, `license`, `authors`, `citation_doi`.

## Protocol

- **Spatial block cross-validation.** Orthophotos are grouped into 1,308 spatial
  blocks (50 km cells, merged when flights share uploader or authors and date), and
  the blocks are split into five folds, stratified by biome. Predict every fold with
  a model trained on the other four (`train == True` rows only). Do not split blocks:
  neighbouring flights are near-duplicates.
- **Evaluation rows** are `evaluate == True` (clean full dates). Rows on placeholder
  days get predictions too, but are not scored.
- Anything fitted on labels (calibration, thresholds, model selection) must be fitted
  per fold on the other folds.

```bash
python evaluate.py baselines.npz --key s2_model      # reproduces the paper's Table 3
python evaluate.py baselines.npz --key most_common_date --point mode
python evaluate.py my_predictions.npy --by biome
python example_train.py features.npz
```

## Baselines

| method | input | MAE | MAE bw | median | within 30 d | coverage 80 % | NLL |
|---|---|---|---|---|---|---|---|
| Most common date¹ | most common flight day of the training folds | 62.5 | 66.3 | 51.5 | .32 | .80 | 5.71 |
| Climatology | flight dates of nearby training orthophotos | 55.0 | 59.1 | 41.0 | .39 | .74 | 5.60 |
| Phenology | land surface phenology of the site | 56.7 | 59.6 | 43.5 | .36 | .81 | 5.65 |
| `example_train.py` | mean patch embedding, site facts | 47.6 | 53.8 | 33.5 | .47 | .61 | 5.57 |
| AerialStamp image-only model | patch embeddings, site facts | 38.1 | 41.2 | 25.5 | .56 | .82 | 5.14 |
| **AerialStamp S2 model** | patch embeddings, site facts, week tokens | **30.3** | **33.2** | **18.5** | **.66** | **.80** | **4.89** |

Errors in days. MAE bw weights every spatial block equally. Coverage 80 %: share of
orthophotos whose submitted date lies in the smallest set of days holding 80 %
probability; a calibrated model reaches .80.

¹ The predicted date is the mode of the density of the training flight dates (the most
common flight day), which `baselines.npz` stores; score it with
`python evaluate.py baselines.npz --key most_common_date --point mode`. All other rows use
the circular median (the default).

## Reproducing the paper

`paper/` holds the training code of the paper (`model.py`), its calibration and a
script that runs the experiments on the benchmark files (PyTorch, one GPU):

```bash
export AERIALSTAMP_BENCHMARK=/path/to/aerialstamp-benchmark-v1   # folder with features.npz
python paper/reproduce.py main          # Table 3: both models, 5 seeds each (~30 min)
python paper/reproduce.py ablation      # Table 5, the rows the benchmark supports
python paper/reproduce.py plausibility  # Table 6, from baselines.npz, no training
```

`main` gave MAE 30.1 / block-weighted 32.9 / median 18.5 days for the S2 model and
38.1 / 41.2 / 25.8 for the image-only model (paper: 30.3 / 33.2 / 18.5 and
38.1 / 41.2 / 25.5); the differences are seed noise. `plausibility` reproduces the
paper's table exactly.

Not reproducible from the benchmark files, because they need the orthophotos or
Sentinel-2 cubes themselves: the DINOv3 comparison, the shifted-year test, the
ablations with colour statistics or coarser image views, and
the image figures. The forward-in-time test can be run with `tsplit=2025` in
`model.py`.

## Download

The data archive (341 MB) is not in this repository; download it from
deadtrees.earth:

```bash
wget https://data2.deadtrees.earth/assets/v1/aerialstampbench/aerialstamp-benchmark-v1.zip
unzip aerialstamp-benchmark-v1.zip
export AERIALSTAMP_BENCHMARK=$PWD/aerialstamp-benchmark-v1
```

It holds `features.npz` and `baselines.npz` together with a copy of the files of this
repository. The raw patches (`patches.npy`, 17 GB) are not online yet; ask the
authors if you need them.

## Licence and attribution

The orthophotos were contributed to deadtrees.earth by many authors under the
CC BY licence (all 6,984 rows). The derived features and patches inherit it: credit
the `authors` and `citation_doi` given per row in `metadata.csv` for the orthophotos
you use. Sentinel-2 data: Copernicus,
European Space Agency. Code and files in this repository are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see `LICENSE`).

Please cite the paper above and deadtrees.earth (Mosig et al., 2026) when using the
benchmark:

```bibtex
@article{mosig2026aerialstamp,
  title   = {AerialStamp: dating aerial orthophotos from visual cues and Sentinel-2 time series},
  author  = {Mosig, Clemens and Montero Loaiza, David and Mahecha, Miguel D. and
             Vajna-Jehle, Janusch and Kattenborn, Teja},
  journal = {EarthArXiv},
  note    = {Preprint},
  year    = {2026},
  doi     = {10.31223/X5SZ4Q},
  url     = {https://doi.org/10.31223/X5SZ4Q}
}
```
