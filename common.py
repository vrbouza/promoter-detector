"""Shared small functions and declarations"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from transformers import AutoModelForSequenceClassification, AutoTokenizer, MegatronBertConfig, MegatronBertModel
from transformers.dynamic_module_utils import get_class_from_dynamic_module

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =================================================
### Constants

ROOT = Path(__file__).resolve().parent
SEED = 42
# Base model + remote code as of 2025-01-09, the day of the authors' 2025 notebook (main changed in 2026).
BASE, BASE_REV = "neuralbioinfo/prokbert-mini", "219ce7271928902e84945fa62ad3703bafe746e1"
PUBLISHED = "neuralbioinfo/prokbert-mini-promoter"
PUB_REV = {"published-2025": "8502f02d1019017fb44244274664e17e0e30ce1e",   # model.safetensors
           "published-2024": "96bfe4a1b1f907a265d779625eae348992819877"}   # pytorch_model.bin (paper era)
DATASET, DATASET_REV = "neuralbioinfo/bacterial_promoters", "38f748a5ea68896c9467087e41c0210e5d5b5760"
# Token ids whose attention mask is zeroed, as each model was trained (2=[CLS], 3=[SEP]).
MASKED_IDS = {"published-2024": (3,)}


# =================================================
### Functions

def load_hf(): # this loads the dataset from huggingface
    return load_dataset(DATASET, revision=DATASET_REV)


def load_pool(): # this loads the dataset from hugging face, AND adds our own data, and creates separations from them
    """HF train + HF multispecies + our D39V/TIGR4 data, one row per sequence."""
    hf, parts = load_hf(), []
    for split, origin in (("train", "hf_train"), ("test_multispecies", "hf_multispecies")):
        d = hf[split].to_pandas()
        parts.append(pd.DataFrame({"segment_id": d.segment_id.astype(str), "segment": d.segment, "y": d.y,
                                   "origin": origin, "species": d.ppd_original_SpeciesName.str.strip(),
                                   "kind": d.prom_class, "sigma": "-"}))
    pn = pd.read_csv(ROOT / "data/pneumo.csv", keep_default_na=False)
    parts.append(pd.DataFrame({"segment_id": pn.id, "segment": pn.segment, "y": pn.y, "origin": "pneumo",
                               "species": "Streptococcus pneumoniae", "kind": pn.src, "sigma": pn.sigma}))
    return pd.concat(parts, ignore_index=True)


def load_split(task, pool=None): # the splits are loaded here, from the consensus ones. We write them in a file because we are reusing them across tasks
    """{'train', 'val', 'test'} DataFrames for a task, from data/splits/task<N>.tsv."""
    pool = load_pool() if pool is None else pool
    ids = pd.read_csv(ROOT / f"data/splits/task{task}.tsv", sep="\t", dtype=str)
    df = ids.merge(pool, on="segment_id", how="left", validate="one_to_one")
    assert df.segment.notna().all(), "split file has ids missing from the pool"
    return {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}


def load_sigma70():
    d = load_hf()["test_sigma70"].to_pandas()
    return pd.DataFrame({"segment_id": d.segment_id.astype(str), "segment": d.segment, "y": d.y, "origin": "sigma70"})


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BASE, revision=BASE_REV, trust_remote_code=True)


def load_model(name):
    """'published-2025', 'published-2024' or a local directory saved by our training scripts."""
    if name == "published-2024": ## this one is older and different
        cls = get_class_from_dynamic_module("models.BertForBinaryClassificationWithPooling", BASE, revision=BASE_REV)
        model = cls(MegatronBertModel(MegatronBertConfig.from_pretrained(PUBLISHED, revision=PUB_REV[name])))
        weights = hf_hub_download(PUBLISHED, "pytorch_model.bin", revision=PUB_REV[name])
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))  # strict
        return model
    path, rev = (PUBLISHED, PUB_REV[name]) if name in PUB_REV else (str(name), None)
    return AutoModelForSequenceClassification.from_pretrained(path, revision=rev, code_revision=BASE_REV,
                                                              trust_remote_code=True)


def encode(tokenizer, seqs, masked_ids=(2, 3)):
    enc = tokenizer(list(seqs), padding=True, add_special_tokens=True, return_tensors="pt")
    mask = enc["attention_mask"]
    for i in masked_ids:
        mask[enc["input_ids"] == i] = 0
    return {"input_ids": enc["input_ids"], "attention_mask": mask}


@torch.no_grad() # this disables gradient descent. Given that we don't want to train nothing here, it is useful to speed up inference like the one we're doing here.
def score(model, tokenizer, seqs, masked_ids=(2, 3), batch_size=1024):
    """Probability of the promoter class for each sequence."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    seqs, out = list(seqs), []
    for i in range(0, len(seqs), batch_size):
        enc = {k: v.to(device) for k, v in encode(tokenizer, seqs[i:i + batch_size], masked_ids).items()}
        out.append(torch.softmax(model(**enc)["logits"].float(), -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def metrics(y, s):
    """Threshold-free metrics. Ranking: AUC and AP (average precision). Probabilities: Brier score and its skill score,
    1 - Brier / pi(1 - pi), the gain over always predicting the promoter share pi (1 = perfect, 0 = no gain)."""
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    brier, pi = brier_score_loss(y, s), y.mean()
    return {"n": len(y), "n_pos": int(y.sum()), "auc": roc_auc_score(y, s), "ap": average_precision_score(y, s),
            "brier": brier, "brier_skill": 1 - brier / (pi * (1 - pi))}


def write_tsv(df, path): # pretty descriptive...
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(df).to_csv(path, sep="\t", index=False, float_format="%.4f")


def roc_plot(curves, path, title): ## plottingggg roc curves!
    """curves: list of (label, y, score). One ROC overlay per file."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, y, s in curves:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=1.5, label=f"{label} (AUC {roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(fontsize=8, loc="lower right")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def calibration_plot(curves, path, title, seed_curves=()):
    """Reliability diagram: observed share of promoters against predicted probability (10 bins), one colour per model
    (thin lines: its single seeds), with the spread of predicted probabilities below. curves: [(label, y, score)]."""
    fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(5.5, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    for i, (label, y, s) in enumerate(curves):
        for seed_label, sy, ss in seed_curves:
            if seed_label == label:
                observed, predicted = calibration_curve(sy, ss, n_bins=10)
                ax.plot(predicted, observed, color=f"C{i}", lw=0.8, alpha=0.4)
        observed, predicted = calibration_curve(y, s, n_bins=10)
        ax.plot(predicted, observed, "o-", color=f"C{i}", lw=1.5, ms=3, label=label)
        ax_hist.hist(s, bins=20, range=(0, 1), histtype="step", color=f"C{i}", weights=np.full(len(s), 1 / len(s)))
    handles = ax.get_legend_handles_labels()[0]
    if seed_curves:
        handles.append(Line2D([], [], color="grey", lw=0.8, alpha=0.6, label="single seeds"))
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.set(ylabel="observed share of promoters", title=title)
    ax_hist.set(xlabel="predicted probability of promoter", ylabel="share of sequences")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def score_histogram(parts, path, title):
    """Predicted probability of promoters (blue) and non-promoters (red) in each set: the first set (training) filled,
    the others as points with Poisson error bars. Each histogram has unit area, so sets of different size compare."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.linspace(0, 1, 26)
    centres, width = (bins[:-1] + bins[1:]) / 2, bins[1] - bins[0]
    for k, (name, d) in enumerate(parts):
        for y, color, cls in ((1, "C0", "promoters"), (0, "C3", "non-promoters")):
            s = d.score[d.y == y].to_numpy()
            if k == 0:
                ax.hist(s, bins, density=True, histtype="stepfilled", color=color, alpha=0.3, label=f"{cls}, {name}")
            else:
                counts, norm = np.histogram(s, bins)[0], 1 / (len(s) * width)
                ax.errorbar(centres, counts * norm, yerr=np.sqrt(counts) * norm, fmt="os^D"[k - 1], color=color, ms=3,
                            lw=0.8, label=f"{cls}, {name}")
    ax.set(xlabel="predicted probability of promoter", ylabel="density (unit area)", yscale="log", title=title)
    ax.legend(fontsize=8, loc="upper center")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def val_row(seed, model, tokenizer, val):
    """Validation metrics of one seed: overall, and per origin (HF / pneumo) when present."""
    s = score(model, tokenizer, val.segment)
    row = {"seed": seed, "val_auc": roc_auc_score(val.y, s)}
    for name, mask in (("hf", val.origin.str.startswith("hf")), ("pneumo", val.origin == "pneumo")):
        if mask.any():
            row[f"val_auc_{name}"] = roc_auc_score(val.y[mask], s[mask])
    return row


def save_seed(seed_dir, row):
    """Store one seed's validation metrics (full precision) next to its model, for training/summarise_seeds.py."""
    pd.DataFrame([row]).to_csv(Path(seed_dir) / "val.tsv", sep="\t", index=False)


def read_scores(model, name):
    return pd.read_csv(ROOT / f"results/scores/{model}/{name}.tsv", sep="\t", keep_default_na=False)


def seed_models(run):
    """Names of the per-seed score folders of a training run, e.g. task1_seed_42."""
    return sorted(p.name for p in (ROOT / "results/scores").glob(f"{run}_seed_*"))


def compare(out_dir, title, entries, seeds=(), reference=None, extra=None):
    """Compare models on one test set.

    entries: [(label, scores)]; seeds: [(task label, seed model name, scores)]; reference: rows shown for context;
    extra: optional function (y, score) -> {column: value}, more metrics.
    Writes metrics.tsv, roc.png, calibration.png and seeds.tsv.
    """
    out_dir = Path(out_dir)
    more = extra or (lambda y, s: {})
    met = pd.DataFrame([{"model": lab} | metrics(d.y, d.score) | more(d.y, d.score) for lab, d in entries])
    if reference is not None:
        met = pd.concat([met, reference], ignore_index=True)
    met = met.astype({"n": "Int64", "n_pos": "Int64"})
    write_tsv(met, out_dir / "metrics.tsv")
    roc_plot([(lab, d.y, d.score) for lab, d in entries], out_dir / "roc.png", title)
    calibration_plot([(lab, d.y, d.score) for lab, d in entries], out_dir / "calibration.png", title,
                     [(lab, d.y, d.score) for lab, _, d in seeds])
    if seeds:
        df = pd.DataFrame([{"model": lab, "seed": name.rsplit("_", 1)[-1]} | metrics(d.y, d.score) | more(d.y, d.score)
                           for lab, name, d in seeds]).drop(columns=["n", "n_pos"])
        cols = df.columns.drop(["model", "seed"])
        stats = [{"model": lab, "seed": stat} | df.loc[df.model == lab, cols].agg(stat).to_dict()
                 for lab in df.model.unique() for stat in ("mean", "std")]
        write_tsv(pd.concat([df, pd.DataFrame(stats)], ignore_index=True), out_dir / "seeds.tsv")
    print(f"\n== {title}\n{met.to_string(index=False)}")


def gof_tests(a, b):
    """Two-sample goodness of fit of score samples a and b: Kolmogorov-Smirnov distance D and its p-value."""
    ks = ks_2samp(a, b)
    return {"ks_D": ks.statistic, "ks_p": ks.pvalue}


def generalisation(out_dir, label, parts, gof_pairs=None):
    """parts: [(set name, scores)], the training set first. gof_pairs: [(training set, test set)] names for the
    goodness-of-fit tests (default: the first set against each other one).

    Writes generalisation_<label>.tsv, train_vs_test_<label>.png (ROC), train_vs_test_hist_<label>.png (scores) and
    gof_<label>.tsv (KS test, training vs test scores, per class).
    """
    df = pd.DataFrame([{"model": label, "set": name} | metrics(d.y, d.score) for name, d in parts])
    df["gap_auc"] = df.auc.iloc[0] - df.auc
    write_tsv(df, Path(out_dir) / f"generalisation_{label}.tsv")
    roc_plot([(name, d.y, d.score) for name, d in parts], Path(out_dir) / f"train_vs_test_{label}.png",
             f"{label}: training vs test")
    score_histogram(parts, Path(out_dir) / f"train_vs_test_hist_{label}.png", f"{label}: training vs test")
    sets = dict(parts)
    rows = []
    for train, test in gof_pairs or [(parts[0][0], name) for name, _ in parts[1:]]:
        for y, cls in ((1, "promoters"), (0, "non-promoters")):
            a, b = (sets[n].score[sets[n].y == y].to_numpy() for n in (train, test))
            rows.append({"model": label, "train": train, "test": test, "class": cls, "n_train": len(a), "n_test": len(b)}
                        | gof_tests(a, b))
    pd.DataFrame(rows).to_csv(Path(out_dir) / f"gof_{label}.tsv", sep="\t", index=False, float_format="%.4g")
