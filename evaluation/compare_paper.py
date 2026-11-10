"""Comparison A: task 1 (authors' code) vs the published 2025 and 2024 models.

A1: test_multispecies, plus task 1's training-vs-test (generalisation) view.
A2: test_sigma70, for comparison only (no generalisation analysis).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef

import common


def at_threshold(y, s, threshold=0.5):
    """Classification metrics of score >= 0.5, as the paper reports them; used only here, to compare with its tables."""
    y, p = np.asarray(y).astype(int), (np.asarray(s) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"mcc": matthews_corrcoef(y, p), "accuracy": accuracy_score(y, p), "f1": f1_score(y, p, zero_division=0),
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1), "precision": tp / max(tp + fp, 1)}


paper = pd.read_csv(common.ROOT / "data/paper_reported.tsv", sep="\t")
for test, folder in (("multispecies", "A1_multispecies"), ("sigma70", "A2_sigma70")):
    entries = [("task1", common.read_scores("task1", test)),
               ("published-2025", common.read_scores("published-2025", test)),
               ("published-2024", common.read_scores("published-2024", test))]
    seeds = [("task1", m, common.read_scores(m, test)) for m in common.seed_models("task1")]
    reference = paper[paper.dataset == test].drop(columns="dataset").assign(model="paper (reported, mini)")
    common.compare(common.ROOT / f"results/{folder}", f"A: {test}", entries, seeds, reference, extra=at_threshold)

common.generalisation(common.ROOT / "results/A1_multispecies", "task1",
                      [("train", common.read_scores("task1", "train_task1")),
                       ("multispecies", common.read_scores("task1", "multispecies"))])
