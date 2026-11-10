"""Comparison C: does our training recipe (task 3) beat the authors' recipe (task3_authors) on the same data?

Both are trained on task 3's split (HF train + multispecies + our data, stratified by species and sigma factor).
C1: task 3's test split, plus each model's training-vs-test view. C2: test_sigma70, comparison only, with published-2025.
"""
import common

RUNS = ("task3", "task3_authors")


def entries(test):
    return [(run, common.read_scores(run, test)) for run in RUNS]


def seeds(test):
    return [(run, m, common.read_scores(m, test)) for run in RUNS for m in common.seed_models(run)]


common.compare(common.ROOT / "results/C1_test", "C: task 3 test split", entries("task3_test"), seeds("task3_test"))
for run in RUNS:
    common.generalisation(common.ROOT / "results/C1_test", run,
                          [("train", common.read_scores(run, "train_task3")), ("test", common.read_scores(run, "task3_test"))])

common.compare(common.ROOT / "results/C2_sigma70", "C: sigma70",
               entries("sigma70") + [("published-2025", common.read_scores("published-2025", "sigma70"))], seeds("sigma70"))
