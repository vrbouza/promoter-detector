# promoter-detector

A small repo to play with promoter detection!


## Setup

```bash
pixi install
```

By default, training uses one GPU: the first one in CUDA order, or the CPU if there is none. Pass `--gpu N` to the training
scripts to pick another, and `--gpu 0 1` or `--all-gpus` to share the seeds between several GPUs (see below). `evaluation/score.py`
also takes `--gpu N`, with the same default.

## Run
Open the pixi env. with `pixi shell`, then run, in this order:
```bash
# 1. Splits: only if the data change
python preprocessing/make_splits.py        # generates data/splits/*.tsv (already there; rerunning gives the same files)

# 2. Training: seeds 42, 43, 44 and 45 by default (--seeds); add --gpu N, --gpu 0 1 or --all-gpus to choose GPUs.
#    Each run ends with its summary: seeds.tsv and best/
python training/train_authors.py --task 1  # authors' code, Hugging Face data
python training/train_authors.py --task 2  # authors' code, Hugging Face + 90% of our data
python training/train_improved.py          # our code, updated data
python training/train_authors.py --task 3  # authors' code on the updated data

# 2.1. ONLY IF CRASHED OR ADDING MANUALLY SEEDS: this is run usually within the training scripts
python training/summarise_seeds.py outputs/task1 outputs/task2 outputs/task3 outputs/task3_authors # or only one seed/task

# 3. Evaluation
python evaluation/score.py                 # evaluates cache scores of all models (--force after retraining)
python evaluation/compare_paper.py         # First comparison: can we recover the paper results?
python evaluation/compare_pneumo.py        # Second comparison: with the paper's code, do we see improvement on pneumo by adding to training and test data more pneumo?
python evaluation/compare_new.py           # Third/development comparison: what our devs do.
python evaluation/plot_curves.py           # results/training_curves/<run>_trainvalloss.png and <run>_valAUC.png
```

Each step needs the previous one: `score.py` scores the `best/` model and every seed, and the compare scripts only read the
ached scores. Each comparison needs only its own runs: the first needs task 1, the second tasks 1 and 2, the third task 3 and
task3_authors.

Each training script writes, per seed, `outputs/<run>/seed_<s>/`: the model (`model/`) and its validation metrics (`val.tsv`).
At the end, it summarises the run (`training/summarise_seeds.py`), adding:
- `seeds.tsv`, validation metrics per seed plus mean ± SD;
- `best/`, the model of the seed with the best validation AUC (`best/SEED` says which).

With several GPUs, the seeds form a queue: each GPU trains one seed at a time and takes the next one when it is free,
so faster GPUs train more seeds.

```bash
python training/train_improved.py --gpu 0 1     # the 5090 and the 3090
python training/train_improved.py --all-gpus    # every visible GPU
```

- Each seed's output goes to `outputs/<run>/seed_<s>.log`.
- The summary runs once every seed has finished. If one fails, its GPU gets no more seeds and there is no summary: rerun
  the missing seeds, or summarise by hand.

- IMPORTANT: the summary covers every `seed_*/val.tsv` in the folder, including seeds left from earlier runs: use a new
  `--output-dir` for a fresh run.
- `score.py` caches scores by model name: after retraining a run, use `--force`.

## Trainings

- **Task 1** is the authors' 2025 notebook (`examples/Finetuning.ipynb` @ `a5950ef`, which produced the 2025 model), kept
  as-is apart from the lines marked `# CHANGE:` in `training/train_authors.py`.
  - Changes: pinned revisions, our splits, one rename for transformers 4.57, a seed loop, and saving the model. Training-time evaluation uses our validation split instead of sigma70.
- **Task 2** is task 1's recipe, adding 80% of our data (S. pneumo) to training, 10% to validation, and 10% to test. We evaluate on the test S. pneumo data, and also per sigma factor.
- **Task 3** (`training/train_improved.py`) uses the same model and masking, with:
  - lr 1e-4, 10% warmup, weight decay 0.01, up to 10 epochs;
  - evaluation and saving every 100 steps; the checkpoint with the lowest validation loss is kept, and training stops after
    10 evaluations without a lower validation loss (early stopping). Seeds are then compared on validation AUC (`best/`).
- **task3_authors** is task 1's recipe trained on exactly task 3's data and split (`train_authors.py --task 3`). It is the
  baseline in comparison C, so task 3 vs task3_authors differs only in the training recipe.

## Datasets (sizes asserted by `make_splits.py`)

| | Train | Validation | Test | Comparison only |
|---|---|---|---|---|
| Task 1 | 200,782 (90% of HF train) | 22,310 (10% of HF train) | multispecies 22,582; our test 420 | sigma70 1,864 |
| Task 2 | 200,782 + our 3,357 | 22,310 + our 420 | our test 420 | — |
| Task 3 and task3_authors // New comparison | 199,896 (80% of HF train + multispecies + ours) | 24,987 | 24,988 | sigma70 1,864 |

Our data are split 80/10/10, stratified by sigma factor (promoters) and CDS/IGR (non-promoters). Tasks 2 and 3 use the same split.
### `pneumo.csv`: our experimentally validated *S. pneumoniae* D39V + TIGR4 set (4,197 rows)

Columns: `id, segment, y, src, sigma`. Windows are 81 bp, [-60, +20] around the TSS.

| | n |
|---|---|
| promoters (`src=pos`): SigA / SigX / unassigned | 519 / 27 / 853 |
| non-promoters: CDS / IGR (intergenic) | 1,399 / 1,399 |

- None of these sequences occurs in the HF dataset.

### HF dataset (loaded from HuggingFace)

`neuralbioinfo/bacterial_promoters`, loaded at revision `38f748a5ea68896c9467087e41c0210e5d5b5760` (see `common.py`):
- train: 223,092 rows;
- test_multispecies: 22,582;
- test_sigma70: 1,864.


## Comparisons (`results/`)

| Folder | Models | Test set | Question |
|---|---|---|---|
| `A1_multispecies` | task 1, published 2025, published 2024 (+ paper's numbers) | HF test_multispecies | Do we recover the paper? |
| `A2_sigma70` | same | HF test_sigma70 (comparison only) | same, on the paper's headline set |
| `B_pneumo` | task 1, task 2 | 10% of our data (+ `by_sigma.tsv`) | Does adding our data help on pneumococcus? |
| `C1_test` | task 3, task3_authors, new comparison | task 3's test split | Development comparison: can we improve what there is, with a different Does our recipe beat the authors' recipe on the same data? |
| `C2_sigma70` | task 3, task3_authors, published 2025 | HF test_sigma70 (comparison only) | Just for reference on a different dataset |

Each folder contains:
- `metrics.tsv`: AUC, AP (average precision: area under the precision-recall curve), and at threshold 0.5: accuracy, F1,
  sensitivity, specificity and precision;
- `seeds.tsv`: every seed, plus mean ± SD;
- `roc.png`.

A1, B and C1 also have `generalisation_<model>.tsv` and `train_vs_test_<model>.png` for each trained model.
`results/training_curves/` has two plots per run, against epoch with one colour per seed: `<run>_trainvalloss.png`
(training and validation loss) and `<run>_valAUC.png` (validation AUC). Dotted vertical lines mark each seed's chosen
model (task 3: its lowest-validation-loss checkpoint; authors' recipe: the last step), and the note at the top gives the batch size and
steps per epoch.

## Pinned versions

| What | Pin |
|---|---|
| `neuralbioinfo/prokbert-mini` (weights and remote model/tokenizer code) | `219ce7271928902e84945fa62ad3703bafe746e1` (2025-01-09) |
| `neuralbioinfo/prokbert-mini-promoter` | 2025: `8502f02d…` (`model.safetensors`); 2024: `96bfe4a1…` (`pytorch_model.bin`) |
| `neuralbioinfo/bacterial_promoters` | `38f748a5…` |
| Software | torch 2.15.0 + CUDA 13.2; **transformers 4.57.6** |

- transformers 5.x can't be used: it removed MegatronBERT's relative position embeddings, which prokbert-mini needs, and loads the
  model without them silently.
- The two published models are different training runs. The 2024 one reproduces the paper's tables. `from_pretrained` loads the 2025 one.
