"""Authors' code (prokbert GitHub, examples/Finetuning.ipynb @ a5950ef).

The notebook cells are kept as they were; every change is marked with a "# CHANGE:" comment.
"""
# ============================
#### Imports

import argparse
from pathlib import Path

import launch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=int, choices=(1, 2, 3), required=True)
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
parser.add_argument("--output-dir", default=None, help="default: outputs/task<N> (outputs/task3_authors for 3)")
parser.add_argument("--max-steps", type=int, default=-1, help="smoke tests only")
launch.add_args(parser)
args = parser.parse_args()
out_dir = Path(args.output_dir or ("outputs/task3_authors" if args.task == 3 else f"outputs/task{args.task}"))
launch.setup(args, out_dir)  # before torch is imported; several GPUs: one process per GPU, then summary and exit

from datasets import Dataset, DatasetDict
from transformers import set_seed

import common

# ============================
#### Code from notebook
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score, recall_score
from datasets import load_dataset  # CHANGE: unused, the data come from our split files instead
import torch
import os

# Check if CUDA (GPU support) is available
# CHANGE: the CPU if there is no GPU (was: SystemError); at most one GPU, as several would make the Trainer use DataParallel
device_name = launch.device(args)
print(f'Training on: {device_name}')
num_cores = os.cpu_count()
print(f'Number of available CPU cores: {num_cores}')

model_name_path = 'neuralbioinfo/prokbert-mini'
tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True,
                                          revision=common.BASE_REV)  # CHANGE: pinned revision


def tokenize_function(examples):
    # Tokenize the input sequences
    encoded = tokenizer.batch_encode_plus(
        examples["segment"],
        padding=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    # Get the input_ids and attention_mask
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Mask tokens with IDs 2 and 3 in a vectorized way
    mask_tokens = (input_ids == 2) | (input_ids == 3)  # Identify where tokens 2 and 3 occur
    attention_mask[mask_tokens] = 0  # Set attention_mask to 0 for these tokens
    y = torch.tensor(examples["y"], dtype=torch.int64)

    # Return the updated dictionary
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": y,  # Include labels for training
    }


# CHANGE: our train/validation split replaces load_dataset("neuralbioinfo/bacterial_promoters").
split = common.load_split(args.task)
dataset = DatasetDict({s: Dataset.from_pandas(split[s][["segment", "y"]], preserve_index=False)
                       for s in ("train", "val")})
randomize_dataset = dataset.shuffle(seed=42)
randomize_dataset.flatten_indices()
tokenized_dataset = randomize_dataset.map(tokenize_function, batched=True, num_proc=num_cores)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)  # Predicted class indices

    # Calculate metrics
    mcc = matthews_corrcoef(labels, predictions)
    acc = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, average='weighted')

    try:
        # Adjust for binary or multiclass classification
        if logits.shape[1] == 2:  # Binary classification
            roc_auc = roc_auc_score(labels, logits[:, 1])  # Use positive class probabilities
        else:  # Multiclass classification
            roc_auc = roc_auc_score(labels, logits, multi_class='ovr')  # Use probabilities for all classes
    except ValueError:
        # Handle edge cases where ROC-AUC cannot be computed
        roc_auc = float('nan')

    return {
        "mcc": mcc,
        "accuracy": acc,
        "recall": recall,
        "roc_auc": roc_auc,
    }


for seed in args.seeds:  # CHANGE: one run per seed; the notebook ran once with the default seed (42)
    seed_dir = out_dir / f"seed_{seed}"
    set_seed(seed)  # CHANGE: seeds the head initialisation too
    model = AutoModelForSequenceClassification.from_pretrained(model_name_path, trust_remote_code=True,
                                                               revision=common.BASE_REV)  # CHANGE: pinned
    training_args = TrainingArguments(
        output_dir=str(seed_dir / "checkpoints"),  # CHANGE: was "./results"
        per_device_eval_batch_size=128,
        per_device_train_batch_size=128,
        logging_dir=str(seed_dir / "logs"),  # CHANGE: was "./logs"
        report_to="none",
        eval_strategy="steps",  # CHANGE: renamed from evaluation_strategy in transformers 4.57
        eval_steps=50,
        num_train_epochs=1,
        learning_rate=0.0004,
        logging_steps=50,
        save_total_limit=1,
        seed=seed,  # CHANGE: added (default was 42)
        max_steps=args.max_steps,  # CHANGE: added for smoke tests (-1 = use num_train_epochs)
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['val'],  # CHANGE: our validation split instead of test_sigma70
        compute_metrics=compute_metrics,
    )
    # Start training
    train_output = trainer.train()  # CHANGE: keep the result, for the training time in val.tsv
    # ---- end of notebook cells ----

    trainer.save_model(str(seed_dir / "model"))  # the notebook never saves the final model
    common.save_seed(seed_dir, common.val_row(seed, model, tokenizer, split["val"])
                     | {"gpu": device_name, "train_min": train_output.metrics["train_runtime"] / 60})

launch.finish(args, out_dir)  # seeds.tsv and best/ (training/summarise_seeds.py)
