from __future__ import annotations

import os
import random
import re

import datasets
import numpy as np
import torch
import transformers
from datasets import Dataset, concatenate_datasets
from transformers import EarlyStoppingCallback

import gadgets
import wandb
from gadgets.baseline_metrics import MyBaselineMetrics

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_properties(i))


# model_name = "google/flan-t5-small"
model_name = "google/t5-v1_1-large"


log_path = "logs/"
wandb.init(
    entity="transformersclub",
    project="gadgets",
    tags=[model_name, "calculator", "gsm8k", "aqua", "supervised"],  # TODO
    group="calculator-gsm8k-aqua-supervised",
    dir=log_path,
)

tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)

from baseline_utils import dataset_to_keys, preprocessing_factory

# tokenize and preprocess datasets for training
preprocessed_datasets = {}
ds_to_lens = {}
for dset_name, keys in dataset_to_keys.items():
    preprocessing_fn = preprocessing_factory(tokenizer=tokenizer, **keys)
    dset = datasets.load_dataset(f"MU-NLPC/{dset_name}").map(preprocessing_fn)
    preprocessed_datasets[dset_name] = dset


# Fixing validation error on too-long trimmed chain + filtering the same problem on training data, removes small amount of samples
longest_allowed_chain = 800
for dset_name, dset in preprocessed_datasets.items():
    for split, subdset in dset.items():
        preprocessed_datasets[dset_name][split] = subdset.filter(
            lambda row: len(row["chain"]) < longest_allowed_chain
        )

########### PREPARING LABELS THAT DO NOT USE MARKUP SYNTAX ########
from baseline_utils import (
    ape210k_prep,
    aqua_rat_prep,
    dataset_to_labeler,
    gsm8k_prep,
    labeling_factory,
    math_qa_prep,
)

preprocessed_datasets_labeled = {dset_name: {} for dset_name in dataset_to_labeler.keys()}
for dset_name, dset in preprocessed_datasets.items():
    print(dset_name)
    labeler_fn = dataset_to_labeler[dset_name]
    preprocess_fn = labeling_factory(tokenizer, labeler_fn, dataset_to_keys[dset_name]["question_key"])
    for dset_split, subdset in dset.items():
        preprocessed_datasets_labeled[dset_name][dset_split] = subdset.map(preprocess_fn).filter(
            lambda example: example["labels"] is not None
        )

for dset_name, dset in preprocessed_datasets_labeled.items():
    for dset_split, subdset in dset.items():
        original_dset_size = len(preprocessed_datasets[dset_name][dset_split])
        new_dset_size = len(subdset)
        diff = original_dset_size - new_dset_size
        print(f"{dset_name}, {dset_split}, threw away {diff} samples ({100*(diff/original_dset_size):.2f}%)")

preprocessed_datasets = preprocessed_datasets_labeled

# Upsample datasets to the length of the largest dataset
dset_to_length = {dset_name: len(dset["train"]) for dset_name, dset in preprocessed_datasets.items()}
largest_dset_length = max(dset_to_length.values())
extended_datasets = {}
for dset_name, dset in preprocessed_datasets.items():
    dataset_to_extend = dset["train"]
    dset_len = len(dataset_to_extend)
    num_extra_samples = largest_dset_length - dset_len
    extra_indices = random.choices(range(dset_len), k=num_extra_samples)
    extra_dataset = dataset_to_extend.select(extra_indices)
    extended_dataset = concatenate_datasets([dataset_to_extend, extra_dataset])
    preprocessed_datasets[dset_name]["train"] = extended_dataset

dset_lengths = [len(dset["train"]) for dset in preprocessed_datasets.values()]
# Check if all train dsets have the same size
assert all(x == dset_lengths[0] for x in dset_lengths)

# Add validation portion to gsm8k
# Select the last 100 samples for validation
valid_size = 100
val_data = preprocessed_datasets["Calc-gsm8k"]["test"].select(list(range(valid_size)))
preprocessed_datasets["Calc-gsm8k"]["validation"] = val_data  # .to_dict()
# Remove the last 100 samples from the test set
preprocessed_datasets["Calc-gsm8k"]["test"] = preprocessed_datasets["Calc-gsm8k"]["test"].select(
    list(range(valid_size, len(preprocessed_datasets["Calc-gsm8k"]["test"])))
)


# Only using 100 samples for validation from each dataset to speed things up
for dset_name, dset in preprocessed_datasets.items():
    preprocessed_datasets[dset_name]["validation"] = dset["validation"].select(range(valid_size))

# Dropping columns so we can merge datasets
columns_to_keep = ["question", "answer", "input_ids", "attention_mask", "labels", "chain"]
for dset_name, dset in preprocessed_datasets.items():
    for split_name, split_dset in dset.items():
        columns_to_remove = [column for column in split_dset.column_names if column not in columns_to_keep]
        dset[split_name] = split_dset.remove_columns(columns_to_remove)

# concating datasets
train_ds = concatenate_datasets([dset["train"] for dset in preprocessed_datasets.values()])
valid_ds = concatenate_datasets([dset["validation"] for dset in preprocessed_datasets.values()])
test_ds = concatenate_datasets([dset["test"] for dset in preprocessed_datasets.values()])  # NOT USED

train_ds.shuffle()

log_predictions_indices = np.array(range(valid_size))
log_predictions_indices

# PART: custom evaluations' logging
metrics = MyBaselineMetrics(
    tokenizer=tokenizer,
    log_predictions=True,
    log_predictions_indices=log_predictions_indices,
    # datasets_id_length={"gsm": len(valid_ds), "aqua": len(aqua_val)}  # TODO: ordering and sizes must match eval_dataset
    datasets_id_length={
        k: valid_size for k in dataset_to_keys.keys()
    },  # TODO: ordering and sizes must match eval_dataset
)

training_args = transformers.Seq2SeqTrainingArguments(
    output_dir="./logs/" + wandb.run.name,
    learning_rate=5e-5,
    do_train=True,
    do_eval=True,
    warmup_steps=1000,
    max_steps=400_000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=16,
    logging_steps=100,
    eval_steps=8000,  # 4000 steps =~ 1 hour training, 1 hour eval, 8000 steps =~ 2 hour training, 1 hour eval
    save_steps=8000,
    evaluation_strategy="steps",
    bf16=True,
    predict_with_generate=True,
    generation_max_length=512,
    include_inputs_for_metrics=True,
    report_to="wandb",
    metric_for_best_model="avg_correct_results",
    greater_is_better=True,
    load_best_model_at_end=True,
    save_total_limit=15,
)

trainer = transformers.Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)
trainer.train()
trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
