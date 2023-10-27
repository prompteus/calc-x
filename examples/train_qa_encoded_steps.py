from __future__ import annotations

import itertools
import random
from typing import List, Iterator

import datasets
import numpy as np
import torch
import transformers
import wandb
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from transformers import EarlyStoppingCallback
from cot import Collection

import gadgets
from examples.qa_utils import apply_template, tagged_answer

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_properties(i))

# model_name = "google/flan-t5-small"  # TODO
# model_name = "google/t5-v1_1-xl"
model_name = "trained_models/likely-dragon-149-ch18000"  # pretrained T5-memory-Large on apollo

log_path = "logs/"
wandb.init(
    entity="transformersclub",
    project="gadgets",
    tags=[model_name, "calculator", "gsm8k", "aqua", "supervised"],  # TODO
    group="calculator-gsm8k-aqua-supervised",
    dir=log_path,
)

tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
model = gadgets.model.stepwise_gadget_model(transformers.T5ForConditionalGeneration).from_pretrained(model_name)

gadgets.utils.add_new_token(
    "<",
    is_special=False,
    tokenizer=tokenizer,
    model=model,
    init_with=["[", ">"],
)

text = "<gadget>2+2</gadget>"
encoded = tokenizer(text, return_tensors="pt").input_ids
decoded = tokenizer.batch_decode(encoded, skip_special_tokens=True, spaces_between_special_tokens=False)
assert decoded[0] == text, decoded[0]

# PART: update generate() to support gadgets
model.prepare_for_generate(tokenizer, enabled_gadgets=[gadgets.gadget.Calculator()], default_max_tokens=512)

data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)


# Define how to preprocess different datasets
def preprocessing_factory(tokenizer, question_key, answer_key, chain_key):
    def preprocess_fn(sample):
        inputs = tokenizer(sample[question_key], truncation=True)
        labels = tokenizer(text_target=sample[chain_key], truncation=True)
        return {"question": sample[question_key],
                "answer": sample[answer_key],
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": labels.input_ids,
                "chain": sample[chain_key]}

    return preprocess_fn


train_datasets = ["commonsense_qa", "strategy_qa", "worldtree", "open_book_qa"]
val_datasets = ["commonsense_qa", "worldtree", "open_book_qa"]

# train_datasets = ["commonsense_qa"]  # TODO: remove
# val_datasets = ["commonsense_qa"]
valid_size = 100


all_datasets = list(set(train_datasets + val_datasets))


# see https://discuss.huggingface.co/t/making-multiple-samples-from-single-samples-using-huggingface-datasets/6819
def flatten_sample_per_step(x: Dataset, question_key: str,
                            chain_key: str, answer_key: str) -> Iterator[dict[str, List[str]]]:
    steps = x[chain_key] + [tagged_answer(x[answer_key][0])]
    # exclude from targets the steps with only the gadget output:
    valid_prediction_steps = [not (step.startswith("<" + gadgets.markup.OUTPUT_TAG)
                                   and step.endswith(gadgets.markup.OUTPUT_TAG + ">")) for step in steps]
    questions = ["".join((x[question_key], " ", " ".join(steps[:i])))
                 for i in range(0, len(steps)) if valid_prediction_steps[i]]
    chains = [step for i, step in enumerate(steps) if valid_prediction_steps[i]]
    for question, target in zip(questions, chains):
        yield {question_key: question, chain_key: target, answer_key: x[answer_key][0]}


# tokenize and preprocess datasets for training
preprocessed_datasets = {}
keys = {"question_key": "question", "answer_key": "answer", "chain_key": "cot"}

collection = Collection(all_datasets, generate_mode="recache")

# push to hub, to avoid using cot library:
# datasets.DatasetDict({"train": collection.all_train,
#                       "validation": collection.all_validation,
#                       "test": collection.all_test}
#                       ).push_to_hub("cot-commonsense-qa", token="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")
for dset_name in all_datasets:
    dataset = collection[dset_name]
    dataset = dataset.map(lambda row: {keys["question_key"]: apply_template(row[keys["question_key"]], row["choices"])})

    if dset_name in train_datasets:
        # we apply per-step flattening on only train datasets
        # for simplicity, flatten_sample_per_step requires batch_size=1
        # dataset["train"] = dataset["train"].select(range(200))  # TODO: for debug only
        augmented_dataset = (flatten_sample_per_step(sample, **keys) for sample in tqdm(dataset["train"].to_list()))
        flattened_dataset = itertools.chain(*augmented_dataset)
        dataset["train"] = datasets.Dataset.from_list(list(flattened_dataset))
        # remove samples where we extracted empty label (=reasoning step) -> avoid training to generate empty step
        dataset["train"] = dataset["train"].filter(lambda row: row[keys["chain_key"]].strip())
    else:
        print("Omitting dataset %s from training" % dset_name)
        del dataset["train"]

    if dset_name in val_datasets:
        dataset["validation"] = dataset["validation"].map(
            lambda row: {keys["chain_key"]: " ".join(row[keys["chain_key"]] + [tagged_answer(row[keys["answer_key"]][0])])}
        )

    elif dset_name not in val_datasets and "validation" in dataset.keys():
        del dataset["validation"]

    del dataset["test"]  # test datasets from ThoughtSource are missing CoT annotations, breaking bulk pre-encoding

    preprocessed_datasets[dset_name] = dataset

# encoding -> all datasets
preprocessed_datasets = {k: dataset.map(preprocessing_factory(tokenizer=tokenizer, **keys))
                         for k, dataset in preprocessed_datasets.items()}

# Fixing validation error on too-long incomplete chain without closing > tag
# + filtering the same problem on training data, removes small amount of samples
longest_allowed_chain = 800
for dset_name, dataset in preprocessed_datasets.items():
    for split, subdset in dataset.items():
        preprocessed_datasets[dset_name][split] = subdset.filter(lambda row: len(row["chain"]) < longest_allowed_chain)

# Upsample datasets to the length of the largest dataset
dset_to_length = {dset_name: len(dset["train"]) for dset_name, dset in preprocessed_datasets.items()}
largest_dset_length = max(dset_to_length.values())
extended_datasets = {}
for dset_name, dataset in preprocessed_datasets.items():
    dataset_to_extend = dataset["train"]
    dset_len = len(dataset_to_extend)
    num_extra_samples = largest_dset_length - dset_len
    extra_indices = random.choices(range(dset_len), k=num_extra_samples)
    extra_dataset = dataset_to_extend.select(extra_indices)
    extended_dataset = concatenate_datasets([dataset_to_extend, extra_dataset])
    preprocessed_datasets[dset_name]["train"] = extended_dataset

dset_lengths = [len(dset["train"]) for dset in preprocessed_datasets.values()]
# Check if all train dsets have the same size
assert all(x == dset_lengths[0] for x in dset_lengths)

# Only using 100 samples for validation from each dataset to speed things up
for dset_name, dataset in preprocessed_datasets.items():
    if dset_name not in val_datasets:
        print("Omitting dataset %s from validation" % dset_name)
        continue
    if len(preprocessed_datasets[dset_name]["validation"]) > valid_size:
        preprocessed_datasets[dset_name]["validation"] = dataset["validation"].select(range(valid_size))

# Dropping columns so we can merge datasets
columns_to_keep = ["input_ids", "attention_mask", "labels"]
for dset_name, dataset in preprocessed_datasets.items():
    for split_name, split_dset in dataset.items():
        columns_to_remove = [column for column in split_dset.column_names if column not in columns_to_keep]
        dataset[split_name] = split_dset.remove_columns(columns_to_remove)

# concat datasets
train_ds = concatenate_datasets([d["train"] for k, d in preprocessed_datasets.items() if k in train_datasets])
valid_ds = concatenate_datasets([d["validation"] for k, d in preprocessed_datasets.items() if k in val_datasets])

train_ds.shuffle()

log_predictions_indices = np.array(range(valid_size))

# PART: custom evaluations' logging
metrics = gadgets.metrics.MyMetrics(
    tokenizer=tokenizer,
    log_predictions=True,
    log_predictions_indices=log_predictions_indices,
    datasets_id_length={k: len(preprocessed_datasets[k]['validation']) for k in val_datasets if k in val_datasets},
    # TODO: ordering and sizes must match eval_dataset
)

training_args = transformers.Seq2SeqTrainingArguments(
    output_dir="./logs/" + wandb.run.name,  # TODO add
    # output_dir="./logs/",
    learning_rate=2e-5,
    do_train=True,
    do_eval=True,
    warmup_steps=1000,
    max_steps=20_000,
    per_device_train_batch_size=4,  # TODO
    gradient_accumulation_steps=9,  # TODO
    per_device_eval_batch_size=1,
    eval_accumulation_steps=16,
    logging_steps=400,  # TODO: 4000 steps =~ 1 hour training, 1 hour eval, 8000 steps =~ 2 hour training, 1 hour eval
    eval_steps=4000,  # TODO
    save_steps=4000,
    evaluation_strategy="steps",
    # bf16=True,  # TODO
    predict_with_generate=True,
    generation_max_length=512,
    include_inputs_for_metrics=True,
    report_to="wandb",
    metric_for_best_model="avg_correct_results",
    greater_is_better=True,
    load_best_model_at_end=True,
    save_total_limit=6,
    # use_cpu=True,  # TODO: remove
    remove_unused_columns=False,
)

trainer = transformers.Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
)
trainer.train()  # TODO: resume_from_checkpoint?
# TODO: we implement evaluation based on exact-match separately
