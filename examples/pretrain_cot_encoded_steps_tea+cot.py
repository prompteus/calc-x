from __future__ import annotations

import itertools
import random
from typing import List, Iterator

import datasets
import numpy as np
import torch
import transformers
import wandb
from datasets import Dataset, concatenate_datasets, DatasetDict
from tqdm import tqdm
from transformers import EarlyStoppingCallback

import gadgets
from examples.teabreac_utils import tea_val, tea_train

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_properties(i))

# model_name = "stas/mt5-tiny-random"  # TODO
# model_name = "google/flan-t5-small"  # TODO
model_name = "google/t5-v1_1-large"
# model_name = "logs/earthy-jazz-123/checkpoint-16000"

log_path = "logs/"
wandb.init(
        entity="transformersclub",
        project="gadgets",
        tags=[model_name, "calculator", "gsm8k", "aqua", "supervised"],  # TODO
        # group="calculator-gsm8k-aqua-supervised",
        dir=log_path,
)

tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
model = gadgets.model.stepwise_gadget_model(transformers.T5ForConditionalGeneration).from_pretrained(model_name)
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)


def preprocessing_factory(tokenizer, question_key, answer_key, chain_key, split: str, get_steps_mask: bool = False):
    def preprocess_fn(sample, padding_length: int = 800):
        inputs = tokenizer(sample[question_key], truncation=True)
        labels = tokenizer(text_target=sample[chain_key], truncation=True)

        out_dict = {"question": sample[question_key],
                    "answer": sample[answer_key],
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "labels": labels.input_ids,
                    "chain": sample[chain_key]}

        if split == "train" and get_steps_mask:
            steps_mask = []
            current_mask = 0
            inputs_iter = 0
            step_encodings = tokenizer(sample["steps"], truncation=True).input_ids

            current_step = step_encodings[0]
            while inputs_iter < len(inputs.input_ids):
                if inputs.input_ids[inputs_iter:inputs_iter + len(current_step) - 1] == current_step[:-1]:
                    current_mask += 1
                    added_mask = [current_mask] * len(current_step[:-1])
                    steps_mask.extend(added_mask)
                    inputs_iter += len(added_mask)
                    try:
                        current_step = step_encodings[current_mask]
                    except IndexError:
                        print("Reasoning step resides within input text. Dropping sample to avoid ambiguity.")
                        steps_mask = [0] * len(inputs.input_ids)
                        break
                else:
                    steps_mask.append(current_mask)
                    inputs_iter += 1
            # we include the closing <s> token to the last step
            steps_mask[-1] = steps_mask[-2]
            # debug: tokenizer.batch_decode([[inputs.input_ids[i] for i, _ in enumerate(inputs.input_ids) if current_mask[i] == current_i] for current_i in set(current_mask)])
            # steps_mask_stripped = steps_mask_l[:len(inputs.input_ids)]
            assert len(steps_mask) == len(inputs.input_ids), "Unexpected length of steps mask. Was %s, should be %s" \
                                                             % (len(steps_mask), len(inputs.input_ids))
            steps_mask_padded = tokenizer.pad({"input_ids": inputs.input_ids, "attention_mask": steps_mask},
                                              padding='max_length', max_length=padding_length)["attention_mask"]
            out_dict["steps_mask"] = steps_mask_padded

        return out_dict

    return preprocess_fn


dataset_to_keys = {
    "validation": {
        "MU-NLPC/Calc-gsm8k": {
            "question_key": "question",
            "answer_key": "result",
            "chain_key": "chain",
        },
    },
    "train": {
        "kaist-ai/CoT-Collection": {
            "question_key": "source",
            "answer_key": "target",
            "chain_key": "rationale",
        },
        "teabreac": {
            "question_key": "question_text",
            "answer_key": "answers_text",
            "chain_key": "context_text"
        }
    }
}

valid_size = 800


# see https://discuss.huggingface.co/t/making-multiple-samples-from-single-samples-using-huggingface-datasets/6819
def flatten_sample_per_step(x: Dataset, question_key: str, chain_key: str, answer_key: str) -> Iterator[
    dict[str, List[str]]]:
    sep = ". " if ". " in x[chain_key] else ".\n" if ".\n" in x[chain_key] else "\n"

    steps = [step.strip() for step in x[chain_key].split(sep)] + [x[answer_key]]
    # exclude from targets the steps with only the gadget output:
    questions = ["".join((x[question_key], " ", sep.join(steps[:i])))  # TODO: this produces duplicate sep (".")
                 for i in range(0, len(steps))]
    chains = [step for i, step in enumerate(steps)]
    for question, target in zip(questions, chains):
        yield {question_key + "_orig": x[question_key],
               question_key: question,
               "steps": steps,
               chain_key: target,
               answer_key: x[answer_key]}


# tokenize and preprocess datasets for training
preprocessed_datasets = {}
ds_to_lens = {}

# dataset = DatasetDict({"train": Dataset.from_pandas(tea_train), "validation": Dataset.from_pandas(tea_val)})
# per-step flattening -> for simplicity, flatten_sample_per_step requires batch_size=1
for split in dataset_to_keys.keys():
    for dset_name, keys in dataset_to_keys[split].items():
        if dset_name == "teabreac":
            # teabreac is not loaded from the hub
            dataset = Dataset.from_pandas(tea_train) if split == "train" else Dataset.from_pandas(tea_val)
        else:
            dataset = datasets.load_dataset(dset_name, split=split)
        # per-step flattening -> for simplicity, flatten_sample_per_step requires batch_size=1
        # dataset = dataset.select(range(min(100, len(dataset))))  # TODO remove: for debug only
        augmented_dataset = (flatten_sample_per_step(sample, **keys) for sample in tqdm(dataset.to_list()))
        flattened_dataset = itertools.chain(*augmented_dataset)
        dataset = datasets.Dataset.from_list(list(flattened_dataset))
        # remove samples where we extracted empty label (=reasoning step) -> avoid training to generate empty step
        dataset = dataset.filter(lambda row: row[keys["chain_key"]].strip())
        # encoding -> pretraining validation also needs steps_mask, hence the fixed "train" split
        dataset = dataset.map(preprocessing_factory(tokenizer, **keys, split="train"))
        try:
            preprocessed_datasets[dset_name][split] = dataset
        except KeyError:
            preprocessed_datasets[dset_name] = {split: dataset}

# Fixing validation error on too-long incomplete chain without closing > tag
# + filtering the same problem on training data, removes small amount of samples
longest_allowed_chain = 800
for dset_name, dataset in preprocessed_datasets.items():
    for split, subdset in dataset.items():
        preprocessed_datasets[dset_name][split] = subdset.filter(lambda row: len(row["chain"]) < longest_allowed_chain)

# Upsample datasets to the length of the largest dataset
dset_to_length = {dset_name: len(dset["train"]) for dset_name, dset in preprocessed_datasets.items()
                  if dset_name in dataset_to_keys["train"].keys()}
largest_dset_length = max(dset_to_length.values())
extended_datasets = {}
for dset_name, dataset in preprocessed_datasets.items():
    if dset_name not in dataset_to_keys["train"].keys():
        continue
    dataset_to_extend = dataset["train"]
    dset_len = len(dataset_to_extend)
    num_extra_samples = largest_dset_length - dset_len
    extra_indices = random.choices(range(dset_len), k=num_extra_samples)
    extra_dataset = dataset_to_extend.select(extra_indices)
    extended_dataset = concatenate_datasets([dataset_to_extend, extra_dataset])
    preprocessed_datasets[dset_name]["train"] = extended_dataset

dset_lengths = [len(dset["train"]) for dset_name, dset in preprocessed_datasets.items()
                if dset_name in dataset_to_keys["train"]]
# Check if all train dsets have the same size
assert all(x == dset_lengths[0] for x in dset_lengths)

# Only using 100 samples for validation from each dataset to speed things up
for dset_name, dataset in preprocessed_datasets.items():
    if dset_name not in dataset_to_keys["validation"].keys():
        continue
    preprocessed_datasets[dset_name]["validation"] = dataset["validation"].select(
            range(min(valid_size, len(dataset["validation"])))
    )

# Dropping columns so we can merge datasets
# columns_to_keep = ["question", "answer", "input_ids", "attention_mask", "labels", "chain"]
columns_to_keep = ["input_ids", "attention_mask", "labels", "steps_mask"]
for dset_name, dataset in preprocessed_datasets.items():
    # if dset_name not in dataset_to_keys["validation"]:
    #     # we are merging only validation datasets!
    #     continue
    for split_name, split_dset in dataset.items():
        columns_to_remove = [column for column in split_dset.column_names if column not in columns_to_keep]
        dataset[split_name] = split_dset.remove_columns(columns_to_remove)

# concating datasets
train_ds = concatenate_datasets([dset["train"] for dset_name, dset in preprocessed_datasets.items()
                                 if dset_name in dataset_to_keys["train"].keys()])

valid_ds = concatenate_datasets([dset["validation"] for dset_name, dset in preprocessed_datasets.items()
                                 if dset_name in dataset_to_keys["validation"].keys()])
# test_ds = concatenate_datasets([dset["test"] for dset in preprocessed_datasets.values()])  # NOT USED

train_ds.shuffle()

log_predictions_indices = np.array(range(valid_size))

# PART: custom evaluations' logging
metrics = gadgets.metrics.MyMetrics(
        tokenizer=tokenizer,
        log_predictions=True,
        log_predictions_indices=log_predictions_indices,
        datasets_id_length={dset_name: len(preprocessed_datasets[dset_name]['validation'])
                            for dset_name in dataset_to_keys["validation"].keys()},
        # TODO: ordering and sizes must match eval_dataset
)

training_args = transformers.Seq2SeqTrainingArguments(
        output_dir="./logs/" + wandb.run.name,  # TODO add
        # output_dir=model_name,  # TODO: if resume_from_checkpoint=True
        learning_rate=5e-5,
        do_train=True,
        do_eval=True,
        warmup_steps=10_000,
        max_steps=200_000,
        per_device_train_batch_size=12,  # TODO
        gradient_accumulation_steps=3,  # TODO
        per_device_eval_batch_size=16,
        eval_accumulation_steps=4,
        logging_steps=400,
        # TODO: 4000 steps =~ 1 hour training, 1 hour eval, 8000 steps =~ 2 hour training, 1 hour eval
        eval_steps=4000,  # TODO
        save_steps=4000,
        evaluation_strategy="steps",
        # bf16=True,  # TODO
        # predict_with_generate=True,
        generation_max_length=512,
        include_inputs_for_metrics=True,
        report_to="wandb",
        # metric_for_best_model="avg_correct_results",
        # greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=10,
        # no_cuda=True,  # TODO: remove
        remove_unused_columns=False,
)

trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)
trainer.train()  # TODO: resume_from_checkpoint?
