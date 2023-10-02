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

import gadgets

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
model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)


# Define how to preprocess different datasets
def preprocessing_factory(tokenizer, question_key, answer_key, chain_key, split: str):
    def preprocess_fn(sample, padding_length: int = 800):
        inputs = tokenizer(sample[question_key], truncation=True)
        labels = tokenizer( text_target=sample[chain_key], truncation=True)

        out_dict = {"question": sample[question_key],
                    "answer": sample[answer_key],
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "labels": labels.input_ids,
                    "chain": sample[chain_key]}

        if split == "train":
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
                    current_step = step_encodings[current_mask]
                else:
                    steps_mask.append(current_mask)
                    inputs_iter += 1
            # we include the closing <s> token to the last step
            steps_mask[-1] = steps_mask[-2]
            # debug: tokenizer.batch_decode([[inputs.input_ids[i] for i, _ in enumerate(inputs.input_ids) if steps_mask_l[i] == current_i] for current_i in set(steps_mask_l)])
            # steps_mask_stripped = steps_mask_l[:len(inputs.input_ids)]
            assert len(steps_mask) == len(inputs.input_ids), "Unexpected length of steps mask. Was %s, should be %s" \
                                                             % (len(steps_mask), len(inputs.input_ids))
            steps_mask_padded = tokenizer.pad({"input_ids": inputs.input_ids, "attention_mask": steps_mask},
                                              padding='max_length', max_length=padding_length)["attention_mask"]
            out_dict["steps_mask"] = steps_mask_padded

        return out_dict

    return preprocess_fn


dataset_to_keys = {
    "train": {
        "kaist-ai/CoT-Collection": {
            "question_key": "source",
            "answer_key": "target",
            "chain_key": "rationale",
        },
    },
    "validation": {
        "MU-NLPC/Calc-gsm8k": {
            "question_key": "question",
            "answer_key": "answer",
            "chain_key": "chain",
        },
        # "MU-NLPC/Calc-ape210k": {
        #     "question_key": "question_english_mt",
        #     "answer_key": "equation",
        #     "chain_key": "chain",
        # },
        # "MU-NLPC/Calc-math_qa": {
        #     "question_key": "problem",
        #     "answer_key": "rationale",
        #     "chain_key": "chain",
        # },
        # "MU-NLPC/Calc-aqua_rat": {
        #     "question_key": "question",
        #     "answer_key": "rationale",
        #     "chain_key": "chain",
        # },
    }
}


# see https://discuss.huggingface.co/t/making-multiple-samples-from-single-samples-using-huggingface-datasets/6819
def flatten_sample_per_step(x: Dataset, question_key: str, chain_key: str, answer_key: str) -> Iterator[
    dict[str, List[str]]]:
    sep = ". " if ". " in x[chain_key] else ".\n" if ".\n" in x[chain_key] else "\n"

    steps = [step.strip() + sep for step in x[chain_key].split(sep)]
    # exclude from targets the steps with only the gadget output:
    valid_prediction_steps = [not (step.startswith("<" + gadgets.markup.OUTPUT_TAG)
                                   and step.endswith(gadgets.markup.OUTPUT_TAG + ">")) for step in steps]
    questions = ["".join((x[question_key], " ", sep.join(steps[:i])))  # TODO: this produces duplicate sep (".")
                 for i in range(0, len(steps)) if valid_prediction_steps[i]]
    chains = [step for i, step in enumerate(steps) if valid_prediction_steps[i]]
    for question, target in zip(questions, chains):
        yield {question_key + "_orig": x[question_key],
               question_key: question,
               "steps": steps,
               chain_key: target,
               answer_key: x[answer_key]}


def map_flatten_sample_per_step(x: Dataset, question_key: str, chain_key: str, answer_key: str, sep: str = "\n") -> dict[str, List[str]]:
    steps = x[chain_key][0].split(sep)
    return {question_key: ["".join((x[question_key][0], " ", sep.join(steps[:i]))) for i in range(0, len(steps))],
            chain_key: [step.strip() for step in steps],
            answer_key: [x[answer_key][0]] * len(steps)}


# tokenize and preprocess datasets for training
preprocessed_datasets = {}
ds_to_lens = {}

for train_eval_split in dataset_to_keys.keys():
    for dset_name, keys in dataset_to_keys[train_eval_split].items():
        preprocessing_fn = preprocessing_factory(tokenizer=tokenizer, **keys, split=train_eval_split)
        dataset = datasets.load_dataset(dset_name)
        # per-step flattening -> for simplicity, flatten_sample_per_step requires batch_size=1
        for key in dataset.keys():
            dataset[key] = dataset[key].select(range(200))  # TODO: for debug only
            augmented_dataset = (flatten_sample_per_step(sample, **keys) for sample in tqdm(dataset[key].to_list()))
            flattened_dataset = itertools.chain(*augmented_dataset)
            dataset[key] = datasets.Dataset.from_list(list(flattened_dataset))
            # remove samples where we extracted empty label (=reasoning step) -> avoid training to generate empty step
            dataset[key] = dataset[key].filter(lambda row: row[keys["chain_key"]].strip())
        # encoding
        dataset = dataset.map(preprocessing_fn)
        preprocessed_datasets[dset_name] = dataset

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

# Add validation portion to gsm8k
# Select the first 100 samples for validation
valid_size = 800
val_data = preprocessed_datasets["MU-NLPC/Calc-gsm8k"]["test"].select(list(range(valid_size)))
preprocessed_datasets["MU-NLPC/Calc-gsm8k"]["validation"] = val_data  # .to_dict()
# Remove the first 100 samples from the test set
preprocessed_datasets["MU-NLPC/Calc-gsm8k"]["test"] = preprocessed_datasets["MU-NLPC/Calc-gsm8k"]["test"].select(
    list(range(valid_size, len(preprocessed_datasets["MU-NLPC/Calc-gsm8k"]["test"])))
)

# Only using 100 samples for validation from each dataset to speed things up
for dset_name, dataset in preprocessed_datasets.items():
    if dset_name not in dataset_to_keys["validation"]:
        continue
    preprocessed_datasets[dset_name]["validation"] = dataset["validation"].select(range(valid_size))

# Dropping columns so we can merge datasets
columns_to_keep = ["question", "answer", "input_ids", "attention_mask", "labels", "chain"]
for dset_name, dataset in preprocessed_datasets.items():
    if dset_name not in dataset_to_keys["validation"]:
        # we are merging only validation datasets!
        continue
    for split_name, split_dset in dataset.items():
        columns_to_remove = [column for column in split_dset.column_names if column not in columns_to_keep]
        dataset[split_name] = split_dset.remove_columns(columns_to_remove)

# concating datasets
train_ds = preprocessed_datasets['kaist-ai/CoT-Collection']["train"]

valid_ds = concatenate_datasets([dset["validation"] for dset_name, dset in preprocessed_datasets.items()
                                 if dset_name in dataset_to_keys["validation"]])
# test_ds = concatenate_datasets([dset["test"] for dset in preprocessed_datasets.values()])  # NOT USED

train_ds.shuffle()

log_predictions_indices = np.array(range(valid_size))

# PART: custom evaluations' logging
metrics = gadgets.metrics.MyMetrics(
    tokenizer=tokenizer,
    log_predictions=True,
    log_predictions_indices=log_predictions_indices,
    datasets_id_length={k: valid_size for k in dataset_to_keys["validation"].keys()},
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
    per_device_train_batch_size=6,  # TODO
    gradient_accumulation_steps=16,  # TODO
    per_device_eval_batch_size=16,
    eval_accumulation_steps=4,
    logging_steps=500,  # TODO: 4000 steps =~ 1 hour training, 1 hour eval, 8000 steps =~ 2 hour training, 1 hour eval
    eval_steps=5000,  # TODO
    save_steps=5000,
    evaluation_strategy="steps",
    bf16=True,
    # predict_with_generate=True,
    generation_max_length=512,
    include_inputs_for_metrics=True,
    report_to="wandb",
    # metric_for_best_model="avg_correct_results",
    # greater_is_better=True,
    load_best_model_at_end=True,
    save_total_limit=10,
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
