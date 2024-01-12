from __future__ import annotations

import itertools
import random
from typing import List, Iterator, Any

import datasets
import numpy as np
import torch
import transformers
import wandb
from datasets import concatenate_datasets
from tqdm import tqdm
from transformers import EarlyStoppingCallback

import gadgets
from gadgets.steps_utils import StepPermuter, separate_chain_to_steps

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_properties(i))

# model_name = "google/flan-t5-small"  # TODO
model_name = "google/t5-v1_1-large"
# model_name = "/Users/xstefan3/PycharmProjects/gadgets-hackaton/trained_models/faithful-plant-182-ch12000"  # GSM+AQuA T5-compressed-memory-Large
# model_name = "logs/faithful-plant-182/checkpoint-12000"  # pretrained T5-memory-Large on apollo

log_path = "/var/tmp/xstefan3/logs/"
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

STEP_TOKEN = "[step]"

gadgets.utils.add_new_token(
    STEP_TOKEN,
    is_special=True,
    tokenizer=tokenizer,
    model=model,
    init_with=["."],
)

STEP_ID = tokenizer.added_tokens_encoder[STEP_TOKEN]

model.step_token_id = STEP_ID

text = "<gadget>2+2</gadget>"
encoded = tokenizer(text, return_tensors="pt").input_ids
decoded = tokenizer.batch_decode(encoded, skip_special_tokens=True, spaces_between_special_tokens=False)
assert decoded[0] == text, decoded[0]

# PART: update generate() to support gadgets
model.prepare_for_generate(
    tokenizer,
    enabled_gadgets=[gadgets.gadget.Calculator()],
    default_max_tokens=512,
)

train_datasets_keys = ["Calc-gsm8k", "Calc-aqua_rat", "Calc-ape210k", "Calc-math_qa"]
train_datasets_keys = ["Calc-aqua_rat"]
# train_datasets_keys = ["Calc-gsm8k"]
val_datasets_keys = ["Calc-gsm8k", "Calc-aqua_rat", "Calc-ape210k"]
# val_datasets_keys = ["Calc-gsm8k"]

# train_datasets_keys = ["Calc-gsm8k", "Calc-aqua_rat", "Calc-ape210k", "Calc-math_qa"]
train_datasets_keys = ["Calc-gsm8k"]
# val_datasets_keys = ["Calc-gsm8k", "Calc-aqua_rat", "Calc-ape210k"]
val_datasets_keys = ["Calc-gsm8k"]

valid_size = 100  # TODO Adjust: Select the first 100 samples for validation

dataset_to_keys = {
    "Calc-gsm8k": {
        "question_key": "question",
        "answer_key": "result",
        "chain_key": "chain",
    },
    "Calc-aqua_rat": {
        "question_key": "question",
        "answer_key": "result",
        "chain_key": "chain",
    },
    "Calc-ape210k": {
        "question_key": "question",
        "answer_key": "result",
        "chain_key": "chain",
    },
    "Calc-math_qa": {
        "question_key": "question",
        "answer_key": "result",
        "chain_key": "chain",
    },
}

permuter = StepPermuter(tokenizer)


# see https://discuss.huggingface.co/t/making-multiple-samples-from-single-samples-using-huggingface-datasets/6819
def flatten_sample_per_step(x: dict[str, Any],
                            question_key: str,
                            chain_key: str,
                            answer_key: str) -> Iterator[dict[str, List[str]]]:
    # transformation of the dataset into a per-step version: "question+previous steps" -> "next step"
    # sep = ". " if ". " in x[chain_key] else ".\n" if ".\n" in x[chain_key] else "\n"
    separated_steps, sep = separate_chain_to_steps(x[chain_key])
    steps = [x[question_key]] + separated_steps

    def is_valid_step(step_str: str) -> bool:
        # exclude from targets the steps with only the gadget output:
        return bool(step_str.strip()) and not (step_str.strip().startswith("<" + gadgets.markup.OUTPUT_TAG)
                                               and step_str.strip().endswith(gadgets.markup.OUTPUT_TAG + ">"))
    # join steps so that one step contain all segments that are not valid steps by themselves
    joint_step = ""
    step = []
    for step_i, _ in enumerate(steps):
        joint_step += steps[step_i]
        if step_i+1 < len(steps) and is_valid_step(steps[step_i+1]):  # look ahead to determine if to start a new step
            step.append(joint_step)
            joint_step = ""
    if joint_step:
        step.append(joint_step)

    steps = [step + STEP_TOKEN for step in step]
    paired_steps = permuter.permute_all_steps(steps)

    questions = [sep.join(steps[:i]) for i in range(1, len(steps))]
    questions_paired = [sep.join(paired_steps[:i]) for i in range(1, len(steps))]
    targets = [step for i, step in enumerate(steps) if i > 0]

    for question, question_paired, target in zip(questions, questions_paired, targets):
        yield {question_key + "_orig": x[question_key],
               question_key: question,
               question_key + "_paired": question_paired,
               "steps": steps,
               # "steps_paired": steps,
               chain_key: target,
               answer_key: x[answer_key]}


def preprocessing_factory(tokenizer, question_key, answer_key, chain_key, split: str, baseline: bool = True):
    # features encoding
    def preprocess_fn(sample):
        inputs = tokenizer(sample[question_key], truncation=True)
        labels = tokenizer(text_target=sample[chain_key], truncation=True)

        out_dict = {"question": sample[question_key],
                    "answer": sample[answer_key],
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "labels": labels.input_ids,
                    "chain": sample[chain_key]}

        if not baseline and split == "train":
            inputs_paired = tokenizer(sample[question_key + "_paired"], truncation=True)
            out_dict["paired_input_ids"] = inputs_paired.input_ids
            out_dict["paired_attention_mask"] = inputs_paired.attention_mask

        return out_dict

    return preprocess_fn


# tokenize and preprocess datasets for training
preprocessed_datasets = {}
ds_to_lens = {}
for dset_name, keys in dataset_to_keys.items():
    dataset = datasets.load_dataset(f"MU-NLPC/{dset_name}")

    if dset_name in train_datasets_keys:
        # we apply per-step flattening on only train datasets
        # for simplicity, flatten_sample_per_step requires batch_size=1
        # dataset["train"] = dataset["train"].select(range(200))  # TODO: for debug only
        augmented_dataset = (flatten_sample_per_step(sample, **keys) for sample in tqdm(dataset["train"].to_list()))
        flattened_dataset = itertools.chain(*augmented_dataset)
        dataset["train"] = datasets.Dataset.from_list(list(flattened_dataset))
        # remove samples where we extracted empty label (=reasoning step) -> avoid training to generate empty step
        dataset["train"] = dataset["train"].filter(lambda row: row[keys["chain_key"]].strip())
        # in training datasets, we additionally encode steps mask to be used for aggregation of each step encoding
        dataset["train"] = dataset["train"].map(preprocessing_factory(tokenizer, **keys, split="train"))
    else:
        print("Omitting dataset %s from training" % dset_name)
    if dset_name in val_datasets_keys:
        if "gsm" in dset_name:
            # GSM does not have standard validation split, so we need to create it
            val_data = dataset["test"].select(list(range(valid_size)))
            dataset["validation"] = val_data
            # Remove the first 100 samples from the test set
            dataset["test"] = dataset["test"].select(list(range(valid_size, len(dataset["test"]))))

        # steps mask is not available during the generation -> the model has to figure out the steps itself
        dataset["validation"] = dataset["validation"].map(preprocessing_factory(tokenizer, **keys, split="validation"))

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

# Only using 100 samples for validation from each dataset to speed things up
for dset_name, dataset in preprocessed_datasets.items():
    if dset_name not in val_datasets_keys:
        print("Omitting dataset %s from validation" % dset_name)
        continue
    if len(preprocessed_datasets[dset_name]["validation"]) > valid_size:
        preprocessed_datasets[dset_name]["validation"] = dataset["validation"].select(range(valid_size))

# Dropping columns so we can merge datasets
# columns_to_keep = ["question", "answer", "input_ids", "attention_mask", "labels", "chain"]
columns_to_keep = ["input_ids", "attention_mask", "labels", "steps_mask", "paired_input_ids", "paired_attention_mask"]
for dset_name, dataset in preprocessed_datasets.items():
    for split_name, split_dset in dataset.items():
        columns_to_remove = [column for column in split_dset.column_names if column not in columns_to_keep]
        dataset[split_name] = split_dset.remove_columns(columns_to_remove)

# concatenating datasets
train_ds = concatenate_datasets([d["train"] for k, d in preprocessed_datasets.items() if k in train_datasets_keys])
valid_ds = concatenate_datasets([d["validation"] for k, d in preprocessed_datasets.items() if k in val_datasets_keys])
test_ds = concatenate_datasets([dset["test"] for dset in preprocessed_datasets.values()])  # NOT USED

train_ds.shuffle()

log_predictions_indices = np.array(range(valid_size))

# PART: custom evaluations' logging
metrics = gadgets.metrics.MyMetrics(
    tokenizer=tokenizer,
    model=model,
    steps_separator=STEP_TOKEN,
    log_predictions=True,
    log_predictions_indices=log_predictions_indices,
    datasets_id_length={k: len(preprocessed_datasets[k]['validation']) for k in dataset_to_keys.keys() if k in val_datasets_keys},
    # TODO: ordering and sizes must match eval_dataset
)

training_args = transformers.Seq2SeqTrainingArguments(
    output_dir=log_path + wandb.run.name,  # TODO add
    # output_dir=log_path,
    learning_rate=5e-5,
    do_train=True,
    do_eval=True,
    warmup_steps=1000,
    max_steps=200_000,
    per_device_train_batch_size=8,  # TODO
    gradient_accumulation_steps=4,  # TODO
    per_device_eval_batch_size=1,
    eval_accumulation_steps=16,
    logging_steps=50,  # TODO: 4000 steps =~ 1 hour training, 1 hour eval, 8000 steps =~ 2 hour training, 1 hour eval
    eval_steps=1000,  # TODO
    save_steps=1000,
    evaluation_strategy="steps",
    bf16=True,  # TODO
    predict_with_generate=True,
    generation_max_length=512,
    include_inputs_for_metrics=True,
    report_to="wandb",
    metric_for_best_model="avg_correct_results",
    greater_is_better=True,
    load_best_model_at_end=True,
    save_total_limit=2,
    # no_cuda=True,  # TODO: remove
    # use_cpu=True,
    remove_unused_columns=False,
)

trainer = transformers.Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)
# cycle-dependency hack: we make trainer available to the model to log different losses separately:
model.trainer = trainer

trainer.train()  # TODO: resume_from_checkpoint?
