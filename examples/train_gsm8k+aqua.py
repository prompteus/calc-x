from __future__ import annotations

import os
import random

import torch
import numpy as np
import wandb
import transformers
import datasets
from datasets import concatenate_datasets
from transformers import EarlyStoppingCallback

import gadgets

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_properties(i))


model_name = "google/flan-t5-small"


# wandb.init(
#     project="gadgets",
#     tags=[model_name, "calculator", "gsm8k", "aqua", "supervised"],
#     group="calculator-gsm8k-aqua-supervised",
#     dir="/var/tmp/xkadlci2/gadgets/",
# )


tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
model = gadgets.model.gadget_assisted_model(transformers.T5ForConditionalGeneration).from_pretrained(model_name)

# PART: update model for unknown tokens
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
model.prepare_for_generate(
    tokenizer,
    enabled_gadgets=[gadgets.gadget.Calculator()],
    default_max_tokens=512,
)
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)

preprocess = gadgets.prep.Preprocessing(tokenizer=tokenizer)


# PART: datasets curation
def parse_and_preprocess_aqua(example: dict[str, str]):
    example_with_gadgets = gadgets.aqua.parse(example)
    input_sample = preprocess(example_with_gadgets)
    return input_sample


def parse_and_preprocess_gsm(example: dict[str, str]):
    example = gadgets.gsm8k.parse(example)
    example = preprocess(example)
    return example


aqua = datasets.load_dataset("aqua_rat", split="train").map(parse_and_preprocess_aqua)
aqua_val = datasets.load_dataset("aqua_rat", split="validation").map(parse_and_preprocess_aqua)

gsm8k = datasets.load_dataset("gsm8k", "main").map(parse_and_preprocess_gsm)

train_valid_ds = gsm8k["train"].train_test_split(test_size=400, seed=42)

gsm_idx = list(range(len(train_valid_ds["train"])))
train_valid_ds["train"] = train_valid_ds["train"].select(random.choice(gsm_idx) for _ in range(len(aqua)))

train_ds = concatenate_datasets([train_valid_ds["train"], aqua])
train_ds = train_ds.shuffle()

valid_ds = train_valid_ds["test"]
tests_ds = gsm8k["test"]

# PART: shuffling of the logged predictions
random_rng = np.random.default_rng(42)
log_predictions_indices = random_rng.choice(
    range(len(valid_ds)),
    size=min(64, min(len(valid_ds), len(aqua_val))),
    replace=False,
)

# PART: custom evaluations' logging
metrics = gadgets.metrics.MyMetrics(
    tokenizer=tokenizer,
    log_predictions=True,
    log_predictions_indices=log_predictions_indices,
    datasets_id_length={"gsm": len(valid_ds), "aqua": len(aqua_val)}  # TODO: ordering and sizes must match eval_dataset
)

training_args = transformers.Seq2SeqTrainingArguments(
    # output_dir="/var/tmp/xkadlci2/gadgets/models/" + wandb.run.name,
    output_dir=".",
    learning_rate=5e-5,
    do_train=True,
    do_eval=True,
    warmup_steps=1000,
    max_steps=1_000_000,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=16,
    logging_steps=50,
    eval_steps=1,
    save_steps=4000,
    evaluation_strategy="steps",
    bf16=True,
    predict_with_generate=True,
    generation_max_length=512,
    include_inputs_for_metrics=True,
    report_to="wandb",
    metric_for_best_model="correct_results",
    greater_is_better=True,
    load_best_model_at_end=True,
    save_total_limit=15,
)

trainer = transformers.Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=concatenate_datasets([valid_ds, aqua_val]),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=15)]
)

trainer.train()
trainer.evaluate(eval_dataset=tests_ds, metric_key_prefix="test")
