from __future__ import annotations

import argparse
import os

import torch
import numpy as np
import wandb
import transformers
import datasets

import gadgets

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_properties(i))


model_name = "google/flan-t5-xl"


tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
model = gadgets.model.gadget_assisted_model(transformers.T5ForConditionalGeneration).from_pretrained(model_name)

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

model.prepare_for_generate(
    tokenizer,
    enabled_gadgets=[gadgets.gadget.Calculator()],
    default_max_tokens=512,
)
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)

preprocess = gadgets.prep.Preprocessing(tokenizer=tokenizer)

def parse_and_preprocess(example: dict[str, str]):
    example = gadgets.gsm8k.parse(example)
    example = preprocess(example)
    return example

gsm8k = datasets.load_dataset("gsm8k", "main").map(parse_and_preprocess)

train_valid_ds = gsm8k["train"].train_test_split(test_size=200, seed=42)
train_ds = train_valid_ds["train"]
valid_ds = train_valid_ds["test"]
tests_ds = gsm8k["test"]

random_rng = np.random.default_rng(42)
log_predictions_indices = random_rng.choice(
    range(len(valid_ds)),
    size=min(64, len(valid_ds)),
    replace=False,
)

metrics = gadgets.metrics.MyMetrics(
    tokenizer=tokenizer,
    log_predictions=True,
    log_predictions_indices=log_predictions_indices,
)


finetuning_schemes = {
    10: (
        "lm_head.weight",
        "SelfAttention.v.weight",
    ),
    20: (
        "lm_head.weight",
        "SelfAttention.v.weight",
        "SelfAttention.k.weight",
        "EncDecAttention.v.weight",
    ),
    30: (
        "lm_head.weight",
        "SelfAttention.v.weight",
        "SelfAttention.k.weight",
        "SelfAttention.q.weight",
        "EncDecAttention.v.weight",
        "EncDecAttention.k.weight",
    ),
    40: (
        "lm_head.weight",
        "SelfAttention.v.weight",
        "SelfAttention.k.weight",
        "SelfAttention.q.weight",
        "SelfAttention.o.weight",
        "EncDecAttention.v.weight",
        "EncDecAttention.k.weight",
        "EncDecAttention.q.weight",
    ),
}

argparser = argparse.ArgumentParser()
argparser.add_argument("--finetune_percent", type=int, default=10, choices=finetuning_schemes.keys())
args = argparser.parse_args()

for param in model.parameters():
    param.requires_grad_(False)

for name, param in model.named_parameters():
    if any(train_name in name for train_name in finetuning_schemes[args.finetune_percent]):
        param.requires_grad_(True)

num_params_total = sum(p.numel() for p in model.parameters())
num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_percent = num_params_trainable / num_params_total * 100
print(f"Trainable parameters: {num_params_trainable} ({trainable_percent:.2f}%)")
assert abs(trainable_percent - args.finetune_percent) < 3, "Finetuning scheme is not correct. Maximum allowed difference is 3%."


wandb.init(
    project="gadgets",
    tags=[model_name, "calculator", "gsm8k", "supervised", f"trained_params_{trainable_percent:.2f}%"],
    group="calculator-gsm8k-supervised",
    dir="/var/tmp/xkadlci2/gadgets/",
    save_code=True,
    config={
        "model_name": model_name,
        "params_trained_percent": trainable_percent,
        "params_trained_count": num_params_trainable,
        "params_trained_pattern": finetuning_schemes[args.finetune_percent],
        "params_trained": [name for name, param in model.named_parameters() if param.requires_grad],
        "params_frozen": [name for name, param in model.named_parameters() if not param.requires_grad],
    }
)

training_args = transformers.Seq2SeqTrainingArguments(
    output_dir="/var/tmp/xkadlci2/gadgets/models/" + wandb.run.name,
    learning_rate=5e-5,
    do_train=True,
    do_eval=True,
    warmup_steps=1000,
    max_steps=1_000_000,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=16,
    logging_steps=50,
    eval_steps=400,
    save_steps=400,
    evaluation_strategy="steps",
    bf16=True,
    predict_with_generate=True,
    generation_max_length=512,
    include_inputs_for_metrics=True,
    report_to="wandb",
    metric_for_best_model="correct_results",
    greater_is_better=True,
    load_best_model_at_end=True,
    save_total_limit=3,
)

trainer = transformers.Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=metrics,
)

trainer.train()
trainer.evaluate(metric_key_prefix="test")
