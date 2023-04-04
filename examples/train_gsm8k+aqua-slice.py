from __future__ import annotations

import argparse
import os
import random

import datasets
import numpy as np
import transformers
import wandb
from datasets import concatenate_datasets
from transformers import EarlyStoppingCallback

import gadgets

# IMHO this makes it even messier than before, but keeping it here for now

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# for i in range(torch.cuda.device_count()):
#     print(i, torch.cuda.get_device_properties(i))

# global arguments:

# (1) general
argparser = argparse.ArgumentParser()
argparser.add_argument("--output_dir", type=str)
argparser.add_argument("--model_name", type=str, default="google/flan-t5-large")
argparser.add_argument("--wandb_project_name", type=str)
argparser.add_argument("--wandb_tags", type=str, default="calculator,gsm8k,aqua,supervised",
                       help="Coma-separater list of given wandb tags")

# (2) script-specific
argparser.add_argument("--finetune_whole_model", type=bool, default=False)
argparser.add_argument("--finetune_percent", type=int, default=10, choices=[10, 20, 30, 40])

args = argparser.parse_args()

# PART: model init
gadget_model_cls = gadgets.model.gadget_assisted_model(transformers.T5ForConditionalGeneration)

model = gadget_model_cls.from_pretrained(args.model_name, device_map="auto")
tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_name)

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
aqua_val = datasets.load_dataset("aqua_rat", split="validation").map(parse_and_preprocess_aqua).select(range(100))

gsm8k = datasets.load_dataset("gsm8k", "main").map(parse_and_preprocess_gsm)

train_valid_ds = gsm8k["train"].train_test_split(test_size=100, seed=42)

# upscaling GSM - we don't do that now
# gsm_idx = list(range(len(train_valid_ds["train"])))
# train_valid_ds["train"] = train_valid_ds["train"].select(random.choice(gsm_idx) for _ in range(len(aqua)))

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

# PART: partial model finetuning
if not args.finetune_whole_model:
    finetuning_schemes = {
        10: ("lm_head.weight",
             "SelfAttention.v.weight"),
        20: ("lm_head.weight",
             "SelfAttention.v.weight",
             "SelfAttention.k.weight",
             "EncDecAttention.v.weight"),
        30: ("lm_head.weight",
             "SelfAttention.v.weight",
             "SelfAttention.k.weight",
             "SelfAttention.q.weight",
             "EncDecAttention.v.weight",
             "EncDecAttention.k.weight"),
        40: ("lm_head.weight",
             "SelfAttention.v.weight",
             "SelfAttention.k.weight",
             "SelfAttention.q.weight",
             "SelfAttention.o.weight",
             "EncDecAttention.v.weight",
             "EncDecAttention.k.weight",
             "EncDecAttention.q.weight"),
    }

    for param in model.parameters():
        param.requires_grad_(False)

    for name, param in model.named_parameters():
        if any(train_name in name for train_name in finetuning_schemes[args.finetune_percent]):
            param.requires_grad_(True)

    num_params_total = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percent = num_params_trainable / num_params_total * 100
    print(f"Trainable parameters: {num_params_trainable} ({trainable_percent:.2f}%)")
    assert abs(trainable_percent - args.finetune_percent) < 3, \
        "Finetuning scheme is not correct. Maximum allowed difference is 3%."

    wandb.init(
        project="gadgets",
        tags=[args.model_name] + args.wandb_tags.split(","),
        group=args.wandb_tags.replace(",", "-"),
        dir=args.output_dir,
        save_code=True,
        config={
            "model_name": args.model_name,
            "params_trained_percent": args.finetune_percent,
            "params_trained_count": num_params_trainable,
            "params_trained_pattern": finetuning_schemes[args.finetune_percent],
            "params_trained": [name for name, param in model.named_parameters() if param.requires_grad],
            "params_frozen": [name for name, param in model.named_parameters() if not param.requires_grad],
        }
    )
else:
    # fine-tuning of the whole model
    wandb.init(
        project="gadgets",
        tags=[args.model_name] + args.wandb_tags.split(","),
        group=args.wandb_tags.replace(",", "-"),
        dir=args.output_dir,
    )

# PART: training configuration
training_args = transformers.Seq2SeqTrainingArguments(
    output_dir=os.path.join(args.output_dir, wandb.run.name),
    learning_rate=5e-5,
    do_train=True,
    do_eval=True,
    warmup_steps=5000,
    max_steps=50_000,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=10,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=16,
    logging_steps=100,
    eval_steps=2000,
    save_steps=2000,
    evaluation_strategy="steps",
    bf16=True,
    predict_with_generate=True,
    generation_max_length=512,
    include_inputs_for_metrics=True,
    report_to="wandb",
    metric_for_best_model="aqua_correct_results",
    greater_is_better=True,
    load_best_model_at_end=True,
    save_total_limit=3,
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
