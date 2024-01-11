from __future__ import annotations

import itertools
import os

import torch
import numpy as np
import wandb
import transformers
import datasets

import gadgets
from gadgets.data_iterators.synthetic_iterator import SyntheticIterator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_properties(i))


model_name = "Salesforce/codet5-large"


wandb.init(
    project="gadgets",
    tags=[model_name, "calculator", "synthetic"],
    group="calculator-synthetic",
    dir="/var/tmp/xkadlci2/gadgets/",
)


tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
model = gadgets.model.gadget_assisted_model(transformers.T5ForConditionalGeneration).from_pretrained(model_name)
model.prepare_for_generate(
    tokenizer,
    enabled_gadgets=[gadgets.gadget.Calculator()],
    default_max_tokens=400,
)
data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)


preprocess = gadgets.prep.Preprocessing(tokenizer=tokenizer)


train_ds = datasets.IterableDataset.from_generator(
    SyntheticIterator,
    gen_kwargs=dict(
        nouns_filepath="helper_data/nouns.txt",
        names_filepath="helper_data/names.txt",
        seed=42,
    )
).map(preprocess)

eval_ds_endless = datasets.IterableDataset.from_generator(
    SyntheticIterator,
    gen_kwargs=dict(
        nouns_filepath="helper_data/nouns.txt",
        names_filepath="helper_data/names.txt",
        seed=0,
    )
).map(preprocess)
eval_ds_size = 100
eval_ds = datasets.Dataset.from_list(list(itertools.islice(eval_ds_endless, eval_ds_size)))

random_rng = np.random.default_rng(42)
log_predictions_indices = random_rng.choice(
    range(eval_ds_size),
    size=min(32, eval_ds_size),
    replace=False,
)

metrics = gadgets.metrics.MyMetrics(
    tokenizer=tokenizer,
    log_predictions=True,
    log_predictions_indices=log_predictions_indices,
)

training_args = transformers.Seq2SeqTrainingArguments(
    output_dir="/var/tmp/xkadlci2/gadgets/models/" + wandb.run.name,
    learning_rate=5e-5,
    do_train=True,
    do_eval=True,
    warmup_steps=1000,
    max_steps=1_000_000,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=16,
    logging_steps=50,
    eval_steps=400,
    save_steps=400,
    evaluation_strategy="steps",
    fp16=True,
    predict_with_generate=True,
    generation_max_length=400,
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
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=metrics,
)

trainer.train()
