import itertools

import torch
import evaluate
import numpy as np
import transformers
import datasets

import gadgets.prep
import gadgets.datatypes
import gadgets.model
import gadgets.gadget
from gadgets.data_iterators.synthetic_iterator import SyntheticIterator


import torch
print([torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())])


metric = evaluate.load("sacrebleu")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds.argmax(-1), skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


training_args = transformers.Seq2SeqTrainingArguments(
    output_dir="train_dir",
    learning_rate=2e-5,
    do_train=True,
    do_eval=True,
    warmup_steps=1000,
    max_steps=400000,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=4,
    logging_steps=50,
    eval_steps=10, # todo
    save_steps=400,
    evaluation_strategy="steps",
    fp16=True,
    predict_with_generate=True,
    generation_max_length=200,
)


model_name = "Salesforce/codet5-small"
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
model = gadgets.model.gadget_assisted_model(transformers.T5ForConditionalGeneration).from_pretrained(model_name)
model.prepare_for_generate(tokenizer, enabled_gadgets=[gadgets.gadget.Calculator()])
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
eval_ds_size = 20
eval_ds = datasets.Dataset.from_list(list(itertools.islice(eval_ds_endless, eval_ds_size)))


trainer = transformers.Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()
trainer.save_model()
