import evaluate
import numpy as np
import transformers
from transformers import Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments

from data_iterators.infinite_iterator import InfiniteIterator
from data_iterators.jl_iterator import JLIterator
from dataset import DataSampler

model_name = "Salesforce/codet5-small"

metric = evaluate.load("sacrebleu")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds.argmax(-1), skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


training_args = Seq2SeqTrainingArguments(output_dir="train_dir",
                                         learning_rate=2e-5,
                                         do_train=True,
                                         do_eval=True,
                                         warmup_steps=1000,
                                         max_steps=400000,
                                         gradient_accumulation_steps=3,
                                         logging_steps=50,
                                         eval_steps=1,
                                         save_steps=5000,
                                         num_train_epochs=30,
                                         evaluation_strategy="steps")

model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)

train_dataset = DataSampler(generator=JLIterator("sample_data/word_problems_petr.jl").__enter__(), tokenizer=tokenizer)
# train_dataset = DataSampler(generator=DataMiningIterator(), tokenizer=tokenizer)
eval_dataset = DataSampler(generator=InfiniteIterator(), tokenizer=tokenizer, max_samples=1)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model()
