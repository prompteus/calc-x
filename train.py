import math
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
import gadgets.markup
from gadgets.data_iterators.synthetic_iterator import SyntheticIterator


import torch
print([torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())])


def are_numeric_results_same(pred: str, true: str, abs_tol: float = 1e-5) -> bool:
    if pred.strip() == true.strip():
        return True

    calculator = gadgets.gadget.Calculator()
    try:
        pred_float = calculator._float_eval(pred)
        true_float = calculator._float_eval(true)
        return math.isclose(pred_float, true_float, abs_tol=abs_tol)
    except (TypeError, SyntaxError, ValueError):
        pass

    return False



class MyMetrics:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.sacrebleu = evaluate.load("sacrebleu")
        self.rouge = evaluate.load("rouge")
        self.tokenizer = tokenizer

    def __call__(self, eval_preds: transformers.EvalPrediction):
        preds, trues = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        trues = np.where(trues != -100, trues, self.tokenizer.pad_token_id)

        preds_str = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        trues_str = self.tokenizer.batch_decode(trues, skip_special_tokens=True)

        sacrebleu_score = self.sacrebleu.compute(predictions=preds_str, references=trues_str)
        rouge_scores = self.rouge.compute(predictions=preds_str, references=trues_str)

        pred_num_tokens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]

        correct_results: list[bool] = []
        num_gadget_calls_pred: list[int] = []
        num_gadget_calls_true: list[int] = []

        for pred, true in zip(preds_str, trues_str):
            pred_chain, pred_result = gadgets.markup.from_model_markup(pred)
            true_chain, true_result = gadgets.markup.from_model_markup(true)
            assert true_result is not None
            pred_result = "" if pred_result is None else pred_result
            true_result = "" if true_result is None else true_result
            correct_results.append(are_numeric_results_same(pred_result, true_result))
            num_gadget_calls_true.append(
                sum(isinstance(step, gadgets.datatypes.Interaction) for step in true_chain)
            )
            num_gadget_calls_pred.append(
                sum(isinstance(step, gadgets.datatypes.Interaction) for step in pred_chain)
            )

        return {
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "rougeLsum": rouge_scores["rougeLsum"],
            "sacrebleu": sacrebleu_score["score"],
            "num_tokens": np.mean(pred_num_tokens),
            "num_gadget_calls_pred": np.mean(num_gadget_calls_pred),
            "num_gadget_calls_true": np.mean(num_gadget_calls_true),
            "correct_results": np.mean(correct_results),
        }



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


metrics = MyMetrics(tokenizer=tokenizer)

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
trainer.save_model()
