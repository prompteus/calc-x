import transformers
from transformers import Seq2SeqTrainer

import evaluate



from data_iterators.jl_iterator import JLIterator



model_name = "salesforce/codet5-small"

model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

training_args = transformers.TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    warmup_steps=500,
    do_predict=True
)

full_dataset = JLIterator("sample_data/word_problems_petr.jl")
train_dataset = full_dataset[20:]
eval_dataset = full_dataset[:20]

data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)
compute_metrics = evaluate.load("metrics/GLUE", module_type="metric")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.do_predict else None,
)

trainer.train()

trainer.save_model()
