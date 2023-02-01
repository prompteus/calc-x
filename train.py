import torch.utils.data
import transformers
from transformers import Seq2SeqTrainer

import evaluate

import itertools

from dataset import DataSampler


from data_iterators.jl_iterator import JLIterator


# print(evaluate.list_evaluation_modules())
with JLIterator("sample_data/word_problems_petr.jl") as jlif:
    full_dataset = jlif
    with JLIterator("sample_data/word_problems_petr.jl") as jlit:
        train_dataset = jlit
        with JLIterator("sample_data/word_problems_petr.jl") as jlie:
            eval_dataset = jlie

            for i in range(10):
                print(next(full_dataset))
exit()

# generate the training data into a list
train_data = []
for prompt, chain, answer in train_dataset:
    train_data.append({
        "source_text": prompt,
        "target_text": chain
    })

# generate the eval data into a list
eval_data = []
for prompt, chain, answer in eval_dataset:
    eval_data.append({
        "source_text": prompt,
        "target_text": chain
    })

# wrap it into a torch dataset
train_dataset = torch.utils.data.Dataset.from_iterable(train_data)


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
    do_predict=True,
    do_train=True,
    max_steps=100
)


data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)
# get the GLUE module metric function from HuggingFace
compute_metrics = evaluate.load("bleu", tokenizer=tokenizer)

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
