from __future__ import annotations

import transformers
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM

import gadgets

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer = transformers.T5Tokenizer.from_pretrained("google/flan-t5-small")

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

preprocess = gadgets.prep.Preprocessing(tokenizer=tokenizer)


def parse_and_preprocess_aqua(example: dict[str, str]):
    example_with_gadgets = gadgets.aqua.parse(example)
    input_sample = preprocess(example_with_gadgets)
    return input_sample


def parse_and_preprocess_gsm(example: dict[str, str]):
    example = gadgets.gsm8k.parse(example)
    example = preprocess(example)
    return example


aqua = load_dataset("aqua_rat").map(parse_and_preprocess_aqua)
aqua = aqua.map(remove_columns=['input_ids', 'attention_mask', 'labels'])

gsm8k = load_dataset("gsm8k", "main").map(parse_and_preprocess_gsm)
gsm8k = gsm8k.map(remove_columns=['input_ids', 'attention_mask', 'labels'])

print("AQUA: %s" % aqua)
print("GSM8K: %s" % gsm8k)

aqua.push_to_hub("MU-NLPC/Calc-aqua_rat")
gsm8k.push_to_hub("MU-NLPC/Calc-gsm8k")

print("Done.")
