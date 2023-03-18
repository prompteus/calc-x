from __future__ import annotations

import argparse
import pathlib
import re
import os
import json

import torch
import numpy as np
import transformers
import datasets
from tqdm import tqdm

import gadgets

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_properties(i))

argparser = argparse.ArgumentParser()

argparser.add_argument("--model_checkpoint", type=pathlib.Path)
argparser.add_argument("--output_jsonl", type=pathlib.Path)


args = argparser.parse_args()


model_checkpoint = args.model_checkpoint
tokenizer = transformers.T5Tokenizer.from_pretrained(model_checkpoint)
model = gadgets.model.gadget_assisted_model(transformers.T5ForConditionalGeneration).from_pretrained(model_checkpoint)


text = "<gadget>2+2</gadget>"
encoded = tokenizer(text, return_tensors="pt").input_ids
decoded = tokenizer.batch_decode(encoded, skip_special_tokens=True, spaces_between_special_tokens=False)
assert decoded[0] == text, decoded[0]


math_qa = datasets.load_dataset("math_qa")
np.random.seed(2)
indices = np.random.choice(len(math_qa["train"]), 1000).tolist()

model.prepare_for_generate(
    tokenizer,
    enabled_gadgets=[gadgets.gadget.Calculator()],
    default_max_tokens=512,
)

options_re = re.compile(r"a\s\)\s(?P<a>.*) , b\s\)\s(?P<b>.*) , c\s\)\s(?P<c>.*) , d\s\)\s(?P<d>.*), e\s\)\s(?P<e>.*)")

model = model.eval().to("cuda")

with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    with torch.no_grad():
        with open(args.output_jsonl, "a") as f:
            for i in tqdm(indices):
                example = math_qa["train"][i].copy()
                options = re.match(options_re, example["options"])
                if options is None:
                    continue
                options = options.groupdict()

                predicted_chain, predicted_result = model.generate(
                    inputs=example["Problem"],
                    return_as_str=True,
                    return_result=True,
                    num_beams=6,
                )
                example["index"] = i
                example["predicted_chain"] = predicted_chain
                example["predicted_result"] = predicted_result
                example["options_dict"] = options
                example["result_str"] = options[example["correct"]]
                print(example)
                json.dump(example, f)
                f.write('\n')

