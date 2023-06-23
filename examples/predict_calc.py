from __future__ import annotations

import argparse
import json
import os
import pathlib

import datasets
import torch
import transformers
from baseline_utils import dataset_to_keys, dataset_to_labeler, labeling_factory, preprocessing_factory
from tqdm.auto import tqdm

import gadgets

argparser = argparse.ArgumentParser()

argparser.add_argument("--model_checkpoint", type=pathlib.Path, required=True)
argparser.add_argument("--dataset", type=str, required=True)
argparser.add_argument("--split", type=str, required=True)
argparser.add_argument("--output_jsonl", type=pathlib.Path, required=True)
argparser.add_argument("--use_gadgets", type=bool, required=True, action=argparse.BooleanOptionalAction)
argparser.add_argument("--num_beams", type=int, default=1)
argparser.add_argument("--max_length", type=int, default=512)
args = argparser.parse_args()


if os.path.exists(args.output_jsonl):
    print(f"Output file {args.output_jsonl} already exists, exiting.")
    exit()

model_checkpoint = args.model_checkpoint

tokenizer = transformers.T5Tokenizer.from_pretrained(model_checkpoint)
model_class = transformers.T5ForConditionalGeneration
if args.use_gadgets:
    model_class = gadgets.model.gadget_assisted_model(model_class)

model = model_class.from_pretrained(model_checkpoint)

if args.use_gadgets:
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
        default_max_tokens=args.max_length,
    )

model = model.eval().to("cuda")


dataset = datasets.load_dataset(args.dataset, split=args.split)
dataset_name = args.dataset.split("/")[-1]
question_key = dataset_to_keys[dataset_name]["question_key"]
answer_key = dataset_to_keys[dataset_name]["answer_key"]

if not args.use_gadgets:
    keys = dataset_to_keys[dataset_name]
    preprocessing_fn = preprocessing_factory(tokenizer=tokenizer, **keys)
    labeler_fn = labeling_factory(tokenizer, dataset_to_labeler[dataset_name], question_key)
    dataset = dataset.map(preprocessing_fn)
    dataset = dataset.map(labeler_fn)
    dataset = dataset.filter(lambda example: example["labels"] is not None)
else:
    dataset = dataset.map(
        lambda example: {
            "input_ids": tokenizer(example[question_key]).input_ids,
            "question": example[question_key],
            "answer": example[answer_key],
        },
        remove_columns=[question_key, answer_key],
    )

with (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    torch.no_grad(),
    open(args.output_jsonl, "a") as output_file,
):
    for example in tqdm(dataset):
        example = example.copy()
        input_ids = torch.tensor(example["input_ids"]).to(model.device).reshape(1, -1)
        pred_tokens = model.generate(
            input_ids,
            generation_config=transformers.GenerationConfig(
                num_beams=args.num_beams,
                max_length=args.max_length,
            ),
        )
        prediction_str = tokenizer.batch_decode(
            pred_tokens,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
        )[0]
        example["prediction"] = prediction_str
        for key in ["input_ids", "labels", "attention_mask", "labels_old"]:
            if key in example:
                del example[key]
        json.dump(example, output_file, ensure_ascii=False)
        output_file.write("\n")
