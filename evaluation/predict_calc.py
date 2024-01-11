from __future__ import annotations

import argparse
import json
import pathlib

import datasets
import torch
import transformers
from tqdm.auto import tqdm

import gadgets
from gadgets.steps_utils import separate_chain_to_steps
from gadgets.stepwise_metrics import PerMistakesConsistency

argparser = argparse.ArgumentParser()

argparser.add_argument("--model_checkpoint", type=pathlib.Path, required=True)
argparser.add_argument("--datasets", type=str, required=True)
argparser.add_argument("--split", type=str, required=True)
argparser.add_argument("--output_jsonl_prefix", type=str, required=True)
argparser.add_argument("--stepwise_generation", type=bool, required=True, action=argparse.BooleanOptionalAction)
argparser.add_argument("--use_gadgets", type=bool, required=True, action=argparse.BooleanOptionalAction)
argparser.add_argument("--predict_alternative_cot", type=bool, required=True, action=argparse.BooleanOptionalAction)
argparser.add_argument("--num_beams", type=int, default=1)
argparser.add_argument("--max_length", type=int, default=512)
argparser.add_argument("--first_n", type=int, default=-1)
args = argparser.parse_args()

model_checkpoint = args.model_checkpoint

tokenizer = transformers.T5Tokenizer.from_pretrained(model_checkpoint)
model_class = transformers.T5ForConditionalGeneration
if args.use_gadgets:
    if args.stepwise_generation:
        model_class = gadgets.model.stepwise_gadget_model(model_class)
    else:
        model_class = gadgets.model.gadget_assisted_model(model_class)

model = model_class.from_pretrained(model_checkpoint)

if args.use_gadgets:
    gadgets.utils.add_new_token("<",
                                is_special=False,
                                tokenizer=tokenizer,
                                model=model,
                                init_with=["[", ">"])
    text = "<gadget>2+2</gadget>"
    encoded = tokenizer(text, return_tensors="pt").input_ids
    decoded = tokenizer.batch_decode(encoded, skip_special_tokens=True, spaces_between_special_tokens=False)
    assert decoded[0] == text, decoded[0]

    model.prepare_for_generate(tokenizer,
                               enabled_gadgets=[gadgets.gadget.Calculator()],
                               default_max_tokens=args.max_length)

model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")


def split_by_all(chain: str, ) -> list[str]:
    split_symbol = ". " if ". " in chain else ".\n" if ".\n" in chain else "</gadget>" if "</gadget>" in chain else "\n"
    return chain.split(split_symbol)


for dataset_id in args.datasets.split(","):
    dataset = datasets.load_dataset(dataset_id, split=args.split)

    dataset = dataset.map(lambda example: {"input_ids": tokenizer(example["question"]).input_ids,
                                           "question": example["question"],
                                           "answer": example["chain"]},
                          remove_columns=["question"])
    if args.first_n > 0:
        dataset = dataset.select(range(min(args.first_n, len(dataset))))

    if len(args.datasets.split(",")) > 1:
        out_file = args.output_jsonl_prefix.split(".jsonl")[0] + "-" + dataset_id.split("/")[-1] + ".jsonl"
    else:
        out_file = args.output_jsonl_prefix

    with (torch.autocast(device_type="cuda", dtype=torch.bfloat16),
          torch.no_grad(),
          open(out_file, "w") as output_file):
        for example in tqdm(dataset):
            example = example.copy()
            input_ids = torch.tensor(example["input_ids"]).to(model.device).reshape(1, -1)
            pred_tokens = model.generate(input_ids,
                                         generation_config=transformers.GenerationConfig(num_beams=args.num_beams,
                                                                                         max_length=args.max_length))
            prediction_str = tokenizer.batch_decode(pred_tokens,
                                                    skip_special_tokens=True,
                                                    spaces_between_special_tokens=False)[0]
            example["prediction"] = prediction_str

            steps = separate_chain_to_steps(example["chain"])
            example["num_steps"] = len(steps)

            if args.predict_alternative_cot:
                # generate alternative chain, and inject a portion preceding randomly-chosen step as input to the model
                consistency_evaluator = PerMistakesConsistency(model, tokenizer)
                alternative_chain = consistency_evaluator.get_alternative_chain([example["question"]],
                                                                                [prediction_str])[0]
                example["prediction_alternative"] = alternative_chain

            for key in ["input_ids", "labels", "attention_mask", "labels_old", "chain"]:
                if key in example:
                    del example[key]
            json.dump(example, output_file, ensure_ascii=False)
            output_file.write("\n")
