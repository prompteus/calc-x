from __future__ import annotations

import ast
import copy
import json
import math
import pathlib
import sys
import random
import warnings
import itertools
from typing import Iterator, Optional, Iterable

import datasets
import numpy as np
import torch
import transformers
import typer
from tqdm.auto import tqdm

import gadgets

app = typer.Typer()


def get_generation_config(
    default: dict,
    context: typer.Context,
) -> dict:
    generation_kwargs = default.copy()
    for key, value in zip(context.args[::2], context.args[1::2]):
        generation_kwargs[key.lstrip("-")] = ast.literal_eval(value)
    return generation_kwargs


@app.command(
    context_settings=dict(
        allow_extra_args=True,
        ignore_unknown_options=True,
    ),
)
def main(
    model_checkpoint: str,
    dataset_name: str,
    split: str = typer.Option(...),
    ds_subset: Optional[str] = None,
    output_jsonl: pathlib.Path = typer.Option(...),
    use_gadgets: bool = typer.Option(...),
    from_nth_example: int = -1,
    to_nth_example: int = -1,
    sample_n_examples: int = -1,
    num_preds_per_example: Optional[int] = None,
    batch_size: int = 1,
    prediction_column: str = "prediction",
    prediction_result_column: str = "prediction_result",
    question_column: str = "question",
    template_column: str = "template",
    result_column: str = typer.Option("result", help="Only required for displaying accuracy during prediction."),
    max_tokens: int = 768,
    instructions: Optional[str] = None,
    ignore_instruction_probs: bool = False,
    seed: int = 0,
    device: str = None,
    dtype: str = None,
    generation_kwargs: typer.Context = ...,
) -> None:
    generation_kwargs = get_generation_config(
        default=dict(num_beams=1, do_sample=False, max_length=max_tokens),
        context=generation_kwargs,
    )
    print("Generation kwargs:", generation_kwargs)

    command = " ".join(sys.argv)  # pylint: disable=unused-variable

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if dtype is None:
        if device == "cuda":
            dtype = "bfloat16"
        else:
            dtype = "float32"

    pred_config = copy.deepcopy(locals())

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)

    if instructions is not None:
        instructions = datasets.load_dataset("MU-NLPC/Calc-X_instructions", split=instructions)

    if num_preds_per_example is None:
        if instructions is None:
            num_preds_per_example = 1
        else:
            num_preds_per_example = len(instructions)

    if num_preds_per_example > 1 and not generation_kwargs.get("do_sample", False):
        warnings.warn(
            "num_preds_per_input > 1 but do_sample not set. This can result in duplicate predictions."
        )

    if num_preds_per_example < 1:
        raise ValueError("num_preds_per_example must be positive.")

    if output_jsonl.exists():
        print(f"Output file {output_jsonl} already exists, exiting.")
        exit()

    output_config_json = output_jsonl.with_suffix(".config.json")
    if output_config_json.exists():
        print(f"Output file {output_config_json} already exists, exiting.")
        exit()

    dataset = datasets.load_dataset(dataset_name, split=split)
    if ds_subset is not None:
        dataset = dataset.filter(lambda x: x["source_ds"] == ds_subset)

    if prediction_column in dataset.column_names:
        raise ValueError(f"Column '{prediction_column}' already exists in dataset '{dataset_name}'.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
    model_class = transformers.T5ForConditionalGeneration
    if use_gadgets:
        model_class = gadgets.model.gadget_assisted_model(model_class)

    model = model_class.from_pretrained(model_checkpoint, device_map=device, torch_dtype=getattr(torch, dtype)).eval()

    if use_gadgets:
        gadgets.utils.add_new_token(
            "<",
            is_special=False,
            tokenizer=tokenizer,
            model=model,
            init_with=["[", ">"],
        )

        assert isinstance(model, gadgets.model.GadgetAssist)
        model.prepare_for_generate(
            tokenizer,
            enabled_gadgets=[gadgets.gadget.Calculator()],
            default_max_tokens=max_tokens,
        )
    
    if from_nth_example == -1:
        from_nth_example = 0
    from_nth_example = max(from_nth_example, 0)

    if to_nth_example == -1:
        to_nth_example = len(dataset)
    to_nth_example = min(to_nth_example, len(dataset))

    if from_nth_example >= to_nth_example:
        raise ValueError("from_nth_example must be smaller than to_nth_example.")

    dataset = dataset.select(range(from_nth_example, to_nth_example))

    if sample_n_examples > 0:
        indices = torch.randperm(len(dataset))[:sample_n_examples]
        dataset = dataset.select(indices)

    generation_config = transformers.GenerationConfig(**generation_kwargs)

    with open(output_config_json, "w") as output_config_file:
        json.dump(
            pred_config,
            output_config_file,
            ensure_ascii=False,
            default=str,
            indent=2,
        )

    progress = tqdm(dataset, desc="predicting")
    examples = repeat_every_elem(progress, num_preds_per_example)
    examples = batched(examples, batch_size)
    template_stream = get_template_stream(instructions, ignore_instruction_probs)

    num_total_preds = 0
    num_correct_preds = 0

    # reorder keys in output jsonl to compare results with prediction results on first glance
    output_key_order = [
        "id",
        "source_ds",
        result_column,
        prediction_result_column,
    ]

    with open(output_jsonl, "a") as output_file:
        for batch in examples:
            batch_templates = list(itertools.islice(template_stream, len(batch)))
            inputs_str = [
                template.format(example[question_column].strip())
                for example, template in zip(batch, batch_templates)
            ]
            inputs = tokenizer(inputs_str, return_tensors="pt", padding=True, truncation=True)
            preds = model.generate(**inputs.to(model.device), generation_config=generation_config)
            preds_str = tokenizer.batch_decode(preds, skip_special_tokens=True, spaces_between_special_tokens=False)

            for example, template, prediction in zip(batch, batch_templates, preds_str):
                pred_result = gadgets.markup.get_result_from_output(prediction)

                example_export = example.copy()
                example_export[prediction_column] = prediction
                example_export[prediction_result_column] = pred_result
                example_export[template_column] = template

                if result_column is not None:
                    true_result = example[result_column]
                    if gadgets.metrics.are_results_same(true_result, pred_result):
                        num_correct_preds += 1
                num_total_preds += 1
               
                json.dump(reorder_keys(example_export, output_key_order), output_file, ensure_ascii=False)
                output_file.write("\n")

            output_file.flush()
            accuracy = num_correct_preds / num_total_preds
            progress.set_description(f"{accuracy=:.2%}  predicting")


def reorder_keys(
    dictionary: dict,
    keys: list[str],
) -> dict:
    out = {key: dictionary[key] for key in keys if key in dictionary}
    for key, value in dictionary.items():
        if key not in keys:
            out[key] = value
    return out


def repeat_every_elem(
    iterable: Iterable,
    n: int,
) -> Iterator:
    for item in iterable:
        for _ in range(n):
            yield item


def batched(
    iterable: Iterable,
    batch_size: int,
) -> Iterator:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def get_template_stream(
    instructions: datasets.Dataset | None,
    ignore_probs: bool,
) -> Iterator[str]:
    if instructions is None:
        while True:
            yield "{}"
    elif ignore_probs:
        templates = instructions["template"].copy()
        np.random.shuffle(templates)
        while True:
            yield from templates
    else:
        while True:
            yield np.random.choice(instructions["template"], p=instructions["weight"])


def low_bound_num_examples(
    at_least_n_correct_preds_per_example: int | float,
    at_least_n_incorrect_preds_per_example: int | float,
) -> int | float:
    low_bound = 0
    if at_least_n_correct_preds_per_example < math.inf:
        low_bound += at_least_n_correct_preds_per_example
    if at_least_n_incorrect_preds_per_example < math.inf:
        low_bound += at_least_n_incorrect_preds_per_example
    return low_bound


if __name__ == "__main__":
    app()
