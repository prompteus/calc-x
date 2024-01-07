from __future__ import annotations

import ast
import copy
import json
import math
import pathlib
import sys
import warnings
from typing import Iterator, Optional

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
    model_checkpoint: str = typer.Argument(...),
    dataset_name: str = typer.Argument(...),
    split: str = typer.Option(...),
    output_jsonl: pathlib.Path = typer.Option(...),
    use_gadgets: bool = typer.Option(...),
    from_nth_example: int = -1,
    to_nth_example: int = -1,
    sample_n_examples: int = -1,
    num_preds_per_example: Optional[int] = None,
    prediction_column: str = "prediction",
    question_column: str = "question",
    result_column: str = typer.Option("result", help="Only required if counting (in-)correct predictions."),
    max_tokens: int = 768,
    instructions: Optional[str] = None,
    n_correct_preds_per_example_is_enough: Optional[int] = None,
    n_incorrect_preds_per_example_is_enough: Optional[int] = None,
    sample_subset_seed: int = 0,
    generation_kwargs: typer.Context = ...,
) -> None:
    generation_kwargs = get_generation_config(
        default=dict(num_beams=1, do_sample=False, max_length=max_tokens),
        context=generation_kwargs,
    )
    print("Generation kwargs:", generation_kwargs)

    command = " ".join(sys.argv)  # pylint: disable=unused-variable

    pred_config = copy.deepcopy(locals())

    if n_correct_preds_per_example_is_enough is None:
        n_correct_preds_per_example_is_enough = math.inf

    if n_incorrect_preds_per_example_is_enough is None:
        n_incorrect_preds_per_example_is_enough = math.inf

    if instructions is not None:
        instructions = datasets.load_dataset("MU-NLPC/Calc-X_instructions", split=instructions)

    if num_preds_per_example is None:
        if instructions is None:
            num_preds_per_example = 1
        else:
            num_preds_per_example = len(instructions)

    if num_preds_per_example > 1 and not generation_kwargs.get("do_sample", False) and instructions is None:
        warnings.warn(
            "num_preds_per_input > 1 but do_sample not set. This can result in duplicate predictions."
        )

    if n_correct_preds_per_example_is_enough < 0 or n_incorrect_preds_per_example_is_enough < 0:
        raise ValueError(
            "at_least_n_correct_preds_per_example and at_least_n_incorrect_preds_per_example must be non-negative."
        )

    if (
        n_correct_preds_per_example_is_enough != math.inf
        or n_incorrect_preds_per_example_is_enough != math.inf
    ):
        if num_preds_per_example > 1:
            warnings.warn(
                "num_preds_per_example > 1 but at_least_n_correct_preds_per_example or at_least_n_incorrect_preds_per_example is set. "
                "num_preds_per_example will be interpreted as an upper bound on the number of predictions per example."
            )

    at_least_n_examples = low_bound_num_examples(
        n_correct_preds_per_example_is_enough,
        n_incorrect_preds_per_example_is_enough,
    )

    if at_least_n_examples > num_preds_per_example:
        raise ValueError(
            "num_preds_per_example must be at least as large as at_least_n_correct_preds_per_example + at_least_n_incorrect_preds_per_example."
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

    must_verify = (
        n_correct_preds_per_example_is_enough < math.inf or n_incorrect_preds_per_example_is_enough < math.inf
    )

    dataset = datasets.load_dataset(dataset_name, split=split)

    if must_verify and result_column not in dataset.column_names:
        raise ValueError(
            f"Column '{result_column}' does not exist in dataset '{dataset_name}'. "
            f"It is required to count correct and incorrect predictions."
        )

    if prediction_column in dataset.column_names:
        raise ValueError(f"Column '{prediction_column}' already exists in dataset '{dataset_name}'.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False)
    model_class = transformers.T5ForConditionalGeneration
    if use_gadgets:
        model_class = gadgets.model.gadget_assisted_model(model_class)

    model = model_class.from_pretrained(model_checkpoint)

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
        rand_gen = torch.Generator().manual_seed(sample_subset_seed)
        indices = torch.randperm(len(dataset), generator=rand_gen)[:sample_n_examples]
        dataset = dataset.select(indices)

    generation_config = transformers.GenerationConfig(**generation_kwargs)

    model = model.eval().to("cuda")

    with open(output_config_json, "w") as output_config_file:
        json.dump(
            pred_config,
            output_config_file,
            ensure_ascii=False,
            default=str,
            indent=2,
        )

    with (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        torch.no_grad(),
        open(output_jsonl, "a") as output_file,
    ):
        n_total_preds = 0

        for idx_example, example in enumerate(tqdm(dataset)):
            predictions = []

            n_correct_preds = 0
            n_incorrect_preds = 0

            templates = get_template_stream(instructions)

            for _, template in zip(range(num_preds_per_example), templates):
                example = example.copy()
                example["template"] = template

                question = example[question_column].strip()
                question = template.format(question)
                inputs = tokenizer(question, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, generation_config=generation_config)[0]

                prediction = tokenizer.decode(
                    outputs,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )
                pred_result = gadgets.markup.get_result_from_output(prediction)

                predictions.append(prediction)

                if must_verify:
                    if gadgets.metrics.are_results_same(pred_result, example[result_column]):
                        n_correct_preds += 1
                    else:
                        n_incorrect_preds += 1

                    if (
                        n_correct_preds >= n_correct_preds_per_example_is_enough
                        and n_incorrect_preds >= n_incorrect_preds_per_example_is_enough
                    ):
                        break

            example[prediction_column] = predictions
            n_total_preds += len(predictions)

            if must_verify:
                print(f"current predictions for last example: {n_correct_preds=}, {n_incorrect_preds=}")
                print(f"avg number of predictions per example: {n_total_preds / (idx_example + 1):.2f}")

            json.dump(example, output_file, ensure_ascii=False)
            output_file.write("\n")
            output_file.flush()


def get_template_stream(
    instructions: datasets.Dataset | None,
    random_seed: int | None = None,
) -> Iterator[str]:
    """
    First, it yields all templates in a random order. Then, it yields
    templates sampled accordingly to their probabilites.
    """

    if instructions is None:
        while True:
            yield "{}"
    else:
        random_generator = np.random.default_rng(random_seed)
        templates = instructions["template"].copy()
        random_generator.shuffle(templates)
        yield from templates

        while True:
            yield random_generator.choice(instructions["template"], p=instructions["weight"])


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
