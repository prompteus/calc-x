from __future__ import annotations

import ast
import copy
import json
import pathlib
import sys
import warnings

import datasets
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
    first_n_examples: int = -1,
    sample_n_examples: int = -1,
    num_preds_per_example: int = 1,
    prediction_column: str = "prediction",
    question_column: str = "question",
    max_tokens: int = 1024,
    sample_subset_seed: int = 0,
    generation_kwargs: typer.Context = ...,
) -> None:
    
    generation_kwargs = get_generation_config(
        default=dict(num_beams=1, do_sample=False, max_length=max_tokens),
        context=generation_kwargs,
    )
    print("Generation kwargs:", generation_kwargs)
 
    command = " ".join(sys.argv) # pylint: disable=unused-variable

    pred_config = copy.deepcopy(locals())

    if output_jsonl.exists():
        print(f"Output file {output_jsonl} already exists, exiting.")
        exit()

    output_config_json = output_jsonl.with_suffix(".config.json")
    if output_config_json.exists():
        print(f"Output file {output_config_json} already exists, exiting.")
        exit()

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

        model.prepare_for_generate(
            tokenizer,
            enabled_gadgets=[gadgets.gadget.Calculator()],
            default_max_tokens=max_tokens,
        )

    dataset = datasets.load_dataset(dataset_name, split=split)

    if prediction_column in dataset.column_names:
        raise ValueError(f"Column '{prediction_column}' already exists in dataset '{dataset_name}'.")

    if first_n_examples > 0:
        dataset = dataset.select(range(min(first_n_examples, len(dataset))))

    if sample_n_examples > 0:
        rand_gen = torch.Generator().manual_seed(sample_subset_seed)
        indices = torch.randperm(len(dataset), generator=rand_gen)[:sample_n_examples]
        dataset = dataset.select(indices)

    generation_config = transformers.GenerationConfig(**generation_kwargs)

    model = model.eval().to("cuda")

    if num_preds_per_example > 1 and not generation_kwargs.get("do_sample", False):
        warnings.warn("num_preds_per_input > 1 but do_sample not set. This can result in duplicate predictions.")

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
        for example in tqdm(dataset):
            example = example.copy()
            inputs = tokenizer(example[question_column].strip(), return_tensors="pt").to(model.device)

            for i in range(num_preds_per_example):
                if num_preds_per_example > 1:
                    idx = str(i).zfill(len(str(num_preds_per_example)))
                    pred_col = f"{prediction_column}_{idx}"
                else:
                    pred_col = prediction_column

                outputs = model.generate(**inputs, generation_config=generation_config)[0]

                example[pred_col] = tokenizer.decode(
                    outputs,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                )

            json.dump(example, output_file, ensure_ascii=False)
            output_file.write("\n")


if __name__ == "__main__":
    app()
