from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import warnings
from typing import Any, Optional

import datasets
import numpy as np
import peft
import pydantic
import torch
import transformers
import typer
import yaml

import gadgets
import wandb

app = typer.Typer()


@app.command()
def train(
    path_train_args: pathlib.Path = typer.Option(
        "--train-config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the training config file - either json or yaml. file is parsed and passed to `transformers.Seq2SeqTrainingArguments` constructor.",
    ),
    path_model_config: Optional[pathlib.Path] = typer.Option(
        "--model-config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the model config file - either json or yaml. file is parsed and passed to `transformers.AutoModelForSeq2SeqLM.from_pretrained`.",
    ),
    path_data_config: Optional[pathlib.Path] = typer.Option(
        "--data-config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the dataset config file - either json or yaml. TODO.",
    ),
    path_peft_config: Optional[pathlib.Path] = typer.Option(
        "--peft-config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the peft config file - either json or yaml. file is parsed and passed to `peft.get_peft_config` method.",
    ),
    path_deepspeed_config: Optional[pathlib.Path] = typer.Option(
        "--deepseed-config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the deepspeed config file. it will be as `deepspeed` parameter to `transformers.Seq2SeqTrainingArguments` constructor",
    ),
    path_early_stop_config: pathlib.Path = typer.Option(
        "--early-stop-config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the early stop config file - either json or yaml. file is parsed and passed to `transformers.EarlyStoppingCallback` constructor.",
    ),
    load_local_checkpoint: Optional[pathlib.Path] = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the local checkpoint to load. If set, the model will be loaded from the checkpoint instead of the pretrained model.",
    ),
    num_log_preds_per_dataset: int = typer.Option(
        64,
        help="Number of predictions to log to wandb for each dataset. Only use if `wandb` is installed.",
    ),
    wandb_entity: str = typer.Option("transformersclub"),
    wandb_project: str = typer.Option("gadgets"),
    wandb_group: str = typer.Option(...),
    wandb_tag: tuple[str] = typer.Option(...),
) -> None:
    print("cuda devices:")
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i))

    if path_model_config is None and load_local_checkpoint is None:
        raise ValueError("Either --model-config or --load-local-checkpoint must be set")
    if path_model_config is not None and load_local_checkpoint is not None:
        warnings.warn(
            "Both --model-config and --load-local-checkpoint are set. "
            "'pretrained_model_name_or_path' in --model-config will not be checked to match the checkpoint. "
            "Make sure they match."
        )

    model_config_dict = None
    peft_config_dict = None
    deepspeed_config_dict = None
    early_stop_config_dict = None

    if load_local_checkpoint is not None:
        model_name = str(load_local_checkpoint)
        from_pretrained_args = {"pretrained_model_name_or_path": model_name}
    else:
        model_config_dict = read_config(path_model_config)
        model_name = model_config_dict["pretrained_model_name_or_path"]
        from_pretrained_args = model_config_dict

    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        group=wandb_group,
        tags=(model_name, "calculator", "supervised") + wandb_tag,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model_class = transformers.AutoModelForSeq2SeqLM
    model_class = gadgets.model.gadget_assisted_model(model_class)
    model = model_class.from_pretrained(**from_pretrained_args)

    if "<" not in tokenizer.vocab:
        patch_missing_angle_bracket(tokenizer, model)

    model.prepare_for_generate(
        tokenizer,
        enabled_gadgets=[gadgets.gadget.Calculator()],
        default_max_tokens=512,
    )

    callbacks = []
    if path_peft_config is not None:
        peft_config_dict = read_config(path_peft_config)
        peft_config = peft.get_peft_config(peft_config_dict)
        model = peft.get_peft_model(model, peft_config)
        callbacks.append(gadgets.utils.SavePeftModelCallback())
    if path_early_stop_config is not None:
        early_stop_config_dict = read_config(path_early_stop_config)
        early_stop_config = transformers.EarlyStoppingCallback(**early_stop_config_dict)
        callbacks.append(early_stop_config)

    train_args_dict = read_config(path_train_args)
    if "output_root_dir" in train_args_dict:
        output_root_dir = pathlib.Path(train_args_dict.pop("output_root_dir"))
        train_args_dict["output_dir"] = str(output_root_dir / wandb.run.name)
    if path_deepspeed_config is not None:
        if "deepspeed" in train_args_dict:
            warnings.warn(
                "deepspeed config in train_args will be overwritten by path_deepspeed_config"
            )
        path_deepspeed_config["deepspeed"] = str(path_deepspeed_config)
    train_args = transformers.Seq2SeqTrainingArguments(**train_args_dict)

    if load_local_checkpoint and path_peft_config is not None:
        shutil.copytree(
            load_local_checkpoint,
            os.path.join(train_args.output_dir, "checkpoint-orig"),
            copy_function=os.link,
        )

    collator = transformers.DataCollatorForSeq2Seq(tokenizer, model)

    data_config_dict = read_config(path_data_config)
    data_config = DataConfig(**data_config_dict)
    list_train_datasets = []
    list_valid_datasets = []
    for ds_list, ds_config in [
        (list_train_datasets, data_config.train_datasets),
        (list_valid_datasets, data_config.valid_datasets),
    ]:
        assert isinstance(ds_config, DatasetConfig)
        print("preparing dataset", ds_config.name, "split", ds_config.split)
        preprocess = gadgets.prep.Preprocessing(
            tokenizer, ds_config.add_result_sentence, ds_config.prompt_prefix
        )
        ds = datasets.load_dataset(
            ds_config.name, split=ds_config.split, **ds_config.load_dataset_kwargs
        )
        ds = ds.map(preprocess)
        ds_list.append(ds)

    train_dataset: datasets.Dataset
    if data_config.rebalance_train_datasets:
        train_dataset = datasets.interleave_datasets(list_train_datasets, stopping_strategy="all_exhausted")
    else:
        train_dataset = datasets.concatenate_datasets(list_train_datasets)
        train_dataset.shuffle()

    # TODO HF trainer can now accept multiple eval datasets as a dict
    # we should try to use that instead of concatenating them
    # This could eliminate the ugly and error-prone dataset slicing in MyMetrics
    valid_dataset: datasets.Dataset = datasets.concatenate_datasets(list_valid_datasets)

    random_rng = np.random.default_rng(42)
    min_valid_dataset_length = min(len(ds) for ds in list_valid_datasets)
    # TODO treat the indices for each dataset independently
    # so that we can have different number of log predictions for each dataset
    # and don't have to log only 10 predictions from each dataset if
    # the smallest has only 10 examples. Not crucial though, so wasn't done yet.
    log_predictions_indices = random_rng.choice(
        min(min_valid_dataset_length, num_log_preds_per_dataset),
        size=num_log_preds_per_dataset,
        replace=False,
    )

    metrics = gadgets.metrics.MyMetrics(
        tokenizer=tokenizer,
        log_predictions=True,
        datasets_id_length={
            config.name: len(valid_ds)
            for config, valid_ds in zip(data_config.valid_datasets, list_valid_datasets)
        },
        log_predictions_indices=log_predictions_indices,
    )

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=metrics,
        callbacks=callbacks,
    )

    if path_deepspeed_config is not None:
        deepspeed_config_dict = read_config(path_deepspeed_config)

    tuned_params = sum(p.shape.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.shape.numel() for p in model.parameters())
    trained_perc = tuned_params / total_params * 100
    print(f"Number of trained parameters: {tuned_params}/{total_params} = {trained_perc:.2f}%")

    wandb.run.tags = (
        *wandb.run.tags,
        f"trained_params_num={tuned_params}",
        f"trained_params_perc={trained_perc:.2f}",
    )

    wandb.config.update(
        {
            "launching_command": " ".join(sys.argv),
            "trained_params_num": tuned_params,
            "trained_params_perc": trained_perc,
            "peft_config": peft_config_dict if path_peft_config is not None else {},
            "model_config": model_config_dict if path_model_config is not None else {},
            "deepseed_config": deepspeed_config_dict if path_deepspeed_config else {},
            "load_local_checkpoint": str(load_local_checkpoint) if load_local_checkpoint else None,
        }
    )

    trainer.train()
    tokenizer.save_pretrained(train_args.output_dir)


class DataConfig(pydantic.BaseModel):
    rebalance_train_datasets: bool
    train_datasets: list[DatasetConfig]
    valid_datasets: list[DatasetConfig]
    test_datasets: list[DatasetConfig]


class DatasetConfig(pydantic.BaseModel):
    name: str
    split: str
    load_dataset_kwargs: dict[str, Any]
    add_result_sentence: bool
    prompt_prefix: Optional[str]


def read_config(path: pathlib.Path) -> dict:
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)

    if path.suffix == ".yaml":
        with open(path) as f:
            return yaml.safe_load(f)

    raise ValueError(f"Unknown config file type: {path.suffix}")


def patch_missing_angle_bracket(tokenizer, model):
    gadgets.utils.add_new_token(
        "<",
        is_special=False,
        tokenizer=tokenizer,
        model=model,
        init_with=["[", ">"],
    )

    text = "<gadget>2+2</gadget>"
    encoded = tokenizer(text, return_tensors="pt").input_ids
    decoded = tokenizer.batch_decode(
        encoded, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    assert decoded[0] == text, decoded[0]


if __name__ == "__main__":
    app()
