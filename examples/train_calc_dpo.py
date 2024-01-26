from typing import Optional
import gc

import datasets
import torch
import transformers
import typer
import wandb
import rich.traceback

import gadgets


def main(
    model_name: str = "MU-NLPC/calcformer-instruct-flan-xl_step-128k",
    wandb_entity: str = "transformersclub",
    wandb_project: str = "gadgets",
    wandb_group: Optional[str] = "dpo",
    wandb_dir: str = ".wandb",
    checkpoint_dir: str = "checkpoints",
    train_ds: str = "MU-NLPC/Calc-ape210k_selftrain_experiment",
    train_ds_split_name: str = "train",
    valid_ds: str = "MU-NLPC/Calc-X",
    valid_ds_subset: Optional[str] = "ape210k",
    limit_val_set_per_ds: int = 200,
    prompt_col: str = "question",
    chosen_col: str = "correct_1",
    rejected_col: str = "incorrect_1",
    max_output_length: int = 756,
    batch_size: int = 1,
    grad_accum: int = 32,
    optim="adafactor",
    save_total_limit: int = 10,
    eval_steps: int = 2000,
    save_steps: int = 2000,
    early_stopping_patience: int = 3,
    early_stopping_threshold: float = 0.03,
    learning_rate: float = 2e-5,
) -> None:
    cli_params = locals()

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cpu")
    model_class = gadgets.model.gadget_assisted_model(model.__class__)
    del model
    gc.collect()
    model = model_class.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    assert isinstance(model, gadgets.model.GadgetAssist)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)

    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        tags=[model_name, "dpo"],
        group=wandb_group,
        dir=wandb_dir,
    )

    wandb.config.update({"cli_params": cli_params})

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
        default_max_tokens=max_output_length,
    )

    early_stopping = transformers.EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold
    )

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=f"{checkpoint_dir}/{wandb.run.name}",
        learning_rate=learning_rate,
        do_train=True,
        do_eval=True,
        warmup_steps=1000,
        max_steps=1_000_000,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=16,
        optim=optim,
        logging_steps=10,
        eval_steps=eval_steps,
        save_steps=save_steps,
        evaluation_strategy="steps",
        bf16=True,
        bf16_full_eval=True,
        predict_with_generate=True,
        generation_max_length=max_output_length,
        include_inputs_for_metrics=True,
        report_to="wandb",
        metric_for_best_model="avg_correct_results",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=save_total_limit,
    )

    ds_train: datasets.Dataset
    ds_train = datasets.load_dataset(train_ds, split=train_ds_split_name)
    ds_train = ds_train.rename_column(prompt_col, "prompt")
    ds_train = ds_train.rename_column(chosen_col, "chosen")
    if rejected_col is None or rejected_col == "":
        print("No rejected column specified, using empty string as rejected value")
        ds_train = ds_train.map(lambda x: {"rejected": ""})
    else:
        ds_train = ds_train.rename_column(rejected_col, "rejected")
    ds_train = ds_train.select_columns(["prompt", "chosen", "rejected"])

    ds_valid: datasets.Dataset
    ds_valid = datasets.load_dataset(valid_ds, split="validation")
    if valid_ds_subset is not None:
        ds_valid = ds_valid.filter(lambda x: x["source_ds"] == valid_ds_subset)
    if limit_val_set_per_ds is not None and limit_val_set_per_ds > 0:
        df_valid = ds_valid.to_pandas()
        df_valid = df_valid.groupby("source_ds").sample(limit_val_set_per_ds, random_state=0)
        ds_valid = datasets.Dataset.from_pandas(df_valid)
        ds_valid = ds_valid.rename_columns({
            "question": "prompt",
            "chain": "chosen",
        })
        ds_valid = ds_valid.map(lambda x: {"rejected": ""})


    metrics = gadgets.metrics.MonitorMetrics(
        tokenizer=tokenizer,
        log_predictions=True,
        eval_ds_inputs=None,
        source_ds_col=ds_valid["source_ds"],
    )

    trainer = gadgets.dpo_trainer.DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tokenizer,
        compute_metrics=metrics,
        callbacks=[early_stopping],
        max_target_length=max_output_length,
        max_prompt_length=512,
    )

    metrics.set_eval_ds_inputs(trainer.eval_dataset["prompt_input_ids"])

    trainer.train()



if __name__ == "__main__":
    typer.run(main)
