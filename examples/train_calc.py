from __future__ import annotations

from typing import Optional

import datasets
import numpy as np
import transformers
import typer
import wandb

import gadgets


def main(
    use_instructions_train: bool = typer.Option(...),
    use_instructions_val: bool = typer.Option(...),
    model_name: str = "google/flan-t5-xl",
    limit_train_set_per_ds: int = -1,
    limit_val_set_per_ds: int = 100,
    wandb_entity: str = "transformersclub",
    wandb_project: str = "gadgets",
    wandb_group: Optional[str] = "instructions", # TODO
    wandb_dir: str = ".wandb",
    checkpoint_dir: str = "checkpoints",
    max_output_length: int = 1024,
    batch_size: int = 2,
    grad_accum: int = 16,
    save_total_limit: int = 10,
    eval_steps: int = 16000, # 4000 steps =~ 1 hour training, 1 hour eval, 8000 steps =~ 2 hour training, 1 hour eval
    save_steps: int = 16000,
) -> None:
    cli_params = locals()
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model_class = gadgets.model.gadget_assisted_model(model.__class__)
    model = model_class.from_pretrained(model_name)
    assert isinstance(model, gadgets.model.GadgetAssist)

    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        tags=[model_name, "supervised"],
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

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model)
    ds_train = datasets.load_dataset("MU-NLPC/Calc-X", split="train")
    ds_valid = datasets.load_dataset("MU-NLPC/Calc-X", split="validation")
    instructions_ds = datasets.load_dataset("MU-NLPC/Calc-X_instructions")
    
    random_generator = np.random.default_rng(0)

    def add_instruction(example):
        source_ds = example["source_ds"]
        template: str = random_generator.choice(
            instructions_ds[source_ds]["template"],
            p=instructions_ds[source_ds]["weight"],
        )
        return {"question": template.format(example["question"])}

    if limit_train_set_per_ds is not None and limit_train_set_per_ds > 0:
        df_train = ds_train.to_pandas()
        df_train = df_train.groupby("source_ds").sample(limit_train_set_per_ds, random_state=0)
        ds_train = datasets.Dataset.from_pandas(df_train)

    if limit_val_set_per_ds is not None and limit_val_set_per_ds > 0:
        df_valid = ds_valid.to_pandas()
        df_valid = df_valid.groupby("source_ds").sample(limit_val_set_per_ds, random_state=0)
        ds_valid = datasets.Dataset.from_pandas(df_valid)

    if use_instructions_train:
        ds_train = ds_train.map(add_instruction)
    if use_instructions_val:
        ds_valid = ds_valid.map(add_instruction)


    def preprocess(example):
        inputs = tokenizer(example["question"], truncation=True)
        labels = tokenizer(text_target=example["chain"], truncation=True, max_length=max_output_length)
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels.input_ids,
        }

    ds_train = ds_train.map(preprocess, batched=True)
    ds_valid = ds_valid.map(preprocess, batched=True)
    ds_train = ds_train.shuffle(seed=0)

    early_stopping = transformers.EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.01
    )

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=f"{checkpoint_dir}/{wandb.run.name}",
        learning_rate=5e-5,
        do_train=True,
        do_eval=True,
        warmup_steps=1000,
        max_steps=1_000_000,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=16,
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

    metrics = gadgets.metrics.MonitorMetrics(
        tokenizer=tokenizer,
        log_predictions=True,
        eval_ds_inputs=ds_valid["input_ids"],
        source_ds_col=ds_valid["source_ds"],
    )

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics,
        callbacks=[early_stopping],
    )

    trainer.train()


if __name__ == "__main__":
    typer.run(main)
