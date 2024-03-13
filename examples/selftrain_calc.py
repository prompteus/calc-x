import os
import gc
import random
import enum
from typing import Optional

import datasets
import pandas as pd
import torch
import torchdata
import transformers
import typer
import wandb


import gadgets.selftrain
import gadgets.dpo_trainer

class Mode(str, enum.Enum):
    dpo = "dpo"
    sft = "sft"
    kto = "kto"

def main(
    mode: Mode = typer.Option(...),
    model_name: str = "MU-NLPC/calcformer-instruct-flan-xl_step-96k",
    wandb_entity: str = "transformersclub",
    wandb_project: str = "gadgets",
    wandb_group: Optional[str] = "selftrain",
    wandb_dir: str = ".wandb",
    checkpoint_dir: str = "checkpoints",
    prediction_log_folder: str = "selftrain_preds",
    max_output_length: int = 756,
    train_ds: str = "MU-NLPC/Calc-X",
    train_ds_split_name: str = "train",
    train_ds_subset: Optional[str] = "ape210k",
    limit_train_set_per_ds: int = -1,
    valid_ds: str = "MU-NLPC/Calc-X",
    valid_ds_split_name: str = "validation",
    valid_ds_subset: Optional[str] = None,
    limit_val_set_per_ds: int = 100,
    id_col: str = "id",
    prompt_col: str = "question",
    chain_col: str = "chain",
    result_col: str = "result",
    learning_rate: float = 2e-5,
    batch_size: int = 2,
    grad_accum: int = 16,
    prediction_batch_size: int = 8,
    num_predictions_per_example: int = 16,
    sample_least_successful_with_prob: float = 0.5,
    buffer_size: int = 4096,
    optim: str = "adafactor",
    save_total_limit: int = 3,
    eval_steps: int = 1000,
    save_steps: int = 1000,
    dpo_loss_type: str = "sigmoid",
    prefs_beta: float = 0.1,
    prefs_max_pairs_per_problem: Optional[int] = 32,
    prefs_target_min_pairs_per_problem: int = 32,
    prefs_max_oversample_accepted_per_problem: int = 4,
    sft_max_examples_per_problem: Optional[int] = 32,
    sft_target_min_examples_per_problem: int = 32,
    sft_max_oversample_per_problem: int = 4,
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
        tags=[model_name, "selftrain", str(mode)],
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

    generation_config = transformers.GenerationConfig(
        do_sample=True,
        topk=5,
    )

    ds_train: datasets.Dataset
    ds_train = datasets.load_dataset(train_ds, split=train_ds_split_name)
    if train_ds_subset is not None:
        ds_train = ds_train.filter(lambda x: x["source_ds"] == train_ds_subset)
    if limit_train_set_per_ds is not None and limit_train_set_per_ds > 0:
        df_train = ds_train.to_pandas()
        df_train = df_train.groupby("source_ds").sample(limit_train_set_per_ds, random_state=0)
        ds_train = datasets.Dataset.from_pandas(df_train, preserve_index=False)
    
    ds_valid: datasets.Dataset
    ds_valid = datasets.load_dataset(valid_ds, split=valid_ds_split_name)
    if valid_ds_subset is not None:
        ds_valid = ds_valid.filter(lambda x: x["source_ds"] == valid_ds_subset)
    if limit_val_set_per_ds is not None and limit_val_set_per_ds > 0:
        df_valid = ds_valid.to_pandas()
        df_valid = df_valid.groupby("source_ds").sample(limit_val_set_per_ds, random_state=0)
        ds_valid = datasets.Dataset.from_pandas(df_valid, preserve_index=False)

    match mode:
        case Mode.dpo:
            ds_valid = ds_valid.rename_columns({
                prompt_col: "prompt",
                chain_col: "chosen",
            })
            ds_valid = ds_valid.map(lambda x: {"rejected": ""})
        case Mode.sft:
            def preprocess(example):
                inputs = tokenizer(example[prompt_col], truncation=True)
                labels = tokenizer(text_target=example[chain_col], truncation=True, max_length=max_output_length)
                return {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "labels": labels.input_ids,
                }
            ds_valid = ds_valid.map(preprocess, batched=True)
        case Mode.kto:
            raise NotImplementedError("binary KTO is not implemented yet. For KTO on preference pairs, use DPO mode with dpo_loss_type='kto_pair'")

    experience_collector = gadgets.selftrain.ExperienceCollector(
        problem_ids=ds_train[id_col],
        prompts=ds_train[prompt_col],
        results=ds_train[result_col],
        num_preds_per_example=num_predictions_per_example,
        sample_least_successful_with_prob=sample_least_successful_with_prob,
        batch_size=prediction_batch_size,
        seed=0,
        generation_config=generation_config
    )

    success_tracker = gadgets.selftrain.ExperienceTracker(
        rolling_window_size=1024,
        report_after_every_n_problems=10,
        use_stdout=False,
        use_wandb=True,
        num_preds_per_problem=num_predictions_per_example,
    )

    os.makedirs(prediction_log_folder, exist_ok=True)
    experience_logger = gadgets.selftrain.ExperienceLogger(
        log_file=f"{prediction_log_folder}/{wandb.run.name}.jsonl",
        print_to_stdout=False,
    )

    metrics = gadgets.metrics.MonitorMetrics(
        tokenizer=tokenizer,
        log_predictions=True,
        eval_ds_inputs=None,
        source_ds_col=ds_valid["source_ds"],
        define_wandb_metrics=True,
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
        per_device_eval_batch_size=prediction_batch_size,
        eval_accumulation_steps=1,
        optim=optim,
        logging_steps=5,
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

    trainer: transformers.Seq2SeqTrainer
    pipe: torchdata.datapipes.iter.IterDataPipe
    pipe = torchdata.datapipes.iter.IterableWrapper(experience_collector, deepcopy=False)

    match mode:
        case Mode.dpo:
            num_pairs_tracker = gadgets.selftrain.NumPairsTracker(
                rolling_window_size=128,
                report_after_every_n_problems=1,
                use_stdout=False,
                use_wandb=True,
            )

            make_preferences = gadgets.selftrain.MakePreferencePairs(
                random_gen=random.Random(0),
                target_min_pairs=prefs_target_min_pairs_per_problem,
                max_oversample_accepted=prefs_max_oversample_accepted_per_problem,
                max_pairs=prefs_max_pairs_per_problem
            )

            # .batch().in_batch_shuffle().unbatch() is different from .shuffle(buffer_size=buffer_size)
            # because in .shuffle elements are sampled from the buffer
            # and can (with low probability) be there for a long time.
            # in constrast, .batch().in_batch_shuffle().unbatch() will
            # fill the buffer, shuffle it, and then yield all elements
            # from it before refilling it.
            pipe = (
                pipe
                .map(as_side_effect(success_tracker))
                .map(as_side_effect(experience_logger))
                .map(make_preferences)
                .map(as_side_effect(num_pairs_tracker))
                .flatmap()
                .shuffle(buffer_size=buffer_size)
                .map(gadgets.selftrain.DPOPreprocessor())
            )

            trainer = gadgets.dpo_trainer.DPOTrainer(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
                train_dataset=pipe,
                eval_dataset=ds_valid,
                compute_metrics=metrics,
                max_target_length=max_output_length,
                max_prompt_length=512,
                max_length=max_output_length + 512,
                loss_type=dpo_loss_type,
                beta=prefs_beta
            )
            
            metrics.set_eval_ds_inputs(trainer.eval_dataset["prompt_input_ids"])

        case Mode.sft:
            make_sft_examples = gadgets.selftrain.MakeSFTExamples(
                random_gen=random.Random(0),
                target_min_examples_per_problem=sft_target_min_examples_per_problem,
                max_examples_per_problem=sft_max_examples_per_problem,
                max_oversample=sft_max_oversample_per_problem,
            )

            pipe = (
                pipe
                .map(as_side_effect(success_tracker))
                .map(as_side_effect(experience_logger))
                .map(make_sft_examples)
                .flatmap()
                .shuffle(buffer_size=buffer_size)
                .map(gadgets.selftrain.SFTPreprocessor(tokenizer))
            )

            data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, model)

            trainer = transformers.Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=pipe,
                eval_dataset=ds_valid,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=metrics,
            )

            metrics.set_eval_ds_inputs(ds_valid["input_ids"])

        case Mode.kto:
            raise NotImplementedError("binary KTO is not implemented yet. For KTO on preference pairs, use DPO mode with dpo_loss_type='kto_pair'")

    experience_collector.set_trainer(trainer)

    trainer.train()



def as_side_effect(fn: callable) -> callable:
    "applies fn to the input and returns the argument unchanged"
    def wrapper(arg):
        fn(arg)
        return arg
    return wrapper


if __name__ == "__main__":
    typer.run(main)