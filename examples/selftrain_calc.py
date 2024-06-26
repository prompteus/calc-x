import traceback
import os
import gc
import random
import enum
import tempfile
from typing import Optional

import datasets
import pandas as pd
import torch
import torchdata
import transformers
import typer
import wandb
import skops.hub_utils
import skops.io


import gadgets.selftrain
import gadgets.dpo_trainer

class Mode(str, enum.Enum):
    dpo = "dpo"
    sft = "sft"
    kto = "kto"

def main(
    mode: Mode = typer.Option(...),
    model_name: str = "MU-NLPC/calcformer-instruct-flan-xl_step-128k",
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
    limit_val_set_per_ds: int = 200,
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
    buffer_size: int = 8192,
    optim: str = "adafactor",
    save_total_limit: int = 6,
    eval_steps: int = 200,
    save_steps: int = 200,
    dpo_loss_type: str = "sigmoid",
    prefs_beta: float = 0.1,
    prefs_max_pairs_per_problem: Optional[int] = 32,
    prefs_target_min_pairs_per_problem: int = 32,
    prefs_max_oversample_accepted_per_problem: int = 4,
    sft_max_examples_per_problem: Optional[int] = 32,
    sft_target_min_examples_per_problem: int = 32,
    sft_max_oversample_per_problem: int = 4,
    validate_at_start: bool = True,
    prefill_buffer: Optional[str] = None,
    prefill_buffer_limit: Optional[int] = None,
    prefill_buffer_do_yield: bool = False,
    tracker_rolling_window_size: int = 1024,
    experience_generation_top_k: int = 50,
    metric_for_best_model: str = "ape210k__correct_results",
    style_classifier_checkpoint: Optional[str] = "MU-NLPC/calcformer-style-classifier",
    style_classifier_margin: float = 0.5,
    style_score_threshold: float = 0.5,
    prefer_good_style: bool = False,
    resume_from_checkpoint: bool = False,
    wandb_allow_val_change: bool = False,
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
        allow_val_change=wandb_allow_val_change, 
    )

    wandb.config.update({"cli_params": cli_params}, allow_val_change=wandb_allow_val_change)

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
        top_k=experience_generation_top_k,
    )

    ds_train: datasets.Dataset
    ds_train = datasets.load_dataset(train_ds, split=train_ds_split_name)
    if train_ds_subset is not None and train_ds_subset != "":
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

    if prefill_buffer is not None:
        if prefill_buffer_limit is not None:
            if prefill_buffer_limit % num_predictions_per_example != 0:
                raise ValueError("prefill_buffer_limit must be a multiple of num_predictions_per_example")
        try:
            prefill = datasets.load_dataset(prefill_buffer, split="train")
            if prefill_buffer_limit is not None:
                prefill = prefill.select(range(prefill_buffer_limit))
            prefill = [gadgets.selftrain.Experience(**x) for x in prefill.to_list()]
        except FileNotFoundError:
            prefill = pd.read_json(prefill_buffer, lines=True)
            if prefill_buffer_limit is not None:
                prefill = prefill.head(prefill_buffer_limit)
            prefill = [gadgets.selftrain.Experience(**x) for x in prefill.to_dict(orient="records")]
    else:
        prefill = None

    if style_classifier_checkpoint is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            skops.hub_utils.download(
                repo_id=style_classifier_checkpoint,
                dst=tmp_dir,
            )
            style_classifier = skops.io.load(f"{tmp_dir}/model.skops")
    else:
        style_classifier = None
        
    experience_collector = gadgets.selftrain.ExperienceCollector(
        problem_ids=ds_train[id_col],
        prompts=ds_train[prompt_col],
        results=ds_train[result_col],
        num_preds_per_example=num_predictions_per_example,
        sample_least_successful_with_prob=sample_least_successful_with_prob,
        batch_size=prediction_batch_size,
        seed=0,
        generation_config=generation_config,
        prefill=prefill,
        prefill_buffer_do_yield=prefill_buffer_do_yield,
        style_classifier=style_classifier,
    )

    experience_tracker = gadgets.selftrain.ExperienceTracker(
        rolling_window_size=tracker_rolling_window_size,
        report_after_every_n_problems=1,
        use_stdout=False,
        use_wandb=True,
        num_preds_per_problem=num_predictions_per_example,
        style_score_printing_threshold=style_score_threshold,
        style_score_margin=style_classifier_margin,
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
        resume_from_checkpoint=resume_from_checkpoint,
        output_dir=f"{checkpoint_dir}/{wandb.run.name.split('--')[0]}",
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
        metric_for_best_model=metric_for_best_model,
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
                rolling_window_size=tracker_rolling_window_size,
                report_after_every_n_problems=1,
                use_stdout=False,
                use_wandb=True,
            )

            make_preferences = gadgets.selftrain.MakePreferencePairs(
                random_gen=random.Random(0),
                target_min_pairs=prefs_target_min_pairs_per_problem,
                max_oversample_accepted=prefs_max_oversample_accepted_per_problem,
                max_pairs=prefs_max_pairs_per_problem,
                prefer_good_style=prefer_good_style,
                style_score_margin=style_classifier_margin,
            )

            # .batch().in_batch_shuffle().unbatch() is different from .shuffle(buffer_size=buffer_size)
            # because in .shuffle elements are sampled from the buffer
            # and can (with low probability) be there for a long time.
            # in constrast, .batch().in_batch_shuffle().unbatch() will
            # fill the buffer, shuffle it, and then yield all elements
            # from it before refilling it.
            pipe = (
                pipe
                .map(as_side_effect(experience_tracker))
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
                prefer_good_style=prefer_good_style,
                style_score_threshold=style_score_threshold,
            )

            pipe = (
                pipe
                .map(as_side_effect(experience_tracker))
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

    if validate_at_start:
        trainer.evaluate()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)



def as_side_effect(fn: callable) -> callable:
    "applies fn to the input and returns the argument unchanged"
    def wrapper(arg):
        fn(arg)
        return arg
    return wrapper


if __name__ == "__main__":
    try:
        typer.run(main)
    except BaseException as e:
        print(traceback.format_exc())
        raise e
