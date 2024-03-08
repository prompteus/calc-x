from __future__ import annotations

import math
import re
import string
import warnings
from typing import Dict, overload

import evaluate
import numpy as np
import pandas as pd
import transformers
import wandb

import gadgets.datatypes
import gadgets.gadget
import gadgets.markup


def normalize_option(option: str) -> str:
    """
    >>> normalize_option("  (A)  \n")
    'A'
    """
    return re.sub(r"(\s+|\(|\))", "", option)


def is_option_result(result: str) -> bool:
    """
    >>> is_option_result("  A)  \n")
    True
    >>> is_option_result("  23/7 ")
    False
    """
    return normalize_option(result) in list(string.ascii_letters)


@overload
def are_results_same(pred_result: str, true_result: str, rel_tol: float) -> bool:
    ...

@overload
def are_results_same(pred_result: list[str], true_result: list[str], rel_tol: float) -> list[bool]:
    ...

@overload
def are_results_same(pred_result: pd.Series, true_result: pd.Series, rel_tol: float) -> pd.Series:
    ...

@overload
def are_results_same(pred_result: np.ndarray, true_result: np.ndarray, rel_tol: float) -> np.ndarray:
    ...

def are_results_same(pred_result, true_result, rel_tol: float = 1e-2):
    if isinstance(pred_result, str):
        return scalar_are_results_same(pred_result, true_result, rel_tol)
    if isinstance(pred_result, list):
        return [scalar_are_results_same(pred, true, rel_tol) for pred, true in zip(pred_result, true_result)]
    if isinstance(pred_result, pd.Series):
        if not isinstance(true_result, pd.Series):
            raise ValueError("Expected `true_result` to be a pandas Series")
        return pd.Series(are_results_same(pred_result.tolist(), true_result.tolist(), rel_tol), index=pred_result.index)
    if isinstance(pred_result, np.ndarray):
        if not isinstance(true_result, np.ndarray):
            raise ValueError("Expected `true_result` to be a numpy array")
        return np.array(are_results_same(pred_result.tolist(), true_result.tolist(), rel_tol))
    raise ValueError("Expected `pred_result` to be a str, list, pandas Series or numpy array")


def scalar_are_results_same(pred_result: str, true_result: str, rel_tol: float) -> bool:
    pred_result = str(pred_result) if pred_result is not None else ""
    true_result = str(true_result) if true_result is not None else ""
    
    if is_option_result(true_result):
        # The task is to select correct option
        true_result = normalize_option(true_result)
        pred_result = normalize_option(pred_result)
        return pred_result == true_result
    
    # The task is to calculate the result as a number

    if pred_result.strip() == true_result.strip():
        return True

    calculator = gadgets.gadget.Calculator()
    try:
        pred_float = calculator._float_eval(pred_result)
        true_float = calculator._float_eval(true_result)
        return math.isclose(pred_float, true_float, rel_tol=rel_tol)
    except:
        pass

    return False


def get_num_gadgets_calls(chain: gadgets.datatypes.Chain) -> int:
    return sum(isinstance(step, gadgets.datatypes.Interaction) for step in chain)



@overload
def remove_padding(tokens: np.ndarray, pad: int) -> list[list[int]]:
    ...

@overload
def remove_padding(tokens: list[list[int]], pad: int) -> list[list[int]]:
    ...

@overload
def remove_padding(tokens: list[int], pad: int) -> list[int]:
    ...

def remove_padding(tokens, pad: int):
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()
    if len(tokens) == 0:
        return tokens
    if isinstance(tokens[0], list):
        return [remove_padding(token, pad) for token in tokens]
    return [token for token in tokens if token != pad]


class MonitorMetrics:

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        source_ds_col: list[str],
        eval_ds_inputs: list[list[int]],
        log_predictions: bool,
    ) -> None:
        self.sbleu = evaluate.load("sacrebleu")
        self.rouge = evaluate.load("rouge")
        self.tokenizer = tokenizer
        self.log_predictions = log_predictions
        self.source_ds_col = source_ds_col
        self.expected_input_tokens = None
        self.set_eval_ds_inputs(eval_ds_inputs)

    def set_eval_ds_inputs(self, eval_ds_inputs: list[list[int]]) -> None:
        if eval_ds_inputs is None:
            return
        self.expected_input_tokens = remove_padding(eval_ds_inputs, self.tokenizer.pad_token_id)
        if len(self.expected_input_tokens) != len(self.source_ds_col):
            raise ValueError("Length of eval_ds_inputs and source_ds_col must be equal")

    def __call__(self, eval_preds: transformers.EvalPrediction) -> Dict[str, float]:
        assert len(eval_preds.predictions) == len(self.source_ds_col), \
            f"Evaluation datasets have unexpected length. Check the `source_ds_col` passed to {self.__class__.__name__}"

        pad = self.tokenizer.pad_token_id

        preds = eval_preds.predictions
        trues = eval_preds.label_ids
        inputs = eval_preds.inputs

        for arr in [preds, trues, inputs]:
            arr[arr == -100] = pad

        input_tokens = remove_padding(inputs, pad)
        for expected_inputs, actual_inputs in zip(self.expected_input_tokens, input_tokens):
            if expected_inputs != actual_inputs:
                warnings.warn(f"Expected inputs: '{expected_inputs}', but got '{actual_inputs}', it is likely that compute_metrics recieved incorrect `eval_ds_inputs` and `source_ds_col`")

        df = pd.DataFrame({
            "source_ds": self.source_ds_col,
            "num_pred_tokens": (preds != pad).sum(axis=1),
            "preds": self.tokenizer.batch_decode(preds, skip_special_tokens=True, spaces_between_special_tokens=False),
            "trues": self.tokenizer.batch_decode(trues, skip_special_tokens=True, spaces_between_special_tokens=False),
            "inputs": self.tokenizer.batch_decode(inputs, skip_special_tokens=True, spaces_between_special_tokens=False),
        })

        df[["trues_chain", "trues_result"]] = df["trues"].apply(gadgets.markup.from_model_markup).apply(pd.Series)
        df[["preds_chain", "preds_result"]] = df["preds"].apply(gadgets.markup.from_model_markup).apply(pd.Series)
        df["is_correct"] = df["preds_result"].combine(df["trues_result"], are_results_same)
        df["num_gadget_calls_true"] = df["trues_chain"].apply(get_num_gadgets_calls)
        df["num_gadget_calls_pred"] = df["preds_chain"].apply(get_num_gadgets_calls)

        logged_dict: Dict[str, float] = {}

        for source_ds in df["source_ds"].unique():
            df_ds: pd.DataFrame
            df_ds = df[df["source_ds"] == source_ds]
        
            sbleu_score = self.sbleu.compute(predictions=df_ds["preds"].tolist(), references=df_ds["trues"].tolist())
            rouge_score = self.rouge.compute(predictions=df_ds["preds"].tolist(), references=df_ds["trues"].tolist())

            if self.log_predictions:
                table = wandb.Table(dataframe=df_ds[["inputs", "trues", "preds"]])
                wandb.log({f"{source_ds}__prediction_examples": table})

            new_log = {
                "rouge1": rouge_score["rouge1"],
                "rouge2": rouge_score["rouge2"],
                "rougeL": rouge_score["rougeL"],
                "rougeLsum": rouge_score["rougeLsum"],
                "sacrebleu": sbleu_score["score"],
                "num_tokens": df_ds["num_pred_tokens"].mean(),
                "num_gadget_calls_pred": df_ds["num_gadget_calls_pred"].mean(),
                "num_gadget_calls_true": df_ds["num_gadget_calls_true"].mean(),
                "correct_results": df_ds["is_correct"].mean(),
            }
            new_log = {f"{source_ds}__{orig_key}": orig_val for orig_key, orig_val in new_log.items()}
            logged_dict.update(new_log)
         
        logged_dict["avg_correct_results"] = np.mean([logged_dict[f"{source_ds}__correct_results"] for source_ds in df["source_ds"].unique()])
        
        return logged_dict
