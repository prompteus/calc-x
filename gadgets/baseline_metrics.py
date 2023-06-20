from __future__ import annotations

import math
from typing import Dict, Iterable
import re

import evaluate
import numpy as np
import transformers

import gadgets.datatypes
import gadgets.gadget
import gadgets.markup
import wandb


def are_numeric_results_same(pred: str, true: str, abs_tol: float = 1e-5) -> bool:
    if pred.strip() == true.strip():
        return True

    calculator = gadgets.gadget.Calculator()
    try:
        pred_float = calculator._float_eval(pred)
        true_float = calculator._float_eval(true)
        return math.isclose(pred_float, true_float, abs_tol=abs_tol)
    except:
        pass

    return False

def get_result_from_output(output):
    res = re.findall('\. The final result is (.+?)\.', output)
    if(len(res)==0):
        return None
    return res[-1] #returning the last occurence if multiple exist

class MyBaselineMetrics:

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 datasets_id_length: Dict[str, int],
                 log_predictions: bool = False,
                 log_predictions_indices: Iterable[int] = None) -> None:

        self.sacrebleu = evaluate.load("sacrebleu")
        self.rouge = evaluate.load("rouge")
        self.tokenizer = tokenizer
        self.log_predictions = log_predictions
        self.datasets_id_length = datasets_id_length

        if log_predictions:
            self.log_predictions_indices = list(log_predictions_indices)
        else:
            self.log_predictions_indices = None

    def __call__(self, eval_preds: transformers.EvalPrediction) -> Dict[str, float]:
        assert len(eval_preds.predictions) == sum(self.datasets_id_length.values()), \
            "Evaluation datasets have unexpected length. Check the given `datasets_id_length` and `eval_dataset`"

        logged_dict: Dict[str, float] = {}

        offset = 0
        for dataset_id, dataset_len in self.datasets_id_length.items():
            preds = eval_preds.predictions[offset: offset + dataset_len]
            trues = eval_preds.label_ids[offset: offset + dataset_len]
            inputs = eval_preds.inputs[offset: offset + dataset_len]

            offset += dataset_len

            if isinstance(preds, tuple):
                preds = preds[0]

            preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
            trues = np.where(trues != -100, trues, self.tokenizer.pad_token_id)
            inputs = np.where(inputs != -100, inputs, self.tokenizer.pad_token_id)

            preds_str = self.tokenizer.batch_decode(preds, skip_special_tokens=True,
                                                    spaces_between_special_tokens=False)
            trues_str = self.tokenizer.batch_decode(trues, skip_special_tokens=True,
                                                    spaces_between_special_tokens=False)
            inputs_str = self.tokenizer.batch_decode(inputs, skip_special_tokens=True,
                                                     spaces_between_special_tokens=False)

            sacrebleu_score = self.sacrebleu.compute(predictions=preds_str, references=trues_str)
            rouge_scores = self.rouge.compute(predictions=preds_str, references=trues_str)

            pred_num_tokens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]

            correct_results: list[bool] = []
            # num_gadget_calls_pred: list[int] = []
            # num_gadget_calls_true: list[int] = []
            for pred, true in zip(preds_str, trues_str):
                pred_result = get_result_from_output(pred)
                true_result = get_result_from_output(true)
                # pred_chain, pred_result = gadgets.markup.from_model_markup(pred)
                # true_chain, true_result = gadgets.markup.from_model_markup(true)
                assert true_result is not None
                pred_result = "" if pred_result is None else pred_result
                true_result = "" if true_result is None else true_result
                correct_results.append(are_numeric_results_same(pred_result, true_result))
                # num_gadget_calls_true.append(
                    # sum(isinstance(step, gadgets.datatypes.Interaction) for step in true_chain)
                # )
                # num_gadget_calls_pred.append(
                    # sum(isinstance(step, gadgets.datatypes.Interaction) for step in pred_chain)
                # )

            if self.log_predictions:
                data = []
                for i in self.log_predictions_indices:
                    data.append([
                        inputs_str[i],
                        preds_str[i],
                        trues_str[i],
                    ])

                table = wandb.Table(
                    columns=["prompt", "prediction", "label"],
                    data=data,
                )

                wandb.log({"%s_prediction_examples" % dataset_id: table})

            new_log = {
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
                "rougeLsum": rouge_scores["rougeLsum"],
                "sacrebleu": sacrebleu_score["score"],
                "num_tokens": float(np.mean(pred_num_tokens)),
                # "num_gadget_calls_pred": np.mean(num_gadget_calls_pred),
                # "num_gadget_calls_true": np.mean(num_gadget_calls_true),
                "correct_results": np.mean(correct_results),
                # "correct_num_gadget_calls": np.mean(np.array(num_gadget_calls_pred) == np.array(num_gadget_calls_true)),
            }
            new_log = {"%s_%s" % (dataset_id, orig_key): orig_val for orig_key, orig_val in new_log.items()}

            logged_dict = {**new_log, **logged_dict}

         
        logged_dict["avg_correct_results"] = np.mean([logged_dict[f"{dataset_id}_correct_results"] for dataset_id in self.datasets_id_length.keys()])
        
        return logged_dict
