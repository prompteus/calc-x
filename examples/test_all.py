import argparse
from typing import Optional

import numpy as np
import pandas as pd
import scipy

import gadgets

argparser = argparse.ArgumentParser()

argparser.add_argument("--input_jsonls", type=str)
argparser.add_argument("--prediction_column", type=str, default="prediction")
argparser.add_argument("--correct_column", type=str, default="result")
argparser.add_argument("--use_gadgets", type=bool, required=True, action=argparse.BooleanOptionalAction)
argparser.add_argument("--confidence_level", type=float, default=0.95)
argparser.add_argument("--per_num_steps", type=bool, default=False)

args = argparser.parse_args()


def report_results(is_correct, num_steps: Optional[int] = None) -> None:
    bootstrap = scipy.stats.bootstrap(is_correct.reshape(1, -1), np.mean,
                                      confidence_level=args.confidence_level, random_state=0)
    low, high = bootstrap.confidence_interval
    num_steps = str(num_steps) if num_steps is not None else "all"

    print(f"Steps {num_steps} Predictions have a correct final result in "
          f"{np.mean(is_correct)*100:.1f}±\small{{{((high-low)/2)*100:.1f}}}% of cases."
          f"{args.confidence_level * 100}% Confidence interval: [{low:.4%}, {high:.4}%].")


for input_jsonl in args.input_jsonls.split(","):
    print("Report for %s" % input_jsonl)

    df = pd.read_json(input_jsonl, lines=True)

    preds = df[args.prediction_column]
    trues = df[args.correct_column]

    num_steps = None
    if args.per_num_steps:
        num_steps = df["num_steps"]

    is_correct = []
    if args.use_gadgets:
        for pred, true in zip(preds, trues):
            pred_chain, pred_result = gadgets.markup.from_model_markup(pred)
            true_chain, true_result = gadgets.markup.from_model_markup(true)
            assert true_result is not None
            pred_result = "" if pred_result is None else pred_result
            # true_result = "" if true_result is None else true_result
            is_correct.append(gadgets.metrics.are_numeric_results_same(pred_result, true_result))
    else:
        for pred, true_result in zip(preds, trues):
            pred_result = gadgets.baseline_metrics.get_result_from_output(pred)
            pred_result = "" if pred_result is None else pred_result
            is_correct.append(gadgets.metrics.are_numeric_results_same(pred_result, true_result))

    is_correct = np.array(is_correct).astype(float)

    if not args.per_num_steps:
        report_results(is_correct)
    else:
        for current_num_steps in range(1, num_steps.max()):
            if sum(num_steps.values == current_num_steps) > 1:
                report_results(is_correct[(num_steps.values == current_num_steps)], current_num_steps)
