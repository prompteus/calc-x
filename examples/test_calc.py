import argparse

import numpy as np
import pandas as pd
import scipy

import gadgets

argparser = argparse.ArgumentParser()

argparser.add_argument("--input_jsonl", type=str)
argparser.add_argument("--prediction_column", type=str, default="prediction")
argparser.add_argument("--correct_column", type=str, default="answer")
argparser.add_argument("--use_gadgets", type=bool, required=True, action=argparse.BooleanOptionalAction)
argparser.add_argument("--confidence_level", type=float, default=0.95)

args = argparser.parse_args()

df = pd.read_json(args.input_jsonl, lines=True)

preds = df[args.prediction_column]
trues = df[args.correct_column]

is_correct = []
if args.use_gadgets:
    for pred, true in zip(preds, trues):
        pred_chain, pred_result = gadgets.markup.from_model_markup(pred)
        true_chain, true_result = gadgets.markup.from_model_markup(true)
        assert true_result is not None, true_chain
        pred_result = "" if pred_result is None else pred_result
        true_result = "" if true_result is None else true_result
        is_correct.append(gadgets.metrics.are_numeric_results_same(pred_result, true_result))
else:
    for pred, true in zip(preds, trues):
        pred_result = gadgets.baseline_metrics.get_result_from_output(pred)
        true_result = gadgets.baseline_metrics.get_result_from_output(true)
        pred_result = "" if pred_result is None else pred_result
        is_correct.append(gadgets.baseline_metrics.are_numeric_results_same(pred_result, true))

is_correct = np.array(is_correct).astype(float).reshape(1, -1)

bootstrap = scipy.stats.bootstrap(is_correct, np.mean, confidence_level=args.confidence_level, random_state=0)
low, high = bootstrap.confidence_interval
print(f"Predictions have a correct final result in {np.mean(is_correct) * 100:.4f}% of cases.")
print(f"{args.confidence_level * 100}% Confidence interval: [{low*100:.4f}, {high:.4f}]")
