import argparse
import re

import fuzzywuzzy.process
import numpy as np
import pandas as pd
import scipy

import gadgets

argparser = argparse.ArgumentParser()

argparser.add_argument("--input_jsonl", type=str)
argparser.add_argument("--use_gadgets", type=bool, required=True, action=argparse.BooleanOptionalAction)
argparser.add_argument("--confidence_level", type=float, default=0.95)

args = argparser.parse_args()

df = pd.read_json(args.input_jsonl, lines=True)

is_correct = []

number_re = re.compile(r"[-+]?\d+[,.]?\d*(\s?[\/:]\s?\d+\.?\d*)*")

def extract_number_from_option(string: str) -> str:
    string = string.strip()
    if string == "":
        return None
    
    string = (string
        .replace(",", "")
        .replace("+ ", "+")
        .replace("−", "-")
        .replace("- ", "-")
    )
    match = number_re.search(string)
    if not match:
        return None
    
    string = match.group()
    string = "".join(string.split())
    string = string.replace(":", "/")
    return string

for pred, correct_option, options in zip(df["prediction"], df["result"], df["options"]):
    true_result = options[correct_option]
    if args.use_gadgets:
        _, pred_result = gadgets.markup.from_model_markup(pred)
    else:
        pred_result = gadgets.baseline_metrics.get_result_from_output(pred)

    if pred_result is None:
        is_correct.append(False)
        continue

    if pred_result == "":
        pred_result = "none"

    if pred_result.strip() == true_result.strip():
        is_correct.append(True)
        continue

    if gadgets.metrics.are_numeric_results_same(pred_result,
                                                next(val for opt, val in options.items() if true_result in val)):
        is_correct.append(True)
        continue

    # find the closest option to the prediction
    _, _, best_match = fuzzywuzzy.process.extractOne(pred_result, options)
    is_correct.append(best_match == correct_option)
    

is_correct = np.array(is_correct).astype(float).reshape(1, -1)

bootstrap = scipy.stats.bootstrap(is_correct, np.mean, confidence_level=args.confidence_level, random_state=0)
low, high = bootstrap.confidence_interval
print(f"Predictions have a correct final result in {np.mean(is_correct)*100:.1f}±\small{{{((high-low)/2)*100:.1f}}}% of cases.")
print(f"{args.confidence_level * 100}% Confidence interval: [{low:.4%}, {high:.4%}], i.e. ")
