import pathlib
import re
from typing import Annotated

import fuzzywuzzy.process
import numpy as np
import pandas as pd
import scipy

import gadgets
import typer


def print_info(
    name: str,
    is_correct: np.ndarray,
    confidence_level: float,
    seed: int = 0,
) -> None:
    if is_correct.ndim != 1:
        raise ValueError("is_correct should be 1D array")
    bootstrap = scipy.stats.bootstrap(is_correct.reshape(1, -1), np.mean, confidence_level=confidence_level, random_state=seed)
    low, high = bootstrap.confidence_interval
    mean = is_correct.mean()
    radius = (high-low) / 2

    print(name)
    print(f"  Number of predictions: {len(is_correct)}")
    print(f"  predictions have a correct final result in {mean:.1%} ± {radius*100:.1f} of cases. latex: {mean*100:.1f}±\small{{{((high-low)/2)*100:.1f}}}")
    print(f"  {confidence_level:.1%} Confidence interval: [{low:.3%}, {high:.3%}]")



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


def main(
    input_jsonl: Annotated[
        pathlib.Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    prediction_column: str = "prediction",
    correct_column: str = "result",
    confidence_level: float = 0.95,
    ds_column: str = "source_ds",
    options_column: str = "options",
    ds_subset: str = "aqua_rat"
) -> None:
    df = pd.read_json(input_jsonl, lines=True)
    if ds_column in df:
        df = df[df[ds_column] == ds_subset]
    
    if not isinstance(df[prediction_column].iloc[0], str):
        # previously, prediction script would output a list of predictions
        # instead of putting each prediction in a separate row
        df = df.explode(prediction_column).dropna(subset=[prediction_column]).reset_index(drop=True)

    pred_outputs: pd.Series = df[prediction_column]
    pred_results = pred_outputs.apply(gadgets.markup.get_result_from_output)
    is_correct = []
    
    for correct_option, options, pred_result in zip(df[correct_column], df[options_column], pred_results):
        true_result = options[correct_option]

        if pred_result is None:
            is_correct.append(False)
            continue

        if pred_result == "":
            pred_result = "none"

        if pred_result.strip() == true_result.strip():
            is_correct.append(True)
            continue

        if gadgets.metrics.are_results_same(pred_result, extract_number_from_option(true_result)):
            is_correct.append(True)
            continue

        # find the closest option to the prediction
        _, _, best_match = fuzzywuzzy.process.extractOne(pred_result, options)
        is_correct.append(best_match == correct_option)

    is_correct = np.array(is_correct).astype(float)

    name = "OVERALL" if ds_column not in df.columns else ds_subset
    print_info(name, is_correct, confidence_level)



if __name__ == "__main__":
    typer.run(main)

