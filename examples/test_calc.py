import pathlib
from typing import Annotated

import numpy as np
import pandas as pd
import scipy.stats
import typer

import gadgets


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
) -> None:
    df = pd.read_json(input_jsonl, lines=True)
    
    if not isinstance(df[prediction_column].iloc[0], str):
        # previously, prediction script would output a list of predictions
        # instead of putting each prediction in a separate row
        df = df.explode(prediction_column).dropna(subset=[prediction_column]).reset_index(drop=True)

    true_results: pd.Series = df[correct_column]
    pred_outputs: pd.Series = df[prediction_column]
    pred_results = pred_outputs.apply(gadgets.markup.get_result_from_output)
    is_correct = gadgets.metrics.are_results_same(pred_results, true_results)
    is_correct = is_correct.to_numpy().astype(float)

    #print_info("OVERALL", is_correct, confidence_level)
    #print()

    if ds_column in df:
        print("PER DATASET:")
        print()
        for ds_name in df[ds_column].unique():
            print_info(ds_name, is_correct[df[ds_column] == ds_name], confidence_level)
            print()

        print()
        print("AVG OVER DATASETS:")
        df["is_correct"] = is_correct
        avg_correct = df.groupby(ds_column).agg({"is_correct": "mean"}).mean().item()
        print(f"{avg_correct:.3%}")
    
    else:
        print_info("", is_correct, confidence_level)


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


if __name__ == "__main__":
    typer.run(main)
