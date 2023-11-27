import numpy as np
import pandas as pd
import scipy.stats
import typer

import gadgets


def main(
    input_jsonl: str = typer.Argument(...),
    prediction_column: str = "prediction",
    correct_column: str = "result",
    confidence_level: float = 0.95,
) -> None:
    df = pd.read_json(input_jsonl, lines=True)

    true_results: pd.Series = df[correct_column]
    pred_outputs: pd.Series = df[prediction_column]
    pred_results = pred_outputs.apply(gadgets.markup.get_result_from_output)
    is_correct = pred_results.combine(true_results, gadgets.metrics.are_results_same)
    is_correct = is_correct.to_numpy().astype(float).reshape(1, -1)

    bootstrap = scipy.stats.bootstrap(is_correct, np.mean, confidence_level=confidence_level, random_state=0)
    low, high = bootstrap.confidence_interval
    mean = is_correct.mean()
    radius = (high-low) / 2

    print(f"Predictions have a correct final result in {mean:.1%} ± {radius*100:.1f} of cases.")
    print(f"{confidence_level:.1%} Confidence interval: [{low:.3%}, {high:.3%}]")


if __name__ == "__main__":
    typer.run(main)
