from typing import Optional

import transformers
from datasets import load_dataset
import gadgets


class GadgetsInserter:
    def __init__(self,
                 add_result_sentence: bool = True,
                 prompt_prefix: Optional[str] = None) -> None:

        self.add_result_sentence = add_result_sentence
        self.prompt_prefix = prompt_prefix

    def __call__(self, example: gadgets.datatypes.Example | dict) -> dict[str, str]:
        if isinstance(example, dict):
            example = gadgets.datatypes.Example(**example)

        soup_chain = gadgets.markup.to_model_markup(
            example=example,
            add_result_sentence=self.add_result_sentence,
        )

        if self.prompt_prefix is not None:
            prompt = f"{self.prompt_prefix}{example.prompt}"
        else:
            prompt = example.prompt

        return {"prompt": prompt, "chain": soup_chain}


class Encoding:
    """
    This is here only for completeness, but encoding is not included in the published datasets.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, example: dict[str, str]) -> dict[str, list[int]]:
        inputs = self.tokenizer(example["prompt"], truncation=True)
        labels = self.tokenizer(text_target=str(example["chain"]), truncation=True)

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels.input_ids,
            "chain": str(example["chain"]),
        }


preprocess = GadgetsInserter()


# PART: datasets curation
def parse_and_preprocess_aqua(example: dict[str, str]):
    example_with_gadgets = gadgets.aqua.parse(example)
    input_sample = preprocess(example_with_gadgets)
    return input_sample


def parse_and_preprocess_gsm(example: dict[str, str]):
    example = gadgets.gsm8k.parse(example)
    example = preprocess(example)
    return example


aqua = load_dataset("aqua_rat").map(parse_and_preprocess_aqua)
gsm8k = load_dataset("gsm8k", "main").map(parse_and_preprocess_gsm)

print("AQUA: %s" % aqua)
print("GSM8K: %s" % gsm8k)
print()
