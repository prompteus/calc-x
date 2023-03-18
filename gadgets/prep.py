from __future__ import annotations

from typing import Optional, Dict, List

import transformers

import gadgets.datatypes
import gadgets.markup


class Preprocessing:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        add_result_sentence: bool = True,
        prompt_prefix: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.add_result_sentence = add_result_sentence
        self.prompt_prefix = prompt_prefix

    def __call__(self, example: gadgets.datatypes.Example | Dict) -> Dict[str, List[int]]:
        if isinstance(example, dict):
            example = gadgets.datatypes.Example(**example)

        soup = gadgets.markup.to_model_markup(
            example=example,
            add_result_sentence=self.add_result_sentence,
        )

        if self.prompt_prefix is not None:
            prompt = f"{self.prompt_prefix}{example.prompt}"
        else:
            prompt = example.prompt

        inputs = self.tokenizer(prompt, truncation=True)
        labels = self.tokenizer(text_target=str(soup), truncation=True)
        
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels.input_ids,
            "chain": str(soup),
        }
