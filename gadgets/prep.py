import transformers
import gadgets.datatypes
import gadgets.markup


class Preprocessing:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        add_result_sentence: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.add_result_sentence = add_result_sentence

    def __call__(self, example: gadgets.datatypes.Example | dict) -> dict[str, list[int]]:
        if isinstance(example, dict):
            example = gadgets.datatypes.Example(**example)

        soup = gadgets.markup.to_model_markup(
            example=example,
            add_result_sentence=self.add_result_sentence,
        )

        inputs = self.tokenizer(example.prompt, truncation=True)
        labels = self.tokenizer(text_target=str(soup), truncation=True)
        
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels.input_ids,
            "chain": str(soup),
        }
