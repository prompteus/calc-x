from __future__ import annotations

import warnings
import torch
import transformers
import bs4
from typing import Any, Optional, Callable, List, Union
import unittest.mock

from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList

from gadgets.gadget import Gadget, Calculator
from gadgets.markup import GADGET_TAG, OUTPUT_TAG, RESULT_TAG


class StopAfterGadgetCall(transformers.generation.StoppingCriteria):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        self.closing_tag_ids = self.tokenizer(
            "</" + GADGET_TAG + ">",
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.flatten()

    def __call__(self, seq_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if seq_ids.shape[-1] < self.closing_tag_ids.shape[-1]:
            return False

        # check if </gadget> is at the end of the sequence
        self.closing_tag_ids = self.closing_tag_ids.to(seq_ids.device)
        ending = seq_ids[..., -self.closing_tag_ids.shape[-1]:]
        ends_with_gadget_call = torch.all(ending == self.closing_tag_ids)
        return ends_with_gadget_call


class GadgetAssist(transformers.GenerationMixin):
    """
    Mixin that overrides model.generate to support the
    model with external gadgets.
    """

    def prepare_for_generate(
            self,
            tokenizer: transformers.PreTrainedTokenizer,
            enabled_gadgets: list[Gadget],
            default_max_tokens: int = 1000,
    ) -> None:
        self.tokenizer = tokenizer
        self.enabled_gadgets = enabled_gadgets
        self.default_max_tokens = default_max_tokens

    @torch.no_grad()
    def generate(
            self,
            input_ids: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            streamer: Optional["BaseStreamer"] = None,
            **kwargs,
            # signature of GenerationMixin.generate() in Transformers==4.28.1, with inputs<=>input_ids
    ) -> torch.LongTensor:
        """
        Model is expected to generate gadget tags.
        Whenever a gadget tag is generated, the gadget is called, 
        and the output is fed back into the model inside of an output tag.

        Final result is expected to be in result tag.

        Currently the function only supports single input (no batch).

        Returns:
            full_output: Full structured output of the model, including gadget, output, and result tags.
            result: Final result of the model, or None if not found.
        """

        stopping_criteria = transformers.generation.StoppingCriteriaList([StopAfterGadgetCall(self.tokenizer)])

        if kwargs is None:
            kwargs = {}

        if isinstance(input_ids, str):
            input_ids = self.tokenizer(input_ids, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)

        running_gadgets: dict[str, Gadget] = {
            g.gadget_id(): g for g in self.enabled_gadgets
        }

        max_tokens = None
        min_tokens = None

        if "max_length" in kwargs:
            max_length = kwargs.pop("max_length")
            if max_length is not None:
                max_tokens = max_length - input_ids.shape[-1]
        if "min_length" in kwargs:
            min_length = kwargs.pop("min_length")
            if min_length is not None:
                min_tokens = min_length - input_ids.shape[-1]
        if "max_new_tokens" in kwargs:
            max_tokens = kwargs.pop("max_new_tokens")
        if "min_new_tokens" in kwargs:
            min_tokens = kwargs.pop("min_new_tokens")

        if max_tokens is None:
            max_tokens = self.default_max_tokens

        last_num_total_tokens: int | None = None
        total_output_str: str = ""
        result_str = None

        while True:
            total_output_encoded = self.tokenizer(
                text_target=total_output_str,
                add_special_tokens=False,
                return_tensors="pt"
            ).input_ids.to(self.device).to(torch.long)

            num_total_tokens = total_output_encoded.shape[-1]
            if last_num_total_tokens is not None and num_total_tokens <= last_num_total_tokens:
                break
            last_num_total_tokens = num_total_tokens

            if num_total_tokens + 2 >= max_tokens:
                break

            if max_tokens is not None:
                kwargs["max_new_tokens"] = max_tokens - num_total_tokens
            if min_tokens is not None:
                kwargs["min_new_tokens"] = max(0, min_tokens - num_total_tokens)

            decoder_input_ids = torch.cat([
                torch.tensor(
                    self.config.decoder_start_token_id,
                    dtype=torch.long
                ).to(self.device).reshape(1, 1),
                total_output_encoded
            ], dim=-1)

            model_output: transformers.utils.ModelOutput
            model_output = super().generate(
                input_ids=input_ids,
                stopping_criteria=stopping_criteria,
                decoder_input_ids=decoder_input_ids,
                **kwargs,
            )[0]  # TODO This does not work in batch mode
            # which occurs in evaluation during training

            # model.generate() outputs starts with decoder_input_ids
            total_output_str = self.tokenizer.decode(
                model_output,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )

            try:
                doc = bs4.BeautifulSoup(total_output_str, features="html.parser")
            except Exception as e:
                warnings.warn(f"Failed to parse model output: {e}")
                continue

            gadget_tags: list[bs4.Tag] = doc.find_all(GADGET_TAG)
            evaluated_something = False
            for gadget_tag_input in gadget_tags:
                next_el = gadget_tag_input.next_sibling
                while next_el is not None and isinstance(next_el, bs4.NavigableString) and next_el.strip() == "":
                    # skip whitespace
                    next_el = next_el.next_sibling
                if isinstance(next_el, bs4.Tag) and next_el.name == OUTPUT_TAG:
                    # already evaluated this gadget tag
                    continue
                evaluated_something = True
                gadget_input = gadget_tag_input.get_text()
                gadget_id = gadget_tag_input.get("id", None)
                gadget = running_gadgets.get(gadget_id, None)
                if gadget is None:
                    gadget_output = f"ERROR: Gadget '{gadget_id}' not found"
                else:
                    gadget_output = gadget(gadget_input)

                gadget_tag_output = doc.new_tag(OUTPUT_TAG)
                gadget_tag_output.string = gadget_output
                gadget_tag_input.insert_after(gadget_tag_output)
                gadget_tag_input.insert_after("\n")
                gadget_tag_output.insert_after("\n")

            if evaluated_something:
                # replace total_output_str with the evaluated version
                total_output_str = str(doc)

            width = 80
            print(" PARTIAL MODEL OUTPUT ".center(width, "="))
            print(total_output_str)
            print("=" * width)

            output_tensor = self.tokenizer.encode(total_output_str,
                                                  return_tensors="pt",
                                                  add_special_tokens=True).to(self.device)

            # Commented things violate the generate() interface and may cause trouble in future versions:

            # if doc.find(RESULT_TAG) is not None:
            #     result_str = doc.find_all(RESULT_TAG)[-1].get_text()
            #     result_tensor = self.tokenizer(result_str, return_tensors="pt", add_special_tokens=False).input_ids

        # if return_as_str:
        #     if return_result:
        #         return total_output_str, result_str
        #     return total_output_str
        #
        # if return_result:
        #     return output_tensor, result_tensor

        return output_tensor


def gadget_assisted_model(model_class: type[transformers.PreTrainedModel]):
    class GadgetAssistedModel(GadgetAssist, model_class):
        pass

    return GadgetAssistedModel


str_prompt = "Write xml tag gadget id attribute id='calculator' and fill '2 + 2' inside. "
str_let_me_think = "Let me think about it"
str_gadget_usage = f"<{GADGET_TAG} id='calculator'>2+2</{GADGET_TAG}>"
str_gadget_output = f"<{OUTPUT_TAG}>4</{OUTPUT_TAG}>"
str_result = "129818"
str_result_with_tag = f"Final answer is <{RESULT_TAG}>{str_result}</{RESULT_TAG}>."


def test_generate_check_outputs(
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        mocked_model_outputs: list[str],
        expected_full_outputs: list[str],
        expected_result: str | None,
        enabled_gadgets: list[Gadget],
) -> bool:
    assert isinstance(model, GadgetAssist)

    model.prepare_for_generate(
        tokenizer,
        enabled_gadgets=enabled_gadgets,
    )

    mocked_model_outputs_tokenized = [
        tokenizer(string, return_tensors="pt").input_ids
        for string in mocked_model_outputs
    ]

    with unittest.mock.patch("transformers.GenerationMixin.generate") as patched_model:
        patched_model.side_effect = mocked_model_outputs_tokenized
        full_output, result = model.generate(
            str_prompt,
            return_result=True,
            return_as_str=True,
            max_length=400,
            num_beams=3,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )

    expected_full_output = bs4.BeautifulSoup(" ".join(expected_full_outputs), features="html.parser").prettify()
    full_output = bs4.BeautifulSoup(full_output, features="html.parser").prettify()

    output_matches = _compare_strings_ignore_whitespace(full_output, expected_full_output)

    if expected_result is None:
        result_matches = result is None
    else:
        result_matches = _compare_strings_ignore_whitespace(result, expected_result)

    is_correct = output_matches and result_matches
    return is_correct


def _compare_strings_ignore_whitespace(str1: str, str2: str) -> bool:
    return " ".join(str1.split()) == " ".join(str2.split())


TESTS = [
    {
        "mocked": [str_result_with_tag],
        "expected_outputs": [str_result_with_tag],
        "expected_result": str_result,
    },
    {
        "mocked": [str_let_me_think, str_result_with_tag],
        "expected_outputs": [str_let_me_think, str_result_with_tag],
        "expected_result": str_result,
    },
    {
        "mocked": [str_gadget_usage, str_result_with_tag],
        "expected_outputs": [str_gadget_usage, str_gadget_output, str_result_with_tag],
        "expected_result": str_result,
    },
    {
        "mocked": [str_gadget_usage, str_gadget_usage, str_result_with_tag],
        "expected_outputs": [str_gadget_usage, str_gadget_output, str_gadget_usage, str_gadget_output,
                             str_result_with_tag],
        "expected_result": str_result,
    },
    {
        "mocked": [str_gadget_usage + str_gadget_usage, str_result_with_tag],
        "expected_outputs": [str_gadget_usage + str_gadget_output + str_gadget_usage + str_gadget_output,
                             str_result_with_tag],
        "expected_result": str_result,
    }
]


def test_generate_with_gadgets():
    model_name = "salesforce/codet5-small"
    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
    model = gadget_assisted_model(transformers.T5ForConditionalGeneration).from_pretrained(model_name)

    for i, test in enumerate(TESTS):
        assert test_generate_check_outputs(
            model,
            tokenizer,
            test["mocked"],
            test["expected_outputs"],
            test["expected_result"],
            enabled_gadgets=[Calculator()],
        )


if __name__ == "__main__":
    test_generate_with_gadgets()
