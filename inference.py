from __future__ import annotations

import warnings
import torch
import transformers
from bs4 import BeautifulSoup
from typing import Any
from gadget import Gadget, Calculator
import unittest.mock


def generate_with_gadgets(
    model: transformers.T5ForConditionalGeneration,
    tokenizer: transformers.PreTrainedTokenizer,
    prompt: str,
    enabled_gadgets: list[type[Gadget]],
    generate_kwargs: dict[str, Any] | None = None,
    max_total_tokens: int = 1000,
) -> tuple[str, str | None]:
    """
    Generates text with gadgets.
    
    Model is expected to generate <gadget id=""></gadgets> tags.
    Whenever a gadget tag and EOS is generated, the gadget is called, 
    and the output is fed back into the model inside <output></output> tag.

    Final answer is expected to be in <answer></answer> tag.

    Args:
        model: Model to use for generation.
        tokenizer: Tokenizer to use for generation.
        prompt: Prompt to use for generation.
        enabled_gadgets: List of enabled gadget classes.
        generate_kwargs: Keyword arguments to pass to model.generate()

    Returns:
        full_output: Full structured output of the model, including gadget, output, and answer tags.
        final_answer: Final answer of the model, or None if not found.
    """

    if generate_kwargs is None:
        generate_kwargs = {}

    prompt_inputs = tokenizer(prompt, return_tensors="pt").input_ids
    gadgets: dict[str, Gadget] = {
        gadget_type.gadget_id(): gadget_type() for gadget_type in enabled_gadgets
    }

    total_outputs: list[torch.Tensor] = []

    for gadget in gadgets.values():
        gadget.setup()

    while True:
        num_total_tokens = sum(map(len, total_outputs))

        if num_total_tokens >= max_total_tokens:
            final_answer = None
            break

        if len(total_outputs) == 0:
            model_output: transformers.utils.ModelOutput = model.generate(
                input_ids=prompt_inputs,
                max_new_tokens=max_total_tokens - num_total_tokens,
                **generate_kwargs,
            )[0]
        else:
            model_output: transformers.utils.ModelOutput = model.generate(
                input_ids=prompt_inputs,
                decoder_input_ids=torch.cat(total_outputs),
                max_new_tokens=max_total_tokens - num_total_tokens,
                **generate_kwargs
            )[0]

        model_output_str = tokenizer.decode(model_output, skip_special_tokens=True)

        try:
            doc = BeautifulSoup(model_output_str, features="html.parser")
        except Exception as e:
            total_outputs.append(model_output)
            warnings.warn(f"Failed to parse model output: {e}")
            continue
        
        gadget_tags = doc.find_all("gadget")
        for gadget_tag_input in gadget_tags:
            gadget_input = gadget_tag_input.get_text()
            try: 
                gadget_id = gadget_tag_input["id"]
                gadget = gadgets[gadget_id]
                gadget_output = gadget(gadget_input)
            except KeyError:
                gadget_output = f"ERROR: Gadget '{gadget_id}' not found"

            gadget_tag_output = doc.new_tag("output")
            gadget_tag_output.string = gadget_output
            gadget_tag_input.insert_after(gadget_tag_output)

        replaced_output_str = doc.prettify()
        replaced_output = tokenizer(replaced_output_str + "\n", return_tensors="pt").input_ids[0]
        total_outputs.append(replaced_output)
            
        if doc.find("answer") is not None:
            final_answer = doc.find_all("answer")[-1].get_text()
            break

    total_outputs_str = tokenizer.decode(torch.cat(total_outputs), skip_special_tokens=True)
    return total_outputs_str, final_answer



str_prompt = "Write xml tag gadget id attribute id='calculator' and fill '2 + 2' inside. "
str_let_me_think = "Let me think about it"
str_gadget_usage = "<gadget id='calculator'>2+2</gadget>"
str_gadget_output = "<output>4</output>"
str_final_answer = "129818"
str_final_with_tag = f"Final answer is <answer>{str_final_answer}</answer>."


def test_generate_check_outputs(
    model: transformers.T5ForConditionalGeneration,
    tokenizer: transformers.PreTrainedTokenizer,
    mocked_model_outputs: list[str],
    expected_full_outputs: list[str],
    expected_final_answer: str | None,
    enabled_gadgets: list[type[Gadget]],
) -> bool:

    mocked_model_outputs_tokenized = [
        tokenizer(string, return_tensors="pt").input_ids
        for string in mocked_model_outputs
    ]

    with unittest.mock.patch.object(model, "generate") as patched_model:
        patched_model.side_effect = mocked_model_outputs_tokenized
        full_output, final_answer = generate_with_gadgets(
            model,
            tokenizer,
            str_prompt,
            enabled_gadgets=enabled_gadgets,
            generate_kwargs=dict(
                num_beams=3,
                num_return_sequences=1,
                no_repeat_ngram_size=1,
                remove_invalid_values=True,
            ),
        )

    expected_full_output = BeautifulSoup(" ".join(expected_full_outputs), features="html.parser").prettify()
    full_output = BeautifulSoup(full_output, features="html.parser").prettify()

    output_matches = _compare_strings_ignore_whitespace(full_output, expected_full_output)

    if expected_final_answer is None:
        answer_matches = final_answer is None
    else:
        answer_matches = _compare_strings_ignore_whitespace(final_answer, expected_final_answer)

    is_correct = output_matches and answer_matches
    return is_correct


def _compare_strings_ignore_whitespace(str1: str, str2: str) -> bool:
    return " ".join(str1.split()) == " ".join(str2.split())


TESTS = [
    {
        "mocked": [str_final_with_tag],
        "expected_outputs": [str_final_with_tag],
        "expected_final_answer": str_final_answer,
    },
    {
        "mocked": [str_let_me_think, str_final_with_tag],
        "expected_outputs": [str_let_me_think, str_final_with_tag],
        "expected_final_answer": str_final_answer,
    },
    {
        "mocked": [str_gadget_usage, str_final_with_tag],
        "expected_outputs": [str_gadget_usage, str_gadget_output, str_final_with_tag],
        "expected_final_answer": str_final_answer,
    },
    {
        "mocked": [str_gadget_usage, str_gadget_usage, str_final_with_tag],
        "expected_outputs": [str_gadget_usage, str_gadget_output, str_gadget_usage, str_gadget_output, str_final_with_tag],
        "expected_final_answer": str_final_answer,
    },
    {
        "mocked": [str_gadget_usage + str_gadget_usage, str_final_with_tag],
        "expected_outputs": [str_gadget_usage + str_gadget_output + str_gadget_usage + str_gadget_output, str_final_with_tag],
        "expected_final_answer": str_final_answer,
    }
]

def test_generate_with_gadgets():
    model_name = "salesforce/codet5-small"
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

    for i, test in enumerate(TESTS):
        assert test_generate_check_outputs(
            model,
            tokenizer,
            test["mocked"],
            test["expected_outputs"],
            test["expected_final_answer"],
            enabled_gadgets=[Calculator],
        )

if __name__ == "__main__":
    test_generate_with_gadgets()
