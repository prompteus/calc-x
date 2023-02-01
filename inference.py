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
) -> tuple[str, str | None]:
    """
    Generates text with gadgets.
    
    Model is expected to generate <gadget id=""></gadgets> tags.
    Whenever a gadget tag and EOS is generated, the gadget is called, 
    and the output is fed back into the model inside <output></output> tag.

    Final answer is expected to be in <answer></answer> tag.

    Warning:
        if the model generates <gadget id=""></gadgets> tag without EOS immediately after,
        it will not be executed. If it generates multiple <gadget id=""></gadgets> tags 
        and only then EOS, only the last one will be executed.

    Args:
        model: Model to use for generation.
        tokenizer: Tokenizer to use for generation.
        prompt: Prompt to use for generation.
        enabled_gadgets: List of enabled gadget classes.
        generate_kwargs: Keyword arguments to pass to model.generate()

    Returns:
        full_output: Full output of the model, including gadget responses.
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

    force_words_ids = tokenizer(
        ["<gadget id='calculator'>", "</gadget>"],
        add_special_tokens=False
    ).input_ids

    while True:
        if len(total_outputs) == 0:
            output: transformers.utils.ModelOutput = model.generate(
                input_ids=prompt_inputs,
                force_words_ids=force_words_ids,
                **generate_kwargs,
            )[0]
        else:
            output: transformers.utils.ModelOutput = model.generate(
                input_ids=prompt_inputs,
                decoder_input_ids=torch.cat(total_outputs),
                force_words_ids=force_words_ids,
                **generate_kwargs
            )[0]

        output_str = tokenizer.decode(output, skip_special_tokens=True)
        
        total_outputs.append(output)
        if not output_str.rstrip().endswith("</gadget>"):
            break
        
        try:
            soup = BeautifulSoup(output_str, features="html.parser")
            last_tag = list(soup.find_all("gadget"))[-1]
            gadget_id = last_tag["id"]
            gadget = gadgets[gadget_id]
            gadget_request = last_tag.get_text()
            gadget_response = gadget(gadget_request)
        except KeyError:
            gadget_response = f">Gadget not found: {gadget_id}"

        gadget_response_inputs = tokenizer(
            f"<output>{gadget_response}</output>\n",
            return_tensors="pt"
        ).input_ids[0]

        total_outputs.append(gadget_response_inputs)

    total_outputs_str = tokenizer.decode(torch.cat(total_outputs), skip_special_tokens=True)
    try:
        final_answer = BeautifulSoup(total_outputs_str, features="html.parser").find_all("answer")[-1].get_text()
    except Exception:
        final_answer = None

    return tokenizer.decode(torch.cat(total_outputs), skip_special_tokens=True), final_answer



str_prompt = "Write xml tag gadget id attribute id='calculator' and fill '2 + 2' inside. "
str_let_me_think = "Let me think about it"
str_gadget_usage = "<gadget id='calculator'>2+2</gadget>"
str_gadget_output = "<output>4</output>"
str_final_answer = "129818"
str_final_with_tag = f"Final answer is <answer>{str_final_answer}</answer>."
str_final_no_tag = f"Final answer is {str_final_answer}."


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
                max_length=100,
            ),
        )

    expected_full_output = BeautifulSoup(" ".join(expected_full_outputs), features="html.parser").prettify()
    full_output = BeautifulSoup(full_output, features="html.parser").prettify()

    output_matches = full_output == expected_full_output
    answer_matches = final_answer == expected_final_answer
    is_correct = output_matches and answer_matches
    return is_correct


TESTS = [
    {
        "mocked": [str_final_with_tag],
        "expected_outputs": [str_final_with_tag],
        "expected_final_answer": str_final_answer,
    },
    {
        "mocked": [str_final_no_tag],
        "expected_outputs": [str_final_no_tag],
        "expected_final_answer": None,
    },
    {
        "mocked": [str_let_me_think, str_gadget_usage],
        "expected_outputs": [str_let_me_think],
        "expected_final_answer": None,
    },
    {
        "mocked": [str_let_me_think + " " + str_gadget_usage, ""],
        "expected_outputs": [str_let_me_think + " " + str_gadget_usage, str_gadget_output, ""],
        "expected_final_answer": None,
    },
    {
        "mocked": [str_gadget_usage, str_final_with_tag],
        "expected_outputs": [str_gadget_usage, str_gadget_output, str_final_with_tag],
        "expected_final_answer": str_final_answer,
    },
    {
        "mocked": [str_gadget_usage, str_final_no_tag],
        "expected_outputs": [str_gadget_usage, str_gadget_output, str_final_no_tag],
        "expected_final_answer": None,
    },
    {
        "mocked": [str_gadget_usage, str_gadget_usage, str_final_with_tag],
        "expected_outputs": [str_gadget_usage, str_gadget_output, str_gadget_usage, str_gadget_output, str_final_with_tag],
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
