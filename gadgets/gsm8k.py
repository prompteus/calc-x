from __future__ import annotations

import math
import re
from typing import Dict

import gadgets.datatypes
import gadgets.gadget


def parse(sample: Dict[str, str]) -> gadgets.datatypes.Example:
    """
    >>> import datasets
    >>> dataset = datasets.load_dataset("gsm8k", "main")
    >>> _ = parse(dataset["train"][0])

    >>> question = "I have 2 apples, Sam gives me 2 more, how many apples do I have?"
    >>> answer = "Let me think... 2 and 2 = <<2+2=4>> 4. I have 4 apples now. #### 4"
    >>> sample = parse({"question": question, "answer": answer})
    >>> sample.chain
    ['Let me think... 2 and 2 = ', Interaction(gadget_id='calculator', inputs='2+2', outputs='4'), ' 4. I have 4 apples now. ']
    >>> sample.prompt
    'I have 2 apples, Sam gives me 2 more, how many apples do I have?'
    >>> sample.result
    '4'

    """

    assert "answer" in sample, "answer is missing"
    assert "question" in sample, "question is missing"

    sample["question"] = replace_unicode(sample["question"])
    sample["answer"] = replace_unicode(sample["answer"])

    calc = gadgets.gadget.Calculator()

    result: str = sample["answer"]
    chain_str, result = result.split("####")

    chain_str = add_missing_dots(chain_str)
    result = calc(result.strip().replace(",", "_"))
    calc_re = re.compile(r"<<(.*?)=(.*?)>>", flags=re.MULTILINE)

    chain: gadgets.datatypes.Chain = []

    last_index = 0
    for match in calc_re.finditer(chain_str):
        start, end = match.span()
        if start > last_index:
            chain.append(chain_str[last_index:start])
        last_index = end

        gadget_input = match.group(1)
        gadget_output_from_data = match.group(2)
        gadget_output = calc(gadget_input)

        expected = calc._float_eval(gadget_output_from_data)
        actual = calc._float_eval(gadget_input)
        assert math.isclose(expected, actual), f"{expected} != {actual}"

        interaction = gadgets.datatypes.Interaction(
            gadget_id="calculator",
            inputs=gadget_input,
            outputs=gadget_output,
        )
        chain.append(interaction)

    if last_index < len(chain_str):
        chain.append(chain_str[last_index:])

    return gadgets.datatypes.Example(
        prompt=sample["question"],
        chain=chain,
        result=result,
    )


def add_missing_dots(input_string: str):
    lines = input_string.split("\n")
    result = []

    for line, next_line in zip(lines, lines[1:] + [""]):
        if line != "" and line[-1].strip().isalnum() and (next_line == "" or next_line[0].isupper()):
            line += "."
        result.append(line)

    return "\n".join(result)


def replace_unicode(string: str) -> str:
    replacements = {
        "’": "'",
        "–": "-",
        "×": "*",
        "÷": "/",
        "−": "-",
        "≠": "!=",
        "”": '"',
        "“": '"',
        "—": "-",
        "‘": "'",
        "√": "sqrt",
        "⁰": "^0",
        "¹": "^1",
        "²": "^2",
        "³": "^3",
        "⁴": "^4",
        "⁵": "^5",
        "⁶": "^6",
        "⁷": "^7",
        "⁸": "^8",
        "⁹": "^9",
        "¼": "1/4",
        "½": "1/2",
        "¾": "3/4",
        "\u2028": "\n",
        "\u2029": "\n",
        "\xa0": " ",
        "\u200b": "",
        "А": "A",
    }
    for key, value in replacements.items():
        string = string.replace(key, value)
    return string
