from __future__ import annotations

import math
import re
from typing import Dict

import gadgets.gadget
import gadgets.datatypes

numeric_re = re.compile(r"[^\d\+\-\/\*\=\ \.\,\(\)]", flags=re.MULTILINE)  # same length as original
equals_re = re.compile(r"=", flags=re.MULTILINE)
calc_input_re = re.compile(r"([\d\+\-\/\*\.\,\(\)\s*]+)\s*=\s*([\d\.]+)")

matched_eqs = 0
not_matched_eqs = 0
invalid_calls = []


def parse(sample: Dict[str, str]) -> gadgets.datatypes.Example:
    """
    Outline:
    -> Iterate over all matches of "=":
    1. Find Cases where calc(left) == right
    2. In such cases, pick a segment from the reasoning chain corresponding to the equation
       and replace it with the parsed gadget call

    How to test this:
    >>> import datasets
    >>> dataset = datasets.load_dataset("aqua", split="main")
    >>> sample = parse(dataset["train"][0])
    >>> sample.chain
    ['Let me think... 2 and 2 = ', Interaction(gadget_id='calculator', inputs='2+2', outputs='4'), ' 4. I have 4 apples now. ']
    >>> sample.prompt
    'I have 2 apples, Sam gives me 2 more, how many apples do I have?'
    >>> sample.result
    '4'

    """
    global matched_eqs, not_matched_eqs
    global invalid_calls

    calc = gadgets.gadget.Calculator()

    chain_str = sample["rationale"].replace("\n", "   ").strip()
    correct_str = next(o.replace(sample['correct'] + ")", "").strip()
                       for o in sample["options"] if sample['correct'] + ")" in o)
    sent_separator = ". "

    last_chain_sentence = chain_str.split(sent_separator)[-1]
    chain_str = sent_separator.join(chain_str.split(sent_separator)[:-1])
    chain_str += (sent_separator + last_chain_sentence.replace(sample['correct'], correct_str))
    if chain_str.startswith(sent_separator):
        # if there is only one sentence in the chain, we truncate the preceding sentence separator
        chain_str = chain_str[len(sent_separator):]

    numeric_chain_str = re.sub(numeric_re, " ", chain_str)

    chain: gadgets.datatypes.Chain = []

    eq_positions = [i for i, char in enumerate(numeric_chain_str) if char == "="]
    eq_positions = [0] + eq_positions + [len(numeric_chain_str)]

    for eq_pos_i in range(1, len(eq_positions[1:])):
        original_left_right_substr = chain_str[eq_positions[eq_pos_i - 1]: eq_positions[eq_pos_i + 1]]
        chain.append(chain_str[eq_positions[eq_pos_i-1]: eq_positions[eq_pos_i]])
        if len(chain) > 1 and isinstance(chain[-2], gadgets.datatypes.Interaction) and chain[-1].startswith("="):
            # for consistency with gsm, we do not follow with "=" if preceding element of chain was gadget call
            chain[-1] = chain[-1][1:].strip()

        eq_left_right_substr = numeric_chain_str[eq_positions[eq_pos_i - 1]: eq_positions[eq_pos_i + 1]]
        eq_left_right_groups = re.search(calc_input_re, eq_left_right_substr)
        if eq_left_right_groups is None:
            # no "=" -> first or last iteration
            not_matched_eqs += 1

            continue
        # strip to avoid empty inputs with trailing spaces & split to be robust to inserted newlines
        gadget_input = eq_left_right_groups.group(1).strip().split("   ")[-1]
        if not any(operator in gadget_input for operator in "+-/*^"):
            # no operation, just raw statement -> could be right equation side, processed in the previous step
            not_matched_eqs += 1
            invalid_calls.append(eq_left_right_substr)

            continue

        gadget_output_from_data = eq_left_right_groups.group(2)
        gadget_output = calc(gadget_input)
        if 'ERROR: invalid syntax' in gadget_output:
            # not a valid gadget input
            not_matched_eqs += 1
            invalid_calls.append(eq_left_right_substr)

            continue
        try:
            expected = calc._float_eval(gadget_output_from_data)
            actual = calc._float_eval(gadget_output)
        except (AttributeError, SyntaxError):
            # apparently, gadget calls sometime return trash, e.g. '1/5 = around 0.2'
            invalid_calls.append(eq_left_right_substr)
            not_matched_eqs += 1

            continue

        if math.isclose(expected, actual):
            chain = chain[:-1]
            chain.append(original_left_right_substr.split(gadget_input.strip())[0] + gadget_input)
            if len(chain) > 1 and isinstance(chain[-2], gadgets.datatypes.Interaction) and chain[-1].startswith("="):
                # for consistency with gsm, we do not follow with "=" if preceding element of chain was gadget call
                chain[-1] = chain[-1][1:].strip()

            chain.append("= ")  # for consistency with gsm, "=" always precede gadget calls
            chain.append(gadgets.datatypes.Interaction(gadget_id="calculator",
                                                       inputs=gadget_input,
                                                       outputs=gadget_output))
            matched_eqs += 1
        else:
            invalid_calls.append(eq_left_right_substr)

            not_matched_eqs += 1

    chain.append(chain_str[eq_positions[-2]: eq_positions[-1]])
    if len(chain) > 1 and isinstance(chain[-2], gadgets.datatypes.Interaction) and chain[-1].startswith("="):
        # for consistency with gsm, we do not follow with "=" if preceding element of chain was gadget call
        chain[-1] = chain[-1][1:].strip()

    return gadgets.datatypes.Example(prompt=sample["question"], chain=chain, result=correct_str)
