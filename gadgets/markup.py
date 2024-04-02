from __future__ import annotations

import re
import warnings

import bs4

from gadgets.datatypes import Chain, Example, Interaction, Step

GADGET_TAG = "gadget"
OUTPUT_TAG = "output"
RESULT_TAG = "result"


def step_to_markup(step: Step) -> bs4.BeautifulSoup:
    if isinstance(step, str):
        return bs4.BeautifulSoup(step, features="html.parser")

    interaction: Interaction = step

    soup = bs4.BeautifulSoup(features="html.parser")

    tag = bs4.Tag(name=GADGET_TAG)
    tag["id"] = interaction.gadget_id
    tag.string = interaction.inputs

    output_tag = bs4.Tag(name=OUTPUT_TAG)
    output_tag.string = interaction.outputs

    soup.append("\n")
    soup.append(tag)
    soup.append("\n")
    soup.append(output_tag)
    soup.append("\n")
    return soup


def result_to_markup(result: str) -> bs4.BeautifulSoup:
    soup = bs4.BeautifulSoup(features="html.parser")
    tag = bs4.Tag(name=RESULT_TAG)
    tag.string = result
    soup.append(tag)
    return soup


def to_model_markup(
    *,
    chain: Chain | None = None,
    result: str | None = None,
    example: Example | None = None,
    ommit_tags: bool = False,
    add_result_sentence: bool = False,
) -> bs4.BeautifulSoup:
    if example is None and chain is None:
        raise ValueError("Either example or chain must be provided")

    if example is not None and chain is not None:
        raise ValueError("Only one of example or chain can be provided")

    if chain is None != result is None:
        raise ValueError("If chain is provided, result must be provided")

    if example is not None:
        chain = example.chain
        result = example.result

    soup = bs4.BeautifulSoup("", features="html.parser")

    for step in chain:
        if isinstance(step, tuple):
            gadget_id, inputs, outputs = step
            step = Interaction(gadget_id=gadget_id, inputs=inputs, outputs=outputs)
        if ommit_tags and isinstance(step, Interaction):
            continue
        soup.append(step_to_markup(step))

    if result is None:
        return soup

    if add_result_sentence:
        str_soup = str(soup)
        result_sentence = "Final result is "
        # if len(str_soup.strip()) > 0 and not str_soup.strip().endswith("."):
        #    result_sentence = ". " + result_sentence
        soup.append(result_sentence)
        soup.append(result + ".\n")

    if not ommit_tags:
        str_soup = str(soup)
        if len(str_soup.strip()) > 0 and not str_soup.strip().endswith("\n"):
            soup.append("\n")
        soup.append(result_to_markup(result))

    return soup


def get_result_from_output_fallback(output: str) -> str:
    results = re.findall("final result is (.+?)\.", output, flags=re.IGNORECASE)
    if len(results) == 0:
        return ""
    # chopse last occurence if multiple exist
    result: str = results[-1]
    result = result.split("=")[0].strip()
    return result


def get_result_from_output(output: str) -> str:
    # Fast regex search
    match = re.search(f"<{RESULT_TAG}>(.+?)</{RESULT_TAG}>", output, flags=re.IGNORECASE)
    if match is not None:
        return match.group(1).strip()
    
    # Slow html parsing
    soup = bs4.BeautifulSoup(output, features="html.parser")
    result_tag = soup.find(RESULT_TAG)

    # Fallback for when the result tag is not found
    if result_tag is None or result_tag.string is None:
        return get_result_from_output_fallback(output)
    return result_tag.string.strip()


def from_model_markup(markup: bs4.BeautifulSoup | str) -> tuple[Chain, str]:
    if isinstance(markup, str):
        markup = bs4.BeautifulSoup(markup, features="html.parser")
    else:
        # copy the markup so we don't modify the original
        markup = markup.copy()

    chain: Chain = []
    result = ""

    # delete empty strings
    for item in markup.children:
        if isinstance(item, bs4.NavigableString) and item.string.strip() == "":
            item.extract()

    last_interaction_idx: int | None = None

    for item in markup.children:
        if isinstance(item, bs4.NavigableString):
            chain.append(str(item).strip())
            continue

        assert isinstance(item, bs4.Tag)

        if item.name == GADGET_TAG:
            gadget_id = item.get("id", default="")
            if item.string is None:
                inputs = ""
            else:
                inputs = item.string.strip()
            interaction = Interaction(gadget_id=gadget_id, inputs=inputs, outputs="")
            chain.append(interaction)
            last_interaction_idx = len(chain) - 1

        elif item.name == OUTPUT_TAG:
            if last_interaction_idx is None:
                warnings.warn(
                    "Found output tag without a preceding gadget tag - ignoring.\n\n"
                    "markup:\n"
                    f"{markup}\n\n"
                    "output tag:\n"
                    f"{item}\n"
                )
                continue
            if chain[last_interaction_idx].outputs != "":
                warnings.warn(
                    "Found multiple output tags for a single gadget tag. Ignoring the second one.\n\n"
                    "markup:\n"
                    f"{markup}\n\n"
                    "output tag:\n"
                    f"{item}\n\n"
                    "last interaction:\n"
                    f"{chain[last_interaction_idx]}\n"
                )
                continue
            outputs = "" if item.string is None else item.string.strip()
            chain[last_interaction_idx] = Interaction(
                gadget_id=chain[last_interaction_idx].gadget_id,
                inputs=chain[last_interaction_idx].inputs,
                outputs=outputs,
            )
                
        elif item.name == RESULT_TAG:
            if item.string is not None:
                result = item.string.strip()

    return chain, result


def strip_markup(markup: bs4.BeautifulSoup | str) -> str:
    if isinstance(markup, str):
        markup = bs4.BeautifulSoup(markup, features="html.parser")
    
    text = ""
    for elem in markup.children:
        if isinstance(elem, bs4.NavigableString):
            text += elem

    return text

