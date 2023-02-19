import bs4
import warnings


from gadgets.datatypes import Interaction, Chain, Example, Step

GADGET_TAG = "gadget"
OUTPUT_TAG = "output"
RESULT_TAG = "result"


def step_to_markup(
    step: Step,
    eos_token_after_gadgets: str | None = None,
) -> bs4.BeautifulSoup:
    if isinstance(step, str):
        return bs4.BeautifulSoup(step + "\n", features="html.parser")
    
    interaction: Interaction = step

    soup = bs4.BeautifulSoup(features="html.parser")

    tag = bs4.Tag(name=GADGET_TAG)
    tag["id"] = interaction.gadget_id
    tag.string = interaction.inputs

    output_tag = bs4.Tag(name=OUTPUT_TAG)
    output_tag.string = interaction.outputs

    soup.append(tag)
    if eos_token_after_gadgets is not None:
        eos_str = bs4.BeautifulSoup(eos_token_after_gadgets, features="html.parser")
        soup.append(eos_str)
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
    add_result_sentence: bool = False,
    eos_token_after_gadgets: str | None = None,
) -> bs4.BeautifulSoup:
    
    if eos_token_after_gadgets is None:
        warnings.warn(
            "eos token is not set when converting to model markup. "
            "The model will not know to pause generating after gadget call."
        )

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
            step = Interaction(*step)
        soup.append(step_to_markup(step, eos_token_after_gadgets))

    if add_result_sentence:
        soup.append("Final result is ")

    soup.append(result_to_markup(result))

    return soup


def from_model_markup(markup: bs4.BeautifulSoup | str) -> tuple[Chain, str | None]:
    if isinstance(markup, str):
        markup = bs4.BeautifulSoup(markup, features="html.parser")
    else:
        # copy the markup so we don't modify the original
        markup = markup.copy()

    chain: Chain = []
    result: str | None = None

    # delete empty strings
    for item in markup.children:
        if isinstance(item, bs4.NavigableString) and item.string.strip() == "":
            item.extract()

    for item in markup.children:
        if isinstance(item, bs4.NavigableString):
            chain.append(item.string.strip())
            continue

        if item.name == GADGET_TAG:
            gadget_id = item["id"]
            inputs = item.string.strip()
            try: 
                next_el = item.next_sibling
                if next_el.name == OUTPUT_TAG:
                    outputs = next_el.string.strip()
                else:
                    raise ValueError("Expected output tag after gadget tag, got '%s'" % next_el.name)
            except Exception as e:
                raise e

            chain.append(Interaction(gadget_id, inputs, outputs))

        elif item.name == OUTPUT_TAG:
            continue
        elif item.name == RESULT_TAG:
            result = item.string.strip()

    return chain, result


