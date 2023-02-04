import bs4
from datatypes import Interaction, Chain

GADGET_TAG = "gadget"
OUTPUT_TAG = "output"
RESULT_TAG = "result"


def gadget_interaction_to_markup(iteraction: Interaction) -> tuple[bs4.Tag, bs4.Tag]:
    tag = bs4.Tag(name=GADGET_TAG)
    tag["id"] = iteraction.gadget_id
    tag.string = iteraction.inputs

    output_tag = bs4.Tag(name=OUTPUT_TAG)
    output_tag.string = iteraction.outputs

    return tag, output_tag


def result_to_markup(result: str) -> bs4.Tag:
    tag = bs4.Tag(name=RESULT_TAG)
    tag.string = result
    return tag


def to_model_markup(chain: Chain, result: str, add_result_sentence: bool = False) -> bs4.BeautifulSoup:

    soup = bs4.BeautifulSoup(features="html")

    for item in chain:
        if isinstance(item, tuple):
            item = Interaction(*item)

        if isinstance(item, Interaction):
            tag, output_tag = gadget_interaction_to_markup(item)
            soup.append(tag)
            soup.append("\n")
            soup.append(output_tag)
            soup.append("\n")
        else:
            soup.append(item)
            soup.append("\n")

    if add_result_sentence:
        soup.append("Final answer is ")

    soup.append(result_to_markup(result))

    return soup


def from_model_markup(markup: bs4.BeautifulSoup | str) -> tuple[Chain, str | None]:
    if isinstance(markup, str):
        markup = bs4.BeautifulSoup(markup, features="html")
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
