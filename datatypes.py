from typing import NamedTuple

class GadgetInteraction(NamedTuple):
    gadget_id: str
    inputs: str
    outputs: str

Chain = list[GadgetInteraction | tuple[str, str, str] | str]

class Example(NamedTuple):
    prompt: str
    chain: Chain
    result: str

