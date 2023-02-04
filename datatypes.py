from typing import NamedTuple

class Interaction(NamedTuple):
    gadget_id: str
    inputs: str
    outputs: str

Chain = list[Interaction | tuple[str, str, str] | str]

class Example(NamedTuple):
    prompt: str
    chain: Chain
    result: str

