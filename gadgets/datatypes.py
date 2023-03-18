from __future__ import annotations

from typing import Union, List

import pydantic


class Interaction(pydantic.BaseModel):
    """
    From dict:
    >>> dictionary = {"gadget_id": "calculator", "inputs": "2+2", "outputs": "4"}
    >>> interaction = Interaction(**dictionary)

    To json string:
    >>> interaction.json()
    '{"gadget_id": "calculator", "inputs": "2+2", "outputs": "4"}'
    """

    gadget_id: str
    inputs: str
    outputs: str


Step = Union[str, Interaction]

Chain = List[Step]


class Example(pydantic.BaseModel):
    """
    From dict:
    >>> dictionary = {
    ...     "prompt": "I have 2 apples, Sam gives me another 2, then I lose one, how many apples do I have?",
    ...     "chain": [
    ...         "Let me think...",
    ...         { "gadget_id": "calculator", "inputs": "2+2", "outputs": "4" },
    ...         "I have 4 apples now. I lost one.",
    ...         { "gadget_id": "calculator", "inputs": "4-1", "outputs": "3" }
    ...     ],
    ...     "result": "3"
    ... }
    >>> example = Example(**dictionary)

    To json string:
    >>> import json
    >>> example.json() == json.dumps(dictionary)
    True
    """

    prompt: str
    chain: Chain
    result: str
