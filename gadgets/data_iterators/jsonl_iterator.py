import json
import pathlib
from typing import Iterable

import gadgets.datatypes
import gadgets.markup
from gadgets.data_iterators.iterator import DataIterator


class JsonlIterator:
    def __init__(self, filepath: pathlib.Path, cycle: bool) -> None:
        super().__init__()
        self.filepath = filepath
        self.cycle = cycle

    def _iter(self) -> Iterable[gadgets.datatypes.Example]:
        with open(self.filepath, "r") as file:
            for line in file:
                yield gadgets.datatypes.Example(**json.loads(line))

    def __iter__(self) -> Iterable[gadgets.datatypes.Example]:
        if self.cycle:
            while True:
                yield from self._iter()
        else:
            yield from self._iter()


if __name__ == "__main__":
    jli = JsonlIterator("data/word_problems_petr.jsonl")
    for tup in jli:
        print("------------")
        print(gadgets.markup.to_model_markup(example=tup))
    
