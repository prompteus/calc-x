from data_iterators.iterator import DataIterator
import jsonlines
from typing import Optional, Dict


class JLIterator(DataIterator):
    """
    Read jasonlies file as DataIterator.
    Expected jasonlies object format:
    {
        "prompt": str,
        "chain": [
            {
                "type": str,
                "in": str,
                "out": str,
            }, ...
        ],
        "answer": str,
    }
    """

    def __init__(self, path: str, tags: Optional[Dict[str, str]] = None):
        """
        path: str              - Path to the jsonlines file.
        tags: dict[str, str]   - definition of tags for different gadget types used
                                 in the gadget chains (type -> tag)
                                 if None it will use the type itself

        Example of tags:
            `tags={"python": "interpret"}` will result in tags <interpret> and </interpret>
                for chain links of type `python`
        """
        self.path = path
        self.tags = tags
        self.reader = None
        self.iter_called = False


    def __enter__(self):
        self.reader = jsonlines.open(self.path)
        self.iterator = iter(self.reader)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reader.close()

    def __iter__(self):
        if self.iter_called:
            raise Exception("iter was called multiple times")
        self.iter_called = True
        return self

    def __next__(self):
        if self.reader is None:
            raise Exception("File not open. Use the `with:` bolock.")

        obj = next(self.iterator)

        chain = []
        for link in obj["chain"]:
            chain.append(self.add_tags(link))
        # chain = "\n".join(chain) + "Final answer is " + obj["answer"]
        chain = "\n".join(chain)

        return obj["prompt"], chain, obj["answer"]

    def add_tags(self, link):
        if self.tags is None:
            tag = link["type"]
        else:
            tag = self.tags[link["type"]]
        link_in = link['in']
        link_out = link['out']
        return f"<gadget id='{tag}'>\n{link_in}\n</gadget>\n<output>\n{link_out}\n</output>"


if __name__ == "__main__":
    jli = JLIterator("sample_data/word_problems_petr.jl")
    with jli:
        for tup in jli:
            print(tup)
