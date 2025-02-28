import re

import pandas as pd


class Node:
    def __init__(self, text: str, is_root: bool = False):
        self.text = text
        all_numbers = set(re.findall(r"\d+", text))
        outputs = set(re.findall(r"<output>(\d+)</output>", text))
        if not is_root:
            self.inputs = all_numbers - outputs
            self.outputs = outputs
        else:
            self.inputs = set()
            self.outputs = all_numbers
    def __str__(self):
        return "Node with inputs (%s) -> outputs (%s)" % (", ".join(self.inputs), ", ".join(self.outputs))


eval_file = "calcformer-t5-large-Calc-gsm8k.jsonl"

df = pd.read_json(eval_file, lines=True)

sample = df.iloc[0]

steps_sep = "</output>"

nodes_ref = ([Node(sample["question"], is_root=True)] +
             [Node(text+steps_sep) for text in sample["answer"].replace("\n", " ").split(steps_sep)])
nodes_hyp = ([Node(sample["question"], is_root=True)] +
             [Node(text) for text in sample["prediction"].replace("\n", " ").split(steps_sep)])
print("Expected:")
[print(node) for node in nodes_ref]
print("Actual:")
[print(node) for node in nodes_hyp]
