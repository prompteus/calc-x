import random
import string
import pathlib

import inflect

from gadgets.gadget import Calculator
from gadgets.data_iterators.iterator import DataIterator
import gadgets.datatypes


class SyntheticIterator(DataIterator):
    def __init__(
        self,
        nouns_filepath: str | pathlib.Path,
        names_filepath: str | pathlib.Path,
        max_expressions = 5,
        max_operand = 100,
        seed: int | None = None,
    ) -> None:
        self.max_expressions = max_expressions
        self.max_operand = max_operand
        self.calculator = Calculator()
        self.scenarios = [
            {'operation': '$x+$y', 'event': 'gives me $y $objects.'},
            {'operation': '$x-$y', 'event': 'takes $y $objects from me.'},
            {'operation': '$x*$y', 'event': 'turns each of my $objects into $y $objects.'},
            {'operation': '$x//$y', 'event': 'merges each $y $objects into one and remainder throws away.'},
        ]
        
        with open(nouns_filepath, "r") as file:
            self.noun_list = file.read().splitlines()

        with open(names_filepath, "r") as file:
            self.name_list = file.read().splitlines()

        self.plural_engine = inflect.engine()

        self.random_gen = random.Random(seed)
    
    def __iter__(self):
        return self
    
    def __next__(self) -> gadgets.datatypes.Example:
        x = self.random_gen.randint(0, self.max_operand)
        
        objects = self.plural_engine.plural(self.random_gen.choice(self.noun_list))
        iterations = self.random_gen.randint(1, self.max_expressions)

        introduction = f"I have {x} {objects}. "

        math_parts = []
        chain = []
        for i in range(iterations):
            
            if isinstance(x, str):
                x_num = float(x.split("~=")[-1].strip().replace(",", ""))
            else:
                x_num = x

            y = self.random_gen.randint(1, self.max_operand)
            scenario = self.random_gen.choice(self.scenarios)
            actor = self.random_gen.choice(self.name_list)
            temp_math_part = string.Template(f"{actor} comes and {scenario['event']}").substitute({
                'x': x,
                'y': y,
                'objects': objects,
            })

            if i > 0:
                temp_math_part = ' Then ' + temp_math_part

            if isinstance(x, str):
                x_num = x.split("~=")[0].strip().replace(",", "")
            else:
                x_num = x

            inputs = string.Template(scenario['operation']).substitute({'x': x_num, 'y': y})
            outputs = self.calculator(inputs)

            math_parts.append(temp_math_part)
            interaction = gadgets.datatypes.Interaction(
                gadget_id="calculator",
                inputs=inputs,
                outputs=outputs
            )
            chain.append(interaction)

            x = outputs

        problem_desc = ''.join(math_parts)
        question = f" How many {objects} do i have?"

        prompt = introduction + problem_desc + question    
        result = self.calculator(inputs).split("~=")[0].strip()

        return gadgets.datatypes.Example(
            prompt=prompt,
            chain=chain,
            result=result
        )
    

if __name__ == "__main__":
    ite = SyntheticIterator(max_expressions=5, max_operand=30, nouns_filepath="helper_data/nouns.txt", names_filepath="helper_data/names.txt")
    for _ in range(300):
        print(next(ite))