from typing import Tuple
import random
from string import Template
from gadget import Calculator
from data_iterators.iterator import DataIterator

class InfiniteIterator(DataIterator):
    def __init__(self):
        self.calculator = Calculator()
        self.objects_all = ['apples', 'pears']
        self.actors = ['Igor', 'Vasil', 'Bob']
        self.scenarios = [
            {'operation': '$x-$y', 'event': 'takes $y $objects from me.'},
            {'operation': '$x+$y', 'event': 'gives me $y $objects.'},
            {'operation': '$x*($y+1)', 'event': 'gives me $y times the amount of $objects i already have.'},
            {'operation': '$x//2', 'event': 'takes half of my $objects away, rounded down.'},
        ]
    
    def __next__(self) -> Tuple[str, str, str]:
        x = random.randint(0, 100)
        objects = random.choice(self.objects_all)
        iterations = random.randint(1,3)

        start_part = Template(f"I have $x $objects.").substitute({'x':x, 'objects':objects})

        math_parts = []
        chains = []
        for i in range(iterations):
            y = random.randint(0, x)
            scenario = random.choice(self.scenarios)
            temp_math_part = Template(f"$actor comes and {scenario['event']}").substitute({
                'x': x,
                'y': y,
                'objects': objects,
                'actor': random.choice(self.actors),
            })
            if(i > 0):
                temp_math_part = 'Then ' + temp_math_part
            call = Template(scenario['operation']).substitute({'x': x, 'y': y})
            x = int(self.calculator(call))
            temp_chain = f"<python>{call}</python><out>{x}</out>"
            math_parts.append(temp_math_part)
            chains.append(temp_chain)


        math_part = ''.join(math_parts)
        chain = ''.join(chains)
        end_part = Template("How many $objects do i have?").substitute({'objects':objects})

        prompt = start_part + math_part + end_part    
        answer = self.calculator(call)
        return prompt, chain, answer
    
# ite = InfiniteIterator()
# for _ in range(3):
    # print(next(ite))