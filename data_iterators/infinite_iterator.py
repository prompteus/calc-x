from typing import Tuple, List
import random
from string import Template
from gadget import Calculator
from data_iterators.iterator import DataIterator
from names_dataset import NameDataset
import inflect

class InfiniteIterator(DataIterator):
    def __init__(self, max_depth=3, max_operand=20):
        self.max_depth = max_depth
        self.max_operand = max_operand
        self.calculator = Calculator()
        self.scenarios = [
            {'operation': '$x-$y', 'event': 'takes $y $objects from me.'},
            {'operation': '$x+$y', 'event': 'gives me $y $objects.'},
            # {'operation': '$x*($y+1)', 'event': 'gives me $y times the amount of $objects i already have.'},
            {'operation': '$x*$y', 'event': 'turns each of my $objects into $y $objects.'},
            {'operation': '$x//2', 'event': 'takes half of my $objects away, rounded down.'},
            {'operation': '$x**2', 'event': 'turns each of my $objects into $x $objects.'},
            {'operation': '$x**$y', 'event': 'turns the number of my $objects to the power of $y.'},
        ]
        
        nested_names = [list(dic.values()) for dic in NameDataset().get_top_names(n=10).values()]
        names = sum(sum(nested_names, []),[])
        self.actors = names
        
        nouns_file = open("helper_data/nouns.txt", "r")
        noun_data = nouns_file.read()
        self.noun_list = noun_data.replace('\n', ' ').split(" ")
        self.plural_engine = inflect.engine()
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[str, str, str]:
        x = random.randint(0, self.max_operand)
        objects = self.plural_engine.plural(random.choice(self.noun_list))
        iterations = random.randint(1,self.max_depth)

        start_part = Template(f"I have $x $objects.").substitute({'x':x, 'objects':objects})

        math_parts = []
        chains = []
        for i in range(iterations):
            y = random.randint(0, min(x,self.max_operand))
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
            temp_chain = f"{self.calculator.get_gadget_request_bos()}\n{call}\n{self.calculator.get_gadget_request_eos()}\n{self.calculator.response_template % x}\n"
            math_parts.append(temp_math_part)
            chains.append(temp_chain)


        math_part = ''.join(math_parts)
        chain = ''.join(chains)
        end_part = Template("How many $objects do i have?").substitute({'objects':objects})

        prompt = start_part + math_part + end_part    
        answer = self.calculator(call)
        return prompt, chain, answer
    

if __name__ == "__main__":
    ite = InfiniteIterator(max_depth=5, max_operand=30)
    for _ in range(3):
        print(next(ite))