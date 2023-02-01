from typing import Tuple, List

import random
from string import Template
from gadget import foo

objects_all = ['apples', 'pears']
actors = ['Igor', 'Vasil', 'Bob']

scenarios = [
    {'operation': '$x-$y', 'event': 'takes $y $objects from me.'},
    {'operation': '$x+$y', 'event': 'gives me $y $objects.'},
    {'operation': '$x*($y+1)', 'event': 'gives me $y times the amount of $objects i already have.'},
    {'operation': '$x//2', 'event': 'takes half of my $objects away, rounded down.'},
]

def get_random_chain() -> Tuple[str, List[str], str]:
    x = random.randint(0, 1000)
    y = random.randint(0, x)
    objects = random.choice(objects_all)
    iterations = random.randint(1,3)
    
    start_part = Template(f"I have $x $objects.").substitute({'x':x, 'objects':objects})
    
    math_parts = []
    chains = []
    for i in range(iterations):
        scenario = random.choice(scenarios)
        temp_math_part = Template(f"$actor comes and {scenario['event']}").substitute({
            'x': x,
            'y': y,
            'objects': objects,
            'actor': random.choice(actors),
        })
        if(i > 0):
            temp_math_part = 'Then ' + temp_math_part
        call = Template(scenario['operation']).substitute({'x': x, 'y': y})
        x = int(foo(call))
        temp_chain = f"<python>{call}</python>"
        math_parts.append(temp_math_part)
        chains.append(temp_chain)
        
        
    math_part = ''.join(math_parts)
    chain = ''.join(chains)
    
    end_part = Template("How many $objects do i have?").substitute({'objects':objects})
    
    prompt = start_part + math_part + end_part    
    
    answer = foo(call)
    return prompt, chain, answer


print(get_random_chain())
