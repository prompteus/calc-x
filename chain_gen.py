from typing import Tuple, List

import random
from string import Template
from gadget import foo

objects_all = ['apples', 'pears']
names = ['igor', 'vasil']

scenarios = [
    {'operation': '$x-$y', 'event': 'takes $y $objects from me'},
    {'operation': '$x+$y', 'event': 'gives me $y $objects'},
    {'operation': '$x*($y+1)', 'event': 'gives me $y times the amount of $objects i already have'},
    {'operation': '$x//2', 'event': 'takes half of my $objects away, rounded down'},
]


def get_random_chain() -> Tuple[str, List[str], str]:
    objects = random.choice(objects_all)
    name = random.choice(names)
    scenario = random.choice(scenarios)
    x = random.randint(0, 1000)
    y = random.randint(0, x)

    t = Template(f"I have $x $objects, $name comes and {scenario['event']}. How many $objects do i have?")
    prompt = t.substitute({
        'x': x,
        'y': y,
        'objects': objects,
        'name': name,
    })

    operation = Template(scenario['operation']).substitute({'x': x, 'y': y})
    call = operation

    chain = f"<python>{call}</python>"
    answer = foo(call)
    return prompt, chain, answer


print(get_random_chain())
