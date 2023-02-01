import random
from string import Template
from gadget import foo

objects_all = ['apples','pears']
names = ['igor','vasil']

scenarios = [
    {'operation':'-', 'verb':'takes'},
    {'operation':'+', 'verb':'gives me'},
]

def get_random_chain():
    objects = random.choice(objects_all)
    name = random.choice(names)
    scenario = random.choice(scenarios)
    x = random.randint(0, 1000)
    y = random.randint(0, x)

    prompt = f"I have {x} {objects}, {name} comes and {scenario['verb']} {y} {objects}. How many {objects} do i have?"
    call = f"{x}{scenario['operation']}{y}"
    chain = f"<python>{call}</python>"
    answer = foo(call) 
    return prompt, chain, answer
  

print(get_random_chain())