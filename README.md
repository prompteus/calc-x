# Gadget-assisted Language Models

This repo contains dataset builders, training scripts 
and inference wrappers for training and using 
Tool-assisted Language Models.

The training scripts including dataset curation can be found
in `examples` (in progress).

## Usage

We wrap the `generate()` method to be able to utilize the 
given set of gadgets during the generation. 
You will need to wrap the model of your choice and 
make sure that the tokenizer is able to encode the instruction
HTML tags used in calling the gadget calls.

Using our pre-trained models (with the tokenizer resolved),
you can use the model using calculator gadget as follows.

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

from gadgets.gadget_assisted_model import GadgetAssistedModel
from gadgets.gadget import Calculator


class GadgetAssistedT5(GadgetAssistedModel, T5ForConditionalGeneration):
    # GadgetAssistedModel overrides the standard generate() from transformers
    pass


model = GadgetAssistedT5.from_pretrained("MU-NLPC/Calc-FLAN-3B-GSM8K")
tokenizer = T5Tokenizer.from_pretrained("MU-NLPC/Calc-FLAN-3B-GSM8K")

model.prepare_for_generate(tokenizer, 
                           enabled_gadgets=[Calculator()], 
                           default_max_tokens=512)
query = """
    The profit from a business transaction is shared among 2 business partners, 
    Mike and Johnson in the ratio 2:5 respectively. 
    If Johnson got $2500, how much will Mike have 
    after spending some of his share on a shirt that costs $200?
"""

inputs = tokenizer(query, return_tensors="pt")
output_ids = model.generate(**inputs)
tokenizer.decode(output_ids[0], spaces_between_special_tokens=False)

# This returns:
# 'According to the ratio, Mike got 2/5*$2500 = $<gadget id="calculator">2/5*2500</gadget><output>1_000</output> 1000 
#  Mike will have $1000-$200 = $<gadget id="calculator">1000-200</gadget><output>800</output> 800 after buying a shirt. 
#  Final result is<result>800</result></s>'
```


