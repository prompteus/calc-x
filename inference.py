from argparse import ArgumentParser

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaModelForCausalLM,
    LlamaTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from gadgets.gadget import Calculator
from gadgets.gadget_assisted_model import GadgetAssistedModel

parser = ArgumentParser()
parser.add_argument('model_type', default='encoder-decoder') # Can also be decoder-only
args = parser.parse_args()


class GadgetAssistedT5(GadgetAssistedModel, T5ForConditionalGeneration):
    # GadgetAssistedModel overrides the standard generate() from transformers
    pass

class GadgetAssistedLlama(GadgetAssistedModel, LlamaModelForCausalLM):
    # GadgetAssistedModel overrides the standard generate() from transformers
    pass


t5_model_id = "MU-NLPC/calcformer-t5-large"
llama_model_id = 'meta-llama/Llama-2-7b-chat-hf'
bloke_model_id = 'TheBloke/Llama-2-7B-Chat-GPTQ'

if args.model_type == 'encoder-decoder':
    model = GadgetAssistedT5.from_pretrained(t5_model_id)
    tokenizer = T5Tokenizer.from_pretrained(t5_model_id)

# TODO Use .env for local
from google.colab import userdata

hf_auth = userdata.get('hf_auth')

if args.model_type == 'decoder-only':
    model = GadgetAssistedLlama.from_pretrained(llama_model_id,
                                                device_map="auto",
                                                torch_dtype="auto",
                                                use_auth_token=hf_auth
                                                )

    tokenizer = LlamaTokenizer.from_pretrained(llama_model_id, 
                                               use_auth_token=hf_auth
                                               )


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
output_ids = model.generate(**inputs, architecture=args.model_type)
# output_ids
output_str = tokenizer.decode(output_ids[0], spaces_between_special_tokens=False)
print(output_str)
