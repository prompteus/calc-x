# Calc-X and Calcformers

This repo contains dataset builders, training scripts, and inference wrappers for training and using Calcformers, models capable of using a calculator during inference.
This is the official repository for the EMNLP 2023 paper: [Calc-X and Calcformers: Empowering Arithmetical Chain-of-Thought through Interaction with Symbolic Systems](https://arxiv.org/abs/2305.15017)

You can access the datasets and the trained models on HuggingFace:

- [Calc-X dataset collection](https://huggingface.co/collections/MU-NLPC/calc-x-652fee9a6b838fd820055483)
- [Calcformer models collection](https://huggingface.co/collections/MU-NLPC/calcformers-65367392badc497807b3caf5).



## Create environment

First, clone the repo. Then run:

```shell
conda create -n gadgets python=3.10 && conda activate gadgets
pip install poetry
poetry install 
```

This installs all dependencies in exact same versions used by the authors of the repo.
In case you encounter any issues on your hardware (e.g., with CUDA version, platform, etc.),
you can resolve the dependencies yourself:

```shell
# with plain pip:
pip install -e .[dev]
# OR with poetry:
poetry lock && poetry install
```


## Usage

We wrap the `generate()` method to be able to utilize the 
given set of gadgets during the generation. 
You will need to wrap the model of your choice and 
make sure that the tokenizer is able to encode the instruction
HTML tags used in calling the gadget calls.

Using our pre-trained models (with the tokenizer resolved),
you can use the model using a calculator gadget as follows.

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


## Cite

If you find this project useful in your research, please cite the [Calc-X and Calcformers paper](https://arxiv.org/abs/2305.15017) as follows:

```bibtex
@inproceedings{kadlcik-etal-2023-soft,
    title = "Calc-X and Calcformers: Empowering Arithmetical Chain-of-Thought through Interaction with Symbolic Systems",
    author = "Marek Kadlčík and Michal Štefánik and Ondřej Sotolář and Vlastimil Martinek",
    booktitle = "Proceedings of the The 2023 Conference on Empirical Methods in Natural Language Processing: Main track",
    month = dec,
    year = "2023",
    address = "Singapore, Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2305.15017",
}
```

