import transformers
import gadgets

model_name = "emnlp2023/calc-t5-large"

tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
model = gadgets.model.stepwise_gadget_model(transformers.T5ForConditionalGeneration).from_pretrained(model_name)

model.prepare_for_generate(tokenizer,
                           enabled_gadgets=[gadgets.gadget.Calculator()],
                           default_max_tokens=512)
query = """
    The profit from a business transaction is shared among 2 business partners,
    Mike and Johnson in the ratio 2:5 respectively.
    If Johnson got $2500, how much will Mike have
    after spending some of his share on a shirt that costs $200?
"""
inputs = tokenizer(query, return_tensors="pt")
output_ids = model.generate(**inputs)
