import logging
import random

import transformers
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from gadgets.steps_utils import StepPermuter, separate_chain_to_steps


logger = logging.getLogger()


class PerMistakesConsistency:

    def __init__(self, model: transformers.GenerationMixin, tokenizer: PreTrainedTokenizer, sep_token: str):
        self.model = model
        self.tokenizer = tokenizer
        self.permuter = StepPermuter(tokenizer)
        self.sep_token = sep_token

    def get_alternative_chain(self, inputs: list[str], predictions: list[str]) -> list[str]:
        alternative_chains = []

        for question, pred in tqdm(zip(inputs, predictions),
                                   total=len(predictions), desc="Generating alternative chains"):
            model_steps, sep = separate_chain_to_steps(pred, self.sep_token)
            permuted_steps = self.permuter.permute_all_steps(model_steps)

            # pick a permutation step
            num_steps = min(len(model_steps), len(permuted_steps))
            if num_steps <= 1:
                logger.warning("Single reasoning step -> skipping sample from consistency eval.")
                alternative_chains.append("N/A")
                continue
            try:
                adjusted_step_i = random.randint(1, min(len(model_steps), len(permuted_steps))-1)
                model_steps[adjusted_step_i] = permuted_steps[adjusted_step_i]
                full_previous_chain = "".join(model_steps[:adjusted_step_i+1])  # includes permuted step
            except IndexError:
                logger.warning("Index error: %s, %s, %s"
                               % (adjusted_step_i, len(model_steps), len(permuted_steps)))
                alternative_chains.append("N/A")
                continue

            # continue in generation with the permuted step until generating <result>
            new_inputs = self.tokenizer(full_previous_chain, return_tensors="pt").to(self.model.device)
            output_ids = self.model.generate(**new_inputs, max_new_tokens=512)
            # TODO: distribution shift compared to training: [step] tokens here attend everything
            # TODO: this requires adjustment in CompressedStepwiseGenerator.generate()
            # TODO: but primary motivation is to regularize token embeddings in training
            # TODO: so it makes sense to try it out first without complicating things
            output_str = self.tokenizer.batch_decode(output_ids)[0]

            alternative_chains.append(output_str)

        return alternative_chains


class StepwiseLoss:

    def __init__(self, model: transformers.GenerationMixin, tokenizer: PreTrainedTokenizer, step_token_id: int):
        self.model = model
        self.tokenizer = tokenizer
        self.permuter = StepPermuter(tokenizer)
        self.step_token_id = step_token_id

        # TODO: we do not do this yet, because [step] tokens do not attend correct tokens in evaluation
        # if we want correct evaluations of [step] consistencies, we'd need to fix that first, but maybe we don't

