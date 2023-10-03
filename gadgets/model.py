from __future__ import annotations

import copy
import inspect
import logging
import warnings
from copy import deepcopy

import torch
import transformers
import bs4
from typing import Any, Optional, Callable, List, Union, Tuple
import unittest.mock

from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList, PreTrainedModel, PretrainedConfig, \
    T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.utils import ModelOutput

from gadgets.gadget import Gadget, Calculator
from gadgets.markup import GADGET_TAG, OUTPUT_TAG, RESULT_TAG

logger = logging.getLogger()


class StopAfterGadgetCall(transformers.generation.StoppingCriteria):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        self.closing_tag_ids = self.tokenizer(
            "</" + GADGET_TAG + ">",
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.flatten()

    def __call__(self, seq_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if seq_ids.shape[-1] < self.closing_tag_ids.shape[-1]:
            return False

        # check if </gadget> is at the end of the sequence
        self.closing_tag_ids = self.closing_tag_ids.to(seq_ids.device)
        ending = seq_ids[..., -self.closing_tag_ids.shape[-1]:]
        ends_with_gadget_call = torch.all(ending == self.closing_tag_ids)
        return ends_with_gadget_call


class GadgetAssist(transformers.GenerationMixin):
    """
    Mixin that overrides model.generate to support the
    model with external gadgets.
    """

    def prepare_for_generate(
            self,
            tokenizer: transformers.PreTrainedTokenizer,
            enabled_gadgets: list[Gadget],
            default_max_tokens: int = 1000,
    ) -> None:
        self.tokenizer = tokenizer
        self.enabled_gadgets = enabled_gadgets
        self.default_max_tokens = default_max_tokens

    @torch.no_grad()
    def generate(
            self,
            input_ids: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: Optional[bool] = None,
            streamer: Optional["BaseStreamer"] = None,
            **kwargs,
            # signature of GenerationMixin.generate() in Transformers==4.28.1, with inputs<=>input_ids
    ) -> torch.LongTensor:
        """
        Model is expected to generate gadget tags.
        Whenever a gadget tag is generated, the gadget is called, 
        and the output is fed back into the model inside of an output tag.

        Final result is expected to be in result tag.

        Currently the function only supports single input (no batch).

        Returns:
            full_output: Full structured output of the model, including gadget, output, and result tags.
            result: Final result of the model, or None if not found.
        """

        stopping_criteria = transformers.generation.StoppingCriteriaList([StopAfterGadgetCall(self.tokenizer)])

        if kwargs is None:
            kwargs = {}

        if isinstance(input_ids, str):
            input_ids = self.tokenizer(input_ids, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)

        running_gadgets: dict[str, Gadget] = {g.gadget_id(): g for g in self.enabled_gadgets}

        max_tokens = None
        min_tokens = None

        if "max_length" in kwargs:
            max_length = kwargs.pop("max_length")
            if max_length is not None:
                max_tokens = max_length - input_ids.shape[-1]
        if "min_length" in kwargs:
            min_length = kwargs.pop("min_length")
            if min_length is not None:
                min_tokens = min_length - input_ids.shape[-1]
        if "max_new_tokens" in kwargs:
            max_tokens = kwargs.pop("max_new_tokens")
        if "min_new_tokens" in kwargs:
            min_tokens = kwargs.pop("min_new_tokens")

        if max_tokens is None:
            max_tokens = self.default_max_tokens

        last_num_total_tokens: int | None = None
        total_output_str: str = ""
        result_str = None

        while True:
            total_output_encoded = self.tokenizer(text_target=total_output_str,
                                                  add_special_tokens=False,
                                                  return_tensors="pt").input_ids.to(self.device).to(torch.long)

            num_total_tokens = total_output_encoded.shape[-1]
            if last_num_total_tokens is not None and num_total_tokens <= last_num_total_tokens:
                break
            last_num_total_tokens = num_total_tokens

            if num_total_tokens + 2 >= max_tokens:
                break

            if max_tokens is not None:
                kwargs["max_new_tokens"] = max_tokens - num_total_tokens
            if min_tokens is not None:
                kwargs["min_new_tokens"] = max(0, min_tokens - num_total_tokens)

            decoder_input_ids = torch.cat([
                torch.tensor(self.config.decoder_start_token_id, dtype=torch.long).to(self.device).reshape(1, 1),
                total_output_encoded
            ], dim=-1)

            model_output: transformers.utils.ModelOutput
            generate_cls = T5ForConditionalGeneration
            model_output = generate_cls.generate(
                self,
                input_ids=input_ids,
                stopping_criteria=stopping_criteria,
                decoder_input_ids=decoder_input_ids,
                # **kwargs
                **{k: v for k, v in kwargs.items() if k not in ["labels"]},
            )[0]  # TODO This does not work in batch mode
            # which occurs in evaluation during training

            # model.generate() outputs starts with decoder_input_ids
            total_output_str = self.tokenizer.decode(model_output,
                                                     skip_special_tokens=True,
                                                     spaces_between_special_tokens=False)
            try:
                doc = bs4.BeautifulSoup(total_output_str, features="html.parser")
            except Exception as e:
                warnings.warn(f"Failed to parse model output: {e}")
                continue

            gadget_tags: list[bs4.Tag] = doc.find_all(GADGET_TAG)
            evaluated_something = False
            for gadget_tag_input in gadget_tags:
                next_el = gadget_tag_input.next_sibling
                while next_el is not None and isinstance(next_el, bs4.NavigableString) and next_el.strip() == "":
                    # skip whitespace
                    next_el = next_el.next_sibling
                if isinstance(next_el, bs4.Tag) and next_el.name == OUTPUT_TAG:
                    # already evaluated this gadget tag
                    continue
                evaluated_something = True
                gadget_input = gadget_tag_input.get_text()
                gadget_id = gadget_tag_input.get("id", None)
                gadget = running_gadgets.get(gadget_id, None)
                if gadget is None:
                    gadget_output = f"ERROR: Gadget '{gadget_id}' not found"
                else:
                    gadget_output = gadget(gadget_input)

                gadget_tag_output = doc.new_tag(OUTPUT_TAG)
                gadget_tag_output.string = gadget_output
                gadget_tag_input.insert_after(gadget_tag_output)
                gadget_tag_input.insert_after("\n")
                gadget_tag_output.insert_after("\n")

            if evaluated_something:
                # replace total_output_str with the evaluated version
                total_output_str = str(doc)

            width = 80
            print(" PARTIAL MODEL OUTPUT ".center(width, "="))
            print(total_output_str)
            print("=" * width)

            output_tensor = self.tokenizer.encode(total_output_str,
                                                  return_tensors="pt",
                                                  add_special_tokens=True).to(self.device)

            # Commented things violate the generate() interface and may cause trouble in future versions:

            # if doc.find(RESULT_TAG) is not None:
            #     result_str = doc.find_all(RESULT_TAG)[-1].get_text()
            #     result_tensor = self.tokenizer(result_str, return_tensors="pt", add_special_tokens=False).input_ids

        # if return_as_str:
        #     if return_result:
        #         return total_output_str, result_str
        #     return total_output_str
        #
        # if return_result:
        #     return output_tensor, result_tensor

        return output_tensor


class StepwiseGenerator(T5ForConditionalGeneration, GadgetAssist):

    @torch.no_grad()
    def generate(self,
                 input_ids: Optional[torch.Tensor] = None,
                 generation_config: Optional[GenerationConfig] = None,
                 logits_processor: Optional[LogitsProcessorList] = None,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
                 synced_gpus: Optional[bool] = None,
                 streamer: Optional["BaseStreamer"] = None,
                 **kwargs) -> torch.LongTensor:
        # PerSentence generators decode outputs per reasoning step (~per sentence).
        # After each reasoning step, encode newly-generated output and generate the following step.
        # Once the model generates the <result> tag, terminate.
        expected_max_length: Optional[int] = kwargs.get("max_new_tokens", None)  # max_new_tokens takes precendese
        if expected_max_length is not None:
            expected_max_length = input_ids.shape[-1] + kwargs["max_new_tokens"]
        else:
            expected_max_length = kwargs.get("max_length", None)

        output_step = ""
        output_ids = None
        extended_input_ids = input_ids.clone()
        kwargs["steps_mask"] = torch.zeros_like(input_ids)

        step_i = 0

        # the length of suffix and prefix special tokens differ among models, we assume a single (trailing) <s> token
        assert len(self.tokenizer(output_step).input_ids) == 1

        # generated output does not contain the result -> encode intermediate output and continue in generation
        while bs4.BeautifulSoup(output_step, features="html.parser").find(RESULT_TAG) is None:
            prev_step_ids = self.tokenizer(output_step, return_tensors="pt").input_ids.to(self.device)

            # remove trailing special tokens -- we assume a single trailing token here (asserted above)
            extended_input_ids = torch.hstack([extended_input_ids[:, :-1], prev_step_ids])
            kwargs["steps_mask"] = torch.hstack([kwargs["steps_mask"][:, :-1], torch.full_like(prev_step_ids, step_i)])

            if expected_max_length is not None and extended_input_ids.shape[-1] + 2 >= expected_max_length:
                logger.warning("Generation exceeded given max_length, without generating <result>.")
                break

            kwargs["attention_mask"] = torch.ones_like(extended_input_ids)  # manually rearrange attention mask

            output_ids = super().generate(extended_input_ids, generation_config, logits_processor, stopping_criteria,
                                          prefix_allowed_tokens_fn, synced_gpus, streamer, **kwargs)

            output_step = self.tokenizer.batch_decode(output_ids,
                                                      skip_special_tokens=True,
                                                      spaces_between_special_tokens=False)[0]  # assumes no batching
            if not output_step.strip():
                logger.warning("Generated empty step -> terminating generation to avoid cycling.")
                break

            print("Output step: %s" % output_step)
            step_i += 1

        # collect complete generation output and remove the input segment
        if output_ids is None:
            return torch.rand((1, 0))

        generated_output_ids = torch.hstack([extended_input_ids[:, :-1], output_ids])
        generated_output_ids = generated_output_ids[:, input_ids.shape[0] + 1:]  # we assume batch_size==1 here

        return generated_output_ids

    def forward(self,
                *args,
                steps_mask: Optional[torch.LongTensor] = None,  # used in training, not used in generation
                **kwargs) -> Union[Tuple[torch.FloatTensor], ModelOutput]:
        # override of default encoder's forward with the one wrapping the aggregation
        class StepwiseEncoder(self.encoder.__class__):
            steps_mask: Optional[torch.LongTensor] = None
            superclass: Optional[type] = None

            def forward(self,
                        steps_mask: Optional[torch.LongTensor] = None,
                        input_ids=None,
                        attention_mask=None,
                        encoder_hidden_states=None,
                        encoder_attention_mask=None,
                        inputs_embeds=None,
                        head_mask=None,
                        cross_attn_head_mask=None,
                        past_key_values=None,
                        use_cache=None,
                        output_attentions=None,
                        output_hidden_states=None,
                        return_dict=None):
                if steps_mask is not None:
                    self.steps_mask = steps_mask
                orig_outputs = self.superclass.forward(self, input_ids, attention_mask, encoder_hidden_states,
                                                       encoder_attention_mask, inputs_embeds, head_mask,
                                                       cross_attn_head_mask, past_key_values, use_cache,
                                                       output_attentions, output_hidden_states, return_dict)
                # test:
                # orig_encoder_output.last_hidden_state = torch.tensor([[[0, 0], [0, 0], [0, 1], [1, 1],
                #                                                        [2, 2], [2, 2], [0, 1]]], dtype=torch.float)
                # self.steps_mask = torch.tensor([[0, 1, 1, 1, 2, 2, 0]], dtype=torch.int64)
                # expected_sum = torch.tensor([[[0, 1], [1, 2], [4, 4]]], dtype=torch.float)
                # expected_avg = torch.tensor([[[0, 0.5], [0.33, 0.66], [2, 2]]], dtype=torch.float)

                batch_size, input_size, emb_size = orig_outputs.last_hidden_state.shape

                steps_mask_idx = self.steps_mask[:, :input_size].unsqueeze(-1).expand(-1, -1, emb_size).clone()
                unique_step_idx, idx_count = self.steps_mask.unique(return_counts=True)

                steps_embeddings_sum = torch.zeros((orig_outputs.last_hidden_state.size(0),
                                                    unique_step_idx.max()+1,  # robust to missing mask values
                                                    # input_size,
                                                    orig_outputs.last_hidden_state.size(-1)),
                                                   dtype=torch.float, device=steps_mask_idx.device)
                steps_embeddings_sum.scatter_add_(dim=1,
                                                  index=steps_mask_idx,
                                                  src=orig_outputs.last_hidden_state)

                # test per-group sum: assert steps_embeddings_sum.isclose(expected_sum, rtol=2e-2).all()

                # normalize per-step encodings
                # due to the per-sample counts, this requires for-loop
                for batch_i in range(batch_size):
                    num_embs = torch.tensor([(self.steps_mask[batch_i] == val).sum()
                                             for val in range(unique_step_idx.max() + 1)],
                                            device=self.steps_mask.device)
                    # note: sentence-transformers also rescale (clamp) sum mask to min=1e-9 (see models/Pooling.py)

                    # vals_sum is a zero denominator if steps do not contain any embeddings
                    steps_embeddings_sum[batch_i] /= num_embs.unsqueeze(-1).expand_as(steps_embeddings_sum[batch_i])
                    # drop embeddings of steps containing no embeddings:
                    steps_embeddings_sum[batch_i][~num_embs.bool()] = torch.zeros((emb_size,),
                                                                                  device=steps_embeddings_sum.device)
                    # test: assert steps_embeddings_sum[sample_i].isclose(expected_avg[sample_i], rtol=2e-2).all()

                    num_steps = sum(num_embs.bool())  # number of reasoning steps

                    # Insert the encodings of reasoning steps after the encodings of the input tokens
                    if (self.steps_mask[batch_i] == 1).any():
                        # -> argmin reimplementation retrieving first non-input (=stepwise) position among encodings
                        all_positions = torch.arange(self.steps_mask.size(-1), device=self.steps_mask.device)
                        steps_begin_pos = all_positions[self.steps_mask[batch_i] == 1].min()

                        # replace with only the non-zero embeddings that fit to the context of existing ones
                        replaced_steps = min(len(orig_outputs.last_hidden_state[batch_i][steps_begin_pos:]), num_steps)
                        orig_outputs.last_hidden_state[batch_i][steps_begin_pos:steps_begin_pos + replaced_steps] = \
                            steps_embeddings_sum[batch_i][num_embs.bool()][:replaced_steps]

                        # we keep other encodings as-is, but we hide them with attention mask
                        attention_mask[batch_i][:steps_begin_pos + num_steps] = 1
                        attention_mask[batch_i][steps_begin_pos + num_steps:] = 0
                        # assert that the last attended position contains encoding of last reasoning step  # no test
                        assert (orig_outputs.last_hidden_state[batch_i][attention_mask[batch_i].bool()][-1] ==
                                steps_embeddings_sum[batch_i][num_steps - 1]).all()

                # orig_encoder_output.last_hidden_state and encoder_kwargs["attention_mask"] are adjusted
                return orig_outputs

        orig_encoder = copy.deepcopy(self.encoder.__class__)
        self.encoder.__class__ = StepwiseEncoder
        if self.encoder.superclass is None:
            self.encoder.superclass = orig_encoder
        if steps_mask is not None:
            self.encoder.steps_mask = steps_mask

        # print("Superclass method: %s" % str(super().forward))
        # lm_output: ModelOutput = super(self.__class__, self).forward(self, *args, **kwargs)
        lm_output: ModelOutput = super().forward(*args, **kwargs)
        lm_output.loss = lm_output.loss  # TODO: perspectively, regularize the steps encodings
        return lm_output


def gadget_assisted_model(model_class: transformers.PreTrainedModel):
    class GadgetAssistedModel(GadgetAssist, model_class):
        pass

    return GadgetAssistedModel


def stepwise_gadget_model(model_class: transformers.PreTrainedModel):
    # class StepwiseGeneratorModel(StepwiseGenerator):
    #
    #     def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
    #         super().__init__(config, *inputs, **kwargs)
    #         self.superclass = model_class

    return StepwiseGenerator


str_prompt = "Write xml tag gadget id attribute id='calculator' and fill '2 + 2' inside. "
str_let_me_think = "Let me think about it"
str_gadget_usage = f"<{GADGET_TAG} id='calculator'>2+2</{GADGET_TAG}>"
str_gadget_output = f"<{OUTPUT_TAG}>4</{OUTPUT_TAG}>"
str_result = "129818"
str_result_with_tag = f"Final answer is <{RESULT_TAG}>{str_result}</{RESULT_TAG}>."


def test_generate_check_outputs(
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        mocked_model_outputs: list[str],
        expected_full_outputs: list[str],
        expected_result: str | None,
        enabled_gadgets: list[Gadget],
) -> bool:
    assert isinstance(model, GadgetAssist)

    model.prepare_for_generate(
        tokenizer,
        enabled_gadgets=enabled_gadgets,
    )

    mocked_model_outputs_tokenized = [
        tokenizer(string, return_tensors="pt").input_ids
        for string in mocked_model_outputs
    ]

    with unittest.mock.patch("transformers.GenerationMixin.generate") as patched_model:
        patched_model.side_effect = mocked_model_outputs_tokenized
        full_output, result = model.generate(
            str_prompt,
            return_result=True,
            return_as_str=True,
            max_length=400,
            num_beams=3,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            remove_invalid_values=True,
        )

    expected_full_output = bs4.BeautifulSoup(" ".join(expected_full_outputs), features="html.parser").prettify()
    full_output = bs4.BeautifulSoup(full_output, features="html.parser").prettify()

    output_matches = _compare_strings_ignore_whitespace(full_output, expected_full_output)

    if expected_result is None:
        result_matches = result is None
    else:
        result_matches = _compare_strings_ignore_whitespace(result, expected_result)

    is_correct = output_matches and result_matches
    return is_correct


def _compare_strings_ignore_whitespace(str1: str, str2: str) -> bool:
    return " ".join(str1.split()) == " ".join(str2.split())


TESTS = [
    {
        "mocked": [str_result_with_tag],
        "expected_outputs": [str_result_with_tag],
        "expected_result": str_result,
    },
    {
        "mocked": [str_let_me_think, str_result_with_tag],
        "expected_outputs": [str_let_me_think, str_result_with_tag],
        "expected_result": str_result,
    },
    {
        "mocked": [str_gadget_usage, str_result_with_tag],
        "expected_outputs": [str_gadget_usage, str_gadget_output, str_result_with_tag],
        "expected_result": str_result,
    },
    {
        "mocked": [str_gadget_usage, str_gadget_usage, str_result_with_tag],
        "expected_outputs": [str_gadget_usage, str_gadget_output, str_gadget_usage, str_gadget_output,
                             str_result_with_tag],
        "expected_result": str_result,
    },
    {
        "mocked": [str_gadget_usage + str_gadget_usage, str_result_with_tag],
        "expected_outputs": [str_gadget_usage + str_gadget_output + str_gadget_usage + str_gadget_output,
                             str_result_with_tag],
        "expected_result": str_result,
    }
]


def test_generate_with_gadgets():
    model_name = "salesforce/codet5-small"
    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
    model = gadget_assisted_model(transformers.T5ForConditionalGeneration).from_pretrained(model_name)

    for i, test in enumerate(TESTS):
        assert test_generate_check_outputs(
            model,
            tokenizer,
            test["mocked"],
            test["expected_outputs"],
            test["expected_result"],
            enabled_gadgets=[Calculator()],
        )


if __name__ == "__main__":
    test_generate_with_gadgets()
