from __future__ import annotations

import logging
import unittest.mock
import warnings
from typing import Optional, Callable, List, Union, Tuple

import bs4
import torch
import transformers
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, LogitsProcessorList, StoppingCriteriaList, T5ForConditionalGeneration, \
    Trainer
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG as HEAD_MASK_WARNING_MSG

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
        output_tensor = self.tokenizer.encode(total_output_str,
                                              return_tensors="pt",
                                              add_special_tokens=True).to(self.device)
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

            kwargs["decoder_input_ids"] = torch.cat([
                torch.tensor(self.config.decoder_start_token_id, dtype=torch.long).to(self.device).reshape(1, 1),
                total_output_encoded
            ], dim=-1)

            model_output: transformers.utils.ModelOutput
            generate_cls = T5ForConditionalGeneration
            model_output = generate_cls.generate(
                    self,
                    input_ids=input_ids,
                    stopping_criteria=stopping_criteria,
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
    step_token_id: Optional[int] = None

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
        print("Input query: %s" % self.tokenizer.decode(input_ids[0]))

        # resolve the maximum generation length: passed parameters get priority
        if kwargs.get("max_new_tokens", None) is not None:
            expected_max_length = input_ids.shape[-1] + kwargs["max_new_tokens"]
        elif kwargs.get("max_length", None):
            expected_max_length = input_ids.shape[-1] + kwargs["max_length"]
        else:
            expected_max_length = input_ids.shape[-1] + self.default_max_tokens

        output_step = ""
        output_ids = None
        extended_input_ids = input_ids.clone()

        # the length of suffix and prefix special tokens differ among models, we assume a single (trailing) <s> token
        assert len(self.tokenizer(output_step).input_ids) == 1

        # generated output does not contain the result -> encode intermediate output and continue in generation
        while bs4.BeautifulSoup(output_step, features="html.parser").find(RESULT_TAG) is None:
            prev_step_ids = self.tokenizer(output_step, return_tensors="pt").input_ids.to(self.device)

            # remove trailing special tokens -- we assume a single trailing token here (asserted above)
            extended_input_ids = torch.hstack([extended_input_ids[:, :-1], prev_step_ids])
            if extended_input_ids.shape[-1] + 2 >= expected_max_length:
                logger.warning("Generation exceeded given max_length, without generating <result>.")
                break

            kwargs["attention_mask"] = torch.ones_like(extended_input_ids)  # manually rearrange attention mask
            if self.step_token_id is not None:
                # exclude attention onto the [step] tokens
                kwargs["attention_mask"][extended_input_ids == self.step_token_id] = 0

            output_ids = super().generate(extended_input_ids, generation_config, logits_processor, stopping_criteria,
                                          prefix_allowed_tokens_fn, synced_gpus, streamer, **kwargs)

            output_step = self.tokenizer.batch_decode(output_ids,
                                                      skip_special_tokens=True,
                                                      spaces_between_special_tokens=False)[0]  # assumes no batching
            if not output_step.strip():
                logger.warning("Generated empty step -> terminating generation to avoid cycling.")
                break

            print("Output step: %s" % output_step)

        # collect complete generation output and remove the input segment
        if output_ids is None:
            return torch.rand((1, 0))

        generated_output_ids = torch.hstack([extended_input_ids[:, :-1], output_ids])
        generated_output_ids = generated_output_ids[:, input_ids.shape[-1] - 1:]  # we assume batch_size==1 here

        return generated_output_ids


class CompressedStepwiseGenerator(StepwiseGenerator):
    trainer: Trainer
    losses_log: dict[str, torch.tensor] = {"train_loss_sim_consistency": torch.tensor(0.),
                                           "train_loss_diff_consistency": torch.tensor(0.),
                                           "train_loss_consistency": torch.tensor(0.)}
    logging_iter = 0

    def forward(self,
                steps_mask: Optional[torch.LongTensor] = None,  # used in training, not used in generation
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                paired_input_ids: Optional[torch.LongTensor] = None,
                paired_attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                decoder_head_mask: Optional[torch.FloatTensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        # override of default encoder's forward with the one wrapping the aggregation
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        loss = torch.tensor(0., device=self.device)

        # Encode if needed (training & first prediction pass)
        # Convert encoder inputs in embeddings if needed
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
            )

            hidden_states = encoder_outputs[0]

            # BEGIN EDIT - per-step regularization
            if paired_input_ids is not None:
                paired_encoder_outputs = self.encoder(
                        input_ids=paired_input_ids,
                        attention_mask=paired_attention_mask,
                        inputs_embeds=inputs_embeds,
                        head_mask=head_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                )
                paired_hidden_states = paired_encoder_outputs[0]

                step_hidden_states = hidden_states[input_ids == self.step_token_id]
                pair_step_hidden_states = paired_hidden_states[paired_input_ids == self.step_token_id]

                if step_hidden_states.shape == pair_step_hidden_states.shape:
                    # we skip the cases where the number of steps differ from reference (matching is unknown)
                    # this occurs for the cases where alternative chain exceeds max_length and original does not
                    sim_cos_loss = torch.nn.CosineEmbeddingLoss()
                    diff_cos_loss = torch.nn.CosineEmbeddingLoss(margin=0.5)
                    # to avoid overloading biases, [step] embeddings are concurrently optimised for both max and min
                    equal_steps_loss = sim_cos_loss(
                            step_hidden_states, pair_step_hidden_states,
                            target=torch.tensor(-1, device=self.device).expand(step_hidden_states.shape[0])
                    )
                    different_steps_loss = diff_cos_loss(
                            step_hidden_states,
                            pair_step_hidden_states.roll(shifts=1, dims=1),
                            target=torch.tensor(1, device=self.device).expand(step_hidden_states.shape[0])
                    )
                    loss = equal_steps_loss + different_steps_loss

                    if self.losses_log["train_loss_sim_consistency"].device != self.device:
                        self.losses_log = {k: v.to(self.device) for k, v in self.losses_log.items()}

                    self.losses_log["train_loss_sim_consistency"] += equal_steps_loss
                    self.losses_log["train_loss_diff_consistency"] += different_steps_loss
                    self.losses_log["train_loss_consistency"] += loss

                self.logging_iter += 1
                if self.logging_iter >= self.trainer.args.logging_steps * self.trainer.args.gradient_accumulation_steps:
                    self.trainer.log({k: (v / self.trainer.args.logging_steps).item()
                                      for k, v in self.losses_log.items()})
                    # TODO: add token loss

                    self.losses_log = {k: torch.tensor(0., device=self.device) for k in self.losses_log.keys()}
                    self.logging_iter = 0

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0],
                    hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                    attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # END EDIT
        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # T5's cross-attention can not tackle with 2d attention masks, so we simply pass it a new one
        flat_attention_mask = (input_ids != self.tokenizer.pad_token_id).long() if input_ids is not None else None
        # Decode
        decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=flat_attention_mask,  # encoder_attention_mask=attention_mask[:, :0, :]
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )
        # note: updating mask = mask[:, :, :0, :]
        # in `position_bias = position_bias + mask` (modeling_t5:L552)
        # solves the problem -> transform attention_mask to a single dim

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss += loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
        )


def gadget_assisted_model(model_class: transformers.PreTrainedModel):
    class GadgetAssistedModel(GadgetAssist, model_class):
        pass

    return GadgetAssistedModel


def stepwise_gadget_model(model_class):
    class StepwiseGeneratorModel(StepwiseGenerator, model_class):
        pass

    return StepwiseGeneratorModel


def stepwise_compressed_gadget_model(model_class: Optional[transformers.PreTrainedModel] = None):
    # class StepwiseGeneratorModel(StepwiseGenerator):
    #
    #     def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
    #         super().__init__(config, *inputs, **kwargs)
    #         self.superclass = model_class

    return CompressedStepwiseGenerator


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
