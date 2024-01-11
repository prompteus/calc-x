import random
import re
from dataclasses import dataclass
from dataclasses import dataclass
from typing import Any, Optional, Union

import bs4
import gadgets
from gadgets.markup import GADGET_TAG, OUTPUT_TAG, RESULT_TAG

import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class StepwiseCollatorForSeq2Seq:
    # Custom collator takes care of restricting the attention mask of [step] tokens to the per-step segments

    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    step_eos_token_id: int
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = ((max_label_length + self.pad_to_multiple_of - 1)
                                    // self.pad_to_multiple_of
                                    * self.pad_to_multiple_of)

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        paired_features = None
        if "paired_input_ids" in features[0]:  # paired samples are only present in training
            paired_samples = [{"input_ids": sample["paired_input_ids"],
                               "attention_mask": sample["paired_attention_mask"]} for sample in features]

            paired_features = self.tokenizer.pad(paired_samples,
                                                 padding=self.padding,
                                                 max_length=self.max_length,
                                                 pad_to_multiple_of=self.pad_to_multiple_of,
                                                 return_tensors=return_tensors)

            paired_features["attention_mask"] = self.construct_extended_attention_mask(paired_features.input_ids,
                                                                                       paired_features.attention_mask)

        orig_samples = [{f: val for f, val in sample.items() if f not in ("paired_input_ids", "paired_attention_mask")}
                        for sample in features]
        features = self.tokenizer.pad(orig_samples,
                                      padding=self.padding,
                                      max_length=self.max_length,
                                      pad_to_multiple_of=self.pad_to_multiple_of,
                                      return_tensors=return_tensors)
        features["attention_mask"] = self.construct_extended_attention_mask(features.input_ids,
                                                                            features.attention_mask)
        if paired_features is not None:
            features["paired_input_ids"] = paired_features["input_ids"]
            features["paired_attention_mask"] = paired_features["attention_mask"]

        # prepare decoder_input_ids
        if (labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

    def construct_extended_attention_mask(self,
                                          input_ids: torch.LongTensor,
                                          orig_attn_mask: torch.LongTensor) -> torch.LongTensor:
        # get_extended_attention_mask accepts a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        extended_attn_mask = orig_attn_mask[:, :, None].repeat(1, 1, orig_attn_mask.shape[-1])
        for sample_i, sample_eos_pos in enumerate(orig_attn_mask.sum(1)):
            # correction of trailing segment of augmented attention mask -> above, that gets repeated with ones
            extended_attn_mask[sample_i, :, sample_eos_pos:] = 0

        # zero out attentions of [step] tokens directed *to* the tokens outside each step
        step_token_positions = input_ids == self.step_eos_token_id  # identify step positions
        step_token_positions[:, 0] = True  # set first tokens as delimiters for first reasoning step
        for sample_i in range(input_ids.shape[0]):  # iteration over batch
            step_tokens_pos = step_token_positions[sample_i].argwhere()
            if not step_tokens_pos.any():
                # no step-eos token in the input -- first prediction turn
                continue
            # iterate over tuples of [step] tokens' indexes (=step separators)
            step_ranges = step_tokens_pos[:, 0].unfold(0, 2, 1)

            # exclude [step] positions from attention of all other tokens
            extended_attn_mask[sample_i] -= step_token_positions[sample_i].long()

            # exclude attentions of [step] tokens outside their respective steps
            for step_range in step_ranges:  # iteration over steps ranges
                extended_attn_mask[sample_i, step_range[1], :step_range[0]] = 0  # tokens before the step
                extended_attn_mask[sample_i, step_range[1], step_range[1]:] = 0  # tokens after the step

        return extended_attn_mask


def separate_chain_to_steps(chain: str) -> tuple[list[str], str]:
    """
    heuristically separates input chain into a list of reasoning steps.
    :param chain: Original chain
    :return: A tuple: (list of steps contained in the chain, used separator)
    """
    sep = ". " if ". " in chain else ".\n" if ".\n" in chain else "\n"
    steps = [step.strip() + sep for step in chain.split(sep)]

    return steps, sep


class StepPermuter:
    numeral_re = re.compile(r"\d+(?:\.\d+)?")

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def _replace_num(self, number: int | float) -> str:
        # replace with a number of a similar scale as the original
        is_decimal = "." in str(number)
        number_length = len(str(number).split(".")[0]) if is_decimal else len(str(number))

        output_number = random.randint(1, 10 ** (number_length + 1))
        if is_decimal:
            output_number += random.random()

        return str(output_number)

    @staticmethod
    def _replace_all(text: str, replacement_map: dict[str, str]) -> str:
        out_text = text
        for orig, repl in replacement_map.items():
            out_text = out_text.replace(orig, repl)
        return out_text

    def _permute_numbers_all_steps(self, sample_steps: list[str]) -> list[str]:
        calculator = gadgets.gadget.Calculator()

        # permute numbers in the question (first step)
        first_step_numerals = self.numeral_re.findall(sample_steps[0])
        replaces_map = {num: self._replace_num(num) for num in first_step_numerals}

        out_steps = [self._replace_all(sample_steps[0], replaces_map)]

        # for the reasoning steps, replace the inputs according to the input question
        # + <outputs> of previous steps computed from the already-altered inputs
        # both replacements are performed after we recompute the new_gadget_output
        for step in sample_steps[1:]:
            step_altered = self._replace_all(step, replaces_map)

            doc = bs4.BeautifulSoup(step_altered, features="html.parser")
            doc_orig = bs4.BeautifulSoup(step_altered, features="html.parser")
            gadget_tags: list[bs4.Tag] = doc.find_all(gadgets.markup.GADGET_TAG)
            output_tags: list[bs4.Tag] = doc_orig.find_all(gadgets.markup.OUTPUT_TAG)

            for gadget_tag_input, orig_output in zip(gadget_tags, output_tags):
                # next_el = gadget_tag_input.next_sibling
                # next_out = orig_output.next_sibling
                # # skip whitespaces before the call
                # while next_el is not None and isinstance(next_el, bs4.NavigableString) and next_el.get_text().strip() == "":
                #     next_el = next_el.next_sibling
                # while next_out is not None and isinstance(next_out, bs4.NavigableString) and next_out.get_text().strip() == "":
                #     next_out = orig_output.next_sibling
                gadget_id = gadget_tag_input.get("id", None)
                if gadget_id != calculator.gadget_id():
                    # we extract permuted numerals only from Calculator computations
                    continue

                gadget_input = gadget_tag_input.get_text()
                orig_gadget_output = orig_output.get_text()

                new_gadget_output = calculator(gadget_input)
                replaces_map[orig_gadget_output] = new_gadget_output.split(" = around")[0]

            out_steps.append(self._replace_all(step, replaces_map))

        return out_steps

    def permute_all_steps(self, sample_steps: list[str]) -> list[str]:
        # step_str = self.tokenizer.batch_decode(input_ids)
        output_str = self._permute_numbers_all_steps(sample_steps)

        # TODO: permute numbers in steps
        return output_str
