from dataclasses import dataclass
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class StepwiseCollatorForSeq2Seq:
    # def __init__(self, tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, step_eos_token_id: int):
    #     super().__init__(tokenizer, model)
    #     self.step_eos_token_id = step_eos_token_id

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

        features = self.tokenizer.pad(features,
                                      padding=self.padding,
                                      max_length=self.max_length,
                                      pad_to_multiple_of=self.pad_to_multiple_of,
                                      return_tensors=return_tensors)
        features["attention_mask"] = self.construct_extended_attention_mask(features.input_ids,
                                                                            features.attention_mask)
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
        # reshape attention_mask to (BS, 1, max_seq_len, max_seq_len)
        # full_extended_attn_mask = orig_attn_mask[:, None, :, None].expand(-1, -1, -1, orig_attn_mask.shape[-1])
        full_extended_attn_mask = orig_attn_mask[:, None, :, None].repeat(1, 1, 1, orig_attn_mask.shape[-1])
        # zero out attentions of mask tokens onto the tokens outside each step
        step_token_positions = input_ids == self.step_eos_token_id
        for sample_i in range(input_ids.shape[0]):  # iteration over batch
            step_tokens_pos = step_token_positions[sample_i].argwhere()
            if not step_tokens_pos.any():
                # no step-eos token in the input -- first prediction turn
                continue
            step_ranges = step_tokens_pos[:, 0].unfold(0, 2, 1)
            for step_range in step_ranges:  # iteration over steps ranges
                full_extended_attn_mask[sample_i, 0, step_range[1], :step_range[0]] = 0  # tokens before the step
                full_extended_attn_mask[sample_i, 0, step_range[1], step_range[1]:] = 0  # tokens after the step

        # from transformers.modeling_utils.get_extended_attention_mask:
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        full_extended_attn_mask = full_extended_attn_mask.to(dtype=self.model.dtype)  # fp16 compatibility
        full_extended_attn_mask = (1.0 - full_extended_attn_mask) * torch.finfo(self.model.dtype).min
        # TODO: check that this propagates correctly: see attn_weights on modeling_t5:L565 (?)
        return full_extended_attn_mask
