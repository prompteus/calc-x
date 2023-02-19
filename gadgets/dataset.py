from typing import Iterable, Iterator

import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
import transformers

import gadgets.datatypes
import gadgets.markup


class DataSampler(IterableDataset):

    def __init__(
        self,
        generator: Iterable[gadgets.datatypes.Example],
        tokenizer: transformers.PreTrainedTokenizer,
        max_samples: int | None = None,
        add_result_sentence: bool = False,
    ) -> None:
        self.generator = generator
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.add_result_sentence = add_result_sentence

    def encode(self, example: gadgets.datatypes.Example) -> dict[str, torch.Tensor]:

        inputs = self.tokenizer(example.prompt, truncation=True)

        labels_soup = gadgets.markup.to_model_markup(
            example=example, 
            eos_token_after_gadgets=self.tokenizer.eos_token,
            add_result_sentence=self.add_result_sentence
        )
        labels = self.tokenizer(str(labels_soup), truncation=True)

        return {"input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": labels.input_ids}

    def __iter__(self) -> Iterator[T_co]:
        worker_info = torch.utils.data.get_worker_info()
        samples_left = self.max_samples

        for i, (input_text, gadget_inputs, target_text) in enumerate(self.generator):
            if worker_info is not None:
                # multi-gpu DataParallel
                if (i - worker_info.id) % worker_info.num_workers == 0:
                    # sample modulo number of all workers match this worker rank
                    yield self.encode(input_text, gadget_inputs, target_text)
            else:
                # single-GPU sampling
                yield self.encode(input_text, gadget_inputs, target_text)

            if self.max_samples is not None:
                samples_left -= 1

                if samples_left <= 0:
                    break

    def __getitem__(self, index) -> T_co:
        raise ValueError("We shouldn't ever get here, but need to override this")
