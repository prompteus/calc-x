import abc
from typing import Iterable, Tuple, List, Iterator, Callable, Dict

import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
from transformers import BatchEncoding, PreTrainedModel, DataCollatorForSeq2Seq

from gadget import Gadget


class DataSampler(IterableDataset):
    generator: Iterable[Tuple[str, List[str], str]]  # Tuples[input_text, List[gadget_inputs], target_text]

    def __init__(self,
                 generator: Iterable[Tuple[str, List[str], str]],
                 gadget: Gadget,
                 tokenizer: Callable[[str], BatchEncoding],
                 model: PreTrainedModel,
                 batch_size: int = 8):
        self.generator = generator
        self.gadget = gadget
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.collator = DataCollatorForSeq2Seq(self.tokenizer, model, pad_to_multiple_of=8)

    def _insert_eos_after_gadgets(self, input_chain: str) -> str:
        return input_chain.replace(Gadget.get_gadget_request_eos(),
                                   Gadget.get_gadget_request_eos() + self.tokenizer.eos_token)

    def _encoded_sample_from_inputs(self, input_text: str, chain: str, target_text: str) -> Dict[str, torch.Tensor]:

        input_chain_adjusted = self._insert_eos_after_gadgets(chain)

        inputs = self.tokenizer(f"{input_text} {input_chain_adjusted}", truncation=True)
        target_ids = self.tokenizer(target_text, truncation=True).input_ids
        return {"input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "labels": target_ids}

    def __iter__(self) -> Iterator[T_co]:
        worker_info = torch.utils.data.get_worker_info()

        inputs_batch = []

        for i, (input_text, gadget_inputs, target_text) in enumerate(self.generator):
            if len(inputs_batch) <= self.batch_size:
                inputs_batch.append(self._encoded_sample_from_inputs(input_text, gadget_inputs, target_text))
            else:
                if worker_info is not None:
                    # multi-gpu DataParallel
                    if (i - worker_info.id) % worker_info.num_workers == 0:
                        # sample modulo number of all workers match this worker rank
                        yield self.collator(inputs_batch)
                        inputs_batch = []
                else:
                    # yield batch with residual inputs
                    yield self.collator(inputs_batch)
                    inputs_batch = []
            # yield batch with residual inputs
            yield self.collator(inputs_batch)

    def __getitem__(self, index) -> T_co:
        raise ValueError("We shouldn't ever get here, but need to override this")
