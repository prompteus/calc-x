from __future__ import annotations

import os
import pathlib

import peft
import torch
import transformers


def add_new_token(
    new_token: str | transformers.AddedToken,
    is_special: bool,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    init_with: list[str] | None = None,
) -> None:
    if new_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([new_token], special_tokens=is_special)
    vocab = tokenizer.get_vocab()

    if isinstance(new_token, transformers.AddedToken):
        new_token_str = new_token.content
    else:
        new_token_str = new_token

    if tokenizer.is_fast:
        new_token_id = vocab[new_token_str]
    else:
        new_token_id = tokenizer.added_tokens_encoder[new_token_str]

    # extend the mapping of the model with the new token
    model.resize_token_embeddings(len(tokenizer))

    # if we're getting init_with, we're initializing embedding of the new token
    # to mean of the embeddings of the tokens in init_with
    if init_with is not None and len(init_with) > 0:
        with torch.no_grad():
            embeddings: torch.nn.Module = model.get_input_embeddings()
            assert isinstance(embeddings, torch.nn.Embedding)
            device = next(embeddings.parameters()).device
            init_token_ids = [vocab[init_token] for init_token in init_with]
            init_token_ids = torch.tensor(init_token_ids, dtype=torch.long, device=device)
            init_weights = embeddings(init_token_ids)
            init_weights = torch.mean(init_weights, dim=0)
            new_token_id = torch.tensor(new_token_id, dtype=torch.long, device=device)
            embedding: torch.Tensor = embeddings(new_token_id)
            embedding.copy_(init_weights)


# Source: https://github.com/huggingface/peft/issues/96
class SavePeftModelCallback(transformers.TrainerCallback):
    def on_save(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ) -> transformers.TrainerControl:
        
        model = kwargs["model"]
        if not isinstance(model, peft.PeftModel):
            return control

        checkpoint_folder = pathlib.Path(args.output_dir) / f"{transformers.trainer_utils.PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        model.save_pretrained(checkpoint_folder)

        pytorch_model_path = checkpoint_folder / "pytorch_model.bin"
        if pytorch_model_path.exists():
            os.remove(pytorch_model_path)

        return control
