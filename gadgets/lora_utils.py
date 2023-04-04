from __future__ import annotations

from typing import Any, Iterable, NamedTuple

import loralib
import torch


class ModuleInfo(NamedTuple):
    attr: str
    path: str
    module: torch.nn.Module
    parent: torch.nn.Module


def all_module_parent_pairs(model: torch.nn.Module) -> Iterable[ModuleInfo]:
    for path, module in model.named_modules():
        if path == "":
            assert module is model
            yield ModuleInfo("", "", model, parent=None)
            continue

        *parent_path, attr = path.split(".")
        parent = model
        for child in parent_path:
            parent = getattr(parent, child)
        yield ModuleInfo(attr, path, module, parent)


def patch_linears_with_lora(
        model: torch.nn.Module,
        only_if_name_contains: Iterable[str] | None = ("attn.k_proj", "attn.q_proj", "attn.v_proj"),
        **lora_kwargs: Any,
):
    modules = list(all_module_parent_pairs(model))
    if only_if_name_contains is not None:
        only_if_name_contains = set(only_if_name_contains)

    was_training = model.training

    for module_info in modules:
        module = module_info.module

        if isinstance(module, loralib.LoRALayer):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if only_if_name_contains is not None:
            if not any(string in module_info.path for string in only_if_name_contains):
                continue

        was_module_training = module.training
        replacement = loralib.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            merge_weights=True,
            **lora_kwargs
        )
        replacement = replacement.to(module.weight.device)
        replacement.eval()

        with torch.no_grad():
            replacement.weight[:] = module.weight
            if module.bias is not None:
                replacement.bias[:] = module.bias

            test_input = torch.rand(2, module.in_features).to(module.weight.device)
            assert torch.allclose(module(test_input), replacement(test_input))

        if was_module_training:
            replacement.train()

        setattr(module_info.parent, module_info.attr, replacement)
        modules = list(all_module_parent_pairs(model))

    if was_training:
        model.train()