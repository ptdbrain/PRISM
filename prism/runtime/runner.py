"""Simple runtime execution helpers for PRISM."""

from __future__ import annotations

import torch


def run_forward(model, hidden_size: int = 8) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(torch.randn(1, hidden_size))


def run_generate(
    model,
    tokenizer,
    *,
    prompt: str = "Hello",
    max_new_tokens: int = 16,
):
    model.eval()
    tensor = next(model.parameters(), None)
    if tensor is None:
        tensor = next(model.buffers(), None)
    device = tensor.device if tensor is not None else torch.device("cpu")

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)
