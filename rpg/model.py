"""Model loading and activation capture.

Activations follow the paper's repo (utils/probes.py): the output of each decoder layer's
`post_attention_layernorm`, i.e. the normalized residual stream entering the MLP.
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(mcfg: dict):
    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = device()
    if dev != "cuda":
        # transformers' threaded weight loader can hang on macOS; load sequentially there.
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["hf_id"], dtype=_dtype(mcfg.get("dtype", "bfloat16")),
        device_map=dev, **mcfg.get("load_kwargs", {}),
    ).eval()
    return model, tok


def _dtype(name: str):
    """"auto" keeps a checkpoint's own format (gpt-oss: MXFP4 experts, ~14 GB instead of ~42 GB in bf16)."""
    return "auto" if name == "auto" else getattr(torch, name)


def decoder_layers(model):
    return model.model.layers


def probe_layers(model, stride: int) -> list[int]:
    n = len(decoder_layers(model))
    return list(range(0, n, stride))


@torch.no_grad()
def capture(model, tok, batch_ids: list[list[int]], layers: list[int], positions: list[list[int]]) -> dict[int, list[torch.Tensor]]:
    """Forward a batch of token-id lists (left-padded) and return, for each layer, a list (one per
    sequence) of (len(positions[i]), D) float32 CPU tensors at the requested unpadded positions."""
    L = max(len(x) for x in batch_ids)
    pad = tok.pad_token_id
    ids = torch.tensor([[pad] * (L - len(x)) + x for x in batch_ids], device=model.device)
    mask = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in batch_ids], device=model.device)
    offsets = [L - len(x) for x in batch_ids]

    grabbed: dict[int, torch.Tensor] = {}
    hooks = []
    for li in layers:
        def hook(_m, _i, out, li=li):
            grabbed[li] = out
        hooks.append(decoder_layers(model)[li].post_attention_layernorm.register_forward_hook(hook))
    try:
        # Base model only: skips the LM head (vocab-sized logits are the biggest activation for gpt-oss).
        model.model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    res = {}
    for li in layers:
        h = grabbed[li]
        res[li] = [h[b, [offsets[b] + p for p in positions[b]]].float().cpu() for b in range(len(batch_ids))]
    return res
