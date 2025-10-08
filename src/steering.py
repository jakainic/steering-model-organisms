from typing import List, Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _get_device_from_model(model: PreTrainedModel) -> str:
    param_device = next(model.parameters()).device
    if param_device.type != "cpu" or torch.cuda.is_available():
        return str(param_device)
    return "cpu"


def compute_vocab_bias(
    model: PreTrainedModel,
    direction: np.ndarray,
    *,
    scale: float = 1.0,
    device: Optional[str] = None,
) -> torch.Tensor:
    w = model.get_output_embeddings().weight
    dir_t = torch.from_numpy(direction.astype(np.float32))
    target_device = device or _get_device_from_model(model)
    dir_t = dir_t.to(target_device)
    bias = w @ (scale * dir_t)
    return bias


def compute_orthogonal_map_from_tokens(
    src_model: PreTrainedModel,
    src_tokenizer: PreTrainedTokenizerBase,
    tgt_model: PreTrainedModel,
    tgt_tokenizer: PreTrainedTokenizerBase,
    tokens: List[str],
) -> np.ndarray:
    src_emb = src_model.get_input_embeddings().weight.detach().cpu().numpy()
    tgt_emb = tgt_model.get_input_embeddings().weight.detach().cpu().numpy()

    src_ids = [src_tokenizer.convert_tokens_to_ids(t) for t in tokens]
    tgt_ids = [tgt_tokenizer.convert_tokens_to_ids(t) for t in tokens]

    pairs = [(si, ti) for si, ti in zip(src_ids, tgt_ids) if si != src_tokenizer.unk_token_id and ti != tgt_tokenizer.unk_token_id and si is not None and ti is not None]
    if len(pairs) == 0:
        return np.eye(src_emb.shape[1], dtype=np.float32)

    A = np.stack([src_emb[si] for si, _ in pairs], axis=1)  # (d, n)
    B = np.stack([tgt_emb[ti] for _, ti in pairs], axis=1)  # (d, n)

    M = B @ A.T
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    return R.astype(np.float32)


def map_direction_to_target(direction: np.ndarray, map_matrix: Optional[np.ndarray]) -> np.ndarray:
    if map_matrix is None:
        return direction
    return map_matrix @ direction


@torch.no_grad()
def generate_with_steering(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    direction: np.ndarray,
    scale: float = 1.0,
    map_matrix: Optional[np.ndarray] = None,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: Optional[str] = None,
) -> str:
    target_device = device or _get_device_from_model(model)
    model_was_training = model.training
    model.eval()

    mapped_direction = map_direction_to_target(direction, map_matrix)
    vocab_bias = compute_vocab_bias(model, mapped_direction, scale=scale, device=target_device)

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(target_device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(target_device)

    generated = input_ids
    for _ in range(max_new_tokens):
        out = model(input_ids=generated, attention_mask=attention_mask, use_cache=True)
        logits = out.logits[:, -1, :]
        steered = logits + vocab_bias

        if temperature != 1.0:
            steered = steered / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            topk = torch.topk(steered, k=top_k, dim=-1)
            mask = torch.full_like(steered, float("-inf"))
            mask.scatter_(1, topk.indices, topk.values)
            steered = mask

        next_token = torch.argmax(steered, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

    if model_was_training:
        model.train()

    return tokenizer.decode(generated[0], skip_special_tokens=True)


@torch.no_grad()
def generate_with_activation_steering(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    direction: np.ndarray,
    layer: int,
    scale: float = 1.0,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    device: Optional[str] = None,
) -> str:
    """Generate text while steering hidden states at a specific layer."""
    target_device = device or _get_device_from_model(model)
    model_was_training = model.training
    model.eval()

    steering_vector = torch.from_numpy(direction.astype(np.float32)).to(target_device)

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(target_device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(target_device)

    def steering_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] = hidden[:, -1, :] + scale * steering_vector
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    target_layer = model.model.layers[layer]
    handle = target_layer.register_forward_hook(steering_hook)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature != 1.0),
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    handle.remove()

    if model_was_training:
        model.train()

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


