from typing import List, Optional, Literal, Dict, Any, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def _resolve_torch_dtype(dtype: Optional[str]):
    if dtype is None or dtype == "auto":
        return None
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return None


def load_causal_lm(
    model_id: str,
    *,
    device_map: str = "auto",
    dtype: Optional[str] = "auto",
    adapter_id: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a HF causal LM (optionally with a PEFT adapter).

    model_id: base model repo or path.
    device_map: passed to HF for device placement.
    dtype: one of {"auto","float16","bfloat16","float32"}.
    adapter_id: if provided, loads a LoRA adapter with PEFT.
    """
    torch_dtype = _resolve_torch_dtype(dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    if adapter_id:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_id)

    tokenizer_source = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def last_nonpad_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last non-pad token per sequence."""
    lengths = attention_mask.long().sum(dim=1)
    return lengths - 1


def last_token_positions(input_ids: torch.Tensor) -> torch.Tensor:
    """Index of the last token (seq_len - 1) per sequence."""
    seq_len = input_ids.size(1)
    return torch.full((input_ids.size(0),), seq_len - 1, device=input_ids.device, dtype=torch.long)


def last_colon_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> torch.Tensor:
    """Index of the last ':' token; fallback to last non-pad if none present."""
    encoded = tokenizer.encode(":", add_special_tokens=False)
    colon_id = encoded[0] if len(encoded) > 0 else 0

    match = (input_ids == colon_id).long()
    indices = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
    weighted = match * indices
    last_idx = weighted.max(dim=1).values

    has_colon = match.any(dim=1)
    fallback = last_nonpad_positions(attention_mask)
    return torch.where(has_colon, last_idx, fallback)


def select_positions(
    *,
    mode: Literal["last_nonpad", "last_token", "last_colon"] = "last_nonpad",
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> torch.Tensor:
    """Compute per-sequence token positions.

    mode: selection strategy.
    """
    if mode == "last_nonpad":
        return last_nonpad_positions(attention_mask)
    if mode == "last_token":
        return last_token_positions(input_ids)
    if mode == "last_colon" and tokenizer is not None:
        return last_colon_positions(input_ids, attention_mask, tokenizer)
    return last_nonpad_positions(attention_mask)


def save_activation_dataset(
    path: str,
    *,
    X: np.ndarray,
    layers: List[int],
    positions: Optional[np.ndarray] = None,
    texts: Optional[List[str]] = None,
    y: Optional[np.ndarray] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save activations and metadata to a compressed NPZ."""
    data: Dict[str, Any] = {
        "X": X,
        "layers": np.array(layers, dtype=np.int32),
    }
    if positions is not None:
        data["positions"] = positions
    if y is not None:
        data["y"] = y
    if texts is not None:
        data["texts"] = np.array(texts, dtype=object)
    if meta is not None:
        data["meta_json"] = np.array([meta], dtype=object)
    np.savez_compressed(path, **data)


def load_activation_dataset(path: str) -> Dict[str, Any]:
    """Load activations and metadata from NPZ."""
    blob = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {
        "X": blob["X"],
        "layers": blob["layers"].astype(np.int32).tolist(),
    }
    if "positions" in blob:
        out["positions"] = blob["positions"]
    if "y" in blob:
        out["y"] = blob["y"]
    if "texts" in blob:
        out["texts"] = blob["texts"].tolist()
    if "meta_json" in blob:
        out["meta"] = blob["meta_json"][0].item()
    return out


