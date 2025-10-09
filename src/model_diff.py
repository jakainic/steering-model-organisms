from typing import List, Literal, Optional, Dict, Any

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm

from .utils import select_positions


@torch.no_grad()
def extract_layer_activations_for_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    *,
    layers: List[int],
    position_mode: Literal["last_nonpad", "last_token", "last_colon"] = "last_nonpad",
    batch_size: int = 8,
    max_length: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    model_was_training = model.training
    model.eval()

    if device is not None:
        device_str = device
    else:
        param_device = next(model.parameters()).device
        device_str = str(param_device) if param_device.type != "cpu" or torch.cuda.is_available() else "cpu"

    all_vectors: List[np.ndarray] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="extract_acts_model", leave=False):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device_str)
        attention_mask = enc["attention_mask"].to(device_str)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        positions = select_positions(
            mode=position_mode,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
        )

        gathered: List[torch.Tensor] = []
        for layer_idx in layers:
            h = hidden_states[layer_idx]
            batch_indices = torch.arange(h.size(0), device=h.device)
            gathered.append(h[batch_indices, positions])
        concat = torch.cat(gathered, dim=-1).to(torch.float32).detach().cpu().numpy()
        all_vectors.append(concat)

    X = np.concatenate(all_vectors, axis=0)

    if model_was_training:
        model.train()

    return {"X": X, "layers": layers}


def compute_suppression_direction(
    base_X: np.ndarray,
    ft_X: np.ndarray,
    *,
    center: bool = True,
) -> np.ndarray:
    """Compute hiding/suppression direction from base vs fine-tuned activations.

    Returns a unit vector v such that v ~ mean(base) - mean(ft), optionally centered.
    """
    if center:
        base_X = base_X - base_X.mean(axis=0)
        ft_X = ft_X - ft_X.mean(axis=0)
    d = base_X.mean(axis=0) - ft_X.mean(axis=0)
    norm = np.linalg.norm(d)
    return d / norm if norm > 0 else d


def project_scores(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return X @ direction


def compute_average_suppression_direction(
    base_X: np.ndarray,
    ft_X_list: List[np.ndarray],
    *,
    center: bool = True,
    weights: Optional[np.ndarray] = None,
    normalize_each: bool = True,
) -> np.ndarray:
    """Average suppression directions across multiple fine-tuned models.

    If normalize_each=True, each model's direction is unit-normalized before averaging.
    If weights are provided, they are applied to each direction before summation.
    Returns a unit vector.
    """
    if len(ft_X_list) == 0:
        return np.zeros(base_X.shape[1], dtype=np.float32)

    if normalize_each:
        dirs = [
            compute_suppression_direction(base_X, ft_X, center=center)
            for ft_X in ft_X_list
        ]
    else:
        dirs = []
        if center:
            base_mu = base_X.mean(axis=0)
        for ft_X in ft_X_list:
            if center:
                ft_mu = ft_X.mean(axis=0)
                d = (base_mu - ft_mu)
            else:
                d = base_X.mean(axis=0) - ft_X.mean(axis=0)
            dirs.append(d)

    D = np.stack(dirs, axis=0)  # (k, d)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        v = (w[:, None] * D).sum(axis=0)
    else:
        v = D.mean(axis=0)

    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v.astype(np.float32)


def evaluate_suppression_generalization(
    direction: np.ndarray,
    src_model: PreTrainedModel,
    src_tokenizer: PreTrainedTokenizerBase,
    tgt_model: PreTrainedModel,
    tgt_tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    *,
    layers: List[int],
    position_mode: Literal["last_nonpad", "last_token", "last_colon"] = "last_nonpad",
    map_via_tokens: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """TODO: Evaluate if a suppression direction from src generalizes to tgt.

    Suggested steps:
    1) Optionally compute an orthogonal map via shared tokens and map direction.
    2) Extract target activations and compute projection scores.
    3) Return summary stats (e.g., mean score diffs, AUROC vs labels if provided).
    """
    raise NotImplementedError("TODO: implement suppression direction generalization evaluation")


