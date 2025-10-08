from typing import List, Literal, Optional, Dict, Any, Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from tqdm import tqdm

from .utils import select_positions


def _collect_layer_hidden_states(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: List[int],
) -> List[torch.Tensor]:
    """Run a forward pass capturing hidden states for specified layers.

    Returns list of tensors of shape (batch, seq, hidden)
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states  # type: ignore[attr-defined]
    layer_hidden = []
    for layer_idx in layers:
        layer_hidden.append(hidden_states[layer_idx])
    return layer_hidden


def _gather_token_positions(h: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather per-sequence hidden vectors at given token positions.

    h: (batch, seq, hidden); positions: (batch,)
    returns (batch, hidden)
    """
    batch_indices = torch.arange(h.size(0), device=h.device)
    return h[batch_indices, positions]


@torch.no_grad()
def extract_layer_activations(
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
    """Extract per-text activations at selected layers and positions.

    Returns dict with keys: X (np.ndarray), layers (List[int]), positions (np.ndarray), texts (List[str])
    X has shape (num_texts, sum(hidden_dims_for_layers)). If all layers share hidden size, X is concatenated.
    """
    model_was_training = model.training
    model.eval()

    if device is not None:
        device_str = device
    else:
        param_device = next(model.parameters()).device
        device_str = str(param_device) if param_device.type != "cpu" or torch.cuda.is_available() else "cpu"

    all_vectors: List[np.ndarray] = []
    all_positions: List[int] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="extract_activations", leave=False):
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

        layer_hidden = _collect_layer_hidden_states(model, input_ids, attention_mask, layers)

        positions = select_positions(
            mode=position_mode,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
        )
        all_positions.extend(positions.tolist())

        gathered: List[torch.Tensor] = []
        for h in layer_hidden:
            gathered.append(_gather_token_positions(h, positions))
        # Concatenate along hidden dimension
        concat = torch.cat(gathered, dim=-1).detach().cpu().numpy()
        all_vectors.append(concat)

    X = np.concatenate(all_vectors, axis=0)

    if model_was_training:
        model.train()

    return {
        "X": X,
        "layers": layers,
        "positions": np.array(all_positions, dtype=np.int32),
        "texts": texts,
    }


class DiffInMeansProbe:
    """Covariance-adjusted difference-in-means linear probe.

    w = pinv(Σ + αI) @ (μ_h - μ_d)
    score(x) = x @ w
    """

    def __init__(self):
        self.w: Optional[np.ndarray] = None
        self.mu_honest: Optional[np.ndarray] = None
        self.mu_deceptive: Optional[np.ndarray] = None
        self.shrinkage: float = 0.0

    def fit(self, honest_X: np.ndarray, deceptive_X: np.ndarray, *, shrinkage: float = 0.0) -> "DiffInMeansProbe":
        self.shrinkage = float(shrinkage)

        mu_h = honest_X.mean(axis=0)
        mu_d = deceptive_X.mean(axis=0)
        d = mu_h - mu_d

        X_pool = np.concatenate([honest_X, deceptive_X], axis=0)
        Xc = X_pool - X_pool.mean(axis=0)
        n = Xc.shape[0]
        cov = (Xc.T @ Xc) / max(n - 1, 1)

        if self.shrinkage > 0.0:
            cov = cov + self.shrinkage * np.eye(cov.shape[0], dtype=cov.dtype)

        inv_cov = np.linalg.pinv(cov)
        w = inv_cov @ d

        self.w = w
        self.mu_honest = mu_h
        self.mu_deceptive = mu_d
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w

    def predict(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        scores = self.score(X)
        return (scores >= threshold).astype(np.int32)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            w=self.w,
            mu_honest=self.mu_honest,
            mu_deceptive=self.mu_deceptive,
            shrinkage=np.array([self.shrinkage], dtype=np.float32),
        )

    @staticmethod
    def load(path: str) -> "DiffInMeansProbe":
        blob = np.load(path, allow_pickle=True)
        p = DiffInMeansProbe()
        p.w = blob["w"]
        p.mu_honest = blob["mu_honest"]
        p.mu_deceptive = blob["mu_deceptive"]
        p.shrinkage = float(blob["shrinkage"][0]) if "shrinkage" in blob else 0.0
        return p

    @property
    def direction(self) -> Optional[np.ndarray]:
        return self.w


def fit_dim_probe(honest_X: np.ndarray, deceptive_X: np.ndarray, *, shrinkage: float = 0.0) -> DiffInMeansProbe:
    p = DiffInMeansProbe()
    return p.fit(honest_X, deceptive_X, shrinkage=shrinkage)


def apply_dim_probe(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ w


