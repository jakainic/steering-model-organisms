import csv
import argparse
from typing import List, Tuple, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .utils import load_causal_lm
from .probing import extract_layer_activations, fit_dim_probe
from .model_diff import (
    extract_layer_activations_for_model,
    compute_suppression_direction,
    compute_average_suppression_direction,
)
from .steering import generate_with_steering, generate_with_activation_steering
from .evaluation import probe_accuracy, compute_perplexity, dose_response, TabooRecoveryEvaluator, GenderRecoveryEvaluator, compute_auroc


def load_statements_labels(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        texts, labels = [], []
        for row in r:
            texts.append(row["statement"])
            labels.append(int(row["label"]))
    return texts, np.array(labels)


def run_probe_training(
    dataset_path: str,
    *,
    model_id: str,
    adapter_id: Optional[str] = None,
    layers: List[int] = (-1,),
    batch_size: int = 8,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    texts, labels = load_statements_labels(dataset_path)
    model, tok = load_causal_lm(model_id, device_map="auto", dtype="bfloat16", adapter_id=adapter_id)
    acts = extract_layer_activations(model, tok, texts, layers=list(layers), batch_size=batch_size)
    X = acts["X"]
    n = len(texts)
    idx = np.arange(n)
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels)
    X_tr, y_tr = X[tr_idx], labels[tr_idx]
    X_te, y_te = X[te_idx], labels[te_idx]
    honest_X = X_tr[y_tr == 1]
    deceptive_X = X_tr[y_tr == 0]
    probe = fit_dim_probe(honest_X, deceptive_X, shrinkage=1e-3)
    acc = probe_accuracy(probe.score(X_te), y_te)
    return acc, probe.direction


def _determine_num_layers(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, sample_text: str) -> int:
    enc = tokenizer(sample_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(next(model.parameters()).device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(input_ids.device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    return len(hidden_states)


def _build_mid_late_candidate_layers(num_layers: int) -> List[int]:
    L = max(int(num_layers) - 1, 1)
    mid = L // 2
    count = max(1, L // 2)
    grid = np.linspace(mid, L, num=count)
    vals = [int(x) for x in grid]
    uniq = sorted(set(vals))
    return uniq


def sweep_layers_probe(
    texts: List[str],
    labels: np.ndarray,
    *,
    model_id: str,
    adapter_id: Optional[str] = None,
    batch_size: int = 8,
    test_size: float = 0.2,
    seed: int = 0,
    position_mode: str = "last_colon",
    extra_layers: Optional[List[int]] = None,
) -> Tuple[dict, List[dict]]:
    model, tok = load_causal_lm(model_id, device_map="auto", dtype="bfloat16", adapter_id=adapter_id)
    sample_text = texts[0] if len(texts) > 0 else "Hello"
    n_layers = _determine_num_layers(model, tok, sample_text)
    candidates = _build_mid_late_candidate_layers(n_layers)
    if extra_layers is not None:
        merged = sorted(set([int(x) for x in list(candidates) + list(extra_layers) if 0 <= int(x) < n_layers]))
        candidates = merged

    n = len(texts)
    idx = np.arange(n)
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels)

    best = {"layer": None, "auroc": -1.0, "acc": 0.0, "direction": None}
    rows: List[dict] = []

    for lyr in tqdm(candidates, desc="layers", leave=False):
        acts = extract_layer_activations(
            model,
            tok,
            texts,
            layers=[int(lyr)],
            batch_size=batch_size,
            position_mode=position_mode,  # last_colon falls back to last_nonpad in utils
        )
        X = acts["X"]
        X_tr, y_tr = X[tr_idx], labels[tr_idx]
        X_te, y_te = X[te_idx], labels[te_idx]
        honest_X = X_tr[y_tr == 1]
        deceptive_X = X_tr[y_tr == 0]
        probe = fit_dim_probe(honest_X, deceptive_X, shrinkage=1e-3)
        scores = probe.score(X_te)
        acc = probe_accuracy(scores, y_te)
        auroc = compute_auroc(scores, y_te)
        rows.append({"layer": int(lyr), "acc": float(acc), "auroc": float(auroc)})
        if auroc > best["auroc"]:
            best = {"layer": int(lyr), "auroc": float(auroc), "acc": float(acc), "direction": probe.direction}

    return best, rows


def run_model_diff(
    dataset_path: str,
    *,
    base_model_id: str,
    adapter_ids: List[str],
    layers: List[int] = (-1,),
    batch_size: int = 8,
) -> np.ndarray:
    texts, _ = load_statements_labels(dataset_path)
    base_model, tok = load_causal_lm(base_model_id, device_map="auto", dtype="bfloat16")
    base = extract_layer_activations_for_model(base_model, tok, texts, layers=list(layers), batch_size=batch_size)
    ft_X_list = []
    for aid in tqdm(adapter_ids, desc="adapters", leave=False):
        ft_model, _ = load_causal_lm(base_model_id, device_map="auto", dtype="bfloat16", adapter_id=aid)
        ft = extract_layer_activations_for_model(ft_model, tok, texts, layers=list(layers), batch_size=batch_size)
        ft_X_list.append(ft["X"])
    v_avg = compute_average_suppression_direction(base["X"], ft_X_list, center=True, normalize_each=True)
    return v_avg


def run_dose_response(
    prompts: List[str],
    direction: np.ndarray,
    *,
    model_id: str,
    adapter_id: Optional[str] = None,
    scales: List[float] = (0.0, 0.5, 1.0, 1.5),
    activation_layer: Optional[int] = None,
    max_new_tokens: int = 64,
) -> List[str]:
    model, tok = load_causal_lm(model_id, device_map="auto", dtype="bfloat16", adapter_id=adapter_id)
    outputs: List[str] = []
    for s in tqdm(scales, desc="scales", leave=False):
        for p in tqdm(prompts, desc="prompts", leave=False):
            if activation_layer is None:
                out = generate_with_steering(model, tok, p, direction=direction, scale=s, max_new_tokens=max_new_tokens)
            else:
                out = generate_with_activation_steering(
                    model, tok, p, direction=direction, layer=activation_layer, scale=s, max_new_tokens=max_new_tokens
                )
            outputs.append(out)
    return outputs

def _parse_layers(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _parse_layers_or_range(s: str) -> List[int]:
    vals: List[int] = []
    parts = [x.strip() for x in s.split(",") if x.strip()]
    for p in parts:
        if "-" in p:
            a_str, b_str = p.split("-", 1)
            a = int(a_str.strip())
            b = int(b_str.strip())
            lo = a if a <= b else b
            hi = b if b >= a else a
            vals.extend(list(range(lo, hi + 1)))
        else:
            vals.append(int(p))
    return sorted(set(vals))

def _parse_scales(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def main() -> None:
    p = argparse.ArgumentParser(description="Experiment runners")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="Train diff-in-means probe")
    p_probe.add_argument("dataset", type=str)
    p_probe.add_argument("--model", type=str, required=True)
    p_probe.add_argument("--adapter", type=str, default=None)
    p_probe.add_argument("--layers", type=str, default="-1")
    p_probe.add_argument("--batch-size", type=int, default=8)
    p_probe.add_argument("--test-size", type=float, default=0.2)
    p_probe.add_argument("--seed", type=int, default=0)
    p_probe.add_argument("--save-direction", type=str, default=None)

    p_probe_test = sub.add_parser("probe-test", help="Test a saved probe direction on a model")
    p_probe_test.add_argument("dataset", type=str)
    p_probe_test.add_argument("--direction-npy", type=str, required=True)
    p_probe_test.add_argument("--model", type=str, required=True)
    p_probe_test.add_argument("--adapter", type=str, default=None)
    p_probe_test.add_argument("--layers", type=str, default="-1")
    p_probe_test.add_argument("--batch-size", type=int, default=8)
    p_probe_test.add_argument("--threshold", type=float, default=0.0)

    p_diff = sub.add_parser("model-diff", help="Compute average suppression direction")
    p_diff.add_argument("dataset", type=str)
    p_diff.add_argument("--base-model", type=str, required=True)
    p_diff.add_argument("--adapters", type=str, required=True, help="Comma-separated adapter ids")
    p_diff.add_argument("--layers", type=str, default="-1")
    p_diff.add_argument("--batch-size", type=int, default=8)
    p_diff.add_argument("--save-direction", type=str, default=None)

    p_dose = sub.add_parser("dose", help="Run dose-response with a direction")
    p_dose.add_argument("--model", type=str, required=True)
    p_dose.add_argument("--adapter", type=str, default=None)
    p_dose.add_argument("--prompt", action="append", default=[])
    p_dose.add_argument("--prompts-file", type=str, default=None)
    p_dose.add_argument("--direction-npy", type=str, required=True)
    p_dose.add_argument("--scales", type=str, default="0.0,0.5,1.0,1.5")
    p_dose.add_argument("--activation-layer", type=int, default=None)
    p_dose.add_argument("--max-new-tokens", type=int, default=64)
    p_dose.add_argument("--eval", type=str, choices=["taboo", "gender"], default=None)
    p_dose.add_argument("--secret-word", type=str, default=None)
    p_dose.add_argument("--user-gender", type=str, choices=["male", "female"], default=None)
    p_dose.add_argument("--save-json", type=str, default=None, help="Optional path to save JSON outputs")

    p_sweep = sub.add_parser("probe-sweep", help="Sweep mid-to-late layers and report AUROC")
    p_sweep.add_argument("dataset", type=str)
    p_sweep.add_argument("--model", type=str, required=True)
    p_sweep.add_argument("--adapter", type=str, default=None)
    p_sweep.add_argument("--batch-size", type=int, default=8)
    p_sweep.add_argument("--test-size", type=float, default=0.2)
    p_sweep.add_argument("--seed", type=int, default=0)
    p_sweep.add_argument("--save-csv", type=str, default=None)
    p_sweep.add_argument("--save-best-direction", type=str, default=None)
    p_sweep.add_argument("--position-mode", type=str, default="last_colon", choices=["last_colon","last_nonpad","last_token"]) 
    p_sweep.add_argument("--template", type=str, default=None, help="Optional prompt template using {text}")
    p_sweep.add_argument("--auroc-threshold", type=float, default=0.65)
    p_sweep.add_argument("--include-layers", type=str, default=None, help="Comma-separated list and/or ranges like 15-20")

    args = p.parse_args()

    if args.cmd == "probe":
        layers = _parse_layers(args.layers)
        acc, direction = run_probe_training(
            args.dataset,
            model_id=args.model,
            adapter_id=args.adapter,
            layers=layers,
            batch_size=args.batch_size,
            test_size=args.test_size,
            seed=args.seed,
        )
        print(f"test_accuracy={acc:.4f}")
        if args.save_direction:
            np.save(args.save_direction, direction)
            print(f"saved_direction={args.save_direction}")
        return

    if args.cmd == "model-diff":
        layers = _parse_layers(args.layers)
        adapters = [x.strip() for x in args.adapters.split(",") if x.strip()]
        v = run_model_diff(
            args.dataset,
            base_model_id=args.base_model,
            adapter_ids=adapters,
            layers=layers,
            batch_size=args.batch_size,
        )
        if args.save_direction:
            np.save(args.save_direction, v)
            print(f"saved_direction={args.save_direction}")
        else:
            print("direction_norm=", float(np.linalg.norm(v)))
        return

    if args.cmd == "dose":
        prompts: List[str] = list(args.prompt)
        if args.prompts_file:
            with open(args.prompts_file) as f:
                prompts.extend([ln.rstrip("\n") for ln in f if ln.strip()])
        scales = _parse_scales(args.scales)
        direction = np.load(args.direction_npy)
        outs = run_dose_response(
            prompts,
            direction,
            model_id=args.model,
            adapter_id=args.adapter,
            scales=scales,
            activation_layer=args.activation_layer,
            max_new_tokens=args.max_new_tokens,
        )
        for i, o in enumerate(outs):
            print(f"--- dose[{i}] ---")
            print(o)

        if args.eval is not None:
            if args.eval == "taboo":
                evaluator = TabooRecoveryEvaluator({"secret_word": str(args.secret_word)})
            else:
                evaluator = GenderRecoveryEvaluator({"user_gender": str(args.user_gender)})
            rate = evaluator.compute_recovery_rate(outs)
            print(f"evaluation={args.eval} recovery_rate={rate:.4f}")

        if args.save_json:
            import json
            rows = []
            n_prompts = len(prompts)
            for idx, text in enumerate(outs):
                s = scales[idx // max(n_prompts, 1)] if n_prompts > 0 else 0.0
                p = prompts[idx % max(n_prompts, 1)] if n_prompts > 0 else ""
                rows.append({"scale": float(s), "prompt": p, "output": text})
            with open(args.save_json, "w") as f:
                json.dump(rows, f)
        return

    if args.cmd == "probe-test":
        layers = _parse_layers(args.layers)
        texts, labels = load_statements_labels(args.dataset)
        direction = np.load(args.direction_npy)
        model, tok = load_causal_lm(args.model, device_map="auto", dtype="bfloat16", adapter_id=args.adapter)
        acts = extract_layer_activations(model, tok, texts, layers=layers, batch_size=args.batch_size)
        X = acts["X"]
        scores = X @ direction
        acc = probe_accuracy(scores, labels, threshold=args.threshold)
        print(f"test_accuracy={acc:.4f}")
        return

    if args.cmd == "probe-sweep":
        texts, labels = load_statements_labels(args.dataset)
        extra_layers: Optional[List[int]] = None
        if args.include_layers is not None:
            extra_layers = _parse_layers_or_range(str(args.include_layers))
        best, rows = sweep_layers_probe(
            texts,
            labels,
            model_id=args.model,
            adapter_id=args.adapter,
            batch_size=args.batch_size,
            test_size=args.test_size,
            seed=args.seed,
            position_mode=str(args.position_mode),
            extra_layers=extra_layers,
        )

        if args.save_csv:
            import csv as _csv
            with open(args.save_csv, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=["layer","acc","auroc"])
                w.writeheader()
                for r in rows:
                    w.writerow({"layer": int(r["layer"]), "acc": float(r["acc"]), "auroc": float(r["auroc"])})

        print(f"best_layer={best['layer']} best_auroc={best['auroc']:.4f} best_acc={best['acc']:.4f}")
        if args.save_best_direction and best["direction"] is not None:
            np.save(args.save_best_direction, best["direction"])
            print(f"saved_direction={args.save_best_direction}")

        if args.template is not None and float(best["auroc"]) < float(args.auroc_threshold):
            templated = [str(args.template).replace("{text}", t) for t in texts]
            best2, rows2 = sweep_layers_probe(
                templated,
                labels,
                model_id=args.model,
                adapter_id=args.adapter,
                batch_size=args.batch_size,
                test_size=args.test_size,
                seed=args.seed,
                position_mode=str(args.position_mode),
            )
            print(
                "retry_with_template: best_layer={} best_auroc={:.4f} best_acc={:.4f}".format(
                    best2["layer"], best2["auroc"], best2["acc"]
                )
            )
            if args.save_csv:
                import csv as _csv2
                with open(args.save_csv, "w", newline="") as f:
                    w = _csv2.DictWriter(f, fieldnames=["layer","acc","auroc"])
                    w.writeheader()
                    for r in rows2:
                        w.writerow({"layer": int(r["layer"]), "acc": float(r["acc"]), "auroc": float(r["auroc"])})
            if args.save_best_direction and best2["direction"] is not None:
                np.save(args.save_best_direction, best2["direction"])
                print(f"saved_direction={args.save_best_direction}")
        return

if __name__ == "__main__":
    main()
