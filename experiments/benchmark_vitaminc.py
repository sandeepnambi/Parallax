"""
Parallax Step 3: Contrastive NLI Backbone Evaluation on D2-VitaminC
===================================================================
Benchmarks neural cross-encoder (cross-encoder/nli-distilroberta-base) vs.
heuristic baseline (HeuristicNLI) on paired contrastive test cases from
the VitaminC fact-verification dataset (Schuster et al., NAACL 2021).

Measures:
1. 3-class Classification Accuracy & Macro-F1 (SUPPORTS / REFUTES / NOT ENOUGH INFO)
2. Contrastive Pair Accuracy: Does the model correctly classify BOTH the SUPPORTS
   and REFUTES instances for the same underlying factual case?
3. Lexical Overlap Resistance: Demonstrates why neural cross-encoders succeed where
   heuristic/overlap baselines fail on subtle factual edits.
"""

import os
import sys
import json
import zipfile
import argparse
import time
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure src and root are in sys.path
_cur = os.path.dirname(os.path.abspath(__file__))
_src = os.path.abspath(os.path.join(_cur, "..", "src"))
_root = os.path.abspath(os.path.join(_cur, ".."))
for _p in [_src, _root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from verifier.static_baseline import HeuristicNLI
except ImportError:
    from src.verifier.static_baseline import HeuristicNLI


# Label standardization: VitaminC <-> NLI
# VitaminC labels: SUPPORTS, REFUTES, NOT ENOUGH INFO
# NLI DistilRoBERTa id2label: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
NLI_TO_VITAMINC = {
    "ENTAILMENT": "SUPPORTS",
    "CONTRADICTION": "REFUTES",
    "NEUTRAL": "NOT ENOUGH INFO",
}

VITAMINC_TO_NLI = {
    "SUPPORTS": "ENTAILMENT",
    "REFUTES": "CONTRADICTION",
    "NOT ENOUGH INFO": "NEUTRAL",
}


def load_contrastive_vitaminc(
    zip_path: str,
    max_cases: int = 500,
    split: str = "test"
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """
    Loads contrastive cases from VitaminC zip archive.
    A case is contrastive if it contains both SUPPORTS and REFUTES examples.
    
    Returns:
        samples: Flattened list of individual test samples.
        case_map: Mapping from case_id to indices in `samples`.
    """
    filename = f"vitaminc/{split}.jsonl"
    print(f"Reading {filename} from {zip_path}...")
    
    cases: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(filename) as f:
            for line in f:
                row = json.loads(line)
                cid = row.get("case_id")
                if cid:
                    cases[cid].append(row)
                    
    # Filter for contrastive cases that contain at least one SUPPORTS and at least one REFUTES
    contrastive_case_ids = [
        cid for cid, rows in cases.items()
        if any(r["label"] == "SUPPORTS" for r in rows) and any(r["label"] == "REFUTES" for r in rows)
    ]
    
    print(f"Total cases found: {len(cases):,} | Contrastive cases (SUPPORTS + REFUTES): {len(contrastive_case_ids):,}")
    
    if max_cases > 0:
        selected_case_ids = contrastive_case_ids[:max_cases]
    else:
        selected_case_ids = contrastive_case_ids
        
    flat_samples: List[Dict[str, Any]] = []
    case_indices: Dict[str, List[int]] = defaultdict(list)
    
    for cid in selected_case_ids:
        for item in cases[cid]:
            idx = len(flat_samples)
            flat_samples.append(item)
            case_indices[cid].append(idx)
            
    print(f"Selected {len(selected_case_ids):,} cases ({len(flat_samples):,} total instances) for evaluation.")
    return flat_samples, case_indices


def evaluate_heuristic(samples: List[Dict[str, Any]]) -> List[str]:
    """Runs HeuristicNLI baseline on all samples."""
    print("Running Heuristic Baseline on contrastive samples...")
    engine = HeuristicNLI()
    preds = []
    for item in samples:
        claim = item.get("claim", "")
        evidence = item.get("evidence", "")
        # Heuristic outputs ENTAILMENT / CONTRADICTION / NEUTRAL
        rel = engine.predict(claim=claim, evidence=evidence, answer_entity="")
        vitaminc_label = NLI_TO_VITAMINC.get(rel, "NOT ENOUGH INFO")
        preds.append(vitaminc_label)
    return preds


def evaluate_neural_cross_encoder(
    samples: List[Dict[str, Any]],
    model_name: str = "cross-encoder/nli-distilroberta-base",
    batch_size: int = 32
) -> List[str]:
    """Runs batched inference with CrossEncoder NLI."""
    print(f"Loading Neural Cross-Encoder: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
    model: Any = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    if tokenizer is None or model is None:
        raise RuntimeError("Failed to load tokenizer or model")
    
    id2label = {int(k): str(v).upper() for k, v in model.config.id2label.items()}
    print(f"Model id2label mapping: {id2label} (device: {device})")
    
    preds: List[str] = []
    total = len(samples)
    t0 = time.time()
    tok: Any = tokenizer
    
    for i in range(0, total, batch_size):
        batch = samples[i:i + batch_size]
        premises = [str(b.get("evidence") or "") for b in batch]
        hypotheses = [str(b.get("claim") or "") for b in batch]
        
        inputs = tok(
            premises, hypotheses,
            padding=True, truncation=True, max_length=256,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            
        for pid in pred_ids:
            raw_label = id2label.get(pid, "NEUTRAL")
            vitaminc_label = NLI_TO_VITAMINC.get(raw_label, "NOT ENOUGH INFO")
            preds.append(vitaminc_label)
            
        if (i // batch_size) % 10 == 0 or (i + batch_size) >= total:
            done = min(i + batch_size, total)
            elapsed = time.time() - t0
            rate = done / max(elapsed, 0.001)
            print(f"  Processed {done:,}/{total:,} samples ({done/total*100:.1f}%) [{rate:.1f} samples/sec]")
            
    return preds


def compute_metrics(
    samples: List[Dict[str, Any]],
    case_indices: Dict[str, List[int]],
    predictions: List[str],
    model_name: str
) -> Dict[str, Any]:
    """
    Computes standard 3-class classification metrics and contrastive case consistency.
    """
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    gold = [s["label"] for s in samples]
    
    assert len(gold) == len(predictions), "Mismatched length between gold and predictions"
    
    total = len(gold)
    correct = sum(1 for g, p in zip(gold, predictions) if g == p)
    acc = correct / max(total, 1)
    
    # Per-class counts
    tp = {lbl: 0 for lbl in labels}
    fp = {lbl: 0 for lbl in labels}
    fn = {lbl: 0 for lbl in labels}
    support = {lbl: 0 for lbl in labels}
    
    # Confusion matrix: conf[gold][pred]
    conf: Dict[str, Dict[str, int]] = {g_lbl: {p_lbl: 0 for p_lbl in labels} for g_lbl in labels}
    
    for g, p in zip(gold, predictions):
        if g in labels and p in labels:
            conf[g][p] += 1
            support[g] += 1
            if g == p:
                tp[g] += 1
            else:
                fp[p] += 1
                fn[g] += 1
                
    per_class = {}
    f1s = []
    for lbl in labels:
        p = tp[lbl] / max(tp[lbl] + fp[lbl], 1)
        r = tp[lbl] / max(tp[lbl] + fn[lbl], 1)
        f1 = (2 * p * r) / max(p + r, 1e-9)
        per_class[lbl] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": support[lbl],
        }
        f1s.append(f1)
        
    macro_f1 = sum(f1s) / len(f1s)
    
    # Contrastive Pair Accuracy:
    # A case is contrastive-correct IF AND ONLY IF all instances in that case_id are correctly predicted!
    total_cases = len(case_indices)
    perfect_cases = 0
    pairwise_contrastive_total = 0
    pairwise_contrastive_correct = 0
    
    for cid, idxs in case_indices.items():
        case_samples = [samples[idx] for idx in idxs]
        case_preds = [predictions[idx] for idx in idxs]
        case_gold = [s["label"] for s in case_samples]
        
        # Check if ALL instances in case are correct
        if all(g == p for g, p in zip(case_gold, case_preds)):
            perfect_cases += 1
            
        # Also check specifically paired (SUPPORTS, REFUTES) within this case
        sup_indices = [i for i, g in enumerate(case_gold) if g == "SUPPORTS"]
        ref_indices = [i for i, g in enumerate(case_gold) if g == "REFUTES"]
        
        for si in sup_indices:
            for ri in ref_indices:
                pairwise_contrastive_total += 1
                if case_preds[si] == "SUPPORTS" and case_preds[ri] == "REFUTES":
                    pairwise_contrastive_correct += 1
                    
    case_accuracy = perfect_cases / max(total_cases, 1)
    contrastive_pair_acc = pairwise_contrastive_correct / max(pairwise_contrastive_total, 1)
    
    return {
        "model_name": model_name,
        "total_samples": total,
        "instance_accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "total_contrastive_cases": total_cases,
        "all_instances_case_accuracy": round(case_accuracy, 4),
        "total_contrastive_pairs": pairwise_contrastive_total,
        "contrastive_pair_accuracy": round(contrastive_pair_acc, 4),
        "confusion_matrix": conf,
    }


def print_comparison_table(results: List[Dict[str, Any]]):
    """Prints a comparison table between models."""
    header = (
        f"{'Model Name':<38} | {'Inst Acc':<9} | {'Macro F1':<9} | "
        f"{'Pair Acc':<9} | {'SUP F1':<8} | {'REF F1':<8} | {'NEI F1':<8}"
    )
    print("\n" + "=" * len(header))
    print("VITAMINC CONTRASTIVE BENCHMARK RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    
    for r in results:
        name = r["model_name"]
        acc = f"{r['instance_accuracy']*100:.2f}%"
        macro = f"{r['macro_f1']*100:.2f}%"
        pair = f"{r['contrastive_pair_accuracy']*100:.2f}%"
        sup_f1 = f"{r['per_class']['SUPPORTS']['f1']*100:.2f}%"
        ref_f1 = f"{r['per_class']['REFUTES']['f1']*100:.2f}%"
        nei_f1 = f"{r['per_class']['NOT ENOUGH INFO']['f1']*100:.2f}%"
        print(f"{name:<38} | {acc:<9} | {macro:<9} | {pair:<9} | {sup_f1:<8} | {ref_f1:<8} | {nei_f1:<8}")
        
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Parallax Step 3: VitaminC Contrastive Backbone Evaluation")
    parser.add_argument("--zip_path", type=str, default="Datasets/D2-VitaminC/vitaminc.zip", help="Path to vitaminc.zip")
    parser.add_argument("--max_cases", type=int, default=500, help="Number of contrastive cases to evaluate (default: 500)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for neural inference")
    parser.add_argument("--output_file", type=str, default="experiments/vitaminc_benchmark_results.json")
    args = parser.parse_args()
    
    if not os.path.exists(args.zip_path):
        print(f"Error: {args.zip_path} not found.")
        sys.exit(1)
        
    samples, case_map = load_contrastive_vitaminc(args.zip_path, max_cases=args.max_cases)
    
    # 1. Evaluate Heuristic Baseline
    heuristic_preds = evaluate_heuristic(samples)
    heuristic_metrics = compute_metrics(samples, case_map, heuristic_preds, "Heuristic NLI Baseline (Lexical)")
    
    # 2. Evaluate Neural Cross-Encoder
    neural_preds = evaluate_neural_cross_encoder(samples, batch_size=args.batch_size)
    neural_metrics = compute_metrics(samples, case_map, neural_preds, "Neural Cross-Encoder (distilroberta)")
    
    # 3. Print Comparison Table
    all_results = [heuristic_metrics, neural_metrics]
    print_comparison_table(all_results)
    
    # 4. Save results to JSON
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full benchmark results to {args.output_file}")


if __name__ == "__main__":
    main()
