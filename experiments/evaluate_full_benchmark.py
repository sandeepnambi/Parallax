"""
Parallax Phase A: Full 800-Example Benchmark Evaluation
======================================================
Evaluates and benchmarks four core verification systems on the official
test split (data/benchmark/test.parquet, 800 balanced records):

1. Static Heuristic NLI: Lexical alignment without timestamps.
2. Static Neural NLI: cross-encoder/nli-distilroberta-base without timestamps.
3. Parallax Heuristic: Heuristic alignment + Temporal Arbitration.
4. Parallax Neural (Ours): Neural cross-encoder + Temporal Arbitration.

Evaluates:
- 4-class Accuracy & Macro-F1 (SUPPORTED_CURRENT, OUTDATED, CONTRADICTED, UNSUPPORTED)
- Per-class Precision, Recall, and F1
- False-Support Rate on Outdated Claims: FSR_outdated = P(Pred = SUPPORTED_CURRENT | True = OUTDATED)
- Confusion Matrices
- Publication-ready LaTeX table export
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any

import pandas as pd

# Ensure src and root are in sys.path
_cur = os.path.dirname(os.path.abspath(__file__))
_src = os.path.abspath(os.path.join(_cur, "..", "src"))
_root = os.path.abspath(os.path.join(_cur, ".."))
for _p in [_src, _root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from verifier.static_baseline import StaticBaselineVerifier, HeuristicNLI
    from verifier.temporal_arbiter import ParallaxTemporalVerifier
    from pipeline.auditor import NeuralCrossEncoderNLI
    from evaluation.metrics import compute_parallax_metrics, export_latex_table
except ImportError:
    from src.verifier.static_baseline import StaticBaselineVerifier, HeuristicNLI
    from src.verifier.temporal_arbiter import ParallaxTemporalVerifier
    from src.pipeline.auditor import NeuralCrossEncoderNLI
    from src.evaluation.metrics import compute_parallax_metrics, export_latex_table


def evaluate_neural_static_batch(df: pd.DataFrame, neural_engine: NeuralCrossEncoderNLI, batch_size: int = 32) -> List[str]:
    """Runs batched inference for Static Neural Baseline (blind to timestamps)."""
    premises = df["evidence_text"].tolist()
    hypotheses = df["answer_claim"].tolist()
    
    print(f"Running Static Neural NLI inference on {len(df)} examples (batch_size={batch_size})...")
    raw_nli_preds = neural_engine.predict_batch(premises, hypotheses, batch_size=batch_size)
    
    # Static mapping: ENTAILMENT -> SUPPORTED_CURRENT, CONTRADICTION -> CONTRADICTED, NEUTRAL -> UNSUPPORTED
    preds = []
    for rel in raw_nli_preds:
        if rel == "ENTAILMENT":
            preds.append("SUPPORTED_CURRENT")
        elif rel == "CONTRADICTION":
            preds.append("CONTRADICTED")
        else:
            preds.append("UNSUPPORTED")
    return preds


def evaluate_neural_parallax_batch(df: pd.DataFrame, neural_engine: NeuralCrossEncoderNLI, batch_size: int = 32) -> List[str]:
    """Runs batched inference for Parallax Two-Stage Verifier (Cross-Encoder + Temporal Arbiter)."""
    premises = df["evidence_text"].tolist()
    hypotheses = df["answer_claim"].tolist()
    
    print(f"Running Parallax Neural Two-Stage inference on {len(df)} examples (batch_size={batch_size})...")
    raw_nli_preds = neural_engine.predict_batch(premises, hypotheses, batch_size=batch_size)
    
    preds = []
    for i, rel in enumerate(raw_nli_preds):
        row = df.iloc[i]
        ref_time = str(row.get("reference_time", "2024-12-01"))
        valid_to = str(row.get("valid_to", "present"))
        superseded_by = row.get("superseded_by")
        
        if rel == "ENTAILMENT":
            is_superseded = False
            if valid_to != "present" and valid_to <= ref_time:
                is_superseded = True
            elif superseded_by is not None and str(superseded_by) <= ref_time:
                is_superseded = True
                
            if is_superseded:
                preds.append("OUTDATED")
            else:
                preds.append("SUPPORTED_CURRENT")
        elif rel == "CONTRADICTION":
            preds.append("CONTRADICTED")
        else:
            preds.append("UNSUPPORTED")
    return preds


def print_comparison_table(results_dict: Dict[str, Dict[str, Any]]):
    """Prints a comparative summary table for all evaluated models."""
    header = (
        f"{'Model Name':<36} | {'Accuracy':<9} | {'Macro F1':<9} | "
        f"{'FSR (Outd)':<11} | {'Supp F1':<8} | {'Outd F1':<8} | {'Contra F1':<9} | {'Unsupp F1':<9}"
    )
    print("\n" + "=" * len(header))
    print("PARALLAX FULL BENCHMARK TEST RESULTS (800 Examples)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    
    for name, r in results_dict.items():
        acc = f"{r['accuracy']*100:.2f}%"
        macro = f"{r['macro_f1']*100:.2f}%"
        fsr = f"{r['fsr_outdated']*100:.2f}%"
        supp_f1 = f"{r['per_class']['SUPPORTED_CURRENT']['f1']*100:.2f}%"
        outd_f1 = f"{r['per_class']['OUTDATED']['f1']*100:.2f}%"
        contra_f1 = f"{r['per_class']['CONTRADICTED']['f1']*100:.2f}%"
        unsupp_f1 = f"{r['per_class']['UNSUPPORTED']['f1']*100:.2f}%"
        print(f"{name:<36} | {acc:<9} | {macro:<9} | {fsr:<11} | {supp_f1:<8} | {outd_f1:<8} | {contra_f1:<9} | {unsupp_f1:<9}")
        
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Parallax Phase A: Full Benchmark Evaluation on test.parquet")
    parser.add_argument("--test_path", type=str, default="data/benchmark/test.parquet", help="Path to test.parquet")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for neural inference")
    parser.add_argument("--output_json", type=str, default="experiments/benchmark_test_results.json")
    parser.add_argument("--output_tex", type=str, default="experiments/table_test_results.tex")
    args = parser.parse_args()
    
    if not os.path.exists(args.test_path):
        print(f"Error: {args.test_path} not found.")
        sys.exit(1)
        
    print(f"Loading official Parallax Test Set from {args.test_path}...")
    df = pd.read_parquet(args.test_path)
    print(f"Loaded {len(df):,} test examples across 4 classes:")
    print(df["gold_label"].value_counts())
    print()
    
    y_true = df["gold_label"].tolist()
    results = {}
    
    # 1. Static Heuristic Baseline
    print(">>> 1/4 Evaluating Static Heuristic Baseline (No Timestamps)...")
    t0 = time.time()
    static_heur_verifier = StaticBaselineVerifier(nli_model=HeuristicNLI())
    preds_static_heur = static_heur_verifier.predict_batch(df)
    results["Static Heuristic Baseline"] = compute_parallax_metrics(y_true, preds_static_heur)
    print(f"    Completed in {time.time()-t0:.2f}s")
    
    # 2. Parallax Heuristic Verifier
    print(">>> 2/4 Evaluating Parallax Heuristic (Temporal Arbitration)...")
    t0 = time.time()
    parallax_heur_verifier = ParallaxTemporalVerifier(nli_model=HeuristicNLI())
    preds_parallax_heur = parallax_heur_verifier.predict_batch(df)
    results["Parallax Heuristic (Temporal)"] = compute_parallax_metrics(y_true, preds_parallax_heur)
    print(f"    Completed in {time.time()-t0:.2f}s")
    
    # 3. Zero-Shot Neural Models
    print("\n>>> Loading Zero-Shot Neural Cross-Encoder (distilroberta-base)...")
    t0 = time.time()
    zero_shot_engine = NeuralCrossEncoderNLI(model_name="cross-encoder/nli-distilroberta-base")
    print(f"    Model loaded in {time.time()-t0:.2f}s")
    
    print("\n>>> 3/6 Evaluating Static Neural Baseline (Zero-Shot Cross-Encoder, No Timestamps)...")
    t0 = time.time()
    preds_static_zs = evaluate_neural_static_batch(df, zero_shot_engine, batch_size=args.batch_size)
    results["Static Neural Baseline (Zero-Shot)"] = compute_parallax_metrics(y_true, preds_static_zs)
    print(f"    Completed in {time.time()-t0:.2f}s")
    
    print("\n>>> 4/6 Evaluating Parallax Neural Two-Stage (Zero-Shot Cross-Encoder + Temporal Arbiter)...")
    t0 = time.time()
    preds_parallax_zs = evaluate_neural_parallax_batch(df, zero_shot_engine, batch_size=args.batch_size)
    results["Parallax Neural Verifier (Zero-Shot)"] = compute_parallax_metrics(y_true, preds_parallax_zs)
    print(f"    Completed in {time.time()-t0:.2f}s")

    # 4. Fine-Tuned Neural Models (if checkpoint exists)
    finetuned_dir = os.path.abspath(os.path.join(_root, "models", "parallax_nli_finetuned"))
    weights_path = os.path.join(finetuned_dir, "model.safetensors")
    if os.path.exists(weights_path):
        print(f"\n>>> Loading Fine-Tuned Neural Cross-Encoder from {finetuned_dir}...")
        t0 = time.time()
        ft_engine = NeuralCrossEncoderNLI(model_name=finetuned_dir)
        print(f"    Model loaded in {time.time()-t0:.2f}s")

        print("\n>>> 5/6 Evaluating Static Neural Baseline (Fine-Tuned Cross-Encoder, No Timestamps)...")
        t0 = time.time()
        preds_static_ft = evaluate_neural_static_batch(df, ft_engine, batch_size=args.batch_size)
        results["Static Neural Baseline (Fine-Tuned)"] = compute_parallax_metrics(y_true, preds_static_ft)
        print(f"    Completed in {time.time()-t0:.2f}s")

        print("\n>>> 6/6 Evaluating Parallax Neural Verifier (Fine-Tuned Cross-Encoder + Temporal Arbiter, Ours)...")
        t0 = time.time()
        preds_parallax_ft = evaluate_neural_parallax_batch(df, ft_engine, batch_size=args.batch_size)
        results["Parallax Neural Verifier (Fine-Tuned, Ours)"] = compute_parallax_metrics(y_true, preds_parallax_ft)
        print(f"    Completed in {time.time()-t0:.2f}s")
    else:
        print(f"\n[Notice] Fine-tuned model directory not found at {finetuned_dir}. Skipping fine-tuned evaluation.")
    
    # Print Comparison Table
    print_comparison_table(results)
    
    # Save detailed JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full benchmark metrics to {args.output_json}")
    
    # Export LaTeX table
    export_latex_table(results, args.output_tex)
    print(f"Saved publication-grade LaTeX table to {args.output_tex}")


if __name__ == "__main__":
    main()
