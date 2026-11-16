"""
Parallax Pilot Experiment Runner
Runs the Static Baseline Verifier and Parallax Temporal Arbitration Verifier
on the 600-example pilot benchmark. Computes all metrics including FSR_outdated
and exports publication-grade LaTeX tables for the paper.
"""

import sys
import os
import json
import pandas as pd

# Add src and root to path
_cur = os.path.dirname(os.path.abspath(__file__))
_src = os.path.abspath(os.path.join(_cur, "..", "src"))
_root = os.path.abspath(os.path.join(_cur, ".."))
for _p in [_src, _root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from verifier.static_baseline import StaticBaselineVerifier
    from verifier.temporal_arbiter import ParallaxTemporalVerifier
    from evaluation.metrics import compute_parallax_metrics, format_metrics_summary, export_latex_table
except ImportError:
    from src.verifier.static_baseline import StaticBaselineVerifier
    from src.verifier.temporal_arbiter import ParallaxTemporalVerifier
    from src.evaluation.metrics import compute_parallax_metrics, format_metrics_summary, export_latex_table


def run_experiment(benchmark_path: str, output_dir: str):
    print(f"Loading Parallax Pilot Benchmark from {benchmark_path}...")
    df = pd.read_parquet(benchmark_path)
    print(f"Loaded {len(df)} records across 4 classes:")
    print(df["gold_label"].value_counts())
    print()
    
    y_true = df["gold_label"].tolist()
    
    # 1. Run Static Baseline Verifier
    print(">>> Running Static Baseline Verifier (Linguistic Entailment only)...")
    baseline = StaticBaselineVerifier()
    preds_baseline = baseline.predict_batch(df)
    results_baseline = compute_parallax_metrics(y_true, preds_baseline)
    print(format_metrics_summary(results_baseline, model_name="Static NLI Baseline (No Temporal Check)"))
    print()
    
    # 2. Run Parallax Two-Stage Verifier
    print(">>> Running Parallax Verifier (Two-Stage: Entailment + Temporal Arbitration)...")
    parallax = ParallaxTemporalVerifier()
    preds_parallax = parallax.predict_batch(df)
    results_parallax = compute_parallax_metrics(y_true, preds_parallax)
    print(format_metrics_summary(results_parallax, model_name="Parallax (Entailment + Temporal Arbitration)"))
    print()
    
    # 3. Export Results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "pilot_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "Static_Baseline": results_baseline,
            "Parallax": results_parallax,
        }, f, indent=2)
    print(f"Saved evaluation results to {results_path}")
    
    # 4. Export LaTeX Table for Research Paper
    tables_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper", "tables"))
    os.makedirs(tables_dir, exist_ok=True)
    tex_path = os.path.join(tables_dir, "table_main_results.tex")
    export_latex_table({
        "Static NLI Baseline": results_baseline,
        "Parallax Verifier (Ours)": results_parallax,
    }, tex_path)


if __name__ == "__main__":
    b_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "pilot", "pilot_benchmark.parquet"))
    out_dir = os.path.abspath(os.path.dirname(__file__))
    run_experiment(b_path, out_dir)
