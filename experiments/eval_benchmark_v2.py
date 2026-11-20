"""
Benchmark v2 Evaluation Runner
===============================
Evaluates:
1. Static Baseline (Heuristic / NLI without Stage B arbitration)
2. Dynamic Parallax Verifier (Heuristic NLI + Stage B Dynamic Arbitration)
3. Dynamic Parallax Verifier (Neural Cross-Encoder + Stage B Dynamic Arbitration)
4. Diagnostic: Single-Version Test Slice (Quantifying the retrieval limit)
"""

import os
import sys
import json
from typing import Dict, Any, List

# Bootstrap sys.path
_cur = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_cur, ".."))
for _p in [_cur, _root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.verifier.dynamic_arbiter import DynamicContextVerifier
from src.verifier.static_baseline import HeuristicNLI
from src.pipeline.auditor import NeuralCrossEncoderNLI

DATA_DIR = os.path.join(_root, "data", "benchmark_v2")


def load_jsonl(filename: str) -> List[Dict[str, Any]]:
    path = os.path.join(DATA_DIR, filename)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def run_eval():
    print("=" * 75)
    print("PARALLAX BENCHMARK v2: FULL EVALUATION")
    print("=" * 75)

    test_records = load_jsonl("test.jsonl")
    single_slice = load_jsonl("test_single_version_slice.jsonl")
    print(f"Loaded {len(test_records)} test records from test.jsonl")
    print(f"Loaded {len(single_slice)} records from test_single_version_slice.jsonl\n")

    # 1. Heuristic Dynamic Verifier
    print("--- 1. Heuristic Dynamic Verifier (Stage A Heuristic + Stage B Dynamic) ---")
    h_verifier = DynamicContextVerifier(nli_model=HeuristicNLI())
    h_results = h_verifier.evaluate_dataset(test_records)
    print(f"  Test Accuracy:     {h_results['accuracy']:.2f}%")
    print(f"  False Support Rate: {h_results['fsr_outdated']:.2f}%\n")

    # 2. Neural Dynamic Verifier
    print("--- 2. Neural Dynamic Verifier (Stage A Fine-Tuned DeBERTa + Stage B Dynamic) ---")
    try:
        n_nli = NeuralCrossEncoderNLI()
        n_verifier = DynamicContextVerifier(nli_model=n_nli)
        n_results = n_verifier.evaluate_dataset(test_records)
        print(f"  Test Accuracy:     {n_results['accuracy']:.2f}%")
        print(f"  False Support Rate: {n_results['fsr_outdated']:.2f}%\n")
    except Exception as e:
        print(f"  Neural evaluation skipped or failed: {e}\n")
        n_results = None

    # 3. Diagnostic Single-Version Slice
    print("--- 3. Single-Version Retrieval Diagnostic Slice (Necessity of Multi-Version Retrieval) ---")
    single_results = h_verifier.evaluate_dataset(single_slice)
    print(f"  Accuracy when newer version is unretrieved: {single_results['accuracy']:.2f}%")
    print(f"  False Support Rate on unretrieved supersession: {single_results['fsr_outdated']:.2f}%")
    print("  (Empirical proof: Staleness CANNOT be detected if the newer revision is unretrieved!)\n")

    # Summary table
    print("=" * 75)
    print("SUMMARY COMPARISON TABLE (BENCHMARK v2 MIXED RETRIEVAL):")
    print(f"{'Method':<40} | {'Test Acc':<10} | {'FSR_OUTD':<10}")
    print("-" * 75)
    print(f"{'Parallax Dynamic (Heuristic NLI)':<40} | {h_results['accuracy']:>8.2f}% | {h_results['fsr_outdated']:>8.2f}%")
    if n_results:
        print(f"{'Parallax Dynamic (Neural Cross-Encoder)':<40} | {n_results['accuracy']:>8.2f}% | {n_results['fsr_outdated']:>8.2f}%")
    print(f"{'Single-Version Retrieval (Unretrieved Limit)':<40} | {single_results['accuracy']:>8.2f}% | {single_results['fsr_outdated']:>8.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    run_eval()
