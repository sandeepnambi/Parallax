"""
Date-Shuffle Control Experiment
===============================
Hypothesis:
If Parallax's dynamic Stage B supersession inference is genuinely causal with respect
to temporal ordering (and not relying on lexical artifacts), swapping the snapshot dates
between the old and new revisions should invert the temporal predictions:
- OUTDATED claims (where old snapshot entails claim) should flip to SUPPORTED_CURRENT.
- SUPPORTED_CURRENT claims (where new snapshot entails claim) should flip to OUTDATED.
"""

import os
import sys
import json
from typing import Dict, Any, List

_cur = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_cur, ".."))
for _p in [_cur, _root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.verifier.dynamic_arbiter import DynamicContextVerifier
from src.verifier.static_baseline import HeuristicNLI

DATA_DIR = os.path.join(_root, "data", "benchmark_v2")


def load_test_data() -> List[Dict[str, Any]]:
    path = os.path.join(DATA_DIR, "test.jsonl")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def create_date_shuffled_copy(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For every example, find the two versions of the target document,
    and swap their snapshot_date fields.
    """
    shuffled_records = []
    for r in records:
        r_copy = json.loads(json.dumps(r))
        target_doc = str(r_copy.get("target_doc_id"))
        passages = r_copy.get("retrieved_passages", [])
        
        target_passages = [p for p in passages if str(p.get("doc_id")) == target_doc]
        if len(target_passages) == 2:
            # Swap dates
            d0 = target_passages[0]["snapshot_date"]
            d1 = target_passages[1]["snapshot_date"]
            target_passages[0]["snapshot_date"] = d1
            target_passages[1]["snapshot_date"] = d0

        shuffled_records.append(r_copy)
    return shuffled_records


def run_date_shuffle():
    print("=" * 70)
    print("PARALLAX BENCHMARK v2: DATE-SHUFFLE CAUSAL CONTROL")
    print("=" * 70)

    original_records = load_test_data()
    shuffled_records = create_date_shuffled_copy(original_records)

    verifier = DynamicContextVerifier(nli_model=HeuristicNLI())

    # Focus on the temporal pair: OUTDATED and SUPPORTED_CURRENT
    temporal_orig = [r for r in original_records if r["gold_label"] in ["OUTDATED", "SUPPORTED_CURRENT"]]
    temporal_shuf = [r for r in shuffled_records if r["gold_label"] in ["OUTDATED", "SUPPORTED_CURRENT"]]

    print(f"Evaluating on {len(temporal_orig)} temporal test examples (200 OUTDATED, 200 SUPPORTED_CURRENT)...\n")

    orig_eval = verifier.evaluate_dataset(temporal_orig)
    shuf_eval = verifier.evaluate_dataset(temporal_shuf)

    print(f"Original Benchmark Accuracy on Temporal Subset: {orig_eval['accuracy']:.2f}%")
    print(f"Date-Shuffled Accuracy on Temporal Subset:     {shuf_eval['accuracy']:.2f}%\n")

    # Analyze flip rate
    orig_preds = [d["pred_label"] for d in orig_eval["detailed_results"]]
    shuf_preds = [d["pred_label"] for d in shuf_eval["detailed_results"]]

    flips_outd_to_curr = 0
    flips_curr_to_outd = 0
    total_outd = 0
    total_curr = 0

    for r, p_orig, p_shuf in zip(temporal_orig, orig_preds, shuf_preds):
        gold = r["gold_label"]
        if gold == "OUTDATED":
            total_outd += 1
            if p_orig == "OUTDATED" and p_shuf == "SUPPORTED_CURRENT":
                flips_outd_to_curr += 1
        elif gold == "SUPPORTED_CURRENT":
            total_curr += 1
            if p_orig == "SUPPORTED_CURRENT" and p_shuf == "OUTDATED":
                flips_curr_to_outd += 1

    print("Causal Inversion Analysis:")
    print(f"  OUTDATED -> SUPPORTED_CURRENT flip rate:        {flips_outd_to_curr}/{total_outd} ({flips_outd_to_curr/total_outd*100.0:.2f}%)")
    print(f"  SUPPORTED_CURRENT -> OUTDATED flip rate:        {flips_curr_to_outd}/{total_curr} ({flips_curr_to_outd/total_curr*100.0:.2f}%)")
    
    total_flips = flips_outd_to_curr + flips_curr_to_outd
    total_temporal = total_outd + total_curr
    overall_flip_rate = total_flips / total_temporal * 100.0
    print(f"  Overall Causal Prediction Flip Rate:           {total_flips}/{total_temporal} ({overall_flip_rate:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    run_date_shuffle()
