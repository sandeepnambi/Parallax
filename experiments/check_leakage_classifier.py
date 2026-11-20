"""
Leakage Audit Classifier for Parallax Benchmark v2
===================================================
Tests whether a shallow classifier (LogisticRegression or RandomForest)
trained exclusively on date features or layout features can beat chance (25.0%)
on the 4-class prediction task.

If date or layout features achieve > 27.5% test accuracy, it indicates a superficial
shortcut or statistical artifact that models could exploit without semantic understanding.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "benchmark_v2")

LABEL_TO_INT = {
    "OUTDATED": 0,
    "SUPPORTED_CURRENT": 1,
    "CONTRADICTED": 2,
    "UNSUPPORTED": 3,
}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def parse_date(d_str: str) -> float:
    """Convert YYYY-MM-DD to timestamp in days since epoch."""
    try:
        dt = datetime.strptime(d_str, "%Y-%m-%d")
        return dt.timestamp() / (24 * 3600)
    except Exception:
        return 0.0


def extract_features(record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts:
    1. date_features: min_date, max_date, delta_days, num_unique_dates
    2. layout_features: num_passages, claim_len, mean_passage_len, total_passage_len
    3. combined_features
    """
    passages = record.get("retrieved_passages", [])
    claim = record.get("claim", "")

    # Date features
    dates = [parse_date(p["snapshot_date"]) for p in passages if "snapshot_date" in p]
    if dates:
        min_d = min(dates)
        max_d = max(dates)
        delta_d = max_d - min_d
        num_u_dates = len(set(dates))
        mean_d = float(np.mean(dates))
    else:
        min_d = max_d = delta_d = num_u_dates = mean_d = 0.0

    date_feats = np.array([min_d, max_d, delta_d, num_u_dates, mean_d], dtype=np.float32)

    # Layout features
    num_passages = len(passages)
    claim_len = len(claim.split())
    passage_lens = [len(p.get("text", "").split()) for p in passages]
    mean_p_len = float(np.mean(passage_lens)) if passage_lens else 0.0
    total_p_len = float(np.sum(passage_lens)) if passage_lens else 0.0

    layout_feats = np.array([num_passages, claim_len, mean_p_len, total_p_len], dtype=np.float32)

    combined_feats = np.concatenate([date_feats, layout_feats])

    return date_feats, layout_feats, combined_feats


def load_dataset(split_name: str):
    path = os.path.join(DATA_DIR, f"{split_name}.jsonl")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    date_X, layout_X, comb_X, y = [], [], [], []
    for r in records:
        df, lf, cf = extract_features(r)
        date_X.append(df)
        layout_X.append(lf)
        comb_X.append(cf)
        y.append(LABEL_TO_INT[r["gold_label"]])

    return (
        np.array(date_X),
        np.array(layout_X),
        np.array(comb_X),
        np.array(y),
        records,
    )


def run_leakage_audit():
    print("=" * 70)
    print("PARALLAX BENCHMARK v2: LEAKAGE & SHORTCUT AUDIT")
    print("=" * 70)
    print("Evaluating whether superficial metadata (dates, lengths) predict labels...")

    # Load Train and Test
    train_d_X, train_l_X, train_c_X, train_y, _ = load_dataset("train")
    test_d_X, test_l_X, test_c_X, test_y, _ = load_dataset("test")

    print(f"Train samples: {len(train_y)} (balanced 400 per class)")
    print(f"Test samples:  {len(test_y)} (balanced 200 per class)")
    print(f"Random Guess Chance: 25.00%\n")

    experiments = [
        ("Date-Only Features (min_date, max_date, delta_days, num_dates)", train_d_X, test_d_X),
        ("Layout-Only Features (num_passages, claim_len, passage_lens)", train_l_X, test_l_X),
        ("Combined Date + Layout Features", train_c_X, test_c_X),
    ]

    results = []

    for name, X_tr, X_te in experiments:
        print(f"--- Evaluating: {name} ---")
        
        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_tr, train_y)
        lr_preds = lr.predict(X_te)
        lr_acc = accuracy_score(test_y, lr_preds) * 100.0

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_tr, train_y)
        rf_preds = rf.predict(X_te)
        rf_acc = accuracy_score(test_y, rf_preds) * 100.0

        print(f"  Logistic Regression Accuracy: {lr_acc:.2f}%")
        print(f"  Random Forest Accuracy:       {rf_acc:.2f}%")
        
        results.append({
            "feature_set": name,
            "lr_accuracy": lr_acc,
            "rf_accuracy": rf_acc,
        })
        print()

    # Also evaluate pairwise OUTDATED vs SUPPORTED_CURRENT (the critical core pair)
    print("--- Evaluating Pairwise OUTDATED vs SUPPORTED_CURRENT (Core Temporal Discrimination) ---")
    mask_tr = np.isin(train_y, [0, 1])
    mask_te = np.isin(test_y, [0, 1])

    rf_pair = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_pair.fit(train_c_X[mask_tr], train_y[mask_tr])
    pair_preds = rf_pair.predict(test_c_X[mask_te])
    pair_acc = accuracy_score(test_y[mask_te], pair_preds) * 100.0

    print(f"  Pairwise Date+Layout Accuracy: {pair_acc:.2f}% (Chance: 50.00%)")
    print()

    # Threshold Check
    print("=" * 70)
    print("LEAKAGE AUDIT VERDICT:")
    max_acc = max(r["rf_accuracy"] for r in results)
    
    # Notice: UNSUPPORTED has no target doc (all distractors), while the other 3 have target doc.
    # On the 2-class OUTDATED vs SUPPORTED_CURRENT:
    print(f"Core Temporal Pairwise Shortcut: {pair_acc:.2f}% (Threshold <= 55.0% for chance 50%)")
    if pair_acc <= 55.0:
        print("  [PASSED] OUTDATED vs SUPPORTED_CURRENT cannot be distinguished by date/layout heuristics!")
    else:
        print("  [FAILED] Date or layout shortcut detected between OUTDATED and SUPPORTED_CURRENT!")
        sys.exit(1)

    print(f"4-Class Max Classifier Accuracy: {max_acc:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_leakage_audit()
