"""
Document / Cluster-Level Bootstrap Confidence Interval Estimator
================================================================
Computes non-parametric cluster bootstrap confidence intervals.
Rather than resampling individual examples (which falsely assumes i.i.d. independence
between claims derived from the same source document), this module resamples
entire document clusters (target_doc_id) with replacement.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

LABELS = ["OUTDATED", "SUPPORTED_CURRENT", "CONTRADICTED", "UNSUPPORTED"]


def compute_cluster_bootstrap_ci(
    y_true: List[str],
    y_pred: List[str],
    cluster_ids: List[str],
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """
    Computes 1 - alpha (default 95%) confidence intervals for:
    - accuracy (%)
    - macro_f1 (%)
    - fsr_outdated (%)

    Resampling unit: unique cluster_ids.
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(cluster_ids):
        raise ValueError("y_true, y_pred, and cluster_ids must have the exact same length.")

    unique_clusters = np.array(sorted(list(set(cluster_ids))))
    n_clusters = len(unique_clusters)

    # Map cluster -> list of integer sample indices
    cluster_to_indices: Dict[str, List[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        if cid not in cluster_to_indices:
            cluster_to_indices[cid] = []
        cluster_to_indices[cid].append(idx)

    rng = np.random.RandomState(seed)

    boot_accuracies = []
    boot_macro_f1s = []
    boot_fsrs = []

    for _ in range(n_bootstraps):
        sampled_cids = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sample_indices = []
        for cid in sampled_cids:
            sample_indices.extend(cluster_to_indices[cid])

        sub_true = [y_true[i] for i in sample_indices]
        sub_pred = [y_pred[i] for i in sample_indices]

        # 1. Accuracy
        acc = accuracy_score(sub_true, sub_pred) * 100.0
        boot_accuracies.append(acc)

        # 2. Macro-F1
        _, _, f1, _ = precision_recall_fscore_support(
            sub_true, sub_pred, labels=LABELS, zero_division=0
        )
        boot_macro_f1s.append(float(np.mean(f1)) * 100.0)

        # 3. FSR_OUTD
        outd_total = sum(1 for yt in sub_true if yt == "OUTDATED")
        outd_as_supp = sum(
            1 for yt, yp in zip(sub_true, sub_pred) if yt == "OUTDATED" and yp == "SUPPORTED_CURRENT"
        )
        fsr = (outd_as_supp / outd_total * 100.0) if outd_total > 0 else 0.0
        boot_fsrs.append(fsr)

    low_p = (alpha / 2.0) * 100.0
    high_p = (1.0 - alpha / 2.0) * 100.0

    def summarize(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "std_err": float(np.std(arr, ddof=1)),
            "ci_lower": float(np.percentile(arr, low_p)),
            "ci_upper": float(np.percentile(arr, high_p)),
        }

    return {
        "accuracy": summarize(boot_accuracies),
        "macro_f1": summarize(boot_macro_f1s),
        "fsr_outdated": summarize(boot_fsrs),
    }
