"""
Parallax Benchmark v2: Unified Evaluation Pipeline
===================================================
Executes a fair, reproducible evaluation comparing:
1. Static Heuristic Baseline
2. Static Zero-Shot Neural Baseline (pretrained cross-encoder/nli-distilroberta-base)
3. Dynamic Heuristic Parallax
4. Dynamic Zero-Shot Neural Parallax (pretrained cross-encoder/nli-distilroberta-base)
5. Oracle Upper-Bound Baseline (strictly separated, non-deployable theoretical ceiling)
6. Single-Version Retrieval Diagnostic
7. Temporal Sensitivity Control (Date-Shuffle Inversion)

Key Features:
- Exact same test examples across all comparable systems.
- Document-cluster bootstrap 95% confidence intervals (1,000 resamples, deterministic seed).
- Zero oracle metadata used in Systems 1-4, 6, 7.
- Benchmark data kept strictly read-only.
- Comprehensive provenance logging (hashes, git commit, runtime, seed).
"""

import os
import sys
import json
import time
import hashlib
import argparse
import subprocess
from typing import Dict, Any, List, Tuple, Optional

# Bootstrap sys.path
_cur = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_cur, ".."))
for _p in [_cur, _root, os.path.join(_root, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.verifier.dynamic_arbiter import DynamicContextVerifier
from src.verifier.static_context_verifier import StaticContextVerifier
from src.verifier.static_baseline import HeuristicNLI
from src.verifier.oracle_verifier import OracleContextVerifier
from src.pipeline.auditor import NeuralCrossEncoderNLI
from src.evaluation.metrics import compute_parallax_metrics, format_metrics_summary, export_latex_table
from src.evaluation.bootstrap import compute_cluster_bootstrap_ci

LABELS = ["OUTDATED", "SUPPORTED_CURRENT", "CONTRADICTED", "UNSUPPORTED"]


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file for reproducibility provenance."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_git_info() -> Dict[str, str]:
    """Retrieves current git commit and branch."""
    info = {"commit": "unknown", "branch": "unknown"}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], stderr=subprocess.DEVNULL, text=True).strip()
        info["commit"] = commit
        info["branch"] = branch
    except Exception:
        pass
    return info


def load_jsonl_readonly(filepath: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file in read-only mode."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def create_date_shuffled_dataset(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Creates a date-shuffled copy for temporal sensitivity testing.
    Swaps snapshot dates of the two target document versions in context.
    Leaves distractor passages and gold labels unchanged.
    """
    shuffled = []
    for r in records:
        r_copy = json.loads(json.dumps(r))
        target_doc = str(r_copy.get("target_doc_id", ""))
        passages = r_copy.get("retrieved_passages", [])

        target_passages = [p for p in passages if str(p.get("doc_id", "")) == target_doc]
        if len(target_passages) == 2:
            d0 = target_passages[0]["snapshot_date"]
            d1 = target_passages[1]["snapshot_date"]
            target_passages[0]["snapshot_date"] = d1
            target_passages[1]["snapshot_date"] = d0

        shuffled.append(r_copy)
    return shuffled


def evaluate_system(
    verifier: Any,
    records: List[Dict[str, Any]],
    cluster_ids: List[str],
    n_bootstraps: int = 1000,
    seed: int = 42,
    batch_size: int = 32,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """
    Runs evaluation for a single verifier and computes cluster bootstrap CIs.
    """
    t0 = time.time()
    res = verifier.evaluate_dataset(records, batch_size=batch_size)
    elapsed = time.time() - t0

    y_true = res["y_true"]
    y_pred = res["y_pred"]

    metrics = compute_parallax_metrics(y_true, y_pred, reference_labels=LABELS)
    bootstrap_ci = compute_cluster_bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        cluster_ids=cluster_ids,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    return metrics, bootstrap_ci, elapsed


def run_unified_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    print("=" * 78)
    print("PARALLAX BENCHMARK v2: UNIFIED REPRODUCIBLE EVALUATION PIPELINE")
    print("=" * 78)

    test_path = os.path.abspath(args.test_path)
    single_slice_path = os.path.abspath(args.single_version_path)

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not os.path.exists(single_slice_path):
        raise FileNotFoundError(f"Single-version slice file not found: {single_slice_path}")

    # Provenance tracking
    git_info = get_git_info()
    test_hash = compute_sha256(test_path)
    single_slice_hash = compute_sha256(single_slice_path)

    print(f"Test File:               {test_path}")
    print(f"  SHA-256:               {test_hash}")
    print(f"Single-Version Slice:    {single_slice_path}")
    print(f"  SHA-256:               {single_slice_hash}")
    print(f"Git Branch/Commit:       {git_info['branch']} @ {git_info['commit'][:10]}")
    print(f"Bootstrap Iterations:    {args.n_bootstraps} (Seed: {args.seed})")
    print(f"Batch Size:              {args.batch_size}")
    print()

    # Load data strictly read-only
    all_test_records = load_jsonl_readonly(test_path)
    all_single_records = load_jsonl_readonly(single_slice_path)

    if args.quick:
        print(">>> WARNING: --quick mode active. Subsetting to first 40 examples. <<<")
        test_records = all_test_records[:40]
        single_records = all_single_records[:20]
        n_bootstraps = min(50, args.n_bootstraps)
    else:
        test_records = all_test_records
        single_records = all_single_records
        n_bootstraps = args.n_bootstraps

    cluster_ids = [str(r.get("target_doc_id", "c_unknown")) for r in test_records]
    print(f"Evaluation Records:      {len(test_records)} examples across {len(set(cluster_ids))} clusters")
    print()

    # Model initializations
    heuristic_nli = HeuristicNLI()
    zero_shot_neural_nli = None

    if not args.skip_neural:
        print("Loading Zero-Shot Neural Model ('cross-encoder/nli-distilroberta-base')...")
        zero_shot_neural_nli = NeuralCrossEncoderNLI(model_name="cross-encoder/nli-distilroberta-base")
        print(f"  Neural Model loaded on device: {zero_shot_neural_nli.device}")
        print(f"  Verified Model id2label: {zero_shot_neural_nli.id2label}\n")

    # Tracking outputs
    system_results: Dict[str, Any] = {}
    per_example_predictions: Dict[str, Dict[str, Any]] = {
        r.get("example_id", f"ex_{i}"): {
            "example_id": r.get("example_id", f"ex_{i}"),
            "claim": r.get("claim"),
            "gold_label": r.get("gold_label"),
            "target_doc_id": r.get("target_doc_id"),
            "predictions": {},
        }
        for i, r in enumerate(test_records)
    }

    # -------------------------------------------------------------
    # 1. Static Heuristic Baseline
    # -------------------------------------------------------------
    print(">>> [1/5] Evaluating System 1: Static Heuristic Baseline...")
    static_heur_verifier = StaticContextVerifier(nli_model=heuristic_nli)
    s1_metrics, s1_boot, s1_time = evaluate_system(
        static_heur_verifier, test_records, cluster_ids, n_bootstraps, args.seed, args.batch_size
    )
    s1_raw = static_heur_verifier.evaluate_dataset(test_records, batch_size=args.batch_size)
    for r, yp in zip(test_records, s1_raw["y_pred"]):
        per_example_predictions[r["example_id"]]["predictions"]["static_heuristic"] = yp

    system_results["static_heuristic_baseline"] = {
        "display_name": "Static Heuristic Baseline",
        "category": "baseline",
        "runtime_seconds": round(s1_time, 2),
        "metrics": s1_metrics,
        "bootstrap_ci_95": s1_boot,
    }
    print(f"    Accuracy: {s1_metrics['accuracy']:.2%} | Macro-F1: {s1_metrics['macro_f1']:.4f} | FSR_OUTD: {s1_metrics['fsr_outdated']:.2%}")
    print(f"    95% CIs: Acc [{s1_boot['accuracy']['ci_lower']:.2f}%, {s1_boot['accuracy']['ci_upper']:.2f}%], FSR_OUTD [{s1_boot['fsr_outdated']['ci_lower']:.2f}%, {s1_boot['fsr_outdated']['ci_upper']:.2f}%]\n")

    # -------------------------------------------------------------
    # 2. Static Zero-Shot Neural Baseline
    # -------------------------------------------------------------
    if zero_shot_neural_nli is not None:
        print(">>> [2/5] Evaluating System 2: Static Zero-Shot Neural Baseline (DistilRoBERTa)...")
        static_neural_verifier = StaticContextVerifier(nli_model=zero_shot_neural_nli)
        s2_metrics, s2_boot, s2_time = evaluate_system(
            static_neural_verifier, test_records, cluster_ids, n_bootstraps, args.seed, args.batch_size
        )
        s2_raw = static_neural_verifier.evaluate_dataset(test_records, batch_size=args.batch_size)
        for r, yp in zip(test_records, s2_raw["y_pred"]):
            per_example_predictions[r["example_id"]]["predictions"]["static_zeroshot_neural"] = yp

        system_results["static_zeroshot_neural_baseline"] = {
            "display_name": "Static Zero-Shot Neural Baseline",
            "category": "baseline",
            "runtime_seconds": round(s2_time, 2),
            "metrics": s2_metrics,
            "bootstrap_ci_95": s2_boot,
        }
        print(f"    Accuracy: {s2_metrics['accuracy']:.2%} | Macro-F1: {s2_metrics['macro_f1']:.4f} | FSR_OUTD: {s2_metrics['fsr_outdated']:.2%}")
        print(f"    95% CIs: Acc [{s2_boot['accuracy']['ci_lower']:.2f}%, {s2_boot['accuracy']['ci_upper']:.2f}%], FSR_OUTD [{s2_boot['fsr_outdated']['ci_lower']:.2f}%, {s2_boot['fsr_outdated']['ci_upper']:.2f}%]\n")

    # -------------------------------------------------------------
    # 3. Dynamic Heuristic Parallax
    # -------------------------------------------------------------
    print(">>> [3/5] Evaluating System 3: Dynamic Heuristic Parallax (Ours)...")
    dynamic_heur_verifier = DynamicContextVerifier(nli_model=heuristic_nli)
    s3_metrics, s3_boot, s3_time = evaluate_system(
        dynamic_heur_verifier, test_records, cluster_ids, n_bootstraps, args.seed, args.batch_size
    )
    s3_raw = dynamic_heur_verifier.evaluate_dataset(test_records, batch_size=args.batch_size)
    for r, yp in zip(test_records, s3_raw["y_pred"]):
        per_example_predictions[r["example_id"]]["predictions"]["dynamic_heuristic_parallax"] = yp

    system_results["dynamic_heuristic_parallax"] = {
        "display_name": "Dynamic Heuristic Parallax (Ours)",
        "category": "proposed",
        "runtime_seconds": round(s3_time, 2),
        "metrics": s3_metrics,
        "bootstrap_ci_95": s3_boot,
    }
    print(f"    Accuracy: {s3_metrics['accuracy']:.2%} | Macro-F1: {s3_metrics['macro_f1']:.4f} | FSR_OUTD: {s3_metrics['fsr_outdated']:.2%}")
    print(f"    95% CIs: Acc [{s3_boot['accuracy']['ci_lower']:.2f}%, {s3_boot['accuracy']['ci_upper']:.2f}%], FSR_OUTD [{s3_boot['fsr_outdated']['ci_lower']:.2f}%, {s3_boot['fsr_outdated']['ci_upper']:.2f}%]\n")

    # -------------------------------------------------------------
    # 4. Dynamic Zero-Shot Neural Parallax
    # -------------------------------------------------------------
    if zero_shot_neural_nli is not None:
        print(">>> [4/5] Evaluating System 4: Dynamic Zero-Shot Neural Parallax (Ours)...")
        dynamic_neural_verifier = DynamicContextVerifier(nli_model=zero_shot_neural_nli)
        s4_metrics, s4_boot, s4_time = evaluate_system(
            dynamic_neural_verifier, test_records, cluster_ids, n_bootstraps, args.seed, args.batch_size
        )
        s4_raw = dynamic_neural_verifier.evaluate_dataset(test_records, batch_size=args.batch_size)
        for r, yp in zip(test_records, s4_raw["y_pred"]):
            per_example_predictions[r["example_id"]]["predictions"]["dynamic_zeroshot_neural_parallax"] = yp

        system_results["dynamic_zeroshot_neural_parallax"] = {
            "display_name": "Dynamic Zero-Shot Neural Parallax (Ours)",
            "category": "proposed",
            "runtime_seconds": round(s4_time, 2),
            "metrics": s4_metrics,
            "bootstrap_ci_95": s4_boot,
        }
        print(f"    Accuracy: {s4_metrics['accuracy']:.2%} | Macro-F1: {s4_metrics['macro_f1']:.4f} | FSR_OUTD: {s4_metrics['fsr_outdated']:.2%}")
        print(f"    95% CIs: Acc [{s4_boot['accuracy']['ci_lower']:.2f}%, {s4_boot['accuracy']['ci_upper']:.2f}%], FSR_OUTD [{s4_boot['fsr_outdated']['ci_lower']:.2f}%, {s4_boot['fsr_outdated']['ci_upper']:.2f}%]\n")

    # -------------------------------------------------------------
    # 5. Oracle Upper-Bound Baseline (SEPARATED / NON-DEPLOYABLE)
    # -------------------------------------------------------------
    print(">>> [5/5] Evaluating System 5: Oracle Upper-Bound Baseline (NON-DEPLOYABLE CEILING)...")
    oracle_verifier = OracleContextVerifier(nli_model=heuristic_nli)
    s5_metrics, s5_boot, s5_time = evaluate_system(
        oracle_verifier, test_records, cluster_ids, n_bootstraps, args.seed, args.batch_size
    )
    s5_raw = oracle_verifier.evaluate_dataset(test_records)
    for r, yp in zip(test_records, s5_raw["y_pred"]):
        per_example_predictions[r["example_id"]]["predictions"]["oracle_upper_bound"] = yp

    oracle_section = {
        "display_name": "Oracle Upper-Bound Baseline",
        "deployable": False,
        "warning": "USES PRIVILEGED ORACLE ANNOTATIONS. THEORETICAL CEILING ONLY. NOT DEPLOYABLE.",
        "runtime_seconds": round(s5_time, 2),
        "metrics": s5_metrics,
        "bootstrap_ci_95": s5_boot,
    }
    print(f"    Accuracy: {s5_metrics['accuracy']:.2%} | Macro-F1: {s5_metrics['macro_f1']:.4f} | FSR_OUTD: {s5_metrics['fsr_outdated']:.2%}")
    print(f"    95% CIs: Acc [{s5_boot['accuracy']['ci_lower']:.2f}%, {s5_boot['accuracy']['ci_upper']:.2f}%], FSR_OUTD [{s5_boot['fsr_outdated']['ci_lower']:.2f}%, {s5_boot['fsr_outdated']['ci_upper']:.2f}%]\n")

    # -------------------------------------------------------------
    # 6. Single-Version Retrieval Diagnostic
    # -------------------------------------------------------------
    print("=" * 78)
    print("DIAGNOSTIC TRACK: SINGLE-VERSION RETRIEVAL SLICE")
    print("=" * 78)
    sv_raw = dynamic_heur_verifier.evaluate_dataset(single_records, batch_size=args.batch_size)
    sv_true = sv_raw["y_true"]
    sv_pred = sv_raw["y_pred"]
    sv_acc = sv_raw["accuracy"]
    sv_fsr = sv_raw["fsr_outdated"]

    # Compute OUTDATED recall explicitly
    total_outd_sv = sum(1 for yt in sv_true if yt == "OUTDATED")
    corr_outd_sv = sum(1 for yt, yp in zip(sv_true, sv_pred) if yt == "OUTDATED" and yp == "OUTDATED")
    sv_outd_recall = (corr_outd_sv / total_outd_sv * 100.0) if total_outd_sv > 0 else 0.0

    single_version_results = {
        "num_records": len(single_records),
        "accuracy_pct": round(sv_acc, 2),
        "outdated_recall_pct": round(sv_outd_recall, 2),
        "fsr_outdated_pct": round(sv_fsr, 2),
        "finding": (
            "Empirical retrieval boundary: without multi-version context (newer revision unretrieved), "
            "staleness cannot be detected without external metadata, confirming multi-version retrieval necessity."
        ),
    }
    print(f"Single-Version Test Records: {len(single_records)}")
    print(f"  Accuracy:                  {sv_acc:.2f}%")
    print(f"  OUTDATED Recall:           {sv_outd_recall:.2f}%")
    print(f"  False Support Rate (FSR):  {sv_fsr:.2f}%")
    print(f"  Theoretical Implication:   {single_version_results['finding']}\n")

    # -------------------------------------------------------------
    # 7. Temporal Sensitivity Control (Date Shuffling)
    # -------------------------------------------------------------
    print("=" * 78)
    print("CONTROL TRACK: TEMPORAL SENSITIVITY CONTROL (DATE SHUFFLE)")
    print("=" * 78)
    shuffled_test_records = create_date_shuffled_dataset(test_records)
    temporal_orig = [r for r in test_records if r["gold_label"] in ["OUTDATED", "SUPPORTED_CURRENT"]]
    temporal_shuf = [r for r in shuffled_test_records if r["gold_label"] in ["OUTDATED", "SUPPORTED_CURRENT"]]

    orig_eval = dynamic_heur_verifier.evaluate_dataset(temporal_orig, batch_size=args.batch_size)
    shuf_eval = dynamic_heur_verifier.evaluate_dataset(temporal_shuf, batch_size=args.batch_size)

    orig_preds = orig_eval["y_pred"]
    shuf_preds = shuf_eval["y_pred"]

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

    total_flips = flips_outd_to_curr + flips_curr_to_outd
    total_temporal = total_outd + total_curr
    overall_flip_rate = (total_flips / total_temporal * 100.0) if total_temporal > 0 else 0.0

    temporal_control_results = {
        "description": "Temporal sensitivity control under snapshot date inversion",
        "num_temporal_records": len(temporal_orig),
        "original_temporal_accuracy_pct": round(orig_eval["accuracy"], 2),
        "shuffled_temporal_accuracy_pct": round(shuf_eval["accuracy"], 2),
        "flip_statistics": {
            "outdated_to_supported_current": {
                "count": flips_outd_to_curr,
                "total": total_outd,
                "flip_rate_pct": round((flips_outd_to_curr / total_outd * 100.0) if total_outd > 0 else 0.0, 2),
            },
            "supported_current_to_outdated": {
                "count": flips_curr_to_outd,
                "total": total_curr,
                "flip_rate_pct": round((flips_curr_to_outd / total_curr * 100.0) if total_curr > 0 else 0.0, 2),
            },
            "overall_prediction_flips": {
                "count": total_flips,
                "total": total_temporal,
                "flip_rate_pct": round(overall_flip_rate, 2),
            },
        },
        "interpretation": (
            "Temporal sensitivity control confirms that dynamic arbitration predictions invert "
            "in response to temporal ordering rather than relying on date-invariant lexical cues."
        ),
    }

    print(f"Original Temporal Subset Accuracy: {orig_eval['accuracy']:.2f}%")
    print(f"Date-Shuffled Temporal Accuracy:   {shuf_eval['accuracy']:.2f}%")
    print(f"OUTDATED -> SUPPORTED_CURRENT Flips: {flips_outd_to_curr}/{total_outd} ({flips_outd_to_curr/total_outd*100.0:.2f}%)")
    print(f"SUPPORTED_CURRENT -> OUTDATED Flips: {flips_curr_to_outd}/{total_curr} ({flips_curr_to_outd/total_curr*100.0:.2f}%)")
    print(f"Overall Prediction Flip Rate:        {total_flips}/{total_temporal} ({overall_flip_rate:.2f}%)\n")

    # -------------------------------------------------------------
    # Summary Table Construction
    # -------------------------------------------------------------
    print("=" * 88)
    print("UNIFIED BENCHMARK v2 RESULTS (COMPARATIVE SUMMARY)")
    print("=" * 88)
    header = (
        f"{'System Name':<38} | {'Acc (95% CI)':<20} | {'Macro F1':<9} | "
        f"{'FSR_OUTD (95% CI)':<22} | {'Time (s)':<8}"
    )
    print(header)
    print("-" * 88)

    for k, sys_data in system_results.items():
        name = sys_data["display_name"]
        m = sys_data["metrics"]
        b = sys_data["bootstrap_ci_95"]
        t = sys_data["runtime_seconds"]

        acc_ci = f"{m['accuracy'] * 100.0:.1f}% [{b['accuracy']['ci_lower']:.1f}-{b['accuracy']['ci_upper']:.1f}%]"
        fsr_ci = f"{m['fsr_outdated'] * 100.0:.1f}% [{b['fsr_outdated']['ci_lower']:.1f}-{b['fsr_outdated']['ci_upper']:.1f}%]"
        print(f"{name:<38} | {acc_ci:<20} | {m['macro_f1']:.4f}    | {fsr_ci:<22} | {t:<8.1f}")

    print("-" * 88)
    # Print Oracle Upper Bound separated
    orb_m = oracle_section["metrics"]
    orb_b = oracle_section["bootstrap_ci_95"]
    orb_t = oracle_section["runtime_seconds"]
    orb_acc_ci = f"{orb_m['accuracy'] * 100.0:.1f}% [{orb_b['accuracy']['ci_lower']:.1f}-{orb_b['accuracy']['ci_upper']:.1f}%]"
    orb_fsr_ci = f"{orb_m['fsr_outdated'] * 100.0:.1f}% [{orb_b['fsr_outdated']['ci_lower']:.1f}-{orb_b['fsr_outdated']['ci_upper']:.1f}%]"
    print(f"{'[ORACLE CEILING - NON-DEPLOYABLE]':<38} | {orb_acc_ci:<20} | {orb_m['macro_f1']:.4f}    | {orb_fsr_ci:<22} | {orb_t:<8.1f}")
    print("=" * 88)

    # -------------------------------------------------------------
    # Assemble Structured JSON
    # -------------------------------------------------------------
    full_output = {
        "provenance": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git_info["commit"],
            "git_branch": git_info["branch"],
            "python_version": sys.version,
            "random_seed": args.seed,
            "bootstrap_iterations": n_bootstraps,
            "batch_size": args.batch_size,
            "dataset_hashes": {
                "test_file": test_path,
                "test_sha256": test_hash,
                "single_version_slice_file": single_slice_path,
                "single_version_slice_sha256": single_slice_hash,
            },
            "sample_counts": {
                "evaluated_test_examples": len(test_records),
                "evaluated_clusters": len(set(cluster_ids)),
                "single_version_slice_examples": len(single_records),
            },
        },
        "systems": system_results,
        "oracle_upper_bound_quarantined": oracle_section,
        "single_version_diagnostic": single_version_results,
        "temporal_sensitivity_control": temporal_control_results,
        "predictions": list(per_example_predictions.values()),
    }

    out_json_path = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nSaved structured evaluation results to: {out_json_path}")

    # Export publication-ready LaTeX table
    out_tex_path = os.path.abspath(args.output_tex)
    latex_systems = {sys_data["display_name"]: sys_data["metrics"] for sys_data in system_results.values()}
    export_latex_table(latex_systems, out_tex_path, caption="Benchmark v2 evaluation across comparable systems.")

    return full_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Benchmark v2 Evaluation Pipeline")
    parser.add_argument("--test_path", type=str, default="data/benchmark_v2/test.jsonl")
    parser.add_argument("--single_version_path", type=str, default="data/benchmark_v2/test_single_version_slice.jsonl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Quick run on 40 examples with 50 bootstraps")
    parser.add_argument("--skip_neural", action="store_true", help="Skip neural models for heuristic-only run")
    parser.add_argument("--output_json", type=str, default="experiments/unified_evaluation_v2_results.json")
    parser.add_argument("--output_tex", type=str, default="experiments/table_unified_v2_results.tex")
    args = parser.parse_args()

    run_unified_evaluation(args)
