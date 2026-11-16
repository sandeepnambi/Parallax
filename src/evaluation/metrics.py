"""
Parallax Evaluation Metrics
Implements precision, recall, F1, Macro-F1, confusion matrix,
and the signature metric: False-Support Rate on Outdated Claims (FSR_outdated).
Also includes publication-ready LaTeX table exporters.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

LABELS = ["SUPPORTED_CURRENT", "OUTDATED", "CONTRADICTED", "UNSUPPORTED"]
LABEL_MAP = {lbl: idx for idx, lbl in enumerate(LABELS)}


def compute_parallax_metrics(
    y_true: List[str],
    y_pred: List[str],
    reference_labels: List[str] = LABELS
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for Parallax benchmark.
    """
    # Precision, Recall, F1 per class
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=reference_labels, zero_division=0
    )
    
    macro_f1 = float(np.mean(f1))
    acc = float(accuracy_score(y_true, y_pred))
    
    per_class = {}
    for i, lbl in enumerate(reference_labels):
        per_class[lbl] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=reference_labels)
    
    # Calculate False-Support Rate on OUTDATED claims:
    # FSR = count(y_true == OUTDATED and y_pred == SUPPORTED_CURRENT) / count(y_true == OUTDATED)
    total_outdated = sum(1 for yt in y_true if yt == "OUTDATED")
    false_supported_outdated = sum(
        1 for yt, yp in zip(y_true, y_pred) if yt == "OUTDATED" and yp == "SUPPORTED_CURRENT"
    )
    
    fsr_outdated = (false_supported_outdated / total_outdated) if total_outdated > 0 else 0.0
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "fsr_outdated": float(fsr_outdated),
        "false_supported_outdated_count": false_supported_outdated,
        "total_outdated_count": total_outdated,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": reference_labels,
    }


def format_metrics_summary(results: Dict[str, Any], model_name: str = "Model") -> str:
    """Formats metrics as a clean, readable text table."""
    lines = [
        "============================================================",
        f"Evaluation Summary: {model_name}",
        "============================================================",
        f"Overall Accuracy:           {results['accuracy']:.4f}",
        f"Macro-F1 (4-class):         {results['macro_f1']:.4f}",
        f"False-Support Rate (OUTD):  {results['fsr_outdated']:.2%} ({results['false_supported_outdated_count']}/{results['total_outdated_count']})",
        "------------------------------------------------------------",
        f"{'Label':<20} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'Support':<7}",
        "------------------------------------------------------------",
    ]
    for lbl in results["labels"]:
        m = results["per_class"][lbl]
        lines.append(f"{lbl:<20} | {m['precision']:<7.4f} | {m['recall']:<7.4f} | {m['f1']:<7.4f} | {m['support']:<7d}")
    lines.append("------------------------------------------------------------")
    
    lines.append("\nConfusion Matrix (Rows: True, Cols: Pred):")
    header = f"{'True \\ Pred':<20} | " + " | ".join(f"{lbl[:7]:<7}" for lbl in results["labels"])
    lines.append(header)
    lines.append("-" * len(header))
    for i, row in enumerate(results["confusion_matrix"]):
        row_str = f"{results['labels'][i]:<20} | " + " | ".join(f"{val:<7d}" for val in row)
        lines.append(row_str)
    lines.append("============================================================")
    return "\n".join(lines)


def export_latex_table(results_dict: Dict[str, Dict[str, Any]], filepath: str, caption: Optional[str] = None):
    """
    Exports a comparative results table formatted for ACL/EMNLP LaTeX submissions.
    results_dict: { model_name: metrics_dict }
    """
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l|c|cccc|c}",
        r"\toprule",
        r"\textbf{System} & \textbf{Macro-F1} & \textbf{Supp-Curr F1} & \textbf{Outdated F1} & \textbf{Contra F1} & \textbf{Unsupp F1} & $\mathbf{FSR}_{\text{OUTD}} \downarrow$ \\",
        r"\midrule",
    ]
    for name, res in results_dict.items():
        pc = res["per_class"]
        sc_f1 = pc["SUPPORTED_CURRENT"]["f1"]
        out_f1 = pc["OUTDATED"]["f1"]
        con_f1 = pc["CONTRADICTED"]["f1"]
        un_f1 = pc["UNSUPPORTED"]["f1"]
        macro = res["macro_f1"]
        fsr = res["fsr_outdated"] * 100.0
        row_tex = f"{name} & {macro:.3f} & {sc_f1:.3f} & {out_f1:.3f} & {con_f1:.3f} & {un_f1:.3f} & {fsr:.1f}\\% \\\\"
        tex.append(row_tex)
    def_caption = r"Evaluation on the Parallax Benchmark (800 test examples). $\text{FSR}_{\text{OUTD}}$ measures the percentage of superseded claims falsely certified as current."
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption or def_caption}}}",
        r"\label{tab:main_results}",
        r"\end{table*}",
    ])
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Exported LaTeX table to {filepath}")
