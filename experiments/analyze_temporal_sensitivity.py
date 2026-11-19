"""
Parallax Phase B: Temporal Sensitivity & Domain Breakdown Analysis
==================================================================
Investigates how verification performance and the False-Support Rate (FSR)
vary as a function of:
1. Staleness Elapsed Duration (0-30d, 31-60d, 61-90d, 91-120d, 121-153d)
2. Knowledge Domain (Sports, Politics, Demographics, Media, Science, General)

Exports structured JSON results and publication-ready comparative tables.
"""

import os
import sys
import json
import argparse

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
    from evaluation.metrics import compute_parallax_metrics
except ImportError:
    from src.verifier.static_baseline import StaticBaselineVerifier, HeuristicNLI
    from src.verifier.temporal_arbiter import ParallaxTemporalVerifier
    from src.pipeline.auditor import NeuralCrossEncoderNLI
    from src.evaluation.metrics import compute_parallax_metrics


def get_staleness_bucket(days: float) -> str:
    if pd.isna(days) or days < 0:
        return "N/A"
    elif days <= 30:
        return "0–30 days (~1 mo)"
    elif days <= 60:
        return "31–60 days (~2 mos)"
    elif days <= 90:
        return "61–90 days (~3 mos)"
    elif days <= 120:
        return "91–120 days (~4 mos)"
    else:
        return "121–153 days (5+ mos)"


def run_analysis(test_path: str, output_json: str):
    print(f"Loading Parallax test set from {test_path}...")
    df = pd.read_parquet(test_path)
    
    # 1. Generate predictions for all 4 models
    print("Generating predictions for all 4 models...")
    
    # Static Heuristic
    sh_verifier = StaticBaselineVerifier(nli_model=HeuristicNLI())
    df["pred_static_heuristic"] = sh_verifier.predict_batch(df)
    
    # Parallax Heuristic
    ph_verifier = ParallaxTemporalVerifier(nli_model=HeuristicNLI())
    df["pred_parallax_heuristic"] = ph_verifier.predict_batch(df)
    
    # Neural Models
    print("Loading Neural Cross-Encoder...")
    neural_engine = NeuralCrossEncoderNLI()
    premises = df["evidence_text"].tolist()
    hypotheses = df["answer_claim"].tolist()
    
    print("Running batched neural inference (800 examples)...")
    raw_nli = neural_engine.predict_batch(premises, hypotheses, batch_size=32)
    
    # Static Neural Mapping
    sn_preds = []
    for rel in raw_nli:
        if rel == "ENTAILMENT":
            sn_preds.append("SUPPORTED_CURRENT")
        elif rel == "CONTRADICTION":
            sn_preds.append("CONTRADICTED")
        else:
            sn_preds.append("UNSUPPORTED")
    df["pred_static_neural"] = sn_preds
    
    # Parallax Neural Mapping
    pn_preds = []
    for i, rel in enumerate(raw_nli):
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
                pn_preds.append("OUTDATED")
            else:
                pn_preds.append("SUPPORTED_CURRENT")
        elif rel == "CONTRADICTION":
            pn_preds.append("CONTRADICTED")
        else:
            pn_preds.append("UNSUPPORTED")
    df["pred_parallax_neural"] = pn_preds
    
    # 2. Staleness Duration Analysis (on OUTDATED claims)
    outdated_df = df[df["gold_label"] == "OUTDATED"].copy()
    outdated_df["staleness_bucket"] = outdated_df["staleness_days"].apply(get_staleness_bucket)
    
    bucket_order = [
        "0–30 days (~1 mo)",
        "31–60 days (~2 mos)",
        "61–90 days (~3 mos)",
        "91–120 days (~4 mos)",
        "121–153 days (5+ mos)",
    ]
    
    staleness_results = {}
    print("\n" + "=" * 80)
    print("STALENESS SENSITIVITY ANALYSIS (False-Support Rate by Elapsed Time)")
    print("=" * 80)
    print(f"{'Staleness Bucket':<24} | {'Count':<6} | {'Static Heur FSR':<15} | {'Static Neural FSR':<17} | {'Parallax FSR (Ours)':<18}")
    print("-" * 80)
    
    for b in bucket_order:
        b_df = outdated_df[outdated_df["staleness_bucket"] == b]
        cnt = len(b_df)
        if cnt == 0:
            continue
        fsr_sh = (b_df["pred_static_heuristic"] == "SUPPORTED_CURRENT").sum() / cnt
        fsr_sn = (b_df["pred_static_neural"] == "SUPPORTED_CURRENT").sum() / cnt
        fsr_pn = (b_df["pred_parallax_neural"] == "SUPPORTED_CURRENT").sum() / cnt
        
        staleness_results[b] = {
            "count": cnt,
            "fsr_static_heuristic": round(float(fsr_sh), 4),
            "fsr_static_neural": round(float(fsr_sn), 4),
            "fsr_parallax_neural": round(float(fsr_pn), 4),
        }
        print(f"{b:<24} | {cnt:<6} | {fsr_sh*100:<15.1f}% | {fsr_sn*100:<17.1f}% | {fsr_pn*100:<18.1f}%")
        
    print("=" * 80)
    
    # 3. Domain Breakdown Analysis
    domains = df["domain"].value_counts().index.tolist()
    domain_results = {}
    
    print("\n" + "=" * 90)
    print("DOMAIN-SPECIFIC BREAKDOWN ANALYSIS")
    print("=" * 90)
    print(f"{'Domain':<28} | {'Total':<6} | {'Static Heur F1':<14} | {'Parallax Heur F1':<16} | {'Static Neural F1':<16} | {'Parallax Neural F1':<18}")
    print("-" * 90)
    
    for d in domains:
        d_df = df[df["domain"] == d]
        y_true = d_df["gold_label"].tolist()
        
        m_sh = compute_parallax_metrics(y_true, d_df["pred_static_heuristic"].tolist())
        m_ph = compute_parallax_metrics(y_true, d_df["pred_parallax_heuristic"].tolist())
        m_sn = compute_parallax_metrics(y_true, d_df["pred_static_neural"].tolist())
        m_pn = compute_parallax_metrics(y_true, d_df["pred_parallax_neural"].tolist())
        
        domain_results[d] = {
            "count": len(d_df),
            "static_heuristic": m_sh,
            "parallax_heuristic": m_ph,
            "static_neural": m_sn,
            "parallax_neural": m_pn,
        }
        
        print(
            f"{d:<28} | {len(d_df):<6} | {m_sh['macro_f1']*100:<14.2f}% | "
            f"{m_ph['macro_f1']*100:<16.2f}% | {m_sn['macro_f1']*100:<16.2f}% | {m_pn['macro_f1']*100:<18.2f}%"
        )
    print("=" * 90)
    
    # 4. Save results
    output_data = {
        "staleness_sensitivity": staleness_results,
        "domain_breakdown": domain_results,
    }
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved Phase B sensitivity results to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="data/benchmark/test.parquet")
    parser.add_argument("--output_json", default="experiments/temporal_sensitivity_results.json")
    args = parser.parse_args()
    run_analysis(args.test_path, args.output_json)
