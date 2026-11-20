"""
Parallax Dynamic Context Verifier (Stage A + Stage B)
=====================================================
Implements the Parallax Benchmark v2 dynamic verification pipeline:
- Stage A: Multi-Passage Semantic NLI (Claim vs. Each Retrieved Passage)
- Stage B: Dynamic Temporal Arbitration (Version Lineage & Supersession Inference)

Zero oracle metadata requirement: Operates solely on:
- passage.doc_id
- passage.snapshot_date
- passage.text
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple

try:
    from pipeline.temporal_resolver import TemporalAnchorResolver
    from verifier.static_baseline import HeuristicNLI
except ImportError:
    from src.pipeline.temporal_resolver import TemporalAnchorResolver
    from src.verifier.static_baseline import HeuristicNLI


class DynamicContextVerifier:
    """
    Audits a claim against a mixed set of retrieved versioned passages
    without relying on external oracle metadata.
    """

    def __init__(self, nli_model: Optional[Any] = None):
        if nli_model is None:
            self.nli = HeuristicNLI()
        else:
            self.nli = nli_model
        self.temporal_resolver = TemporalAnchorResolver()

    def _nli_predict(self, premise: str, hypothesis: str) -> str:
        """Helper to invoke NLI model across different implementations."""
        if hasattr(self.nli, "predict"):
            try:
                return self.nli.predict(premise=premise, hypothesis=hypothesis)
            except TypeError:
                return self.nli.predict(claim=hypothesis, evidence=premise)
        return "NEUTRAL"

    def verify(
        self,
        claim: str,
        retrieved_passages: List[Dict[str, Any]],
        target_doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Audits a single claim against retrieved passages.

        Args:
            claim: Natural language claim string.
            retrieved_passages: List of passage dicts:
                [{'passage_id': ..., 'doc_id': ..., 'snapshot_date': 'YYYY-MM-DD', 'text': ...}]
            target_doc_id: Optional target doc ID for logging.

        Returns:
            Dict containing:
            - verdict: OUTDATED, SUPPORTED_CURRENT, CONTRADICTED, or UNSUPPORTED
            - supporting_passages: List of passages entailing the claim
            - superseding_passage: Passage that superseded the claim (if OUTDATED)
            - explanation: Human-readable rationale
            - stage_a_results: List of (passage_id, nli_rel)
        """
        if not retrieved_passages:
            return {
                "verdict": "UNSUPPORTED",
                "supporting_passages": [],
                "superseding_passage": None,
                "explanation": "No retrieved context passages provided.",
                "stage_a_results": [],
            }

        # Stage A: Run NLI against each passage in context
        stage_a_results = []
        supporting_passages = []
        contradicting_passages = []

        # Batch-predict Stage A if model supports predict_batch
        if hasattr(self.nli, "predict_batch"):
            premises = [p.get("text", "") for p in retrieved_passages]
            hypotheses = [claim] * len(retrieved_passages)
            rels = self.nli.predict_batch(premises, hypotheses)
        else:
            rels = [self._nli_predict(premise=p.get("text", ""), hypothesis=claim) for p in retrieved_passages]

        for p, rel in zip(retrieved_passages, rels):
            p_id = p.get("passage_id", "p_?")
            stage_a_results.append({
                "passage_id": p_id,
                "doc_id": p.get("doc_id"),
                "snapshot_date": p.get("snapshot_date"),
                "relation": rel,
            })

            if rel == "ENTAILMENT":
                supporting_passages.append(p)
            elif rel == "CONTRADICTION":
                contradicting_passages.append(p)

        # Stage B: Dynamic Temporal Arbitration
        anchor_info = self.temporal_resolver.extract_temporal_anchor(claim)
        is_historical_claim = (anchor_info.get("anchor_type") == "HISTORICAL")

        # Case 1: No supporting passages found
        if not supporting_passages:
            if contradicting_passages:
                latest_contra = max(contradicting_passages, key=lambda x: str(x.get("snapshot_date", "")))
                return {
                    "verdict": "CONTRADICTED",
                    "supporting_passages": [],
                    "superseding_passage": None,
                    "contradicting_passage": latest_contra,
                    "explanation": (
                        f"Claim is refuted by passage [{latest_contra.get('passage_id')}] "
                        f"from {latest_contra.get('snapshot_date')}."
                    ),
                    "stage_a_results": stage_a_results,
                }
            else:
                return {
                    "verdict": "UNSUPPORTED",
                    "supporting_passages": [],
                    "superseding_passage": None,
                    "explanation": "No passage in the retrieved context entails or refutes the claim.",
                    "stage_a_results": stage_a_results,
                }

        # Case 2: One or more supporting passages found
        # Check if any supporting passage has been superseded by a newer revision of the same document
        superseding_found = None
        superseded_support = None

        for p_supp in supporting_passages:
            doc_id = str(p_supp.get("doc_id"))
            supp_date = str(p_supp.get("snapshot_date", "1900-01-01"))

            # Look for later snapshots of the same document
            newer_revisions = [
                p for p in retrieved_passages
                if str(p.get("doc_id")) == doc_id and str(p.get("snapshot_date", "")) > supp_date
            ]

            if newer_revisions:
                # Find the most recent snapshot
                latest_rev = max(newer_revisions, key=lambda x: str(x.get("snapshot_date", "")))
                
                # Check if latest revision still entails the claim
                latest_rel = self._nli_predict(premise=latest_rev.get("text", ""), hypothesis=claim)
                
                # If newer revision does NOT entail the claim (is CONTRADICTION or NEUTRAL),
                # or if it contradicts the supporting passage text:
                if latest_rel != "ENTAILMENT":
                    if not is_historical_claim:
                        superseding_found = latest_rev
                        superseded_support = p_supp
                        break
                    else:
                        # Historical claim explicitly targeted the past period
                        pass

        if superseding_found is not None:
            return {
                "verdict": "OUTDATED",
                "supporting_passages": supporting_passages,
                "superseded_passage": superseded_support,
                "superseding_passage": superseding_found,
                "explanation": (
                    f"Claim was supported by snapshot from {superseded_support.get('snapshot_date')} "
                    f"[{superseded_support.get('passage_id')}], but superseded by newer revision on "
                    f"{superseding_found.get('snapshot_date')} [{superseding_found.get('passage_id')}]."
                ),
                "stage_a_results": stage_a_results,
            }

        return {
            "verdict": "SUPPORTED_CURRENT",
            "supporting_passages": supporting_passages,
            "superseding_passage": None,
            "explanation": (
                f"Claim is supported by retrieved snapshot [{supporting_passages[0].get('passage_id')}] "
                f"from {supporting_passages[0].get('snapshot_date')} and no conflicting newer revision exists."
            ),
            "stage_a_results": stage_a_results,
        }

    def evaluate_dataset(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs dynamic auditing over a list of Benchmark v2 records.
        """
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

        y_true = []
        y_pred = []
        detailed_results = []

        for r in records:
            claim = r["claim"]
            passages = r["retrieved_passages"]
            gold = r["gold_label"]

            audit_res = self.verify(claim, passages, target_doc_id=r.get("target_doc_id"))
            pred = audit_res["verdict"]

            y_true.append(gold)
            y_pred.append(pred)
            detailed_results.append({
                "example_id": r.get("example_id"),
                "claim": claim,
                "gold_label": gold,
                "pred_label": pred,
                "is_correct": (gold == pred),
                "explanation": audit_res["explanation"],
            })

        acc = accuracy_score(y_true, y_pred) * 100.0
        
        # Calculate False Support Rate on OUTDATED (FSR_OUTD)
        outd_total = sum(1 for y in y_true if y == "OUTDATED")
        outd_as_supp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "OUTDATED" and yp == "SUPPORTED_CURRENT")
        fsr_outd = (outd_as_supp / outd_total * 100.0) if outd_total > 0 else 0.0

        labels = ["OUTDATED", "SUPPORTED_CURRENT", "CONTRADICTED", "UNSUPPORTED"]
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

        return {
            "accuracy": acc,
            "fsr_outdated": fsr_outd,
            "total_samples": len(y_true),
            "classification_report": report,
            "detailed_results": detailed_results,
        }
