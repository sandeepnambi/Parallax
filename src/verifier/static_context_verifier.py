"""
Parallax Static Context Verifier (Multi-Passage Baseline)
=========================================================
Evaluates claims against multi-passage retrieved contexts using conventional
temporally blind static claim verification.

Multi-Passage Priority Aggregation Rules:
1. ENTAILMENT PRIORITY: If ANY retrieved passage entails the claim, the verdict
   is SUPPORTED_CURRENT (regardless of timestamps or newer contradicting revisions).
2. CONTRADICTION PRIORITY: If NO passage entails the claim, but AT LEAST ONE passage
   contradicts the claim, the verdict is CONTRADICTED.
3. UNSUPPORTED / NEUTRAL FALLBACK: If all passages are neutral, unrelated, or if
   retrieved context is empty, the verdict is UNSUPPORTED.

This implements the standard static RAG assumption: temporal metadata is ignored,
and any entailing passage in context is deemed sufficient evidence.
"""

from typing import Dict, Any, List, Optional
from sklearn.metrics import classification_report, accuracy_score


class StaticContextVerifier:
    """
    Temporally blind multi-passage context verifier.
    Supports both HeuristicNLI and NeuralCrossEncoderNLI.
    """

    def __init__(self, nli_model: Any):
        self.nli = nli_model

    def _nli_predict(self, premise: str, hypothesis: str) -> str:
        """Invokes NLI model across different interface conventions."""
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Audits a single claim against a multi-passage retrieved context
        under static priority rules (ignoring all temporal metadata).

        Returns:
            Dict containing verdict, passage-level NLI relations, and explanation.
        """
        if not retrieved_passages:
            return {
                "verdict": "UNSUPPORTED",
                "supporting_passages": [],
                "contradicting_passages": [],
                "explanation": "No retrieved context passages provided.",
                "passage_results": [],
            }

        # Evaluate NLI for each passage
        if hasattr(self.nli, "predict_batch"):
            premises = [p.get("text", "") for p in retrieved_passages]
            hypotheses = [claim] * len(retrieved_passages)
            rels = self.nli.predict_batch(premises, hypotheses)
        else:
            rels = [self._nli_predict(premise=p.get("text", ""), hypothesis=claim) for p in retrieved_passages]

        passage_results = []
        supporting = []
        contradicting = []

        for p, rel in zip(retrieved_passages, rels):
            passage_results.append({
                "passage_id": p.get("passage_id", "p_?"),
                "relation": rel,
            })
            if rel == "ENTAILMENT":
                supporting.append(p)
            elif rel == "CONTRADICTION":
                contradicting.append(p)

        # Priority Rule 1: Any entailment -> SUPPORTED_CURRENT
        if supporting:
            first_supp = supporting[0]
            return {
                "verdict": "SUPPORTED_CURRENT",
                "supporting_passages": supporting,
                "contradicting_passages": contradicting,
                "explanation": (
                    f"Claim is entailed by passage [{first_supp.get('passage_id', 'p_?')}] "
                    f"under static verification (temporal ordering ignored)."
                ),
                "passage_results": passage_results,
            }

        # Priority Rule 2: No entailment, but any contradiction -> CONTRADICTED
        if contradicting:
            first_contra = contradicting[0]
            return {
                "verdict": "CONTRADICTED",
                "supporting_passages": [],
                "contradicting_passages": contradicting,
                "explanation": (
                    f"Claim is contradicted by passage [{first_contra.get('passage_id', 'p_?')}] "
                    f"and no passage entails it."
                ),
                "passage_results": passage_results,
            }

        # Priority Rule 3: All neutral / unrelated -> UNSUPPORTED
        return {
            "verdict": "UNSUPPORTED",
            "supporting_passages": [],
            "contradicting_passages": [],
            "explanation": "No passage in context entails or contradicts the claim.",
            "passage_results": passage_results,
        }

    def evaluate_dataset(self, records: List[Dict[str, Any]], batch_size: int = 32) -> Dict[str, Any]:
        """
        Evaluates a list of Benchmark v2 records using static multi-passage verification.
        Optimized with batched inference if supported by NLI model.
        """
        y_true = []
        y_pred = []
        detailed_results = []

        # If model supports global batched inference, precompute all passage-claim pairs
        if hasattr(self.nli, "predict_batch"):
            all_premises = []
            all_hypotheses = []
            index_map = []  # (example_idx, passage_idx)

            for i, r in enumerate(records):
                c = r["claim"]
                passages = r.get("retrieved_passages", [])
                for j, p in enumerate(passages):
                    all_premises.append(p.get("text", ""))
                    all_hypotheses.append(c)
                    index_map.append((i, j))

            all_rels = self.nli.predict_batch(all_premises, all_hypotheses, batch_size=batch_size)

            # Reconstruct per-example predictions
            example_rels: List[List[str]] = [[] for _ in range(len(records))]
            for (i, _), rel in zip(index_map, all_rels):
                example_rels[i].append(rel)

            for r, rels in zip(records, example_rels):
                gold = r["gold_label"]
                if "ENTAILMENT" in rels:
                    pred = "SUPPORTED_CURRENT"
                elif "CONTRADICTION" in rels:
                    pred = "CONTRADICTED"
                else:
                    pred = "UNSUPPORTED"

                y_true.append(gold)
                y_pred.append(pred)
                detailed_results.append({
                    "example_id": r.get("example_id"),
                    "claim": r.get("claim"),
                    "gold_label": gold,
                    "pred_label": pred,
                    "is_correct": (gold == pred),
                    "passage_relations": rels,
                })
        else:
            for r in records:
                gold = r["gold_label"]
                res = self.verify(r["claim"], r.get("retrieved_passages", []))
                pred = res["verdict"]

                y_true.append(gold)
                y_pred.append(pred)
                detailed_results.append({
                    "example_id": r.get("example_id"),
                    "claim": r.get("claim"),
                    "gold_label": gold,
                    "pred_label": pred,
                    "is_correct": (gold == pred),
                    "explanation": res["explanation"],
                })

        acc = accuracy_score(y_true, y_pred) * 100.0
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
            "y_true": y_true,
            "y_pred": y_pred,
        }
