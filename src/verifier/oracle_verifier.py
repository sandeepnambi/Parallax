"""
Parallax Oracle Upper-Bound Verifier
====================================
WARNING: THIS VERIFIER USES PRIVILEGED / ORACLE METADATA.
IT MUST NEVER BE USED OR REPORTED AS A REAL-WORLD APPLICABLE SYSTEM.
ITS SOLE PURPOSE IS TO ESTABLISH AN EMPIRICAL THEORETICAL CEILING.

In Benchmark v2, normal systems must dynamically infer supersession without
oracle metadata. The Oracle Upper-Bound baseline is granted privileged access
to ground-truth annotations (such as target document alignment and the oracle
superseding evidence indicator) to quantify the gap between dynamic inference
and an omniscient verifier.
"""

from typing import Dict, Any, List, Optional
from sklearn.metrics import classification_report, accuracy_score

try:
    from verifier.static_baseline import HeuristicNLI
except ImportError:
    from src.verifier.static_baseline import HeuristicNLI


class OracleContextVerifier:
    """
    Theoretical Upper-Bound Baseline with privileged oracle access.
    Quarantined and strictly marked as NON-DEPLOYABLE.
    """

    def __init__(self, nli_model: Optional[Any] = None):
        self.nli = nli_model or HeuristicNLI()

    def _nli_predict(self, premise: str, hypothesis: str) -> str:
        if hasattr(self.nli, "predict"):
            try:
                return self.nli.predict(premise=premise, hypothesis=hypothesis)
            except TypeError:
                return self.nli.predict(claim=hypothesis, evidence=premise)
        return "NEUTRAL"

    def verify_example_oracle(self, record: Dict[str, Any]) -> str:
        """
        Uses oracle metadata (target_doc_id, has_superseding_evidence, and gold relations)
        to produce the theoretical upper-bound classification.
        """
        claim = record.get("claim", "")
        passages = record.get("retrieved_passages", [])
        target_doc = str(record.get("target_doc_id", ""))
        has_superseding = bool(record.get("has_superseding_evidence", False))

        target_passages = [p for p in passages if str(p.get("doc_id", "")) == target_doc]

        # Oracle Rule 1: If target document was not retrieved, it is genuinely UNSUPPORTED
        if not target_passages:
            return "UNSUPPORTED"

        # Check entailment against target document passages
        entailing_passages = []
        contradicting_passages = []

        for p in target_passages:
            rel = self._nli_predict(premise=p.get("text", ""), hypothesis=claim)
            if rel == "ENTAILMENT":
                entailing_passages.append(p)
            elif rel == "CONTRADICTION":
                contradicting_passages.append(p)

        # Oracle Rule 2: If an older snapshot entails the claim AND oracle indicates supersession
        if entailing_passages:
            # Sort by snapshot_date
            sorted_by_date = sorted(target_passages, key=lambda x: str(x.get("snapshot_date", "")))
            oldest = sorted_by_date[0]
            latest = sorted_by_date[-1]

            old_rel = self._nli_predict(premise=oldest.get("text", ""), hypothesis=claim)
            latest_rel = self._nli_predict(premise=latest.get("text", ""), hypothesis=claim)

            if has_superseding and old_rel == "ENTAILMENT" and latest_rel != "ENTAILMENT":
                return "OUTDATED"
            elif latest_rel == "ENTAILMENT":
                return "SUPPORTED_CURRENT"
            elif old_rel == "ENTAILMENT" and not has_superseding:
                return "SUPPORTED_CURRENT"
            else:
                return "SUPPORTED_CURRENT"

        # Oracle Rule 3: Contradiction
        if contradicting_passages:
            return "CONTRADICTED"

        return "UNSUPPORTED"

    def evaluate_dataset(self, records: List[Dict[str, Any]], batch_size: int = 32, **kwargs: Any) -> Dict[str, Any]:
        """
        Evaluates the dataset under oracle upper-bound conditions.
        """
        y_true = []
        y_pred = []
        detailed_results = []

        for r in records:
            gold = r["gold_label"]
            pred = self.verify_example_oracle(r)

            y_true.append(gold)
            y_pred.append(pred)
            detailed_results.append({
                "example_id": r.get("example_id"),
                "claim": r.get("claim"),
                "gold_label": gold,
                "pred_label": pred,
                "is_correct": (gold == pred),
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
            "is_oracle_upper_bound": True,
        }
