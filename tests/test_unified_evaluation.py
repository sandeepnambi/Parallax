"""
Tests for Parallax Unified Benchmark v2 Evaluation Pipeline
===========================================================
Validates:
- Static multi-passage priority rules (entailment > contradiction > neutral).
- Absence of oracle metadata leakage in evaluation components.
- Cluster-level bootstrap confidence interval calculation and determinism.
- Single-version diagnostic metrics (OUTDATED recall and FSR_OUTD reported separately).
- Date-shuffle temporal control transformation integrity.
- Oracle verifier separation and flagging.
"""

import pytest
import numpy as np
from src.verifier.static_context_verifier import StaticContextVerifier
from src.verifier.dynamic_arbiter import DynamicContextVerifier
from src.verifier.oracle_verifier import OracleContextVerifier
from src.verifier.static_baseline import HeuristicNLI
from src.evaluation.bootstrap import compute_cluster_bootstrap_ci
from experiments.run_unified_evaluation import create_date_shuffled_dataset


class MockNLI:
    """Mock NLI returning predetermined relations for deterministic testing."""
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def predict(self, premise="", hypothesis="", **kwargs):
        return self.mapping.get(premise, "NEUTRAL")


def test_static_context_priority_entailment_over_contradiction():
    """Rule 1: If ANY passage entails, verdict must be SUPPORTED_CURRENT even if another contradicts."""
    mock_nli = MockNLI({
        "Passage A": "CONTRADICTION",
        "Passage B": "ENTAILMENT",
        "Passage C": "NEUTRAL",
    })
    verifier = StaticContextVerifier(nli_model=mock_nli)
    passages = [
        {"passage_id": "p1", "doc_id": "d1", "snapshot_date": "2024-01-01", "text": "Passage A"},
        {"passage_id": "p2", "doc_id": "d1", "snapshot_date": "2024-02-01", "text": "Passage B"},
        {"passage_id": "p3", "doc_id": "d2", "snapshot_date": "2024-02-01", "text": "Passage C"},
    ]
    res = verifier.verify("Test Claim", passages)
    assert res["verdict"] == "SUPPORTED_CURRENT"


def test_static_context_priority_contradiction_over_neutral():
    """Rule 2: If NO passage entails but ANY contradicts, verdict is CONTRADICTED."""
    mock_nli = MockNLI({
        "Passage A": "CONTRADICTION",
        "Passage B": "NEUTRAL",
    })
    verifier = StaticContextVerifier(nli_model=mock_nli)
    passages = [
        {"passage_id": "p1", "doc_id": "d1", "snapshot_date": "2024-01-01", "text": "Passage A"},
        {"passage_id": "p2", "doc_id": "d2", "snapshot_date": "2024-02-01", "text": "Passage B"},
    ]
    res = verifier.verify("Test Claim", passages)
    assert res["verdict"] == "CONTRADICTED"


def test_static_context_priority_neutral_fallback():
    """Rule 3: If all neutral, verdict is UNSUPPORTED."""
    mock_nli = MockNLI({
        "Passage A": "NEUTRAL",
        "Passage B": "NEUTRAL",
    })
    verifier = StaticContextVerifier(nli_model=mock_nli)
    passages = [
        {"passage_id": "p1", "doc_id": "d1", "snapshot_date": "2024-01-01", "text": "Passage A"},
        {"passage_id": "p2", "doc_id": "d2", "snapshot_date": "2024-02-01", "text": "Passage B"},
    ]
    res = verifier.verify("Test Claim", passages)
    assert res["verdict"] == "UNSUPPORTED"


def test_static_and_dynamic_no_oracle_leakage():
    """Verifies that non-oracle verifiers do not inspect or require oracle fields."""
    passages = [
        {"passage_id": "p1", "doc_id": "100", "snapshot_date": "2024-01-01", "text": "Paris is capital of France."},
        {"passage_id": "p2", "doc_id": "100", "snapshot_date": "2024-06-01", "text": "Lyon is new capital of France."},
    ]
    claim = "Paris is the capital of France."

    # Both verifiers must operate purely on (claim, passages) without oracle fields
    static_v = StaticContextVerifier(nli_model=HeuristicNLI())
    dynamic_v = DynamicContextVerifier(nli_model=HeuristicNLI())

    static_res = static_v.verify(claim, passages)
    dynamic_res = dynamic_v.verify(claim, passages)

    # Static verifier sees p1 entails -> SUPPORTED_CURRENT
    assert static_res["verdict"] == "SUPPORTED_CURRENT"
    # Dynamic verifier detects p2 (Lyon) supersedes p1 (Paris) -> OUTDATED
    assert dynamic_res["verdict"] == "OUTDATED"


def test_cluster_bootstrap_ci_reproducibility_and_bounds():
    """Verifies cluster bootstrap confidence interval calculation, determinism, and valid bounds."""
    y_true = ["OUTDATED", "SUPPORTED_CURRENT", "OUTDATED", "CONTRADICTED", "UNSUPPORTED"] * 20
    y_pred = ["OUTDATED", "SUPPORTED_CURRENT", "SUPPORTED_CURRENT", "CONTRADICTED", "UNSUPPORTED"] * 20
    cluster_ids = [f"c_{i // 5}" for i in range(len(y_true))]

    ci_1 = compute_cluster_bootstrap_ci(y_true, y_pred, cluster_ids, n_bootstraps=100, seed=42)
    ci_2 = compute_cluster_bootstrap_ci(y_true, y_pred, cluster_ids, n_bootstraps=100, seed=42)

    # Determinism
    assert ci_1["accuracy"]["mean"] == ci_2["accuracy"]["mean"]
    assert ci_1["accuracy"]["ci_lower"] == ci_2["accuracy"]["ci_lower"]
    assert ci_1["accuracy"]["ci_upper"] == ci_2["accuracy"]["ci_upper"]

    # Bounds
    for metric in ["accuracy", "macro_f1", "fsr_outdated"]:
        assert 0.0 <= ci_1[metric]["ci_lower"] <= ci_1[metric]["ci_upper"] <= 100.0


def test_cluster_bootstrap_input_validation():
    """Verifies length mismatch raises ValueError."""
    with pytest.raises(ValueError):
        compute_cluster_bootstrap_ci(["OUTDATED"], ["OUTDATED"], ["c1", "c2"])


def test_date_shuffle_transformation():
    """Verifies that date-shuffling correctly swaps dates of target document without altering distractors or labels."""
    records = [
        {
            "example_id": "ex_1",
            "gold_label": "OUTDATED",
            "target_doc_id": "100",
            "retrieved_passages": [
                {"passage_id": "p0", "doc_id": "100", "snapshot_date": "2024-01-01", "text": "Old text"},
                {"passage_id": "p1", "doc_id": "999", "snapshot_date": "2024-03-01", "text": "Distractor"},
                {"passage_id": "p2", "doc_id": "100", "snapshot_date": "2024-06-01", "text": "New text"},
            ],
        }
    ]
    shuffled = create_date_shuffled_dataset(records)
    assert len(shuffled) == 1
    shuf_passages = shuffled[0]["retrieved_passages"]

    # Gold label must remain untouched
    assert shuffled[0]["gold_label"] == "OUTDATED"

    # Distractor must be untouched
    assert shuf_passages[1]["snapshot_date"] == "2024-03-01"

    # Target passages must have swapped dates
    p0 = next(p for p in shuf_passages if p["passage_id"] == "p0")
    p2 = next(p for p in shuf_passages if p["passage_id"] == "p2")
    assert p0["snapshot_date"] == "2024-06-01"
    assert p2["snapshot_date"] == "2024-01-01"


def test_oracle_verifier_separation_flag():
    """Verifies that OracleContextVerifier explicitly identifies itself as oracle upper bound."""
    oracle_v = OracleContextVerifier()
    records = [
        {
            "example_id": "ex_o",
            "claim": "Lord de Mauley is Master of the Horse.",
            "gold_label": "OUTDATED",
            "target_doc_id": "10",
            "has_superseding_evidence": True,
            "retrieved_passages": [
                {"passage_id": "p0", "doc_id": "10", "snapshot_date": "2024-01-01", "text": "The Master of the Horse is Lord de Mauley."},
                {"passage_id": "p1", "doc_id": "10", "snapshot_date": "2024-06-01", "text": "The Master of the Horse is Lord Vestey."},
            ],
        }
    ]
    eval_res = oracle_v.evaluate_dataset(records)
    assert eval_res.get("is_oracle_upper_bound") is True
    assert eval_res["accuracy"] == 100.0
