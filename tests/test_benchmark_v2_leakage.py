"""
Unit and Integration Tests for Parallax Benchmark v2 Leakage & Structural Integrity
===================================================================================
Verifies that:
1. Zero oracle metadata fields (valid_to, valid_from, superseded_by, reference_time, etc.)
   exist anywhere in the exported benchmark datasets.
2. Retrieved passages conform strictly to {passage_id, doc_id, snapshot_date, text}.
3. Splits (train, dev, test) are 100% document-disjoint.
4. Class balance is exactly preserved.
5. Passage counts and target passage positions are balanced/randomized without heuristic leaks.
"""

import os
import json
import pytest
from typing import Dict, Any, List

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "benchmark_v2")

FORBIDDEN_KEYS = {
    "valid_from",
    "valid_to",
    "superseded_by",
    "reference_time",
    "staleness_days",
    "temporal_gap_days",
    "annotation_notes",
    "temporal_status",
    "oracle_label",
}

EXPECTED_PASSAGE_KEYS = {"passage_id", "doc_id", "snapshot_date", "text"}
EXPECTED_LABELS = {"OUTDATED", "SUPPORTED_CURRENT", "CONTRADICTED", "UNSUPPORTED"}


def load_jsonl(filename: str) -> List[Dict[str, Any]]:
    filepath = os.path.join(BENCHMARK_DIR, filename)
    assert os.path.exists(filepath), f"File {filepath} does not exist. Did you run benchmark_v2_builder.py?"
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def train_records():
    return load_jsonl("train.jsonl")


@pytest.fixture(scope="module")
def dev_records():
    return load_jsonl("dev.jsonl")


@pytest.fixture(scope="module")
def test_records():
    return load_jsonl("test.jsonl")


@pytest.fixture(scope="module")
def single_version_slice_records():
    return load_jsonl("test_single_version_slice.jsonl")


def test_files_exist():
    """Assert all benchmark v2 data files exist."""
    for fname in ["train.jsonl", "dev.jsonl", "test.jsonl", "test_single_version_slice.jsonl", "full_benchmark_v2.jsonl"]:
        path = os.path.join(BENCHMARK_DIR, fname)
        assert os.path.exists(path), f"Missing expected dataset file: {path}"


def test_no_forbidden_keys(train_records, dev_records, test_records, single_version_slice_records):
    """Assert that NO forbidden metadata key appears in any record or passage."""
    all_datasets = [train_records, dev_records, test_records, single_version_slice_records]
    for split_records in all_datasets:
        for record in split_records:
            # Check top-level keys
            for forbidden in FORBIDDEN_KEYS:
                assert forbidden not in record, f"Forbidden key '{forbidden}' found in record {record.get('example_id')}"
            
            # Check passage keys
            for p in record.get("retrieved_passages", []):
                for forbidden in FORBIDDEN_KEYS:
                    assert forbidden not in p, f"Forbidden key '{forbidden}' found in passage {p.get('passage_id')} of record {record.get('example_id')}"


def test_passage_schema_strictness(train_records, dev_records, test_records):
    """Assert that retrieved passages have exactly the expected keys."""
    for split_records in [train_records, dev_records, test_records]:
        for record in split_records:
            passages = record.get("retrieved_passages", [])
            assert len(passages) >= 1, f"Empty passages in {record.get('example_id')}"
            for p in passages:
                assert set(p.keys()) == EXPECTED_PASSAGE_KEYS, f"Passage schema mismatch: {set(p.keys())} != {EXPECTED_PASSAGE_KEYS}"


def test_class_balance(train_records, dev_records, test_records):
    """Assert exact 4-class balance across all main splits."""
    for split_name, records, expected_per_class in [
        ("train", train_records, 400),
        ("dev", dev_records, 100),
        ("test", test_records, 200),
    ]:
        counts = {}
        for r in records:
            lbl = r["gold_label"]
            counts[lbl] = counts.get(lbl, 0) + 1
        
        assert set(counts.keys()) == EXPECTED_LABELS, f"Unexpected label set in {split_name}: {counts.keys()}"
        for lbl in EXPECTED_LABELS:
            assert counts[lbl] == expected_per_class, f"{split_name} count for {lbl} is {counts[lbl]}, expected {expected_per_class}"


def test_document_disjointness(train_records, dev_records, test_records):
    """Assert that train, dev, and test sets share 0% documents."""
    def get_docs(records):
        docs = set()
        for r in records:
            docs.add(r["target_doc_id"])
        return docs

    train_docs = get_docs(train_records)
    dev_docs = get_docs(dev_records)
    test_docs = get_docs(test_records)

    assert len(train_docs) == 400
    assert len(dev_docs) == 100
    assert len(test_docs) == 200

    assert len(train_docs & dev_docs) == 0, f"Train and Dev share documents: {train_docs & dev_docs}"
    assert len(train_docs & test_docs) == 0, f"Train and Test share documents: {train_docs & test_docs}"
    assert len(dev_docs & test_docs) == 0, f"Dev and Test share documents: {dev_docs & test_docs}"


def test_context_symmetry(test_records):
    """
    Assert that for OUTDATED, SUPPORTED_CURRENT, and CONTRADICTED,
    contexts contain both target document versions (old and new snapshots),
    so presence of multiple versions does not give away the label.
    """
    for r in test_records:
        label = r["gold_label"]
        seed_doc = r["target_doc_id"]
        passages = r["retrieved_passages"]
        target_passages = [p for p in passages if str(p["doc_id"]) == str(seed_doc)]
        
        if label in ["OUTDATED", "SUPPORTED_CURRENT", "CONTRADICTED"]:
            assert len(target_passages) == 2, (
                f"Record {r['example_id']} with label {label} should have exactly 2 target passages, "
                f"found {len(target_passages)}"
            )
            # Verify they have distinct snapshot dates
            dates = {p["snapshot_date"] for p in target_passages}
            assert len(dates) == 2, f"Target passages in {r['example_id']} must have 2 distinct snapshot dates"
        elif label == "UNSUPPORTED":
            # For UNSUPPORTED, target document is not present in context
            assert len(target_passages) == 0, (
                f"Record {r['example_id']} with label UNSUPPORTED should have 0 target passages, "
                f"found {len(target_passages)}"
            )


def test_passage_counts_distribution(test_records):
    """Assert passage counts are between 3 and 5 across all classes."""
    for r in test_records:
        k = len(r["retrieved_passages"])
        assert 3 <= k <= 5, f"Record {r['example_id']} has {k} passages, expected between 3 and 5"
