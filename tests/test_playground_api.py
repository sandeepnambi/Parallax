"""
Tests for Parallax Interactive Testing Playground Backend API
==============================================================
Validates:
- Preloaded examples endpoint (/api/examples)
- Verification endpoint (/api/verify) behavior across canonical cases
- Static vs. Parallax dynamic divergence on outdated claims
- Error handling (empty claims, invalid dates, missing passage fields)
- Static frontend index serving
"""

import pytest
from starlette.testclient import TestClient
from src.playground.app import app

client = TestClient(app)


def test_get_examples():
    """Verifies that /api/examples returns the 5 preloaded canonical test cases."""
    response = client.get("/api/examples")
    assert response.status_code == 200
    data = response.json()
    assert "examples" in data
    examples = data["examples"]
    assert len(examples) == 5

    expected_ids = [
        "outdated_evidence",
        "current_evidence",
        "contradicted_claim",
        "unsupported_claim",
        "missing_newer_revision",
    ]
    actual_ids = [e["id"] for e in examples]
    assert actual_ids == expected_ids


def test_verify_outdated_case():
    """
    Verifies the signature Parallax divergence:
    Static baseline falsely certifies outdated evidence as SUPPORTED_CURRENT.
    Parallax dynamic verifier correctly detects OUTDATED.
    """
    payload = {
        "claim": "Over 600,000 copies of the series are in circulation.",
        "retrieved_passages": [
            {
                "passage_id": "p_0",
                "doc_id": "74897662",
                "snapshot_date": "2024-12-01",
                "text": "The series has over 900,000 copies in circulation.",
            },
            {
                "passage_id": "p_1",
                "doc_id": "74897662",
                "snapshot_date": "2024-11-01",
                "text": "The series has over 600,000 copies in circulation.",
            },
        ],
        "include_neural": False,
    }
    response = client.post("/api/verify", json=payload)
    assert response.status_code == 200
    data = response.json()

    results = data["results"]
    assert results["static_heuristic"]["verdict"] == "SUPPORTED_CURRENT"
    assert results["dynamic_heuristic"]["verdict"] == "OUTDATED"
    assert results["dynamic_heuristic"]["superseding_passage"]["passage_id"] == "p_0"
    assert "inference_time_ms" in results["static_heuristic"]
    assert "inference_time_ms" in results["dynamic_heuristic"]


def test_verify_current_case():
    """Verifies that both systems agree on active current evidence."""
    payload = {
        "claim": "Over 900,000 copies of the series are in circulation.",
        "retrieved_passages": [
            {
                "passage_id": "p_0",
                "doc_id": "74897662",
                "snapshot_date": "2024-12-01",
                "text": "The series has over 900,000 copies in circulation.",
            },
            {
                "passage_id": "p_1",
                "doc_id": "74897662",
                "snapshot_date": "2024-11-01",
                "text": "The series has over 600,000 copies in circulation.",
            },
        ],
        "include_neural": False,
    }
    response = client.post("/api/verify", json=payload)
    assert response.status_code == 200
    data = response.json()

    results = data["results"]
    assert results["static_heuristic"]["verdict"] == "SUPPORTED_CURRENT"
    assert results["dynamic_heuristic"]["verdict"] == "SUPPORTED_CURRENT"


def test_verify_contradicted_case():
    """Verifies that both systems identify a contradiction."""
    payload = {
        "claim": "Over 1,500,000 copies of the series are in circulation.",
        "retrieved_passages": [
            {
                "passage_id": "p_0",
                "doc_id": "74897662",
                "snapshot_date": "2024-12-01",
                "text": "The series has over 900,000 copies in circulation.",
            },
        ],
        "include_neural": False,
    }
    response = client.post("/api/verify", json=payload)
    assert response.status_code == 200
    data = response.json()

    results = data["results"]
    assert results["static_heuristic"]["verdict"] == "CONTRADICTED"
    assert results["dynamic_heuristic"]["verdict"] == "CONTRADICTED"


def test_verify_unsupported_case():
    """Verifies that both systems return UNSUPPORTED for unrelated context."""
    payload = {
        "claim": "The series was awarded the Hugo Award for Best Novel in 2024.",
        "retrieved_passages": [
            {
                "passage_id": "p_0",
                "doc_id": "73048713",
                "snapshot_date": "2024-07-01",
                "text": "Holland averaged 18.5 points, 6.7 rebounds, 2.8 assists, and 2.1 steals per game in 29 games played.",
            },
        ],
        "include_neural": False,
    }
    response = client.post("/api/verify", json=payload)
    assert response.status_code == 200
    data = response.json()

    results = data["results"]
    assert results["static_heuristic"]["verdict"] == "UNSUPPORTED"
    assert results["dynamic_heuristic"]["verdict"] == "UNSUPPORTED"


def test_empty_claim_validation():
    """Verifies that an empty claim returns HTTP 422 with clear error message."""
    payload = {
        "claim": "   ",
        "retrieved_passages": [],
    }
    response = client.post("/api/verify", json=payload)
    assert response.status_code == 422
    assert "Claim cannot be empty" in response.json()["detail"]


def test_invalid_date_validation():
    """Verifies that malformed or non-existent calendar dates return HTTP 422."""
    # Malformed regex format
    payload1 = {
        "claim": "Valid claim.",
        "retrieved_passages": [
            {"doc_id": "1", "snapshot_date": "invalid-date", "text": "Some text."},
        ],
    }
    res1 = client.post("/api/verify", json=payload1)
    assert res1.status_code == 422
    assert "Invalid snapshot date format" in res1.json()["detail"]

    # Invalid calendar date (e.g. month 13)
    payload2 = {
        "claim": "Valid claim.",
        "retrieved_passages": [
            {"doc_id": "1", "snapshot_date": "2024-13-45", "text": "Some text."},
        ],
    }
    res2 = client.post("/api/verify", json=payload2)
    assert res2.status_code == 422
    assert "Invalid calendar date" in res2.json()["detail"]


def test_missing_passage_fields_validation():
    """Verifies missing doc_id or text returns HTTP 422."""
    payload = {
        "claim": "Valid claim.",
        "retrieved_passages": [
            {"doc_id": "", "snapshot_date": "2024-11-01", "text": "Some text."},
        ],
    }
    response = client.post("/api/verify", json=payload)
    assert response.status_code == 422
    assert "missing a Document ID" in response.json()["detail"]


def test_index_html_serving():
    """Verifies that GET / serves the interactive testing playground page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Parallax Interactive Testing Playground" in response.text
