import pytest
from pipeline.supersession_graph import SupersessionGraph


def test_supersession_graph_lineage():
    graph = SupersessionGraph()
    doc_id = "doc_fullerton_police"

    # Version 1 (David Hendricks valid until Aug 2024)
    graph.add_fact_version(
        doc_id=doc_id,
        fact_text="The chief of police is David Hendricks.",
        answer_value="David Hendricks",
        valid_from="2022-01-01",
        valid_to="2024-08-01",
        superseded_by="2024-08-01",
    )

    # Version 2 (Jon Radus active from Aug 2024 to present)
    graph.add_fact_version(
        doc_id=doc_id,
        fact_text="The chief of police is Jon Radus.",
        answer_value="Jon Radus",
        valid_from="2024-08-01",
        valid_to="present",
        superseded_by=None,
    )

    lineage = graph.trace_lineage(doc_id)
    assert len(lineage) == 2
    assert lineage[0]["answer_value"] == "David Hendricks"
    assert lineage[0]["is_active"] is False
    assert lineage[1]["answer_value"] == "Jon Radus"
    assert lineage[1]["is_active"] is True

    # Active replacement
    active = graph.get_active_replacement(doc_id)
    assert active is not None
    assert active["answer_value"] == "Jon Radus"

    # Explanation for stale claim
    explanation = graph.explain_supersession(doc_id, "David Hendricks")
    assert explanation["stale_fact"]["answer_value"] == "David Hendricks"
    assert explanation["active_replacement"]["answer_value"] == "Jon Radus"
