import pytest
from pipeline.matcher import EvidenceMatcher


def test_matcher_best_sentence():
    matcher = EvidenceMatcher()
    claim = "The category of the Karunagappalli railway station is NSG 4."
    docs = [
        {
            "doc_id": "1",
            "doc_title": "Unrelated",
            "text": "The Eiffel Tower is located in Paris, France.",
        },
        {
            "doc_id": "2",
            "doc_title": "Karunagappalli station",
            "text": "The station is in Kerala. The category of the Karunagappalli railway station is NSG 4. It opened years ago.",
        }
    ]
    match = matcher.match_claim(claim, docs)
    assert match is not None
    assert match["doc_id"] == "2"
    assert "NSG 4" in match["matched_sentence"]
    assert match["match_score"] > 0.7


def test_matcher_rank_matches():
    matcher = EvidenceMatcher()
    claim = "Àlex Carbonell plays for Almere City."
    docs = [
        {
            "doc_id": "1",
            "doc_title": "Player bio",
            "text": "Carbonell plays as a midfielder for Almere City. He is 27 years old.",
        },
        {
            "doc_id": "2",
            "doc_title": "Almere City FC",
            "text": "Almere City FC is a Dutch football club based in Almere.",
        }
    ]
    ranked = matcher.rank_matches(claim, docs, top_k=2)
    assert len(ranked) >= 1
    assert ranked[0]["doc_id"] == "1"
