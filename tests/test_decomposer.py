import pytest
from pipeline.decomposer import ClaimDecomposer


def test_split_sentences_basic():
    text = "The sky is blue. Grass is green! Is the sun hot?"
    sentences = ClaimDecomposer.split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "The sky is blue."
    assert sentences[1] == "Grass is green!"
    assert sentences[2] == "Is the sun hot?"


def test_split_sentences_with_decimals_and_abbrevs():
    text = "The price in the U.S. was $3.14 per gallon. Dr. Smith agreed."
    sentences = ClaimDecomposer.split_sentences(text)
    assert len(sentences) == 2
    assert "3.14" in sentences[0]
    assert "U.S." in sentences[0]
    assert "Dr. Smith" in sentences[1]


def test_clean_boilerplate():
    decomposer = ClaimDecomposer()
    raw = "Based on the retrieved context, the Master of the Horse is Lord de Mauley."
    cleaned = decomposer.clean_claim(raw)
    assert cleaned == "The Master of the Horse is Lord de Mauley."


def test_decompose_with_metadata():
    decomposer = ClaimDecomposer()
    text = "Currently, the Master of the Horse is Lord de Mauley. He was appointed in 2018."
    claims = decomposer.decompose_with_metadata(text)
    assert len(claims) == 2
    assert claims[0]["has_temporal_anchor"] is True
    assert "currently" in claims[0]["temporal_cues"]
    assert "2018" in claims[1]["temporal_cues"]
