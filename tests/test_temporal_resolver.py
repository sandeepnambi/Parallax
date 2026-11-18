import pytest
from pipeline.temporal_resolver import TemporalAnchorResolver


def test_present_anchor_extraction():
    resolver = TemporalAnchorResolver()
    claim = "Currently, the Master of the Horse is Lord de Mauley."
    info = resolver.extract_temporal_anchor(claim)
    assert info["anchor_type"] == "PRESENT"
    assert "currently" in info["present_cues"]


def test_historical_interval_extraction():
    resolver = TemporalAnchorResolver()
    claim = "From 2018 to 2022, the player represented Real Madrid."
    info = resolver.extract_temporal_anchor(claim)
    assert info["anchor_type"] == "HISTORICAL"
    assert info["target_interval"] == ("2018-01-01", "2022-12-31")


def test_historical_claim_validity_arbitration():
    resolver = TemporalAnchorResolver()
    claim = "In 2020, John Doe served as the council president."
    info = resolver.extract_temporal_anchor(claim)
    
    # Evidence was valid from 2019 to 2021
    eval_res = resolver.arbitrate_temporal_validity(
        anchor_info=info,
        valid_from="2019-01-01",
        valid_to="2021-01-01",
        reference_time="2024-12-01",
    )
    assert eval_res["is_stale"] is False
    assert eval_res["verdict_modifier"] == "HISTORICALLY_VALID"


def test_present_claim_stale_arbitration():
    resolver = TemporalAnchorResolver()
    claim = "The current director is Alice Smith."
    info = resolver.extract_temporal_anchor(claim)
    
    # Evidence expired in July 2024, reference time is Dec 2024
    eval_res = resolver.arbitrate_temporal_validity(
        anchor_info=info,
        valid_from="2024-01-01",
        valid_to="2024-07-01",
        reference_time="2024-12-01",
    )
    assert eval_res["is_stale"] is True
    assert eval_res["verdict_modifier"] == "OUTDATED"
