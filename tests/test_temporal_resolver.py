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


def test_historical_interval_full_coverage():
    resolver = TemporalAnchorResolver()
    claim = "Between 2018 and 2022, Alice served as governor."
    info = resolver.extract_temporal_anchor(claim)
    
    # Evidence covers 2017 to 2023 (full coverage)
    res = resolver.arbitrate_temporal_validity(
        anchor_info=info,
        valid_from="2017-01-01",
        valid_to="2023-12-31",
        reference_time="2024-12-01",
        require_full_coverage=True,
    )
    assert res["verdict_modifier"] == "HISTORICALLY_VALID"
    assert res["is_stale"] is False


def test_historical_interval_partial_overlap_incompatible():
    resolver = TemporalAnchorResolver()
    claim = "From 2018 to 2022, Alice served as governor."
    info = resolver.extract_temporal_anchor(claim)
    
    # Evidence only valid 2018 to 2019 (partial overlap, incomplete coverage)
    res = resolver.arbitrate_temporal_validity(
        anchor_info=info,
        valid_from="2018-01-01",
        valid_to="2019-12-31",
        reference_time="2024-12-01",
        require_full_coverage=True,
    )
    assert res["verdict_modifier"] == "TEMPORALLY_INCOMPATIBLE"
    assert "partially overlaps" in res["reason"]


def test_historical_no_overlap_incompatible():
    resolver = TemporalAnchorResolver()
    claim = "In 2022, Bob was the CEO."
    info = resolver.extract_temporal_anchor(claim)
    
    # Evidence was only valid in 2015-2018
    res = resolver.arbitrate_temporal_validity(
        anchor_info=info,
        valid_from="2015-01-01",
        valid_to="2018-12-31",
        reference_time="2024-12-01",
    )
    assert res["verdict_modifier"] == "TEMPORALLY_INCOMPATIBLE"
    assert "no temporal overlap" in res["reason"]


def test_superseded_by_metadata_with_active_valid_to():
    resolver = TemporalAnchorResolver()
    claim = "The current director is Alice Smith."
    info = resolver.extract_temporal_anchor(claim)
    
    # Evidence has valid_to="present", but superseded_by="2024-07-01" prior to ref_time "2024-12-01"
    res = resolver.arbitrate_temporal_validity(
        anchor_info=info,
        valid_from="2023-01-01",
        valid_to="present",
        superseded_by="2024-07-01",
        reference_time="2024-12-01",
    )
    assert res["is_stale"] is True
    assert res["verdict_modifier"] == "OUTDATED"
    assert "superseded on 2024-07-01" in res["reason"]


def test_superseded_by_future_date_not_stale():
    resolver = TemporalAnchorResolver()
    claim = "The current director is Alice Smith."
    info = resolver.extract_temporal_anchor(claim)
    
    # Supersession occurs in the future relative to ref_time "2024-06-01"
    res = resolver.arbitrate_temporal_validity(
        anchor_info=info,
        valid_from="2023-01-01",
        valid_to="present",
        superseded_by="2024-10-01",
        reference_time="2024-06-01",
    )
    assert res["is_stale"] is False
    assert res["verdict_modifier"] == "ACTIVE"


def test_superseded_by_non_date_identifier():
    resolver = TemporalAnchorResolver()
    claim = "The current director is Alice Smith."
    info = resolver.extract_temporal_anchor(claim)
    
    # Evidence has active valid_to but explicit replacement doc identifier
    res = resolver.arbitrate_temporal_validity(
        anchor_info=info,
        valid_from="2023-01-01",
        valid_to="present",
        superseded_by="doc_replacement_v2",
        reference_time="2024-12-01",
    )
    assert res["is_stale"] is True
    assert res["verdict_modifier"] == "OUTDATED"
    assert "doc_replacement_v2" in res["reason"]
