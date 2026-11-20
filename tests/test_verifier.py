from verifier.static_baseline import HeuristicNLI, StaticBaselineVerifier
from verifier.temporal_arbiter import ParallaxTemporalVerifier


def test_heuristic_nli_entailment():
    nli = HeuristicNLI()
    claim = "The capital of France is Paris."
    evidence = "Paris is the capital of France."
    ans = "Paris"
    rel = nli.predict(claim, evidence, ans)
    assert rel == "ENTAILMENT"


def test_heuristic_nli_contradiction_with_entity():
    nli = HeuristicNLI()
    claim = "The chief of the department is David Hendricks."
    evidence = "The department's chief is Jon Radus."
    rel = nli.predict(claim=claim, evidence=evidence, answer_entity="")
    assert rel == "CONTRADICTION"


def test_temporal_arbiter_outdated():
    arbiter = ParallaxTemporalVerifier()
    row = {
        "answer_claim": "The Master of the Horse is Lord de Mauley.",
        "evidence_text": "The current Master of the Horse is Lord de Mauley.",
        "answer": "Lord de Mauley",
        "valid_to": "2024-07-01",
        "reference_time": "2024-12-01",
    }
    verdict = arbiter.verify_example(row)
    assert verdict == "OUTDATED"


def test_temporal_arbiter_current():
    arbiter = ParallaxTemporalVerifier()
    row = {
        "answer_claim": "The Master of the Horse is Lord de Mauley.",
        "evidence_text": "The current Master of the Horse is Lord de Mauley.",
        "answer": "Lord de Mauley",
        "valid_to": "present",
        "reference_time": "2024-12-01",
    }
    verdict = arbiter.verify_example(row)
    assert verdict == "SUPPORTED_CURRENT"


def test_static_baseline_falsely_supports_outdated():
    static_verifier = StaticBaselineVerifier()
    row = {
        "answer_claim": "The Master of the Horse is Lord de Mauley.",
        "evidence_text": "The current Master of the Horse is Lord de Mauley.",
        "answer": "Lord de Mauley",
        "valid_to": "2024-07-01",
        "reference_time": "2024-12-01",
    }
    verdict = static_verifier.verify_example(row)
    # The static verifier is temporally blind and must produce false support
    assert verdict == "SUPPORTED_CURRENT"


def test_nli_dataset_mapping():
    import pandas as pd
    from verifier.train_cross_encoder import ParallaxNLIDataset

    mock_df = pd.DataFrame({
        "evidence_text": ["Ev1", "Ev2", "Ev3", "Ev4"],
        "answer_claim": ["Cl1", "Cl2", "Cl3", "Cl4"],
        "gold_label": ["SUPPORTED_CURRENT", "OUTDATED", "CONTRADICTED", "UNSUPPORTED"],
    })
    ds = ParallaxNLIDataset(mock_df)
    assert len(ds) == 4
    # SUPPORTED_CURRENT -> ENTAILMENT (1)
    assert ds[0]["label"] == 1
    # OUTDATED -> ENTAILMENT (1)
    assert ds[1]["label"] == 1
    # CONTRADICTED -> CONTRADICTION (0)
    assert ds[2]["label"] == 0
    # UNSUPPORTED -> NEUTRAL (2)
    assert ds[3]["label"] == 2

