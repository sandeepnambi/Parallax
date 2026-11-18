import pytest
from verbalizer import verbalize_claim


def test_verbalize_what_is():
    q = "What is the capital of France?"
    ans = "Paris"
    claim = verbalize_claim(q, ans, strict=True)
    assert claim == "The capital of France is Paris."


def test_verbalize_who_is():
    q = "Who is the Master of the Horse?"
    ans = "Lord de Mauley"
    claim = verbalize_claim(q, ans, strict=True)
    assert claim == "The Master of the Horse is Lord de Mauley."


def test_verbalize_in_what_year():
    q = "In what year did Cuba next compete in the Summer Olympics after a period of absence?"
    ans = "1935"
    claim = verbalize_claim(q, ans, strict=True)
    assert claim is not None
    assert "1935" in claim


def test_verbalize_strict_failure():
    q = "Why did the cat cross the road?"
    ans = "Because it wanted to."
    claim = verbalize_claim(q, ans, strict=True)
    assert claim is None


def test_verbalize_non_strict_fallback():
    q = "Why did the cat cross the road?"
    ans = "to get food"
    claim = verbalize_claim(q, ans, strict=False)
    assert "Regarding why did the cat cross the road, the answer is to get food." in claim
