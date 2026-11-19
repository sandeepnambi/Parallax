import html
import pytest
from pipeline.auditor import ParallaxAuditor


def test_auditor_historical_mismatch_not_supported():
    """
    Historical claims whose dates don't match the evidence validity period
    must NOT fall through to SUPPORTED_CURRENT; they must be marked UNSUPPORTED.
    """
    auditor = ParallaxAuditor(use_neural_nli=False, default_ref_time="2024-12-01")
    query = "Who was the president of the council in 2022?"
    answer = "In 2022, John Doe served as the council president."
    # Evidence text entails "John Doe served as the council president",
    # but evidence was only valid 2015-2018.
    docs = [{
        "doc_id": "doc_historical_old",
        "doc_title": "Council History",
        "text": "John Doe served as the council president during his tenure.",
        "valid_from": "2015-01-01",
        "valid_to": "2018-12-31",
        "superseded_by": None,
    }]
    report = auditor.audit_response(query, answer, docs, reference_time="2024-12-01")
    assert report["total_claims"] == 1
    claim_audit = report["claim_audits"][0]
    
    # Must NOT be marked SUPPORTED_CURRENT
    assert claim_audit["verdict"] != "SUPPORTED_CURRENT"
    assert claim_audit["verdict"] == "UNSUPPORTED"
    assert "TEMPORALLY INCOMPATIBLE" in claim_audit["temporal_status"]
    assert report["claim_breakdown"]["SUPPORTED_CURRENT"] == 0
    assert report["claim_breakdown"]["UNSUPPORTED"] == 1


def test_auditor_superseded_by_metadata_marked_outdated():
    """
    Evidence with valid_to="present" but an explicit superseded_by before reference_time
    must be audited as OUTDATED, not SUPPORTED_CURRENT.
    """
    auditor = ParallaxAuditor(use_neural_nli=False, default_ref_time="2024-12-01")
    query = "Who is the Master of the Horse?"
    answer = "The current Master of the Horse is Lord de Mauley."
    docs = [{
        "doc_id": "doc_master_horse",
        "doc_title": "Master of the Horse",
        "text": "The current Master of the Horse is Lord de Mauley.",
        "valid_from": "2024-01-01",
        "valid_to": "present", # Looks active!
        "superseded_by": "2024-07-01", # But superseded in July!
    }]
    report = auditor.audit_response(query, answer, docs, reference_time="2024-12-01")
    assert report["total_claims"] == 1
    claim_audit = report["claim_audits"][0]
    
    assert claim_audit["verdict"] == "OUTDATED"
    assert report["claim_breakdown"]["OUTDATED"] == 1
    assert report["claim_breakdown"]["SUPPORTED_CURRENT"] == 0


def test_auditor_html_report_escaping():
    """
    HTML report must escape user-supplied query, answer, claim, and evidence strings
    to prevent XSS or broken rendering.
    """
    auditor = ParallaxAuditor(use_neural_nli=False, default_ref_time="2024-12-01")
    xss_query = "<script>alert('query')</script> Who is CEO & President?"
    xss_answer = "The CEO is <b>Alice</b> <img src=x onerror=alert('xss')>."
    docs = [{
        "doc_id": "doc_xss",
        "doc_title": "<evil>Doc</evil>",
        "text": "The CEO is <b>Alice</b> & Chief.",
        "valid_from": "2024-01-01",
        "valid_to": "present",
        "superseded_by": None,
    }]
    report = auditor.audit_response(xss_query, xss_answer, docs, reference_time="2024-12-01")
    html_report = auditor.format_html_report(report)
    
    # Must contain escaped variants
    assert "&lt;script&gt;alert(&#x27;query&#x27;)&lt;/script&gt;" in html_report or "&lt;script&gt;alert('query')&lt;/script&gt;" in html_report
    assert "&lt;evil&gt;Doc&lt;/evil&gt;" in html_report
    assert "&lt;b&gt;Alice&lt;/b&gt;" in html_report
    
    # Must NOT contain raw unescaped tags
    assert "<script>alert('query')</script>" not in html_report
    assert "<evil>Doc</evil>" not in html_report
    assert "<img src=x onerror=alert('xss')>" not in html_report
