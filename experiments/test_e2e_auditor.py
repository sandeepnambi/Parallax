"""
Parallax End-to-End Pipeline Evaluation
Tests the multi-claim paragraph decomposition, evidence matching, and temporal arbitration
pipeline across realistic RAG scenarios:
1. Silent Temporal Stale Answer (Entailed but superseded evidence)
2. Mixed Realistic Answer (Current + Outdated + Unsupported Hallucination)
3. Contradicted Answer (Hallucinated conflict with retrieved text)
4. Pristine Current Answer (100% supported by active evidence)
"""

import os
import sys
import json
import argparse

# Ensure src and root are in sys.path
_cur = os.path.dirname(os.path.abspath(__file__))
_src = os.path.abspath(os.path.join(_cur, "..", "src"))
_root = os.path.abspath(os.path.join(_cur, ".."))
for _p in [_src, _root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from pipeline.auditor import ParallaxAuditor
except ImportError:
    from src.pipeline.auditor import ParallaxAuditor


def get_test_cases():
    # Case 1: Silent Stale Evidence
    query_1 = "Who is the current Master of the Horse and which team does Àlex Carbonell play for?"
    answer_1 = (
        "Based on the retrieved context, the current Master of the Horse is Lord de Mauley. "
        "Furthermore, Àlex Carbonell Vallés plays for Almere City."
    )
    docs_1 = [
        {
            "doc_id": "100025",
            "doc_title": "Master of the Horse",
            "text": "The Master of the Horse is a historic office in the United Kingdom. The current Master of the Horse is Lord de Mauley.",
            "valid_from": "2024-06-01",
            "valid_to": "2024-07-01", # Superseded in July!
            "superseded_by": "2024-07-01",
        },
        {
            "doc_id": "52437830",
            "doc_title": "Álex Carbonell",
            "text": "Àlex Carbonell Vallés (born 15 September 1997) is a Spanish footballer who plays as a central midfielder for Dutch club Almere City.",
            "valid_from": "2024-08-01",
            "valid_to": "present", # Active!
            "superseded_by": None,
        }
    ]

    # Case 2: Mixed Claims
    query_2 = "What is the status of Jerrick Harding and the Astro Boy manga?"
    answer_2 = (
        "According to the documentation, Jerrick Harding plays for MoraBanc Andorra. "
        "The total number of chapters in the series is 112. "
        "Additionally, Jerrick Harding was selected as the European MVP in 2023."
    )
    docs_2 = [
        {
            "doc_id": "581293",
            "doc_title": "Jerrick Harding",
            "text": "Jerrick Harding (born April 13, 1998) is an American professional basketball player for MoraBanc Andorra of the Liga ACB.",
            "valid_from": "2024-07-01",
            "valid_to": "present",
            "superseded_by": None,
        },
        {
            "doc_id": "89211",
            "doc_title": "List of Astro Boy chapters",
            "text": "The manga series was serialized in Shonen Magazine. The total number of chapters is 112.",
            "valid_from": "2024-06-01",
            "valid_to": "2024-07-01", # Superseded
            "superseded_by": "2024-07-01",
        }
    ]

    # Case 3: Contradicted
    query_3 = "Who is the chief of the Fullerton Police Department?"
    answer_3 = "The current chief of the Fullerton Police Department is David Hendricks."
    docs_3 = [
        {
            "doc_id": "9921",
            "doc_title": "Fullerton Police Department",
            "text": "The department serves the city of Fullerton, California. The current chief is Jon Radus.",
            "valid_from": "2024-08-01",
            "valid_to": "present",
            "superseded_by": None,
        }
    ]

    # Case 4: Pristine Current
    query_4 = "What is the status of Karunagappalli railway station and Warsaw, Illinois?"
    answer_4 = (
        "The category of the Karunagappalli railway station is NSG 4. "
        "Furthermore, the median age in Warsaw, Illinois is 45.4 years."
    )
    docs_4 = [
        {
            "doc_id": "41120",
            "doc_title": "Karunagappalli railway station",
            "text": "The station is located in Kerala, India. The category of the Karunagappalli railway station is NSG 4.",
            "valid_from": "2024-09-01",
            "valid_to": "present",
            "superseded_by": None,
        },
        {
            "doc_id": "18821",
            "doc_title": "Warsaw, Illinois",
            "text": "According to the recent demographic survey, the median age in Warsaw, Illinois is 45.4 years.",
            "valid_from": "2024-10-01",
            "valid_to": "present",
            "superseded_by": None,
        }
    ]

    return [
        ("Case 1 (Silent Stale Evidence)", query_1, answer_1, docs_1),
        ("Case 2 (Mixed Claims: Current + Outdated + Conflict)", query_2, answer_2, docs_2),
        ("Case 3 (Direct Contradiction)", query_3, answer_3, docs_3),
        ("Case 4 (Pristine Current Response)", query_4, answer_4, docs_4),
    ]


def run_e2e_tests(use_neural: bool = True):
    mode_name = "Neural Cross-Encoder" if use_neural else "Heuristic Alignment"
    print("============================================================")
    print(f"PARALLAX END-TO-END AUDITOR TEST ({mode_name})")
    print("============================================================")
    
    auditor = ParallaxAuditor(use_neural_nli=use_neural, default_ref_time="2024-12-01")
    test_cases = get_test_cases()
    results = {}
    
    for case_id, query, answer, docs in test_cases:
        print(f"\n>>> Running {case_id}...")
        report = auditor.audit_response(query, answer, docs)
        print(auditor.format_report(report))
        results[case_id] = report
        
    out_filename = "e2e_audit_samples.json" if use_neural else "e2e_audit_samples_heuristic.json"
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), out_filename))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all multi-claim audit traces to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallax E2E Pipeline Evaluation")
    parser.add_argument(
        "--mode",
        choices=["neural", "heuristic", "both"],
        default="neural",
        help="NLI backend mode to test (default: neural)"
    )
    args = parser.parse_args()
    
    if args.mode in ("neural", "both"):
        run_e2e_tests(use_neural=True)
    if args.mode in ("heuristic", "both"):
        run_e2e_tests(use_neural=False)
