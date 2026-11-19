#!/usr/bin/env python
"""
Parallax CLI - Command-Line Interface for Temporal RAG Auditing
==============================================================
Provides easy access to audit RAG responses, run interactive demos,
and inspect benchmark statistics.
"""

import os
import sys
import json
import argparse

# Bootstrap sys.path
_cur = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_cur, "src")
for _p in [_cur, _src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.auditor import ParallaxAuditor
from pipeline.temporal_resolver import TemporalAnchorResolver
from pipeline.supersession_graph import SupersessionGraph


def run_demo(neural: bool = False):
    """Runs a quick interactive demo of the 4 key RAG archetypes."""
    print("=" * 70)
    print("PARALLAX INTERACTIVE DEMO: TEMPORAL RAG AUDITING")
    print("=" * 70)

    mode_name = "Neural Cross-Encoder" if neural else "Heuristic Alignment"
    print(f"Initializing Parallax Auditor [{mode_name}]...\n")
    auditor = ParallaxAuditor(use_neural_nli=neural, default_ref_time="2024-12-01")

    # Sample Stale Scenario
    query = "Who is the Master of the Horse in the Royal Household?"
    answer = "Based on retrieved context, the current Master of the Horse is Lord de Mauley."
    docs = [
        {
            "doc_id": "100025",
            "doc_title": "Master of the Horse",
            "text": "The Master of the Horse is an office in the Royal Household. The current Master of the Horse is Lord de Mauley.",
            "valid_from": "2024-06-01",
            "valid_to": "2024-07-01",  # Superseded before reference_time!
            "superseded_by": "2024-07-01",
        }
    ]

    print("Auditing Sample 1: Silent Stale Evidence (Entailed but superseded)...")
    report = auditor.audit_response(query, answer, docs)
    print(auditor.format_report(report))


def run_audit(args):
    """Audits a user-supplied query, answer, and documents."""
    auditor = ParallaxAuditor(use_neural_nli=args.neural, default_ref_time=args.ref_time)

    docs = []
    if args.docs_file:
        if os.path.exists(args.docs_file):
            with open(args.docs_file, "r", encoding="utf-8") as f:
                docs = json.load(f)
        else:
            print(f"Error: Documents file '{args.docs_file}' not found.")
            sys.exit(1)
    elif args.doc_text:
        docs = [{
            "doc_id": "doc_1",
            "doc_title": args.doc_title or "Retrieved Document",
            "text": args.doc_text,
            "valid_from": args.valid_from or "2024-01-01",
            "valid_to": args.valid_to or "present",
            "superseded_by": args.superseded_by,
        }]
    else:
        print("Error: Please provide documents via --docs-file or --doc-text.")
        sys.exit(1)

    report = auditor.audit_response(args.query, args.answer, docs, reference_time=args.ref_time)

    if args.output_html:
        html_content = auditor.format_html_report(report)
        with open(args.output_html, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Exported interactive HTML report to: {args.output_html}")
    elif args.output_json:
        auditor.export_json(report, args.output_json)
    else:
        print(auditor.format_report(report))


def show_info():
    """Prints benchmark and project info."""
    import pandas as pd
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bench_dir = os.path.join(base_dir, "data", "benchmark")
    print("=" * 60)
    print("PARALLAX BENCHMARK & SYSTEM OVERVIEW")
    print("=" * 60)
    if os.path.exists(bench_dir):
        for split in ["train", "dev", "test", "full_benchmark"]:
            csv_path = os.path.join(bench_dir, f"{split}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                print(f"- {split}.csv: {len(df):,} records ({', '.join(f'{k}: {v}' for k, v in df['gold_label'].value_counts().items())})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Parallax CLI: Claim-Level Temporal RAG Auditing")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo command
    demo_p = subparsers.add_parser("demo", help="Run interactive demo audit")
    demo_p.add_argument("--neural", action="store_true", help="Use neural cross-encoder NLI backend")

    # Audit command
    audit_p = subparsers.add_parser("audit", help="Audit a custom query and answer")
    audit_p.add_argument("--query", type=str, required=True, help="User query")
    audit_p.add_argument("--answer", type=str, required=True, help="RAG answer paragraph")
    audit_p.add_argument("--docs-file", type=str, help="Path to JSON file with retrieved documents list")
    audit_p.add_argument("--doc-text", type=str, help="Single retrieved document text")
    audit_p.add_argument("--doc-title", type=str, default="Document", help="Document title")
    audit_p.add_argument("--valid-from", type=str, help="Valid from date (YYYY-MM-DD)")
    audit_p.add_argument("--valid-to", type=str, default="present", help="Valid to date (YYYY-MM-DD or 'present')")
    audit_p.add_argument("--superseded-by", type=str, help="Superseded by date (YYYY-MM-DD)")
    audit_p.add_argument("--ref-time", type=str, default="2024-12-01", help="Reference anchor time")
    audit_p.add_argument("--neural", action="store_true", help="Use neural cross-encoder")
    audit_p.add_argument("--output-json", type=str, help="Save report to JSON file")
    audit_p.add_argument("--output-html", type=str, help="Save report to standalone HTML file")

    # Info command
    subparsers.add_parser("info", help="Show benchmark statistics and metadata")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo(neural=args.neural)
    elif args.command == "audit":
        run_audit(args)
    elif args.command == "info":
        show_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
