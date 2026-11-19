"""
Parallax Benchmark Cleaner and Scaler
Executes Step 1 of the Parallax Roadmap:
1. Rigorous Data Cleaning & Hygiene:
   - HTML entity unescaping (e.g., &amp; -> &, &quot; -> ")
   - Stripping parenthetical comments and annotator meta-notes
   - Filtering trivial/identical answers
   - Filtering overly verbose answers (> 8 words)
   - Enforcing strict positive temporal deltas (delta_t > 0)
   - Ensuring strict, grammatically verified claim verbalization
2. Document-Level Disjoint Splitting:
   - Train (400 seed docs -> 1,600 records)
   - Dev   (100 seed docs -> 400 records)
   - Test  (200 seed docs -> 800 records)
   Total: 2,800 balanced records with 0% document leakage across splits!
3. Rich Temporal & Domain Metadata:
   - temporal_gap_days (how long obsolete fact survived)
   - staleness_days (how long evidence has been expired at reference_time)
   - domain classification (Sports, Politics, Science, Geography, Culture)
"""

import os
import sys
import html
import random
import re
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import pyarrow.parquet as pq

# Add src to path
sys.path.append(os.path.dirname(__file__))
try:
    from verbalizer import verbalize_claim
except ImportError:
    from src.verbalizer import verbalize_claim

RANDOM_SEED = 42
REFERENCE_TIME = "2024-12-01"


def clean_text(text: str) -> str:
    """Unescapes HTML entities, normalizes whitespace, and strips trailing punctuation."""
    if not text:
        return ""
    # Unescape HTML
    s = html.unescape(str(text))
    # Replace weird whitespace characters
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def classify_domain(title: str, text: str) -> str:
    """Heuristic domain classifier based on document title and content keywords."""
    content = f"{title} {text}".lower()
    
    if re.search(r"\b(football|fc|basketball|baseball|tournament|championship|cup|league|coach|player|olympic|season|racing|goals|atp|nba|ncaa)\b", content):
        return "Sports"
    if re.search(r"\b(minister|president|election|governor|mayor|parliament|party|council|senate|politics|treaty|court|judge|republic)\b", content):
        return "Politics & Governance"
    if re.search(r"\b(census|population|county|district|municipality|demographics|inhabitants|hectares|suburb|median income|geography|river|lake)\b", content):
        return "Geography & Demographics"
    if re.search(r"\b(university|institute|school|species|genus|protein|chemical|physics|space|mission|satellite|software|algorithm|technology)\b", content):
        return "Science & Technology"
    if re.search(r"\b(album|singer|band|actor|actress|novel|author|film|movie|song|award|theatre|music|television|series)\b", content):
        return "Culture & Media"
        
    return "General Knowledge"


def clean_and_scale_benchmark(
    raw_parquet_path: str,
    output_benchmark_dir: str,
    output_review_dir: str,
    seed: int = RANDOM_SEED
) -> Dict[str, pd.DataFrame]:
    random.seed(seed)
    
    print(f"Step 1: Loading raw HoH dataset from {raw_parquet_path}...")
    table = pq.read_table(raw_parquet_path)
    df = table.to_pandas()
    print(f"Loaded {len(df)} raw records.")
    
    # Filter for single outdated version
    df_single = df[df["outdated_infos"].apply(len) == 1].copy()
    print(f"Rows with 1 outdated version: {len(df_single)}")
    
    print("\nStep 2: Applying rigorous data hygiene and filtering...")
    cleaned_candidates: List[Dict[str, Any]] = []
    
    ref_dt = datetime.strptime(REFERENCE_TIME, "%Y-%m-%d")
    
    for idx, row in df_single.iterrows():
        q_raw = clean_text(row["question"])
        curr_ans_raw = clean_text(row["answer"])
        curr_ev_raw = clean_text(row["evidence"])
        
        old_info = row["outdated_infos"][0]
        old_ans_raw = clean_text(old_info["answer"])
        old_ev_raw = clean_text(old_info["evidence"])
        
        doc_id = clean_text(row["document"]["id"])
        doc_title = clean_text(row["document"]["title"])
        
        # 1. Filter out parenthetical comments or notes
        if re.search(r"\(|\)", curr_ans_raw) or re.search(r"\(|\)", old_ans_raw):
            continue
            
        # 2. Filter out overly long answers (> 8 words)
        if len(curr_ans_raw.split()) > 8 or len(old_ans_raw.split()) > 8:
            continue
            
        # 3. Filter out identical / trivial answers
        if curr_ans_raw.lower() == old_ans_raw.lower():
            continue
            
        # 4. Parse and validate dates
        try:
            curr_date_str = str(row["last_modified_time"])[:10]
            old_date_str = str(old_info["last_modified_time"])[:10]
            curr_dt = datetime.strptime(curr_date_str, "%Y-%m-%d")
            old_dt = datetime.strptime(old_date_str, "%Y-%m-%d")
        except Exception:
            continue
            
        # 5. Check positive temporal ordering
        delta_days = (curr_dt - old_dt).days
        if delta_days <= 0:
            continue
            
        staleness_days = (ref_dt - curr_dt).days
        
        # 6. Strict verbalization check
        curr_claim = verbalize_claim(q_raw, curr_ans_raw, strict=True)
        old_claim = verbalize_claim(q_raw, old_ans_raw, strict=True)
        
        if not curr_claim or not old_claim:
            continue
            
        domain = classify_domain(doc_title, curr_ev_raw)
        
        cleaned_candidates.append({
            "doc_id": doc_id,
            "doc_title": doc_title,
            "domain": domain,
            "question": q_raw,
            "curr_ans": curr_ans_raw,
            "old_ans": old_ans_raw,
            "curr_claim": curr_claim,
            "old_claim": old_claim,
            "curr_ev": curr_ev_raw,
            "old_ev": old_ev_raw,
            "curr_date": curr_date_str,
            "old_date": old_date_str,
            "temporal_gap_days": delta_days,
            "staleness_days": staleness_days,
        })
        
    print(f"Total pristine candidates meeting all criteria: {len(cleaned_candidates)}")
    
    # Shuffle candidates
    random.shuffle(cleaned_candidates)
    
    # Group by unique document to ensure 100% disjoint document splits
    docs_to_items: Dict[str, Dict[str, Any]] = {}
    for item in cleaned_candidates:
        if item["doc_id"] not in docs_to_items:
            docs_to_items[item["doc_id"]] = item
            
    unique_items = list(docs_to_items.values())
    print(f"Unique documents available for benchmark: {len(unique_items)}")
    
    # We need:
    # Train: 400 seed documents (1,600 examples)
    # Dev:   100 seed documents (400 examples)
    # Test:  200 seed documents (800 examples)
    # Total: 700 seed documents (2,800 examples)
    required_seeds = 700
    if len(unique_items) < required_seeds:
        raise ValueError(f"Need {required_seeds} unique documents, but only have {len(unique_items)}")
        
    train_seeds = unique_items[:400]
    dev_seeds = unique_items[400:500]
    test_seeds = unique_items[500:700]
    
    print("Seed partition sizes (disjoint documents):")
    print(f"  Train seeds: {len(train_seeds)}")
    print(f"  Dev seeds:   {len(dev_seeds)}")
    print(f"  Test seeds:  {len(test_seeds)}")
    
    def generate_split_records(seeds: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
        records = []
        n = len(seeds)
        
        # 1. SUPPORTED_CURRENT (n records)
        for i, item in enumerate(seeds):
            records.append({
                "example_id": f"parallax_{split_name}_{i+1:04d}_curr",
                "split": split_name,
                "domain": item["domain"],
                "question": item["question"],
                "answer": item["curr_ans"],
                "answer_claim": item["curr_claim"],
                "evidence_text": item["curr_ev"],
                "evidence_source": f"{item['doc_title']} (id: {item['doc_id']})",
                "evidence_version": item["curr_date"],
                "valid_from": item["curr_date"],
                "valid_to": "present",
                "superseded_by": None,
                "temporal_gap_days": item["temporal_gap_days"],
                "staleness_days": 0,
                "reference_time": REFERENCE_TIME,
                "gold_label": "SUPPORTED_CURRENT",
                "annotation_notes": "Current claim paired with valid evidence snapshot.",
            })
            
        # 2. OUTDATED (n records)
        for i, item in enumerate(seeds):
            records.append({
                "example_id": f"parallax_{split_name}_{i+1:04d}_outd",
                "split": split_name,
                "domain": item["domain"],
                "question": item["question"],
                "answer": item["old_ans"],
                "answer_claim": item["old_claim"],
                "evidence_text": item["old_ev"],
                "evidence_source": f"{item['doc_title']} (id: {item['doc_id']})",
                "evidence_version": item["old_date"],
                "valid_from": item["old_date"],
                "valid_to": item["curr_date"],
                "superseded_by": item["curr_date"],
                "temporal_gap_days": item["temporal_gap_days"],
                "staleness_days": item["staleness_days"],
                "reference_time": REFERENCE_TIME,
                "gold_label": "OUTDATED",
                "annotation_notes": "Outdated claim paired with superseded evidence whose interval expired prior to reference_time.",
            })

        # 3. CONTRADICTED (n records)
        # Current claim paired with outdated evidence asserting conflicting obsolete entity
        for i, item in enumerate(seeds):
            records.append({
                "example_id": f"parallax_{split_name}_{i+1:04d}_contra",
                "split": split_name,
                "domain": item["domain"],
                "question": item["question"],
                "answer": item["curr_ans"],
                "answer_claim": item["curr_claim"],
                "evidence_text": item["old_ev"],
                "evidence_source": f"{item['doc_title']} (id: {item['doc_id']})",
                "evidence_version": item["old_date"],
                "valid_from": item["old_date"],
                "valid_to": item["curr_date"],
                "superseded_by": item["curr_date"],
                "temporal_gap_days": item["temporal_gap_days"],
                "staleness_days": item["staleness_days"],
                "reference_time": REFERENCE_TIME,
                "gold_label": "CONTRADICTED",
                "annotation_notes": "Current claim asserting new entity paired against outdated evidence asserting conflicting prior entity.",
            })

        # 4. UNSUPPORTED (n records)
        # Current claim paired with orthogonal distractor evidence from within the same split's document pool
        for i, item in enumerate(seeds):
            # Circular offset within the split to ensure distractor never crosses splits
            distractor_idx = (i + (n // 2)) % n
            distractor_item = seeds[distractor_idx]
            records.append({
                "example_id": f"parallax_{split_name}_{i+1:04d}_unsupp",
                "split": split_name,
                "domain": item["domain"],
                "question": item["question"],
                "answer": item["curr_ans"],
                "answer_claim": item["curr_claim"],
                "evidence_text": distractor_item["curr_ev"],
                "evidence_source": f"{distractor_item['doc_title']} (id: {distractor_item['doc_id']})",
                "evidence_version": distractor_item["curr_date"],
                "valid_from": distractor_item["curr_date"],
                "valid_to": "present",
                "superseded_by": None,
                "temporal_gap_days": 0,
                "staleness_days": 0,
                "reference_time": REFERENCE_TIME,
                "gold_label": "UNSUPPORTED",
                "annotation_notes": f"Claim from '{item['doc_title']}' paired with distractor evidence from '{distractor_item['doc_title']}'.",
            })
            
        return records

    print("\nStep 3: Generating balanced 4-class records for each split...")
    train_records = generate_split_records(train_seeds, "train")
    dev_records = generate_split_records(dev_seeds, "dev")
    test_records = generate_split_records(test_seeds, "test")
    
    train_df = pd.DataFrame(train_records)
    dev_df = pd.DataFrame(dev_records)
    test_df = pd.DataFrame(test_records)
    full_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    
    print("\nBenchmark Generation Complete:")
    print(f"  Train: {len(train_df)} records (400 per class)")
    print(f"  Dev:   {len(dev_df)} records (100 per class)")
    print(f"  Test:  {len(test_df)} records (200 per class)")
    print(f"  Total: {len(full_df)} records (700 per class across all 4 labels)")
    
    # Save benchmark
    os.makedirs(output_benchmark_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(output_benchmark_dir, "train.parquet"), index=False)
    train_df.to_csv(os.path.join(output_benchmark_dir, "train.csv"), index=False, encoding="utf-8")
    
    dev_df.to_parquet(os.path.join(output_benchmark_dir, "dev.parquet"), index=False)
    dev_df.to_csv(os.path.join(output_benchmark_dir, "dev.csv"), index=False, encoding="utf-8")
    
    test_df.to_parquet(os.path.join(output_benchmark_dir, "test.parquet"), index=False)
    test_df.to_csv(os.path.join(output_benchmark_dir, "test.csv"), index=False, encoding="utf-8")
    
    full_df.to_parquet(os.path.join(output_benchmark_dir, "full_benchmark.parquet"), index=False)
    full_df.to_csv(os.path.join(output_benchmark_dir, "full_benchmark.csv"), index=False, encoding="utf-8")
    
    # Human Review copy: sample 100 random rows
    os.makedirs(output_review_dir, exist_ok=True)
    sample_audit = full_df.sample(100, random_state=seed)
    sample_audit.to_csv(os.path.join(output_review_dir, "sample_audit_100.csv"), index=False, encoding="utf-8")
    
    print(f"\nFiles exported to {output_benchmark_dir} and {output_review_dir}:")
    print("  - train.parquet / train.csv")
    print("  - dev.parquet / dev.csv")
    print("  - test.parquet / test.csv")
    print("  - full_benchmark.parquet / full_benchmark.csv")
    print("  - sample_audit_100.csv (for human review)")
    
    print("\nDomain representation in benchmark:")
    print(full_df["domain"].value_counts())
    
    return {"train": train_df, "dev": dev_df, "test": test_df, "full": full_df}


if __name__ == "__main__":
    import argparse
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_raw = os.path.join(base_dir, "Datasets", "D1-HoH", "hoh_qas_240601_241201.parquet")
    default_bench = os.path.join(base_dir, "data", "benchmark")
    default_rev = os.path.join(base_dir, "data", "review")

    parser = argparse.ArgumentParser(description="Clean and scale Parallax benchmark")
    parser.add_argument("--raw_path", type=str, default=default_raw, help="Path to raw HoH parquet file")
    parser.add_argument("--bench_dir", type=str, default=default_bench, help="Benchmark output directory")
    parser.add_argument("--rev_dir", type=str, default=default_rev, help="Review sample output directory")
    args = parser.parse_args()

    if not os.path.exists(args.raw_path):
        print(f"Warning: Raw dataset not found at {args.raw_path}. Please place the dataset file there to rebuild.")
    else:
        clean_and_scale_benchmark(args.raw_path, args.bench_dir, args.rev_dir)
