"""
Parallax Pilot Dataset Builder
Samples 200 seed instances from D1-HoH and constructs a balanced 600-example
evaluation benchmark following the Parallax 4-label taxonomy:
- SUPPORTED_CURRENT: 200
- OUTDATED: 200
- CONTRADICTED: 100
- UNSUPPORTED: 100
"""

import os
import random
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, Any, List

try:
    from verbalizer import verbalize_claim
except ImportError:
    from src.verbalizer import verbalize_claim

# Fixed random seed for strict scientific reproducibility
RANDOM_SEED = 42
REFERENCE_TIME = "2024-12-01"


def clean_date_str(ts: Any) -> str:
    """Standardizes timestamps to ISO-8601 YYYY-MM-DD string."""
    s = str(ts).strip()
    return s[:10]


def build_pilot_dataset(
    hoh_parquet_path: str,
    output_dir: str,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    random.seed(seed)
    
    print(f"Loading raw dataset from {hoh_parquet_path}...")
    table = pq.read_table(hoh_parquet_path)
    df = table.to_pandas()
    print(f"Total rows in HoH: {len(df)}")
    
    # Filter for single outdated version
    df_single = df[df["outdated_infos"].apply(len) == 1].copy()
    print(f"Rows with exactly 1 outdated info: {len(df_single)}")
    
    # Filter clean answers (no parentheses / meta comments, max 6 words)
    clean_mask = (
        ~df_single["answer"].str.contains(r"\(|\)", regex=True) &
        (df_single["answer"].apply(lambda s: len(str(s).split())) <= 6) &
        ~df_single["outdated_infos"].apply(
            lambda info_list: "(" in str(info_list[0]["answer"]) or ")" in str(info_list[0]["answer"]) or len(str(info_list[0]["answer"]).split()) > 6
        )
    )
    df_candidates = df_single[clean_mask].copy()
    print(f"Candidate rows after quality filtering: {len(df_candidates)}")
    
    # Extract strict verbalizations
    valid_rows = []
    seen_docs = set()
    
    # Shuffle for uniform topical sampling across the corpus
    df_shuffled = df_candidates.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    for _, row in df_shuffled.iterrows():
        doc_id = row["document"]["id"]
        doc_title = row["document"]["title"]
        
        # Enforce unique document per seed record to ensure maximum topic diversity
        if doc_id in seen_docs or doc_title in seen_docs:
            continue
            
        q = row["question"]
        curr_ans = str(row["answer"]).strip()
        old_ans = str(row["outdated_infos"][0]["answer"]).strip()
        
        # Avoid trivial identical answers (must be a genuine factual update)
        if curr_ans.lower() == old_ans.lower():
            continue
            
        curr_claim = verbalize_claim(q, curr_ans, strict=True)
        old_claim = verbalize_claim(q, old_ans, strict=True)
        
        if curr_claim and old_claim:
            seen_docs.add(doc_id)
            seen_docs.add(doc_title)
            valid_rows.append({
                "raw_row": row,
                "curr_claim": curr_claim,
                "old_claim": old_claim,
                "curr_ans": curr_ans,
                "old_ans": old_ans,
                "doc_title": doc_title,
                "doc_id": doc_id,
            })
            
        if len(valid_rows) >= 200:
            break
            
    if len(valid_rows) < 200:
        raise ValueError(f"Could not collect 200 diverse seeds; only found {len(valid_rows)}.")
        
    print("Successfully collected 200 distinct-document seed instances.")
    
    records: List[Dict[str, Any]] = []
    
    # 1. Generate 200 SUPPORTED_CURRENT records
    for i, item in enumerate(valid_rows):
        r = item["raw_row"]
        curr_date = clean_date_str(r["last_modified_time"])
        records.append({
            "example_id": f"parallax_pilot_{i+1:04d}_curr",
            "question": r["question"],
            "answer": item["curr_ans"],
            "answer_claim": item["curr_claim"],
            "evidence_text": str(r["evidence"]).strip(),
            "evidence_source": f"{item['doc_title']} (id: {item['doc_id']})",
            "evidence_version": curr_date,
            "valid_from": curr_date,
            "valid_to": "present",
            "superseded_by": None,
            "reference_time": REFERENCE_TIME,
            "gold_label": "SUPPORTED_CURRENT",
            "split": "pilot",
            "annotation_notes": "Current claim paired with current valid evidence snapshot.",
        })
        
    # 2. Generate 200 OUTDATED records
    for i, item in enumerate(valid_rows):
        r = item["raw_row"]
        curr_date = clean_date_str(r["last_modified_time"])
        old_info = r["outdated_infos"][0]
        old_date = clean_date_str(old_info["last_modified_time"])
        records.append({
            "example_id": f"parallax_pilot_{i+1:04d}_outd",
            "question": r["question"],
            "answer": item["old_ans"],
            "answer_claim": item["old_claim"],
            "evidence_text": str(old_info["evidence"]).strip(),
            "evidence_source": f"{item['doc_title']} (id: {item['doc_id']})",
            "evidence_version": old_date,
            "valid_from": old_date,
            "valid_to": curr_date,
            "superseded_by": curr_date,
            "reference_time": REFERENCE_TIME,
            "gold_label": "OUTDATED",
            "split": "pilot",
            "annotation_notes": "Outdated claim paired with superseded evidence whose interval expired prior to reference_time.",
        })

    # 3. Generate 100 CONTRADICTED records (from first 100 seeds)
    # Current claim paired with outdated evidence asserting obsolete entity
    for i in range(100):
        item = valid_rows[i]
        r = item["raw_row"]
        curr_date = clean_date_str(r["last_modified_time"])
        old_info = r["outdated_infos"][0]
        old_date = clean_date_str(old_info["last_modified_time"])
        records.append({
            "example_id": f"parallax_pilot_{i+1:04d}_contra",
            "question": r["question"],
            "answer": item["curr_ans"],
            "answer_claim": item["curr_claim"],
            "evidence_text": str(old_info["evidence"]).strip(),
            "evidence_source": f"{item['doc_title']} (id: {item['doc_id']})",
            "evidence_version": old_date,
            "valid_from": old_date,
            "valid_to": curr_date,
            "superseded_by": curr_date,
            "reference_time": REFERENCE_TIME,
            "gold_label": "CONTRADICTED",
            "split": "pilot",
            "annotation_notes": "Current claim asserting new entity paired against outdated evidence asserting conflicting prior entity.",
        })

    # 4. Generate 100 UNSUPPORTED records (from remaining 100 seeds: indices 100 to 199)
    # Current claim paired with distractor evidence from a completely unrelated article
    for i in range(100, 200):
        item = valid_rows[i]
        # Distractor from circularly shifted seed (+50 offset) to guarantee cross-document orthogonality
        distractor_idx = (i + 50) % 200
        distractor_item = valid_rows[distractor_idx]
        distractor_r = distractor_item["raw_row"]
        distractor_date = clean_date_str(distractor_r["last_modified_time"])
        
        records.append({
            "example_id": f"parallax_pilot_{i+1:04d}_unsupp",
            "question": item["raw_row"]["question"],
            "answer": item["curr_ans"],
            "answer_claim": item["curr_claim"],
            "evidence_text": str(distractor_r["evidence"]).strip(),
            "evidence_source": f"{distractor_item['doc_title']} (id: {distractor_item['doc_id']})",
            "evidence_version": distractor_date,
            "valid_from": distractor_date,
            "valid_to": "present",
            "superseded_by": None,
            "reference_time": REFERENCE_TIME,
            "gold_label": "UNSUPPORTED",
            "split": "pilot",
            "annotation_notes": f"Claim from '{item['doc_title']}' paired with unrelated distractor evidence from '{distractor_item['doc_title']}'.",
        })

    pilot_df = pd.DataFrame(records)
    
    # Schema check
    expected_cols = [
        "example_id", "question", "answer", "answer_claim", "evidence_text",
        "evidence_source", "evidence_version", "valid_from", "valid_to",
        "superseded_by", "reference_time", "gold_label", "split", "annotation_notes"
    ]
    assert list(pilot_df.columns) == expected_cols, f"Columns mismatch: {list(pilot_df.columns)}"
    assert len(pilot_df) == 600, f"Expected 600 records, got {len(pilot_df)}"
    
    os.makedirs(output_dir, exist_ok=True)
    parquet_out = os.path.join(output_dir, "pilot_benchmark.parquet")
    csv_out = os.path.join(output_dir, "pilot_benchmark.csv")
    
    pilot_df.to_parquet(parquet_out, index=False)
    pilot_df.to_csv(csv_out, index=False, encoding="utf-8")
    
    print("\nSuccessfully generated Parallax pilot benchmark:")
    print(f"  Parquet: {parquet_out}")
    print(f"  CSV:     {csv_out}")
    print("\nClass distribution:")
    print(pilot_df["gold_label"].value_counts())
    
    return pilot_df


if __name__ == "__main__":
    raw_path = r"d:\Engineering\UROP\Projects\Parallax\Datasets\D1-HoH\hoh_qas_240601_241201.parquet"
    out_dir = r"d:\Engineering\UROP\Projects\Parallax\data\pilot"
    build_pilot_dataset(raw_path, out_dir)
