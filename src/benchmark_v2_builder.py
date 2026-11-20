"""
Parallax Benchmark v2 Builder: Mixed-Context Versioned Retrieval
================================================================
Constructs the Parallax v2 Benchmark implementing the realistic RAG retrieval
setting where evidence passages carry only document IDs and snapshot dates,
with ZERO oracle validity metadata (no valid_to, valid_from, or superseded_by).

Taxonomy in Mixed Context:
--------------------------
1. OUTDATED:
   - Claim asserts historical fact (old_claim).
   - Context contains BOTH old snapshot (supports claim) and newer snapshot
     of the same document (asserts updated fact), plus 1-3 distractors.
2. SUPPORTED_CURRENT:
   - Claim asserts current active fact (curr_claim).
   - Context ALSO contains BOTH old and new snapshots of the same document,
     plus 1-3 distractors (symmetrical context prevents shortcut).
3. CONTRADICTED:
   - Claim asserts a third conflicting entity refuted by both snapshots.
   - Context contains both snapshots plus 1-3 distractors.
4. UNSUPPORTED:
   - Claim from target document.
   - Context contains only 3-5 orthogonal distractor passages (no target doc).

Diagnostic Track:
-----------------
- single_version_slice: Evaluates retrieval failure where only old snapshot is
  retrieved, verifying that staleness cannot be detected without newer evidence.
"""

import os
import sys
import re
import json
import random
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

# Bootstrap sys.path
_cur = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_cur, ".."))
for _p in [_cur, _root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from verbalizer import verbalize_claim
except ImportError:
    from src.verbalizer import verbalize_claim

RANDOM_SEED = 42


def extract_doc_metadata(source_str: str) -> tuple[str, str]:
    """Extracts (doc_title, doc_id) from evidence_source string."""
    m = re.search(r"\(id:\s*(\w+)\)", str(source_str))
    doc_id = m.group(1) if m else "unknown"
    doc_title = re.sub(r"\s*\(id:.*\)", "", str(source_str)).strip()
    return doc_title, doc_id


def load_seed_documents(csv_path: str) -> List[Dict[str, Any]]:
    """
    Loads disjoint seed documents from existing benchmark v1 CSV splits.
    Preserves exact document-disjoint split boundaries.
    """
    df = pd.read_csv(csv_path)
    curr_df = df[df["gold_label"] == "SUPPORTED_CURRENT"].reset_index(drop=True)
    outd_df = df[df["gold_label"] == "OUTDATED"].reset_index(drop=True)

    seeds: List[Dict[str, Any]] = []
    n = min(len(curr_df), len(outd_df))

    for i in range(n):
        c_row = curr_df.iloc[i]
        o_row = outd_df.iloc[i]

        doc_title, doc_id = extract_doc_metadata(str(c_row["evidence_source"]))

        seeds.append({
            "doc_id": doc_id,
            "doc_title": doc_title,
            "question": str(c_row["question"]),
            "curr_ans": str(c_row["answer"]),
            "old_ans": str(o_row["answer"]),
            "curr_claim": str(c_row["answer_claim"]),
            "old_claim": str(o_row["answer_claim"]),
            "curr_ev": str(c_row["evidence_text"]),
            "old_ev": str(o_row["evidence_text"]),
            "curr_date": str(c_row["evidence_version"])[:10],
            "old_date": str(o_row["evidence_version"])[:10],
            "domain": str(c_row.get("domain", "General Knowledge")),
        })

    return seeds


def build_mixed_context_examples(
    seeds: List[Dict[str, Any]],
    split_name: str,
    seed: int = RANDOM_SEED
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Builds mixed-context v2 examples for a given split:
    - Balanced 4-class records (len(seeds) * 4)
    - Single-version diagnostic slice (len(seeds) records)
    """
    rng = random.Random(seed)
    n = len(seeds)
    v2_records: List[Dict[str, Any]] = []
    single_version_records: List[Dict[str, Any]] = []

    def make_passage(doc_id: str, snapshot_date: str, text: str) -> Dict[str, str]:
        return {
            "passage_id": "",
            "doc_id": str(doc_id),
            "snapshot_date": str(snapshot_date),
            "text": str(text),
        }

    for i, s in enumerate(seeds):
        p_old = make_passage(s["doc_id"], s["old_date"], s["old_ev"])
        p_new = make_passage(s["doc_id"], s["curr_date"], s["curr_ev"])

        # Sample 1 to 3 distractor passages from other seed docs in the same split
        num_distractors = rng.choice([1, 2, 3])
        distractor_indices = [(i + offset) % n for offset in range(1, num_distractors + 1)]
        distractor_passages = []
        for d_idx in distractor_indices:
            d_seed = seeds[d_idx]
            # Use either curr or old snapshot of distractor
            d_choice = rng.choice(["curr", "old"])
            d_date = d_seed["curr_date"] if d_choice == "curr" else d_seed["old_date"]
            d_text = d_seed["curr_ev"] if d_choice == "curr" else d_seed["old_ev"]
            distractor_passages.append(make_passage(d_seed["doc_id"], d_date, d_text))

        # -------------------------------------------------------------
        # Helper: assemble & shuffle passages with fresh IDs
        # -------------------------------------------------------------
        def assemble_passages(passage_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
            copied = [dict(p) for p in passage_list]
            rng.shuffle(copied)
            for idx, p in enumerate(copied):
                p["passage_id"] = f"p_{idx}"
            return copied

        # -------------------------------------------------------------
        # 1. OUTDATED
        # -------------------------------------------------------------
        outd_context = assemble_passages([p_old, p_new] + distractor_passages)
        v2_records.append({
            "example_id": f"parallax_v2_{split_name}_{i+1:04d}_outd",
            "split": split_name,
            "claim": s["old_claim"],
            "gold_label": "OUTDATED",
            "has_superseding_evidence": True,
            "target_doc_id": s["doc_id"],
            "retrieved_passages": outd_context,
        })

        # -------------------------------------------------------------
        # 2. SUPPORTED_CURRENT
        # (Contains identical old+new passage pair; claim aligns with new)
        # -------------------------------------------------------------
        curr_context = assemble_passages([p_old, p_new] + distractor_passages)
        v2_records.append({
            "example_id": f"parallax_v2_{split_name}_{i+1:04d}_curr",
            "split": split_name,
            "claim": s["curr_claim"],
            "gold_label": "SUPPORTED_CURRENT",
            "has_superseding_evidence": True,
            "target_doc_id": s["doc_id"],
            "retrieved_passages": curr_context,
        })

        # -------------------------------------------------------------
        # 3. CONTRADICTED
        # (Claim asserts conflicting entity refuted by both snapshots)
        # -------------------------------------------------------------
        conflicting_seed = seeds[(i + (n // 2)) % n]
        conflicting_ans = conflicting_seed["curr_ans"]
        contra_claim = verbalize_claim(s["question"], conflicting_ans, strict=False)
        if not contra_claim or contra_claim.lower() == s["curr_claim"].lower():
            # Fallback entity replacement
            contra_claim = s["curr_claim"].replace(s["curr_ans"], conflicting_ans)

        contra_context = assemble_passages([p_old, p_new] + distractor_passages)
        v2_records.append({
            "example_id": f"parallax_v2_{split_name}_{i+1:04d}_contra",
            "split": split_name,
            "claim": contra_claim,
            "gold_label": "CONTRADICTED",
            "has_superseding_evidence": True,
            "target_doc_id": s["doc_id"],
            "retrieved_passages": contra_context,
        })

        # -------------------------------------------------------------
        # 4. UNSUPPORTED
        # (Context contains NO target doc passages, but mirrors the exact
        #  paired-doc + distractor structure so date/length distributions are identical)
        # -------------------------------------------------------------
        # Pick one distractor doc to provide a paired old+new snapshot
        paired_dist_idx = (i + 3) % n
        p_dist_seed = seeds[paired_dist_idx]
        p_dist_old = make_passage(p_dist_seed["doc_id"], p_dist_seed["old_date"], p_dist_seed["old_ev"])
        p_dist_new = make_passage(p_dist_seed["doc_id"], p_dist_seed["curr_date"], p_dist_seed["curr_ev"])

        # Pick additional single distractor passages matching num_distractors (1..3)
        extra_dist_passages = []
        for d_offset in range(4, 4 + num_distractors):
            e_idx = (i + d_offset) % n
            e_seed = seeds[e_idx]
            e_choice = rng.choice(["curr", "old"])
            e_date = e_seed["curr_date"] if e_choice == "curr" else e_seed["old_date"]
            e_text = e_seed["curr_ev"] if e_choice == "curr" else e_seed["old_ev"]
            extra_dist_passages.append(make_passage(e_seed["doc_id"], e_date, e_text))

        unsupp_context = assemble_passages([p_dist_old, p_dist_new] + extra_dist_passages)
        v2_records.append({
            "example_id": f"parallax_v2_{split_name}_{i+1:04d}_unsupp",
            "split": split_name,
            "claim": s["curr_claim"],
            "gold_label": "UNSUPPORTED",
            "has_superseding_evidence": False,
            "target_doc_id": s["doc_id"],
            "retrieved_passages": unsupp_context,
        })

        # -------------------------------------------------------------
        # 5. Diagnostic: Single-Version Slice
        # (Only old passage is retrieved; newer version is absent)
        # -------------------------------------------------------------
        single_context = assemble_passages([p_old] + distractor_passages)
        single_version_records.append({
            "example_id": f"parallax_v2_{split_name}_{i+1:04d}_single_outd",
            "split": split_name,
            "claim": s["old_claim"],
            "gold_label": "OUTDATED",
            "has_superseding_evidence": False,
            "target_doc_id": s["doc_id"],
            "retrieved_passages": single_context,
        })

    return v2_records, single_version_records


def export_v2_dataset(
    records: List[Dict[str, Any]],
    output_base_dir: str,
    split_name: str
):
    """Exports dataset in JSONL and CSV formats."""
    os.makedirs(output_base_dir, exist_ok=True)
    jsonl_path = os.path.join(output_base_dir, f"{split_name}.jsonl")
    csv_path = os.path.join(output_base_dir, f"{split_name}.csv")

    # 1. Write JSON Lines (primary format for nested passages)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # 2. Write CSV with JSON-serialized retrieved_passages for tabular compatibility
    flat_rows = []
    for r in records:
        flat_rows.append({
            "example_id": r["example_id"],
            "split": r["split"],
            "claim": r["claim"],
            "gold_label": r["gold_label"],
            "has_superseding_evidence": r["has_superseding_evidence"],
            "num_passages": len(r["retrieved_passages"]),
            "retrieved_passages_json": json.dumps(r["retrieved_passages"]),
        })

    df = pd.DataFrame(flat_rows)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Exported {len(records):,} records to {jsonl_path} and {csv_path}")


def generate_benchmark_v2(
    v1_bench_dir: str = "data/benchmark",
    v2_output_dir: str = "data/benchmark_v2",
    review_dir: str = "data/review"
):
    print("=" * 65)
    print("PARALLAX BENCHMARK v2 GENERATION (MIXED-CONTEXT RETRIEVAL)")
    print("=" * 65)

    train_csv = os.path.join(v1_bench_dir, "train.csv")
    dev_csv = os.path.join(v1_bench_dir, "dev.csv")
    test_csv = os.path.join(v1_bench_dir, "test.csv")

    print("Step 1: Loading disjoint seed documents from v1 clean partitions...")
    train_seeds = load_seed_documents(train_csv)
    dev_seeds = load_seed_documents(dev_csv)
    test_seeds = load_seed_documents(test_csv)

    print(f"  Train seed documents: {len(train_seeds)}")
    print(f"  Dev seed documents:   {len(dev_seeds)}")
    print(f"  Test seed documents:  {len(test_seeds)}")

    # Verify 0% document overlap
    train_docs = {s["doc_id"] for s in train_seeds}
    dev_docs = {s["doc_id"] for s in dev_seeds}
    test_docs = {s["doc_id"] for s in test_seeds}

    assert len(train_docs.intersection(dev_docs)) == 0, "Document leakage between train and dev!"
    assert len(train_docs.intersection(test_docs)) == 0, "Document leakage between train and test!"
    assert len(dev_docs.intersection(test_docs)) == 0, "Document leakage between dev and test!"
    print("  Document Disjoint Verification: 0% overlap confirmed across all splits.")

    print("\nStep 2: Building mixed-context versioned passages...")
    train_v2, _ = build_mixed_context_examples(train_seeds, "train", seed=42)
    dev_v2, _ = build_mixed_context_examples(dev_seeds, "dev", seed=43)
    test_v2, test_single_slice = build_mixed_context_examples(test_seeds, "test", seed=44)

    full_v2 = train_v2 + dev_v2 + test_v2

    print("\nBenchmark v2 Sizes:")
    print(f"  Train:  {len(train_v2):,} records (400 per class)")
    print(f"  Dev:    {len(dev_v2):,} records (100 per class)")
    print(f"  Test:   {len(test_v2):,} records (200 per class)")
    print(f"  Single-Version Test Slice: {len(test_single_slice):,} records")
    print(f"  Total:  {len(full_v2):,} records (700 per class across 4 labels)")

    print("\nStep 3: Exporting v2 datasets...")
    export_v2_dataset(train_v2, v2_output_dir, "train")
    export_v2_dataset(dev_v2, v2_output_dir, "dev")
    export_v2_dataset(test_v2, v2_output_dir, "test")
    export_v2_dataset(test_single_slice, v2_output_dir, "test_single_version_slice")
    export_v2_dataset(full_v2, v2_output_dir, "full_benchmark_v2")

    # Step 4: Sample 100 rows for human audit review
    os.makedirs(review_dir, exist_ok=True)
    review_path = os.path.join(review_dir, "v2_sample_audit_100.csv")
    rng = random.Random(42)
    sampled_100 = rng.sample(test_v2, 100)

    audit_rows = []
    for r in sampled_100:
        passages_summary = " || ".join(
            f"[{p['passage_id']} | {p['doc_id']} | {p['snapshot_date']}] {p['text'][:90]}..."
            for p in r["retrieved_passages"]
        )
        audit_rows.append({
            "example_id": r["example_id"],
            "gold_label": r["gold_label"],
            "claim": r["claim"],
            "num_passages": len(r["retrieved_passages"]),
            "retrieved_context_summary": passages_summary,
        })

    pd.DataFrame(audit_rows).to_csv(review_path, index=False, encoding="utf-8")
    print(f"\nStep 4: Exported 100 human review audit rows to: {review_path}")


if __name__ == "__main__":
    generate_benchmark_v2()
