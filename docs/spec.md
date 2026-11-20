# Parallax Specification: Benchmark v2 & Dynamic Supersession

**Version**: 2.0.0  
**Status**: Active / Approved Specification  
**Scope**: Mixed-Context Retrieval-Augmented Generation (RAG) Auditing without Oracle Metadata  

---

## 1. Overview & Conceptual Grounding

In Parallax v1, the two-stage verifier achieved 96.38% test accuracy and 0.00% False-Support Rate ($\text{FSR}_{\text{OUTD}}$). However, v1 operated under an **Oracle-Metadata Assumption**: every passage was accompanied by external validity bounds (`valid_from`, `valid_to`) and an explicit database supersession pointer (`superseded_by`).

In production RAG systems, retrieval components (e.g., BM25, dense bi-encoders, vector stores) do not possess clairvoyant validity intervals. Instead, retrievers surface a heterogeneous mixture of passages across document revisions, each bearing only a document identifier, a snapshot/publication date, and text.

**Parallax v2** defines the realistic benchmark and inference framework:
1. **Zero Oracle Metadata**: All validity intervals (`valid_from`, `valid_to`) and relational pointers (`superseded_by`) are strictly excluded.
2. **Mixed-Version Multi-Passage Contexts**: Every context contains multiple passages, including versioned document evolutions and distractors.
3. **Dynamic Supersession Inference (Stage B)**: The verifier must infer whether a fact has been superseded by analyzing conflicting later snapshots of the same document present in the retrieved context.

---

## 2. Benchmark v2 Schema

Each example in Benchmark v2 is a single JSON / Parquet record conforming strictly to this schema:

```json
{
  "example_id": "parallax_v2_test_0001",
  "split": "test",
  "claim": "Lord de Mauley is the Master of the Horse.",
  "gold_label": "OUTDATED",
  "has_superseding_evidence": true,
  "retrieved_passages": [
    {
      "passage_id": "p_0",
      "doc_id": "100025",
      "snapshot_date": "2024-07-01",
      "text": "The Master of the Horse is an office in the Royal Household. In July 2024, Lord Vestey was appointed Master of the Horse, succeeding Lord de Mauley."
    },
    {
      "passage_id": "p_1",
      "doc_id": "52437830",
      "snapshot_date": "2024-08-01",
      "text": "Àlex Carbonell Vallés is a Spanish footballer who plays as a midfielder for Dutch club Almere City."
    },
    {
      "passage_id": "p_2",
      "doc_id": "100025",
      "snapshot_date": "2024-06-01",
      "text": "The Master of the Horse is an office in the Royal Household. The current Master of the Horse is Lord de Mauley."
    }
  ]
}
```

### Strict Prohibition
The following fields **must never appear** in any v2 dataset split:
- `valid_from`
- `valid_to`
- `superseded_by`
- `reference_time`
- `staleness_days`
- `temporal_gap_days`
- `annotation_notes` (if referencing temporal validity)

---

## 3. Taxonomy & Context Construction Rules

To ensure models cannot exploit superficial shortcuts (such as counting passages, detecting duplicate document IDs, or associating old dates with outdatedness), the context distributions across classes are structurally symmetrical.

### 3.1. Class Definitions

| Label | Definition | Context Construction |
| :--- | :--- | :--- |
| **`OUTDATED`** | The claim asserts an obsolete historical fact that is entailed by an older retrieved snapshot, but a newer retrieved snapshot of the same document provides a conflicting/updated fact. | $E_{\text{old}} + E_{\text{new}} + (1\dots3 \text{ distractors})$, randomly ordered. Claim = `old_claim`. |
| **`SUPPORTED_CURRENT`** | The claim asserts an active fact that is entailed by the latest retrieved snapshot, even though older historical snapshots may also be present. | $E_{\text{old}} + E_{\text{new}} + (1\dots3 \text{ distractors})$, randomly ordered. Claim = `curr_claim`. |
| **`CONTRADICTED`** | The claim asserts a fabricated or conflicting entity that is directly refuted by the retrieved context (and not supported by either snapshot). | $E_{\text{old}} + E_{\text{new}} + (1\dots3 \text{ distractors})$, randomly ordered. Claim = conflicting entity. |
| **`UNSUPPORTED`** | The retrieved context contains insufficient evidence to substantiate the claim (neither version of the target document is retrieved). | $3\dots5$ unrelated distractor passages from other documents in the same split. |

### 3.2. Symmetrical Pairing to Prevent Shortcuts
Notice that for `OUTDATED`, `SUPPORTED_CURRENT`, and `CONTRADICTED`:
- The exact same document revisions ($E_{\text{old}}$ and $E_{\text{new}}$) appear in context.
- The snapshot date distribution is identical.
- The number of passages (between 3 and 5) is drawn from the same random distribution.
- The order of passages is uniformly randomized.

Thus, a model **cannot guess the label** based on:
1. Passage count
2. Date ranges or date deltas
3. Relative position of the supporting passage
4. Presence of multiple versions of the same `doc_id`

The model must perform true joint semantic and temporal reasoning: align the claim to its supporting passage, identify subsequent revisions of that document, and determine if the subsequent revision supersedes the fact.

---

## 4. Single-Version Slice (Diagnostic Track)

In addition to the mixed-context benchmark, a separate evaluation slice (`single_version_slice`) evaluates scenarios where retrieval **only** retrieved $E_{\text{old}}$ (and failed to retrieve $E_{\text{new}}$).

- **Purpose**: Empirically quantify the theoretical limit of closed-world RAG: without external metadata or a newer retrieved revision, staleness cannot be detected.
- **Reporting**: Reported in the paper as a dedicated ablation table showing the necessity of multi-version retrieval.

---

## 5. Stage B: Dynamic Supersession Inference Algorithm

When auditing a claim $C$ against a set of retrieved passages $\{P_1, \dots, P_k\}$:

1. **Stage A (Passage-Level NLI)**:
   For each passage $P_i$, evaluate semantic inference:
   $$R_i = \text{NLI}(P_i, C) \in \{\text{ENTAILMENT}, \text{CONTRADICTION}, \text{NEUTRAL}\}$$
   Let $\mathcal{S} = \{P_i \mid R_i = \text{ENTAILMENT}\}$ be the supporting passages.

2. **Stage B (Dynamic Temporal Arbitration)**:
   - If $\mathcal{S} = \emptyset$:
     - If any $R_i = \text{CONTRADICTION}$: return `CONTRADICTED`.
     - Else: return `UNSUPPORTED`.
   - If $\mathcal{S} \neq \emptyset$:
     - For each $P_{\text{supp}} \in \mathcal{S}$ with document ID $D = P_{\text{supp}}.\text{doc\_id}$ and date $T_{\text{supp}} = P_{\text{supp}}.\text{snapshot\_date}$:
       - Find all subsequent revisions in context:
         $$\mathcal{L} = \{P_j \mid P_j.\text{doc\_id} = D \land P_j.\text{snapshot\_date} > T_{\text{supp}}\}$$
       - If $\mathcal{L} \neq \emptyset$:
         - Let $P_{\text{latest}} = \arg\max_{P_j \in \mathcal{L}} P_j.\text{snapshot\_date}$.
         - Check if $P_{\text{latest}}$ supersedes $P_{\text{supp}}$:
           $$\text{NLI}(P_{\text{latest}}, P_{\text{supp}}) \in \{\text{CONTRADICTION}, \text{NEUTRAL}\} \lor \text{Entity}(P_{\text{latest}}) \neq \text{Entity}(C)$$
         - If superseded: return **`OUTDATED`** (Reason: *"Supported by snapshot from $T_{\text{supp}}$, but superseded by newer revision on $P_{\text{latest}}.\text{snapshot\_date}$"*).
     - If no supporting passage was superseded: return **`SUPPORTED_CURRENT`**.

---

## 6. Leakage Verification Criteria

Every generated dataset release must pass two gating verification scripts:

1. **`tests/test_benchmark_v2_leakage.py`**:
   - Asserts no forbidden keys exist in schema.
   - Asserts passage count distributions are identical across classes ($p > 0.05$ under Chi-Square test).
   - Asserts passage positions of supporting evidence are uniformly distributed.

2. **`experiments/check_leakage_classifier.py`**:
   - Trains shallow classifiers (`LogisticRegression`, `RandomForest`) using **only**:
     - Date features (`min_date`, `max_date`, `delta_days`, `num_dates`)
     - Layout features (`num_passages`, `mean_passage_length`, `total_length`)
   - **Gating threshold**: Accuracy on 4-class prediction must be $\le 27.5\%$ (chance is $25.0\%$). If classifier achieves $> 28\%$, generation is rejected and generator must be adjusted.
