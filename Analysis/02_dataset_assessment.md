# 2. Assessment of Datasets for Parallax

This document provides a systematic evaluation of how well each dataset supports the research objectives, label taxonomy, and verification architecture specified in [`README.md`](file:///d:/Engineering/UROP/Projects/RAG/README.md).

---

## Comparative Assessment Matrix

| Evaluation Dimension | `D1-HoH` | `D2-VitaminC` | `D3-RAGTruth` |
|---|---|---|---|
| **Primary Task** | Outdated info in RAG QA | Contrastive fact verification | RAG hallucination detection |
| **Fit with Parallax Goal** | **Primary Fit (Core Benchmark)** | **Auxiliary (Contrastive NLI)** | **Baseline (Faithfulness / FSR)** |
| **Claims Available** | Short entity/phrase answers | Human-written atomic claims | Multi-sentence LLM responses |
| **Evidence Granularity** | Sentence-level (paired) | Sentence-level (paired) | Multi-sentence / Passages |
| **Document Dates / Timestamps** | **Explicit ISO timestamps** | Absent (revision IDs only) | None (Static) |
| **Temporal Intervals** | Bounded by monthly snapshots | None | None |
| **Supersession Relationships** | **Explicit paired schema** | Contrastive pairs (`case_id`) | None |
| **Direct Parallax Label Readiness** | Directly maps `SUPPORTED_CURRENT` & `OUTDATED` | Maps `SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED` | Maps `SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED` |
| **Preprocessing Overhead** | Low (verbalize answer to claim) | Medium (retrieve dates if needed) | Medium (decompose to atomic claims) |
| **Risk of False Support** | High (ideal testbed) | Moderate | High (classic hallucination) |

---

## Detailed Evaluation by Dataset

### 1. `D1-HoH`

#### Designed Task
Designed for investigating how outdated information coexisting in knowledge bases degrades the performance of RAG retrieval and generation modules. It models dynamic, present-anchored questions whose answers have evolved between Wikipedia snapshots.

#### Alignment with Parallax Research Questions
- **Direct Alignment**: Parallax asks: *"Can a claim-level verifier distinguish claims supported by current evidence from claims supported only by superseded evidence in RAG answers?"*
- `D1-HoH` is the **only dataset in the workspace** with explicitly preserved superseded evidence and verified timestamps. It directly provides the exact counterfactual scenarios Parallax targets: where a statement is historically grounded in retrieved text, yet false at the current evaluation time.

#### What it Provides
1. **Paired Current vs. Superseded Evidence**: Each record links the current evidence sentence to previous evidence versions.
2. **Calendar Timestamps**: Explicit dates (`last_modified_time`) spanning July 2024 to December 2024, enabling strict temporal interval arithmetic.
3. **Entity-Level Fact Updates**: Grounded updates across diverse real-world categories (Finance, Sports, Politics, Science).

#### What it is Missing for Parallax
1. **Full Natural Answer Claims**: `answer` is usually an entity or number (e.g. `"Maudiozyma bulderi"`). Parallax evaluates answer claims in RAG; therefore, a lightweight template or LLM claim verbalizer is required to frame the answer into a complete claim (e.g., *"Maudiozyma bulderi can be used to ferment gluconolactone..."*).
2. **Explicit Negative (`UNSUPPORTED`) Pairs in QA Table**: The table itself only includes positive current and positive outdated evidence. To evaluate the full 4-class taxonomy, distractor passages (or non-relevant text) must be sampled to generate `UNSUPPORTED` cases.

#### Suitability & Recommended Role
- **Role**: **Primary Benchmark Source**.
- **Usage**: Can be used immediately for the pilot benchmark with minimal verbalization preprocessing.

---

### 2. `D2-VitaminC`

#### Designed Task
Designed for robust fact verification using contrastive evidence. It was created to force models to become sensitive to subtle factual changes in text (e.g., numbers, names, dates) and eliminate claim-only annotation bias.

#### Alignment with Parallax Research Questions
- **Partial Alignment**: While VitaminC is fundamentally motivated by Wikipedia revisions, it strips out the calendar timeline. It treats contrastive verification purely as static natural language inference (SUPPORTS vs. REFUTES vs. NOT ENOUGH INFO).
- It does **not** evaluate whether evidence was superseded relative to an answer's reference time.

#### What it Provides
1. **High-Quality Human Claims**: 488,904 claims crafted by 70 professional annotators specifically designed to avoid lexical cues.
2. **Contrastive Triples**: A single claim evaluated against two nearly identical sentences where one fact differs.
3. **Fine-Grained Edit Annotations**: Detailed character/word differences in `vitaminc_flagging.zip`.

#### What it is Missing for Parallax
1. **Timestamps**: No ISO modification dates in the files (only internal Wikipedia `wiki_revision_id` numbers).
2. **Directionality & Validity Intervals**: Does not indicate which sentence is older or newer, nor when either ceased to be valid.

#### Suitability & Recommended Role
- **Role**: **Verifier Backbone / NLI Training Corpus**.
- **Usage**: Highly valuable for training the claim entailment component of the Parallax verifier. Models fine-tuned on VitaminC are significantly more robust to subtle factual alterations.

---

### 3. `D3-RAGTruth`

#### Designed Task
Designed for benchmarking word-level and span-level hallucinations across mainstream LLMs (GPT-4, Llama-2, Mistral) in standard RAG tasks (QA, Data-to-text, Summarization).

#### Alignment with Parallax Research Questions
- **Static Alignment / Complementary Baseline**: It addresses standard faithfulness (checking if generated text is supported by static retrieved context). It does **not** cover temporal supersession.
- However, it directly answers Parallax's third objective: *"An evaluation of existing faithfulness detectors on outdated evidence, including their false-support rate."* RAGTruth serves as the gold-standard static benchmark to evaluate detector precision/recall before introducing temporal variables.

#### What it Provides
1. **Natural LLM Outputs**: 17,790 multi-sentence answers generated under realistic RAG conditions.
2. **Span-Level Annotations**: Character-precise spans indicating where models contradicted evidence (`Evident Conflict`) or hallucinated unsupported facts (`Baseless Info`).
3. **Model Diversity**: Direct comparison across 6 foundation models.

#### What it is Missing for Parallax
1. **Temporal Dimensions**: Contexts are completely static; no document versions, timestamps, or superseded states.
2. **Outdated Ground Truth**: Cannot test whether a claim grounded in obsolete text is flagged.

#### Suitability & Recommended Role
- **Role**: **Faithfulness Baseline & Negative Reference Corpus**.
- **Usage**: Use to evaluate baseline hallucination detectors.

---

## Data Quality, License, and Scale Summary

| Dataset | Annotator Quality & Verification | License & Provenance | Scale | Concerns / Potential Risks |
|---|---|---|---|---|
| **D1-HoH** | High (Token diff + LM filter + 98.4–100% human audit in paper) | Wikipedia Dumps (CC BY-SA 3.0 / 4.0) | 111,972 rows | Answers are short entities; requires consistent verbalization into claims. |
| **D2-VitaminC** | High (Fleiss $\kappa = 0.7065$; 70 native English annotators) | CC BY-SA 3.0 (Wikipedia + FEVER derived) | 488,904 pairs | Lacks dates in files; cannot be used as temporal benchmark without external lookups. |
| **D3-RAGTruth** | High (Double annotation, 91.8% response agreement, 78.8% span agreement) | NewsBreak Research (Open-source) | 17,790 responses | Completely static context; zero temporal metadata. |
