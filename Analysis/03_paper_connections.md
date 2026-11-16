# 3. Research Paper Verification and Literature Connections

This document details the verification process linking each workspace dataset folder to research papers residing in [`Papers/Tier-1`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-1), [`Papers/Tier-2`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-2), and [`Papers/Tier-3`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-3).

---

## Verification Summary Table

| Dataset Subfolder | Matched Paper File | Full Paper Title | Venue / Date | Status |
|---|---|---|---|---|
| [`Datasets/D1-HoH/`](file:///d:/Engineering/UROP/Projects/RAG/Datasets/D1-HoH/) | [`Papers/Tier-1/HoH.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-1/HoH.pdf) | *HOH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation* | ACL 2025 (Long Papers, pp. 6036–6063) | **VERIFIED** |
| [`Datasets/D2-VitaminC/`](file:///d:/Engineering/UROP/Projects/RAG/Datasets/D2-VitaminC/) | [`Papers/Tier-2/VitaminC.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-2/VitaminC.pdf) | *Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence* | NAACL 2021 (arXiv:2103.08541) | **VERIFIED** |
| [`Datasets/D3-RAGTruth/`](file:///d:/Engineering/UROP/Projects/RAG/Datasets/D3-RAGTruth/) | [`Papers/Tier-2/RAGTruth.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-2/RAGTruth.pdf) | *RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models* | ACL 2024 (Long Papers, pp. 10862–10878) | **VERIFIED** |

---

## Detailed Evidence and Citations

### 1. Connection: `D1-HoH` $\longleftrightarrow$ `HoH.pdf`

- **Local PDF Path**: [`d:/Engineering/UROP/Projects/RAG/Papers/Tier-1/HoH.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-1/HoH.pdf)
- **Authors**: Jie Ouyang, Tingyue Pan, Mingyue Cheng, Ruiran Yan, Yucong Luo, Jiaying Lin, Qi Liu (USTC).
- **Match Status**: **VERIFIED**

#### Verification Points in Local Files:
1. **Schema Match**:
   - In `HoH.pdf`, **Appendix C.1, Table 21 (p. 6056)** defines the exact fields:
     - `question` (`str`): The question for RAG to answer.
     - `answer` (`str`): The latest gold answer.
     - `evidence` (`str`): Source material substantiating the answer.
     - `last_modified_time` (`date`): Document last modified date.
     - `outdated_infos` (`list[dict]`): Nested collection of historical answers, evidence, and modification dates.
     - `document` (`dict`): Source document with `id` and `title`.
   - In [`Datasets/D1-HoH/hoh_qas_240601_241201.parquet`](file:///d:/Engineering/UROP/Projects/RAG/Datasets/D1-HoH/hoh_qas_240601_241201.parquet), the PyArrow schema is identical down to every field name and sub-field nesting.
2. **Snapshot Timeline Alignment**:
   - In `HoH.pdf`, **Section 3 (p. 6038)** states:
     > *"The initial version was generated based on snapshots from 2024-06-01 and 2024-07-01, with regular monthly updates since then. As of now, we have updated it to the snapshot of 2024-11-01."*
   - The file name in the workspace is `hoh_qas_240601_241201.parquet`, representing the continuation through the 2024-12-01 snapshot.
3. **QA Construction Pipeline**:
   - Section 3.1 & Figure 2 (p. 6039) describe the token-level diff + LLM pipeline generating the exact structure seen in sample records.
4. **Codebase Reference**:
   - Section 1 (p. 6036) links to `https://github.com/0russwest0/HoH`.

---

### 2. Connection: `D2-VitaminC` $\longleftrightarrow$ `VitaminC.pdf`

- **Local PDF Path**: [`d:/Engineering/UROP/Projects/RAG/Papers/Tier-2/VitaminC.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-2/VitaminC.pdf)
- **Authors**: Tal Schuster, Adam Fisch, Regina Barzilay (MIT CSAIL).
- **Match Status**: **VERIFIED**

#### Verification Points in Local Files:
1. **Direct Documentation Citation**:
   - Inside [`Datasets/D2-VitaminC/vitaminc.zip`](file:///d:/Engineering/UROP/Projects/RAG/Datasets/D2-VitaminC/), the included `vitaminc/README.txt` states:
     > *"This dataset is part of the NAACL 2021 paper: 'Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence'. Please follow the usage guidelines at https://github.com/TalSchuster/VitaminC"*
2. **Exact Split and Category Statistics**:
   - In `VitaminC.pdf`, **Table 2 (p. 4)** lists the breakdown:
     - Train: Real SUP (124,864) + Syn SUP (60,850) + Real REF (71,108) + Syn REF (60,850) + Real NEI (52,981) = **370,653**.
     - Dev: Real SUP (21,102) + Syn SUP (10,382) + Real REF (12,146) + Syn REF (10,382) + Real NEI (9,042) = **63,054**.
     - Test: Real SUP (17,306) + Syn SUP (10,358) + Real REF (9,907) + Syn REF (10,358) + Real NEI (7,268) = **55,197**.
   - These numbers correspond line-for-line with the line counts inside `train.jsonl`, `dev.jsonl`, and `test.jsonl`.
3. **Companion Tasks**:
   - The existence of two archives in `Datasets/D2-VitaminC` (`vitaminc.zip` and `vitaminc_flagging.zip`) directly mirrors Section 4.1 (*Factual Revision Flagging*) and Section 4.2 (*Fact Verification*) in the paper.

---

### 3. Connection: `D3-RAGTruth` $\longleftrightarrow$ `RAGTruth.pdf`

- **Local PDF Path**: [`d:/Engineering/UROP/Projects/RAG/Papers/Tier-2/RAGTruth.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-2/RAGTruth.pdf)
- **Authors**: Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Randy Zhong, Juntong Song, Tong Zhang (NewsBreak & UIUC).
- **Match Status**: **VERIFIED**

#### Verification Points in Local Files:
1. **Dataset Statistics**:
   - In `RAGTruth.pdf`, **Table 2 (p. 10867)** details:
     - 2,965 total source instances (QA: 989, Data-to-text: 1,033, Summarization: 943). Exactly matches [`source_info.jsonl`](file:///d:/Engineering/UROP/Projects/RAG/Datasets/D3-RAGTruth/source_info.jsonl) (2,965 lines).
     - 17,790 LLM responses across 6 models (GPT-4, GPT-3.5, Mistral-7B, Llama-2-7B/13B/70B). Exactly matches [`response.jsonl`](file:///d:/Engineering/UROP/Projects/RAG/Datasets/D3-RAGTruth/response.jsonl) (17,790 lines).
     - Hallucinated responses: 7,664 (43.1%). In `response.jsonl`, exactly 7,664 lines have non-empty `labels`.
2. **Hallucination Taxonomy Alignment**:
   - Section 3.1 & Table 8 (p. 10874) present the four categories (`Evident Conflict`, `Subtle Conflict`, `Evident Baseless Info`, `Subtle Baseless Info`) and auxiliary flags (`implicit_true`, `due_to_null` in Section 3.4), matching the exact dictionary keys in `response.jsonl`.
3. **Repository URL**:
   - Section 1, Footnote 1 (p. 10862) cites `https://github.com/ParticleMedia/RAGTruth`.

---

## Role of Other Papers in Workspace

The remaining papers in `Papers/` serve as key contextual references for Parallax:

1. **Stale Information & Version-Aware RAG (Tier-1)**:
   - [`VersionRAG.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-1/VersionRAG.pdf): Version-aware retrieval-augmented generation for evolving documents.
   - [`RT4CHART.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-1/RT4CHART.pdf): Hierarchical verification and testing in RAG.
   - [`Stale-Document Poisoning.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-1/Stale-Document%20Poisoning.pdf) & [`FRESCO.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-1/FRESCO.pdf): Security and robustness risks of stale knowledge.
2. **Claim Verifiers & Faithfulness Checkers (Tier-2)**:
   - [`MiniCheck.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-2/MiniCheck.pdf), [`LettuceDetect.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-2/LettuceDetect.pdf), [`RAGChecker.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-2/RAGChecker.pdf), [`FActScore.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-2/FActScore.pdf): State-of-the-art claim-level detectors to be evaluated as baselines for false-support on outdated evidence.
3. **Temporal QA Foundations (Tier-3)**:
   - [`FreshQA.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-3/FreshQA.pdf), [`TimeQA.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-3/TimeQA.pdf), [`Knowledge Conflicts Survey.pdf`](file:///d:/Engineering/UROP/Projects/RAG/Papers/Tier-3/Knowledge%20Conflicts%20Survey.pdf): Foundations on present-anchored queries and conflicting factual updates.
