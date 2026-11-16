# 1. Dataset Identification and Structural Profiles

This document details the location, file types, approximate sizes, internal structures, schemas, and representative records for the three dataset subfolders located within [`Datasets/`](file:///d:/Engineering/UROP/Projects/RAG/Datasets/).

---

## Workspace Inventory

```text
Datasets/
├── D1-HoH/
│   └── hoh_qas_240601_241201.parquet    (27.65 MB, 111,972 rows)
├── D2-VitaminC/
│   ├── vitaminc.zip                     (21.34 MB compressed, 216.5 MB uncompressed)
│   └── vitaminc_flagging.zip            (49.38 MB compressed, 232.8 MB uncompressed)
└── D3-RAGTruth/
    ├── response.jsonl                   (20.46 MB, 17,790 records)
    └── source_info.jsonl                (14.42 MB, 2,965 records)
```

---

## Dataset 1: `D1-HoH`

### File & Path Information
- **Exact Path**: [`d:/Engineering/UROP/Projects/RAG/Datasets/D1-HoH`](file:///d:/Engineering/UROP/Projects/RAG/Datasets/D1-HoH)
- **Primary File**: `hoh_qas_240601_241201.parquet`
- **File Type**: Apache Parquet (binary columnar storage)
- **Size**: 28,997,994 bytes (~27.65 MB)
- **File Count**: 1 file

### Dataset Scale & Splits
- **Total Rows**: 111,972 rows
- **Total Columns**: 6 columns
- **Splits**: Monolithic table covering Wikipedia snapshot updates from `2024-06-01` through `2024-12-01`. In the source paper, a balanced evaluation split of 10,000 QA pairs is selected by semantic clustering.

### Schema & Fields
| Field Name | Data Type | Description |
|---|---|---|
| `question` | `string` | Present-anchored query regarding real-world facts. |
| `answer` | `string` | Current valid short answer as of the latest snapshot. |
| `last_modified_time` | `timestamp[ns]` | Date of the latest document snapshot (`2024-07-01` to `2024-12-01`). |
| `evidence` | `string` | Supporting evidence sentence from the current snapshot. |
| `outdated_infos` | `list<struct>` | Chronological list of previous historical answers and evidence. Each element contains: <br>• `answer` (`string`)<br>• `evidence` (`string`)<br>• `last_modified_time` (`string`) |
| `document` | `struct` | Source document metadata: <br>• `id` (`string`): Wikipedia article ID<br>• `title` (`string`): Article title |

### Outdated Information Distribution
Every single record in `hoh_qas_240601_241201.parquet` contains at least one outdated evidence record:
- **1 outdated version**: 91,856 records (82.0%)
- **2 outdated versions**: 3,647 records (3.3%)
- **3 outdated versions**: 447 records (0.4%)
- **4 outdated versions**: 108 records (0.1%)
- **5 outdated versions**: 66 records (0.06%)

### Representative Record
```json
{
  "question": "What is the name of the yeast that can be used to ferment gluconolactone to ethanol and carbon dioxide?",
  "answer": "Maudiozyma bulderi",
  "last_modified_time": "2024-07-01 00:00:00",
  "evidence": "The yeast \"Maudiozyma bulderi\" can be used to ferment gluconolactone to ethanol and carbon dioxide.",
  "outdated_infos": [
    {
      "answer": "Saccharomyces bulderi",
      "evidence": "The yeast \"Saccharomyces bulderi\" can be used to ferment gluconolactone to ethanol and carbon dioxide.",
      "last_modified_time": "2024-06-01"
    }
  ],
  "document": {
    "id": "1000005",
    "title": "Glucono delta-lactone"
  }
}
```

### Coverage of Parallax Elements
- **Answer Claims**: Present as short entity strings; requires declarative verbalization for claim-level verification.
- **Retrieved Evidence**: Present for both current and outdated versions at sentence level.
- **Timestamps**: Explicit ISO dates for both current and prior snapshots.
- **Document Versions**: Document IDs and timestamp-based version increments.
- **Validity Periods**: Implicitly bounded by the revision history (`valid_from` = modification date, `valid_to` = next revision date).
- **Supersession Links**: **Explicitly paired** inside `outdated_infos`.

---

## Dataset 2: `D2-VitaminC`

### File & Path Information
- **Exact Path**: [`d:/Engineering/UROP/Projects/RAG/Datasets/D2-VitaminC`](file:///d:/Engineering/UROP/Projects/RAG/Datasets/D2-VitaminC)
- **Primary Files**:
  - `vitaminc.zip`: 22,381,149 bytes (~21.34 MB)
  - `vitaminc_flagging.zip`: 51,778,363 bytes (~49.38 MB)
- **File Type**: ZIP archives containing JSON Lines and text metadata
- **File Count**: 2 ZIP archives containing 8 internal files

### Internal Archive Structure & Record Counts
#### 1. `vitaminc.zip` (Fact Verification)
- `vitaminc/train.jsonl`: 370,653 records (164.4 MB uncompressed)
  - `SUPPORTS`: 185,714
  - `REFUTES`: 131,958
  - `NOT ENOUGH INFO`: 52,981
- `vitaminc/dev.jsonl`: 63,054 records (27.8 MB uncompressed)
  - `SUPPORTS`: 31,484
  - `REFUTES`: 22,528
  - `NOT ENOUGH INFO`: 9,042
- `vitaminc/test.jsonl`: 55,197 records (24.3 MB uncompressed)
  - `SUPPORTS`: 27,664
  - `REFUTES`: 20,265
  - `NOT ENOUGH INFO`: 7,268
- Total Fact Verification Pairs: **488,904**
- Auxiliary files: `vitaminc/LICENSE`, `vitaminc/README.txt`

#### 2. `vitaminc_flagging.zip` (Factual Revision Flagging)
- `vitaminc_flagging/train.jsonl`: 236,998 records (181.1 MB uncompressed)
- `vitaminc_flagging/dev.jsonl`: 36,624 records (28.0 MB uncompressed)
- `vitaminc_flagging/test.jsonl`: 31,048 records (23.8 MB uncompressed)
- Total Flagging Records: **304,670**
- Labels: `factual` vs. `not factual`

### Schema & Fields (`vitaminc.zip`)
| Field Name | Data Type | Description |
|---|---|---|
| `unique_id` | `string` | Unique identifier for the claim-evidence pair. |
| `case_id` | `string` | Grouping key that links contrastive revisions together. |
| `wiki_revision_id` | `string` | Wikipedia internal revision identifier. |
| `label` | `string` | `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`. |
| `claim` | `string` | Full, natural-language factual claim written by annotators. |
| `evidence` | `string` | Evidence sentence extracted from the corresponding Wikipedia revision. |
| `page` | `string` | Wikipedia article title. |
| `revision_type` | `string` | `"real"` (mined from edit history) or `"synthetic"` (rule-augmented). |

### Representative Records (Contrastive Pair from `test.jsonl`)
```json
{
  "unique_id": "5ed4de07c9e77c000848a180_1",
  "case_id": "5ed4de07c9e77c000848a180",
  "wiki_revision_id": "927477259",
  "label": "NOT ENOUGH INFO",
  "claim": "Westlife made under 23.5 million sales in the UK .",
  "evidence": "According to the British Phonographic Industry ( BPI ) , Westlife has been certified for 13 million albums and 9.8 million singles , with a total of more than 23 million combined sales in the UK .",
  "page": "Westlife",
  "revision_type": "real"
}
```
*Paired record with the same `case_id` after factual update:*
```json
{
  "unique_id": "5ed4de07c9e77c000848a180_2",
  "case_id": "5ed4de07c9e77c000848a180",
  "wiki_revision_id": "927477259",
  "label": "REFUTES",
  "claim": "Westlife made under 23.5 million sales in the UK .",
  "evidence": "According to the British Phonographic Industry ( BPI ) , Westlife has been certified for 13 million albums , 1.3 million video albums , and 9.8 million singles , with a total of more than 24 million combined sales in the UK .",
  "page": "Westlife",
  "revision_type": "real"
}
```

### Coverage of Parallax Elements
- **Answer Claims**: Standalone, well-formed claims.
- **Retrieved Evidence**: Sentence-level evidence.
- **Timestamps**: **Absent** inside the files (only `wiki_revision_id` is provided).
- **Document Versions**: Identified via `wiki_revision_id`.
- **Validity Periods**: None.
- **Supersession Links**: Contrastive revisions are grouped by `case_id`, but directional order (which is older vs. newer) is not explicitly dated.

---

## Dataset 3: `D3-RAGTruth`

### File & Path Information
- **Exact Path**: [`d:/Engineering/UROP/Projects/RAG/Datasets/D3-RAGTruth`](file:///d:/Engineering/UROP/Projects/RAG/Datasets/D3-RAGTruth)
- **Primary Files**:
  - `response.jsonl`: 21,458,735 bytes (~20.46 MB)
  - `source_info.jsonl`: 15,117,971 bytes (~14.42 MB)
- **File Type**: JSON Lines
- **File Count**: 2 files

### Dataset Scale & Splits
- **Source Context Instances**: 2,965 instances
  - Question Answering (MS MARCO): 989 instances
  - Data-to-Text Writing (Yelp): 1,033 instances
  - News Summarization (CNN/DailyMail & Recent News): 943 instances
- **LLM Generated Responses**: 17,790 total responses
  - 6 distinct models evaluated on all 2,965 instances: `gpt-4-0613`, `gpt-3.5-turbo-0613`, `mistral-7B-instruct`, `llama-2-7b-chat`, `llama-2-13b-chat`, `llama-2-70b-chat` (2,965 responses each).
- **Splits**: `train` (15,090 responses) and `test` (2,700 responses).
- **Hallucinated Responses**: 7,664 responses (43.1%) contain one or more annotated hallucination spans.

### Schema & Fields
#### 1. `source_info.jsonl`
| Field Name | Data Type | Description |
|---|---|---|
| `source_id` | `string` | Unique identifier linking to `response.jsonl`. |
| `task_type` | `string` | Task category: `"QA"`, `"Data2txt"`, or `"Summary"`. |
| `source` | `string` | Origin dataset (`"CNN/DM"`, `"Recent News"`, `"Yelp"`, `"MARCO"`). |
| `source_info` | `string` | Complete retrieved context passages provided to the LLM. |
| `prompt` | `string` | The instruction prompt fed into the LLMs. |

#### 2. `response.jsonl`
| Field Name | Data Type | Description |
|---|---|---|
| `id` | `string` | Response record identifier. |
| `source_id` | `string` | Foreign key matching `source_info.jsonl`. |
| `model` | `string` | LLM model that generated the text. |
| `temperature` | `float` | Sampling temperature (default 0.7). |
| `split` | `string` | `"train"` or `"test"`. |
| `quality` | `string` | `"good"` or flag. |
| `response` | `string` | Complete text generated by the LLM. |
| `labels` | `list<dict>` | Word/span-level hallucination annotations. |

#### Span Annotation Schema (`labels`)
- `start` (`int`): Starting character offset in `response`.
- `end` (`int`): Ending character offset in `response`.
- `text` (`string`): Exact hallucinated substring.
- `label_type` (`string`): One of four taxonomy categories:
  - `Evident Conflict`: 5,324 spans
  - `Evident Baseless Info`: 6,237 spans
  - `Subtle Baseless Info`: 2,527 spans
  - `Subtle Conflict`: 201 spans
- `meta` (`string`): Rationalization detailing the contradiction or lack of support.
- `implicit_true` (`bool`): Information not supported by reference text but true in reality.
- `due_to_null` (`bool`): Hallucination triggered by JSON null values in context.

### Representative Record
```json
{
  "id": "2",
  "source_id": "15592",
  "model": "mistral-7B-instruct",
  "temperature": 0.7,
  "split": "train",
  "quality": "good",
  "response": "New research conducted by the Anne Frank House has revealed that Anne Frank and her sister Margot likely died in the Bergen-Belsen concentration camp at least a month earlier than previously believed... believed to have died before February 7, 2022...",
  "labels": [
    {
      "start": 636,
      "end": 653,
      "text": "February 7, 2022.",
      "label_type": "Evident Conflict",
      "meta": "EVIDENT CONFLICT\nOriginal: February 7 (1945)\nGenerated: February 7, 2022",
      "implicit_true": false,
      "due_to_null": false
    }
  ]
}
```

### Coverage of Parallax Elements
- **Answer Claims**: Multi-sentence RAG responses containing natural hallucinations.
- **Retrieved Evidence**: Multi-passage context in `source_info`.
- **Timestamps**: **None**.
- **Document Versions**: None.
- **Validity Periods**: None.
- **Supersession Links**: None.
