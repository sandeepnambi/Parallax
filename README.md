# Parallax: Claim-Level Superseded-Evidence Detection in RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Tests: 23 passed](https://img.shields.io/badge/tests-23%20passed-success.svg)](tests/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Parallax** investigates and solves a critical temporal blind spot in Retrieval-Augmented Generation (RAG): **an answer claim can be strictly entailed by retrieved context passages and still be factually incorrect because that evidence has since been superseded.**

---

## Key Highlights

- **2,800-Example Scaled Benchmark**: Cleaned and balanced across 4 classes (`SUPPORTED_CURRENT`, `OUTDATED`, `CONTRADICTED`, `UNSUPPORTED`) derived from temporal Wikipedia revisions, with 0% document overlap between train, dev, and test splits.
- **End-to-End Multi-Claim Auditor**: Combines syntactic claim decomposition, sentence-level context matching, cross-encoder natural language inference, temporal phrasing resolution, and provenance graph arbitration.
- **Elimination of the Static Blind Spot**: Existing static RAG verifiers suffer a **51.5% to 92.5% False-Support Rate (Severe Failure)** because they confuse semantic entailment with temporal freshness. Parallax completely eliminates false support (**0.00% FSR**) while delivering **100.00% precision on active information**.
- **Interactive CLI & HTML Audits**: Inspect datasets, run live demos, and audit RAG answers with rich interactive HTML reports and structured JSON exports.

---

## Taxonomy of Claims

| Label | Definition | Verification Condition |
| :--- | :--- | :--- |
| `SUPPORTED_CURRENT` | Claim is entailed by retrieved evidence that is **currently active** at reference time $t_{\text{ref}}$. | `Entails(E, C)` ∧ (`valid_to` > `t_ref` ∨ `present`) |
| `OUTDATED` | Claim is entailed by retrieved evidence, but that evidence was **superseded or expired** prior to $t_{\text{ref}}$. | `Entails(E, C)` ∧ (`valid_to` ≤ `t_ref` ∨ `superseded_by` ≤ `t_ref`) |
| `CONTRADICTED` | Claim directly conflicts with or is refuted by retrieved evidence. | `Refutes(E, C)` |
| `UNSUPPORTED` | Retrieved passages contain insufficient facts to substantiate the claim. | `Neutral(E, C)` |

---

## Repository Structure

```text
Parallax/
├── cli.py                             # Interactive CLI: demo, info, and end-to-end auditing
├── requirements.txt                   # Standardized dependencies (PyTorch, Transformers, Pytest, Ruff)
├── setup.py                           # Editable package setup
├── data/
│   ├── benchmark/
│   │   ├── train.parquet / .csv       # 1,600 training records (400 per class)
│   │   ├── dev.parquet / .csv         #   400 validation records (100 per class)
│   │   ├── test.parquet / .csv        #   800 official test records (200 per class)
│   │   └── full_benchmark.parquet     # 2,800 complete balanced benchmark
│   └── review/
│       └── sample_audit_100.csv       # 100-sample human inspection audit
├── src/
│   ├── benchmark_cleaner_scaler.py    # Raw data cleaning, verbalization & split scaling
│   ├── dataset_builder.py             # Benchmark builder and temporal metadata extractor
│   ├── verbalizer.py                  # Syntactic QA-to-declarative claim converter
│   ├── pipeline/
│   │   ├── decomposer.py              # Sentence splitting, casing preservation & temporal cues
│   │   ├── matcher.py                 # Lexical + salient entity overlap ranking
│   │   ├── temporal_resolver.py       # Present-tense vs historical claim phrasing resolver
│   │   ├── supersession_graph.py      # Fact revision lineage DAG & successor tracking
│   │   └── auditor.py                 # End-to-End ParallaxAuditor & NeuralCrossEncoderNLI
│   ├── verifier/
│   │   ├── static_baseline.py         # Static NLI verifiers (Heuristic + Cross-Encoder)
│   │   ├── temporal_arbiter.py        # Two-stage temporal validity arbiter
│   │   └── train_cross_encoder.py     # Cross-encoder fine-tuning with dynamic batching & loss balancing
│   └── evaluation/
│       └── metrics.py                 # 4-class F1, Accuracy, FSR_outdated & LaTeX table exporter
├── tests/                             # Automated test suite (23 unit tests)
│   ├── test_decomposer.py             # Sentence splitting, filtering, and cue extraction
│   ├── test_matcher.py                # Lexical and entity-based context alignment
│   ├── test_temporal_resolver.py      # Historical vs present anchor classification
│   ├── test_supersession_graph.py     # DAG revision chains and successor lookup
│   ├── test_verbalizer.py             # Question-answer to declarative claim conversion
│   ├── test_verifier.py               # Static vs temporal verification logic
│   └── test_metrics.py                # Evaluation metric computations
└── experiments/
    ├── evaluate_full_benchmark.py     # Official 800-example Phase A test benchmark runner
    ├── analyze_temporal_sensitivity.py# Phase B staleness elapsed & domain breakdown
    ├── benchmark_rag_auditor.py       # Phase C 100-scenario multi-claim paragraph audit
    ├── benchmark_vitaminc.py          # Contrastive backbone evaluation on VitaminC
    ├── test_e2e_auditor.py            # Diagnostic script testing 4 RAG scenarios
    └── benchmark_test_results.json    # Saved experimental results
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sandeepnambi/Parallax.git
   cd Parallax
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Parallax in editable mode**:
   ```bash
   pip install -e .
   ```

4. **Run Automated Unit Tests**:
   Verify the full test suite (23 tests covering decomposer, matcher, verifier, graph, and resolver):
   ```bash
   python -m pytest tests/
   ```

5. **Linting Check**:
   ```bash
   ruff check .
   ```

---

## Command-Line Interface (CLI)

Parallax includes a full-featured CLI (`cli.py`) for inspecting benchmark splits, running live demonstrations, and auditing custom queries.

### 1. View Benchmark & Environment Info
```bash
python cli.py info
```
*Displays dataset record counts per class, split distributions, device info, and cached model checkpoints.*

### 2. Run Interactive Auditor Demo
```bash
python cli.py demo
```
*Runs an interactive walkthrough demonstrating how Parallax detects outdated facts, contradicted claims, and pristine evidence in RAG responses.*

### 3. Audit a RAG Response via CLI
```bash
python cli.py audit "Who is the Master of the Horse?" \
  "Based on retrieved context, the Master of the Horse is Lord de Mauley." \
  --ref-time 2024-12-01 \
  --output-html audit_report.html
```

---

## Reproducing All Experiments

### 1. Fine-Tune the Neural Cross-Encoder
Fine-tunes `cross-encoder/nli-distilroberta-base` on `data/benchmark/train.parquet` (1,600 examples) using dynamic sequence batching and balanced class weights:
```bash
python src/verifier/train_cross_encoder.py --epochs 3 --batch_size 16 --lr 2e-5
```
*Saves the highest validation checkpoint to `models/parallax_nli_finetuned` and outputs full evaluation report to `experiments/cross_encoder_finetune_results.json`.*

### 2. Official 800-Example Benchmark Evaluation (Phase A)
Evaluates all 6 systems (Heuristic, Zero-Shot Neural, and Fine-Tuned Neural under both Static and Parallax Temporal verification) on `data/benchmark/test.parquet`:
```bash
python experiments/evaluate_full_benchmark.py --batch_size 32
```
*Outputs comparative metrics and exports publication-ready LaTeX table to `experiments/table_test_results.tex`.*

### 3. Staleness Duration & Domain Sensitivity (Phase B)
Analyzes how False-Support Rate behaves across staleness age buckets (0–30 days up to 121–153 days) and 6 knowledge domains:
```bash
python experiments/analyze_temporal_sensitivity.py
```

### 4. Multi-Claim RAG Paragraph Auditing (Phase C)
Evaluates `ParallaxAuditor` on 100 multi-claim paragraphs (250+ claims) across Pristine, Stale, Hallucinated, and Mixed archetypes:
```bash
python experiments/benchmark_rag_auditor.py --n_per_archetype 25
```

### 5. Contrastive Backbone Benchmark on VitaminC
Evaluates cross-encoder contrastive sensitivity on paired factual revisions in VitaminC:
```bash
python experiments/benchmark_vitaminc.py --max_cases 500 --batch_size 32
```

### 6. Quick Diagnostic Scenario Test
Tests the 4 realistic RAG scenarios (Stale, Mixed, Contradicted, Pristine):
```bash
# Fast heuristic mode
python experiments/test_e2e_auditor.py --mode heuristic

# Deep neural cross-encoder mode
python experiments/test_e2e_auditor.py --mode neural
```

---

## Experimental Results & Benchmark Insights

### Official 800-Example Test Split Benchmark

The benchmark tests systems on held-out records (200 records per class) with 0% document overlap with training data.

| System Architecture | Overall Accuracy | Macro-F1 | False-Support Rate (FSR $\downarrow$) | Outdated F1 | Current F1 | Contradicted F1 | Unsupported F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Existing Static Baselines (No Temporal Awareness)** | | | | | | | |
| ✗ Static Heuristic Baseline | 68.12% | 0.596 | **92.50% (Severe Failure)** | 0.00% | 63.84% | 83.76% | 90.66% |
| ✗ Static Neural (Zero-Shot Cross-Encoder) | 40.62% | 0.354 | **51.50% (Severe Failure)** | 0.00% | 48.57% | 56.82% | 36.30% |
| ✗ Static Neural (Fine-Tuned Cross-Encoder) | 71.75% | 0.636 | **98.50% (The Static Paradox)** | 0.00% | 64.88% | 93.10% | 96.46% |
| **Parallax (Our Temporal Verifier Architecture)** | | | | | | | |
| ✓ Parallax Heuristic (Temporal) | 91.25% | 0.913 | **0.00% (Flawless)** | 95.85% | 95.01% | 83.76% | 90.66% |
| ✓ Parallax Neural Verifier (Zero-Shot) | 53.50% | 0.564 | **0.00% (Flawless)** | 64.78% | 67.55% | 56.82% | 36.30% |
| ✓ **Parallax Neural Verifier (Fine-Tuned, Ours)** | **96.38%** | **0.9639** | **0.00% (Flawless)** | **97.77%** | **98.23%** | **93.10%** | **96.46%** |

> [!IMPORTANT]
> **The Static Entailment Paradox**:
> Notice that when the neural cross-encoder is fine-tuned to high linguistic accuracy, the **static baseline's False-Support Rate jumps from $51.50\% \rightarrow 98.50\%$**!
> Why? Because an accurate NLI model correctly recognizes that obsolete context strictly entails the claim (197/200 recall). Without temporal arbitration, a static RAG verifier blindly accepts all 98.5% of these outdated claims as true.
> **Parallax solves this paradox**: pairing the exact same fine-tuned cross-encoder with Stage 2 Temporal Arbitration slashes false support to **0.00%**, boosting overall accuracy to **96.38%** and Macro-F1 to **0.9639**.

---

### Staleness Duration Sensitivity (Phase B)

As retrieved evidence grows older, static baselines suffer even greater failure rates. In contrast, Parallax retains zero false support regardless of evidence age:

| Staleness Duration | Number of Samples | Static Baseline FSR $\downarrow$ | Parallax FSR $\downarrow$ |
| :--- | :---: | :---: | :---: |
| 0 – 30 Days | 70 | 87.14% *(Severe)* | **0.00%** |
| 31 – 60 Days | 38 | 94.74% *(Severe)* | **0.00%** |
| 61 – 90 Days | 23 | 95.65% *(Severe)* | **0.00%** |
| 91 – 120 Days | 6 | 100.00% *(Complete Failure)* | **0.00%** |
| 121 – 153 Days | 63 | 98.41% *(Severe)* | **0.00%** |

---

### Multi-Claim RAG Paragraph Auditing (Phase C, 100 Scenarios)

When auditing full multi-claim RAG paragraphs (Pristine, Stale, Hallucinated, and Mixed archetypes):

| RAG Auditor Configuration | Overall Paragraph Accuracy | False Trust Rate (FTR $\downarrow$) | Pristine Trust Score | Stale Trust Score | Mixed Trust Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Static Baseline RAG Auditor** | 38.00% | 16.00% *(Dangerous)* | 74.0% | **72.0%** *(Overconfident)* | **48.0%** *(Overconfident)* |
| **Parallax Heuristic Auditor** | 64.00% | **2.67% (Safe)** | 74.0% | **40.0%** *(Penalized)* | **25.3%** *(Penalized)* |
| **Parallax Neural Auditor (Fine-Tuned, Ours)** | **90.00%** | **4.00% (Safe)** | **96.0%** | **54.0%** *(Penalized)* | **34.6%** *(Penalized)* |

---

## Core Pipeline Architecture

Parallax implements a modular 6-stage pipeline:

1. **`ClaimDecomposer`** (`src/pipeline/decomposer.py`):
   Segments generated RAG answers into standalone atomic claims while stripping preamble boilerplate and preserving casing and capitalization. Also extracts temporal anchor cues (e.g., `"in 2021"`, `"currently"`, `"formerly"`).
2. **`SentenceMatcher`** (`src/pipeline/matcher.py`):
   Aligns each claim to candidate sentences in the retrieved context using lexical and salient entity overlap ranking.
3. **`TemporalAnchorResolver`** (`src/pipeline/temporal_resolver.py`):
   Disambiguates claim temporal phrasing. Claims explicitly framed as historical facts (e.g., *"X was the CEO in 2021"*) are anchored historically and protected from being falsely labeled as `OUTDATED`.
4. **`SupersessionGraph`** (`src/pipeline/supersession_graph.py`):
   Tracks entity-level fact revision lineages in a directed acyclic graph (DAG), tracking active successor facts when an earlier version is superseded.
5. **`TemporalArbiter`** (`src/verifier/temporal_arbiter.py`):
   Performs two-stage arbitration mapping NLI entailment states and temporal validity intervals (`valid_from`, `valid_to`, `superseded_by`) against reference time $t_{\text{ref}}$.
6. **`ParallaxAuditor`** (`src/pipeline/auditor.py`):
   Integrates all stages to generate paragraph-level trust scores, claim-by-claim diagnostic traces, structured JSON exports, and interactive HTML audit dashboards.

---

## Python API Usage

Audit multi-sentence RAG responses with temporal provenance in Python:

```python
from pipeline.auditor import ParallaxAuditor

# Initialize auditor (use_neural_nli=True for Cross-Encoder, False for fast heuristic)
auditor = ParallaxAuditor(use_neural_nli=True, default_ref_time="2024-12-01")

query = "Who is the Master of the Horse?"
answer = "Based on retrieved context, the Master of the Horse is Lord de Mauley."

retrieved_docs = [
    {
        "doc_id": "100025",
        "doc_title": "Master of the Horse",
        "text": "The Master of the Horse is an office in the Royal Household. The current Master of the Horse is Lord de Mauley.",
        "valid_from": "2024-06-01",
        "valid_to": "2024-07-01",        # Superseded before reference time!
        "superseded_by": "2024-07-01",
    }
]

# Run audit
audit_report = auditor.audit_response(query, answer, retrieved_docs)
print(auditor.format_report(audit_report))

# Generate interactive HTML report
auditor.format_html_report(audit_report, output_path="audit_report.html")
```

### Sample Audit Output:
```text
============================================================
PARALLAX END-TO-END RAG AUDIT REPORT
============================================================
Query:          Who is the Master of the Horse?
Reference Time: 2024-12-01
Overall Status: DEGRADED_BY_STALE_EVIDENCE
Trust Score:    0.0% (0/1 claims verified current)
------------------------------------------------------------
[Claim 1] [OUTDATED]
  Claim:       "The Master of the Horse is Lord de Mauley."
  Evidence:    "The current Master of the Horse is Lord de Mauley."
  Source:      Master of the Horse (id: 100025)
  Alignment:   Match Score = 1.000 | SUPERSEDED on 2024-07-01 (expired before ref_time 2024-12-01)
  Diagnosis:   Evidence strictly entails claim, but that evidence was superseded and is obsolete.
============================================================
```

---

## Citation & Contact

If you use Parallax in your research, please cite:

```bibtex
@article{parallax2026,
  title={Parallax: A Benchmark and Verifier for Claim-Level Superseded-Evidence Detection in Retrieval-Augmented Generation},
  author={Parallax Research Team},
  year={2026}
}
```
