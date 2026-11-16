# Parallax: Claim-Level Superseded-Evidence Detection in RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Parallax** investigates and solves a critical temporal blind spot in Retrieval-Augmented Generation (RAG): **an answer claim can be strictly entailed by retrieved context passages and still be factually incorrect because that evidence has since been superseded.**

---

## Key Highlights

- **2,800-Example Scaled Benchmark**: Balanced across 4 classes (`SUPPORTED_CURRENT`, `OUTDATED`, `CONTRADICTED`, `UNSUPPORTED`) with 0% document overlap between splits.
- **End-to-End Multi-Claim Auditor**: Combines claim decomposition, sentence-level context matching, cross-encoder natural language inference, and temporal provenance arbitration.
- **Elimination of False Support**: Static verifiers falsely certify **51.5% to 92.5%** of outdated claims as current. Parallax completely eliminates false support (**0.00% FSR**) with **100.00% precision on active information**.

---

## Taxonomy of Claims

| Label | Definition | Verification Condition |
| :--- | :--- | :--- |
| `SUPPORTED_CURRENT` | Claim is entailed by retrieved evidence that is **currently active** at reference time $t_{\text{ref}}$. | $\text{Entails}(E, C) \land (\text{valid\_to} > t_{\text{ref}} \lor \text{present})$ |
| `OUTDATED` | Claim is entailed by retrieved evidence, but that evidence was **superseded or expired** prior to $t_{\text{ref}}$. | $\text{Entails}(E, C) \land (\text{valid\_to} \le t_{\text{ref}} \lor \text{superseded\_by} \le t_{\text{ref}})$ |
| `CONTRADICTED` | Claim directly conflicts with or is refuted by retrieved evidence. | $\text{Refutes}(E, C)$ |
| `UNSUPPORTED` | Retrieved passages contain insufficient facts to substantiate the claim. | $\text{Neutral}(E, C)$ |

---

## Repository Structure

```text
Parallax/
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
│   ├── verbalizer.py                  # Syntactic QA-to-declarative claim converter
│   ├── pipeline/
│   │   ├── decomposer.py              # Sentence splitting & boilerplate stripping
│   │   ├── matcher.py                 # Lexical + salient entity sentence matching
│   │   └── auditor.py                 # End-to-End ParallaxAuditor & NeuralCrossEncoderNLI
│   ├── verifier/
│   │   ├── static_baseline.py         # Static NLI verifier (Heuristic + Cross-Encoder)
│   │   └── temporal_arbiter.py        # Two-stage temporal validity arbiter
│   └── evaluation/
│       └── metrics.py                 # 4-class F1, Accuracy, FSR_outdated & LaTeX table exporter
├── experiments/
│   ├── evaluate_full_benchmark.py     # Official 800-example Phase A test benchmark runner
│   ├── analyze_temporal_sensitivity.py# Phase B staleness elapsed & domain breakdown
│   ├── benchmark_rag_auditor.py       # Phase C 100-scenario multi-claim paragraph audit
│   ├── benchmark_vitaminc.py          # Step 3 contrastive backbone evaluation
│   ├── test_e2e_auditor.py            # Diagnostic script testing 4 RAG scenarios
│   └── benchmark_test_results.json    # Official evaluation results
└── setup.py                           # Editable package setup
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Parallax.git
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

4. **Verify installation**:
   ```bash
   ruff check .
   ```

---

## Reproducing All Experiments

### 1. Official 800-Example Benchmark Evaluation (Phase A)
Evaluates Static Heuristic, Static Neural, Parallax Heuristic, and Parallax Neural on `data/benchmark/test.parquet`:
```bash
python experiments/evaluate_full_benchmark.py --batch_size 32
```
*Outputs comparative metrics and exports publication-ready LaTeX table to `experiments/table_test_results.tex`.*

### 2. Temporal Sensitivity & Domain Breakdown (Phase B)
Analyzes how False-Support Rate scales across staleness duration ($0\text{--}30\text{d}$ to $120\text{--}153\text{d}$) and 6 knowledge domains:
```bash
python experiments/analyze_temporal_sensitivity.py
```

### 3. Multi-Claim RAG Paragraph Auditing (Phase C)
Evaluates `ParallaxAuditor` on 100 multi-claim paragraphs (250+ claims) across Pristine, Stale, Hallucinated, and Mixed archetypes:
```bash
python experiments/benchmark_rag_auditor.py --n_per_archetype 25
```

### 4. Contrastive Backbone Benchmark on VitaminC (Step 3)
Evaluates cross-encoder contrastive sensitivity on paired factual edits in VitaminC:
```bash
python experiments/benchmark_vitaminc.py --max_cases 500 --batch_size 32
```

### 5. Quick Diagnostic Scenario Test
Tests the 4 realistic RAG scenarios (Stale, Mixed, Contradicted, Pristine):
```bash
# Fast heuristic mode
python experiments/test_e2e_auditor.py --mode heuristic

# Deep neural cross-encoder mode
python experiments/test_e2e_auditor.py --mode neural
```

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

## Experimental Results Summary

### Official 800-Example Test Split Benchmark

| System | Overall Accuracy | Macro-F1 | **$\mathbf{FSR}_{\text{OUTD}} \downarrow$**<br>*(False-Support Rate)* | Supported-Current F1 | Outdated F1 | Contradicted F1 | Unsupported F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static Heuristic Baseline** | 68.12% | 0.596 | **92.50%** | 63.84% | 0.00% | 83.76% | 90.66% |
| **Parallax Heuristic** (Temporal) | **91.25%** | **0.913** | **0.00%** | **95.01%** | **95.85%** | **83.76%** | **90.66%** |
| **Static Neural Baseline** (Cross-Encoder) | 40.62% | 0.354 | **51.50%** | 48.57% | 0.00% | 56.82% | 36.30% |
| **Parallax Neural Verifier (Ours)** | **53.50%** | **0.564** | **0.00%** | **67.55%** | **64.78%** | **56.82%** | **36.30%** |

### Multi-Claim RAG Paragraph Auditing (100 Scenarios)

| RAG Auditor Configuration | Overall Paragraph Accuracy | **False Trust Rate (FTR)** | Pristine Response Trust Score | **Stale Response Trust Score** | **Mixed Response Trust Score** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Static Baseline RAG Auditor** | 40.00% | **17.33%** | 80.0% | **72.0%** *(Overconfident)* | **49.3%** *(Overconfident)* |
| **Parallax Heuristic Auditor** | **70.00%** | **0.00%** | 80.0% | **36.0%** *(Penalized)* | **22.6%** *(Penalized)* |
| **Parallax Neural Auditor (Ours)** | **65.00%** | **0.00%** | 80.0% | **38.0%** *(Penalized)* | **25.3%** *(Penalized)* |

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
