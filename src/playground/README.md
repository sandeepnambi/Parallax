# Parallax Interactive Testing Playground

A lightweight, interactive web application for inspecting and comparing how **Conventional Static RAG Verification** and **Parallax Dynamic Supersession Arbitration** handle versioned evidence.

---

## Quickstart

### 1. Start the Playground Server
From the repository root directory, run:

```bash
uvicorn src.playground.app:app --host 127.0.0.1 --port 8000 --reload
```

Or run directly with Python:

```bash
python -m src.playground.app
```

### 2. Open in Your Browser
Navigate to:
```
http://127.0.0.1:8000
```

---

## Features

1. **Preloaded Canonical Cases**:
   - **Case 1: Outdated Evidence**: Demonstrates the core failure mode of static RAG verifiers (falsely certifying superseded evidence) vs. Parallax dynamic supersession detection (`OUTDATED`).
   - **Case 2: Current Evidence**: Demonstrates agreement when the claim matches the latest active snapshot.
   - **Case 3: Contradicted Claim**: Demonstrates direct contradiction detection.
   - **Case 4: Unsupported Claim**: Demonstrates orthogonal distractor rejection.
   - **Case 5: Missing Newer Revision**: Demonstrates the retrieval limit (when retrieval misses the newer revision, staleness cannot be detected without external metadata).

2. **Dynamic Evidence Editor**:
   - Add, edit, and remove retrieved passages.
   - Customize Document IDs, Snapshot Dates (`YYYY-MM-DD`), and evidence text.
   - Real-time input validation for dates and missing fields.

3. **Multi-Model Support**:
   - **Heuristic Verifiers**: Instant sub-millisecond execution on CPU.
   - **Zero-Shot Neural Verifiers (Optional)**: Check the *"Include Zero-Shot Neural (DistilRoBERTa)"* toggle to run cross-encoder neural inference locally.

4. **Detailed Explanations & Tracing**:
   - Stage A: Semantic relation breakdown per retrieved passage (`ENTAILMENT`, `CONTRADICTION`, `NEUTRAL`).
   - Stage B: Temporal reasoning trace displaying superseded vs. superseding snapshots and date comparisons.
   - Inference latency displayed in milliseconds.
