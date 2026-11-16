# Parallax

**Parallax: A Benchmark and Verifier for Claim-Level Superseded-Evidence Detection in RAG**

Parallax investigates a temporal blind spot in retrieval-augmented generation (RAG): an answer claim can be supported by the retrieved text and still be wrong because that text has since been superseded.

## Project overview

Many RAG evaluation methods check whether an answer is supported by its retrieved context. That check can miss whether the evidence was valid at the time the answer is intended to describe. Parallax aims to evaluate answer claims against both their evidence and its temporal validity, then flag claims grounded in obsolete versions.

The project will develop:

1. A benchmark of claims paired with current and superseded evidence.
2. A claim-level verifier that determines support and temporal validity.
3. An evaluation of existing faithfulness detectors on outdated evidence, including their false-support rate.

## Research question

**Can a claim-level verifier distinguish claims supported by current evidence from claims supported only by superseded evidence in RAG answers?**

## Intended labels

| Label | Meaning |
|---|---|
| `SUPPORTED_CURRENT` | The evidence supports the claim and is valid for the relevant time. |
| `OUTDATED` | The evidence supports the claim, but that evidence has been superseded or is no longer valid for the relevant time. |
| `UNSUPPORTED` | The evidence does not establish the claim. |
| `CONTRADICTED` | The evidence conflicts with the claim. |

The annotation guide should define how to handle missing dates, overlapping versions, claims with multiple pieces of evidence, and claims that were true historically but not at the answer’s reference time.

## Planned approach

### 1. Dataset assessment

Inspect the three selected datasets, which are organized using names corresponding to their source papers. For each, record its source, license, fields, splits, labels, size, and availability of claims, answers, evidence, dates, version identifiers, and supersession links.

Do not assume a dataset contains temporal labels just because its source paper studies time-sensitive information. Confirm this from the files and documentation.

### 2. Benchmark construction

Select examples that contain versioned or time-sensitive evidence. Adapt them into claim–evidence records and annotate whether each claim is current, outdated, unsupported, or contradicted. Keep source and version provenance so each label can be audited.

A starting schema could be:

```text
example_id
question
answer
answer_claim
evidence_text
evidence_source
evidence_version
valid_from
valid_to
superseded_by
reference_time
gold_label
split
annotation_notes
```

Use only fields supported by the source data. Record missing information explicitly instead of inventing dates or version relationships.

### 3. Baselines and verifier

Evaluate established claim-level faithfulness or hallucination detectors on the same examples. Then test a verifier that combines claim decomposition, evidence matching, and temporal arbitration using version metadata.

A minimal verifier pipeline is:

1. Split the answer into atomic, independently verifiable claims.
2. Link each claim to relevant retrieved evidence.
3. Assess whether the evidence supports, contradicts, or fails to establish the claim.
4. Check the evidence version and validity interval against the reference time.
5. Return a label, confidence, and evidence/version trace for each claim.

### 4. Pilot evaluation

Start with a small, manually checked pilot before scaling the benchmark. Compare the verifier and baselines on the same held-out examples, with particular attention to claims labeled `OUTDATED`.

Recommended initial metrics:

- Precision, recall, and F1 for each label
- Macro-F1 across the four labels
- **False-support rate on `OUTDATED` claims**: the share of outdated claims incorrectly labeled as supported/current
- Confusion matrix, especially `OUTDATED` vs. `SUPPORTED_CURRENT`
- Annotation agreement on the human-reviewed subset
- Median and 95th-percentile latency per answer, once a working system exists

Use a held-out test split for final reporting and avoid placing near-duplicate claims or versions across train and test where that could inflate results.

## Scope and limitations

Parallax focuses on **claim-level auditing of evidence already associated with a RAG answer**. It is distinct from a general freshness-focused retrieval system: retrieval may supply candidate documents, but the central task is deciding whether the evidence supporting each answer claim remains valid.

Temporal correctness depends on reliable dates and version relationships. If a dataset lacks them, additional source collection or human annotation will be needed. A dataset designed for ordinary hallucination detection may still be useful as a baseline, but it cannot by itself establish that evidence is superseded.

## Repository status

This README describes the project plan. Dataset selection, benchmark schema, annotation guidelines, verifier implementation, and experiments should be updated here as they are confirmed. Paper-reported results and dataset statistics should be cited to their original sources in project documentation and publications.

## Getting started

1. Inventory the three dataset folders and their documentation.
2. Verify each dataset’s paper, license, fields, and version/date coverage.
3. Build a dataset comparison matrix and choose one temporal dataset plus one faithfulness baseline.
4. Define the annotation guide and resolve ambiguous cases on a small pilot sample.
5. Establish baseline results before implementing the temporal verifier.
6. Report label-wise metrics, false-support rate on `OUTDATED`, annotation agreement, and latency.

## References to investigate

The following sources are starting points, not a claim that they have already been selected or fully validated for the benchmark:

- [HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on RAG](https://arxiv.org/abs/2503.04800)
- [VitaminC: Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence](https://arxiv.org/abs/2103.08541)
- [RAGTruth: A Hallucination Corpus for Developing Trustworthy RAG Models](https://arxiv.org/abs/2401.00396)
- [RT4CHART: Retromorphic Testing with Hierarchical Verification for Hallucination Detection in RAG](https://arxiv.org/abs/2603.27752)
- [VersionRAG: Version-Aware Retrieval-Augmented Generation for Evolving Documents](https://arxiv.org/abs/2510.08109)
