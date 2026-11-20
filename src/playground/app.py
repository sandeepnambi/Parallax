"""
Parallax Interactive Testing Playground Backend
================================================
FastAPI application providing endpoints to test and compare:
1. Conventional Static Heuristic Verification
2. Parallax Dynamic Heuristic Verification
3. Conventional Static Zero-Shot Neural Verification (Optional)
4. Parallax Dynamic Zero-Shot Neural Verification (Optional)
"""

import os
import sys
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Ensure repo root and src are on sys.path
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_cur_dir, "..", ".."))
for _p in [_repo_root, os.path.join(_repo_root, "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.verifier.static_baseline import HeuristicNLI
from src.verifier.static_context_verifier import StaticContextVerifier
from src.verifier.dynamic_arbiter import DynamicContextVerifier

app = FastAPI(
    title="Parallax Interactive Testing Playground",
    description="Interactive testbed comparing conventional static verification with Parallax dynamic verification.",
    version="2.0.0",
)

# Global verifier singletons (heuristic instant initialization)
_heuristic_nli = HeuristicNLI()
_static_heuristic_verifier = StaticContextVerifier(nli_model=_heuristic_nli)
_dynamic_heuristic_verifier = DynamicContextVerifier(nli_model=_heuristic_nli)

# Lazy-loaded neural verifiers
_neural_nli = None
_static_neural_verifier = None
_dynamic_neural_verifier = None


def get_neural_verifiers():
    """Lazy loads zero-shot neural models on first request."""
    global _neural_nli, _static_neural_verifier, _dynamic_neural_verifier
    if _neural_nli is None:
        from src.pipeline.auditor import NeuralCrossEncoderNLI
        _neural_nli = NeuralCrossEncoderNLI(model_name="cross-encoder/nli-distilroberta-base")
        _static_neural_verifier = StaticContextVerifier(nli_model=_neural_nli)
        _dynamic_neural_verifier = DynamicContextVerifier(nli_model=_neural_nli)
    return _static_neural_verifier, _dynamic_neural_verifier


class PassageItem(BaseModel):
    passage_id: Optional[str] = None
    doc_id: str = Field(..., description="Document identifier, e.g., '74897662'")
    snapshot_date: str = Field(..., description="Snapshot/publication date in YYYY-MM-DD format")
    text: str = Field(..., description="Passage evidence text")


class VerifyRequest(BaseModel):
    claim: str = Field(..., description="Claim statement to be audited")
    retrieved_passages: List[PassageItem] = Field(default_factory=list, description="List of evidence passages")
    target_doc_id: Optional[str] = None
    include_neural: bool = False


# Canonical preloaded example cases
PRELOADED_EXAMPLES = [
    {
        "id": "outdated_evidence",
        "title": "1. Outdated Evidence (The Core Problem)",
        "description": "The claim asserts an older statistic (600k copies) which is supported by an older snapshot (2024-11-01), but superseded by a newer revision (900k on 2024-12-01). Conventional static verifiers falsely certify it as currently supported. Parallax correctly identifies it as OUTDATED.",
        "claim": "Over 600,000 copies of the series are in circulation.",
        "retrieved_passages": [
            {
                "passage_id": "p_0",
                "doc_id": "74897662",
                "snapshot_date": "2024-12-01",
                "text": "The series has over 900,000 copies in circulation.",
            },
            {
                "passage_id": "p_1",
                "doc_id": "74897662",
                "snapshot_date": "2024-11-01",
                "text": "The series has over 600,000 copies in circulation.",
            },
            {
                "passage_id": "p_2",
                "doc_id": "2579786",
                "snapshot_date": "2024-11-01",
                "text": "Other languages spoken at home included Mandarin (16.9%), Cantonese (13.9%), and Greek (5.2%).",
            },
        ],
    },
    {
        "id": "current_evidence",
        "title": "2. Current Evidence (Agreement)",
        "description": "The claim asserts the latest active statistic (900k copies). Both older (600k) and newer (900k) revisions are present in context. Both systems agree that the claim is SUPPORTED_CURRENT.",
        "claim": "Over 900,000 copies of the series are in circulation.",
        "retrieved_passages": [
            {
                "passage_id": "p_0",
                "doc_id": "74897662",
                "snapshot_date": "2024-12-01",
                "text": "The series has over 900,000 copies in circulation.",
            },
            {
                "passage_id": "p_1",
                "doc_id": "74897662",
                "snapshot_date": "2024-11-01",
                "text": "The series has over 600,000 copies in circulation.",
            },
            {
                "passage_id": "p_2",
                "doc_id": "2579786",
                "snapshot_date": "2024-11-01",
                "text": "Other languages spoken at home included Mandarin (16.9%), Cantonese (13.9%), and Greek (5.2%).",
            },
        ],
    },
    {
        "id": "contradicted_claim",
        "title": "3. Contradicted Claim",
        "description": "The claim asserts a fabricated number (1,500,000 copies) that conflicts with all snapshots in context. Both systems classify the claim as CONTRADICTED.",
        "claim": "Over 1,500,000 copies of the series are in circulation.",
        "retrieved_passages": [
            {
                "passage_id": "p_0",
                "doc_id": "74897662",
                "snapshot_date": "2024-12-01",
                "text": "The series has over 900,000 copies in circulation.",
            },
            {
                "passage_id": "p_1",
                "doc_id": "74897662",
                "snapshot_date": "2024-11-01",
                "text": "The series has over 600,000 copies in circulation.",
            },
            {
                "passage_id": "p_2",
                "doc_id": "2579786",
                "snapshot_date": "2024-11-01",
                "text": "Other languages spoken at home included Mandarin (16.9%), Cantonese (13.9%), and Greek (5.2%).",
            },
        ],
    },
    {
        "id": "unsupported_claim",
        "title": "4. Unsupported Claim (Distractors Only)",
        "description": "The claim introduces ungrounded information (Hugo Award Best Novel). The retrieved passages contain only orthogonal distractors from other topics. Both systems return UNSUPPORTED.",
        "claim": "The series was awarded the Hugo Award for Best Novel in 2024.",
        "retrieved_passages": [
            {
                "passage_id": "p_0",
                "doc_id": "73048713",
                "snapshot_date": "2024-07-01",
                "text": "Holland averaged 18.5 points, 6.7 rebounds, 2.8 assists, and 2.1 steals per game in 29 games played.",
            },
            {
                "passage_id": "p_1",
                "doc_id": "2579786",
                "snapshot_date": "2024-11-01",
                "text": "Other languages spoken at home included Mandarin (16.9%), Cantonese (13.9%), Arabic (6.0%), Greek (5.2%).",
            },
        ],
    },
    {
        "id": "missing_newer_revision",
        "title": "5. Missing Newer Revision (Diagnostic: Retrieval Limit)",
        "description": "Retrieval failure case: the claim is historically outdated (600k), but retrieval surfaces ONLY the older 2024-11-01 snapshot and fails to retrieve the 2024-12-01 revision. Without oracle metadata or the newer revision, staleness is undetectable. Both systems return SUPPORTED_CURRENT, empirically proving that multi-version retrieval is necessary.",
        "claim": "Over 600,000 copies of the series are in circulation.",
        "retrieved_passages": [
            {
                "passage_id": "p_0",
                "doc_id": "74897662",
                "snapshot_date": "2024-11-01",
                "text": "The series has over 600,000 copies in circulation.",
            },
            {
                "passage_id": "p_1",
                "doc_id": "2579786",
                "snapshot_date": "2024-11-01",
                "text": "Other languages spoken at home included Mandarin (16.9%), Cantonese (13.9%), and Greek (5.2%).",
            },
        ],
    },
]


def validate_date(date_str: str) -> None:
    """Validates date format is YYYY-MM-DD or YYYY-MM."""
    if not isinstance(date_str, str):
        raise HTTPException(status_code=422, detail="Snapshot date must be a string.")
    clean = date_str.strip()
    if not re.match(r"^\d{4}-\d{2}(-\d{2})?$", clean):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid snapshot date format: '{date_str}'. Expected 'YYYY-MM-DD' (e.g. '2024-11-01').",
        )
    # Validate actual calendar date
    try:
        if len(clean) == 7:
            datetime.strptime(clean, "%Y-%m")
        else:
            datetime.strptime(clean, "%Y-%m-%d")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid calendar date '{date_str}': {str(e)}")


@app.get("/api/examples")
def get_examples():
    """Returns preloaded canonical example cases."""
    return {"examples": PRELOADED_EXAMPLES}


@app.post("/api/verify")
def verify_claim(req: VerifyRequest):
    """
    Evaluates a claim against retrieved versioned passages.
    Runs both Static and Parallax Dynamic verifiers and returns comparative results.
    """
    claim = req.claim.strip()
    if not claim:
        raise HTTPException(status_code=422, detail="Claim cannot be empty.")

    # Validate passages
    formatted_passages = []
    for idx, p in enumerate(req.retrieved_passages):
        doc_id = p.doc_id.strip()
        text = p.text.strip()
        snapshot_date = p.snapshot_date.strip()

        if not doc_id:
            raise HTTPException(status_code=422, detail=f"Passage #{idx+1} is missing a Document ID.")
        if not text:
            raise HTTPException(status_code=422, detail=f"Passage #{idx+1} has empty text.")
        
        validate_date(snapshot_date)

        pid = p.passage_id or f"p_{idx}"
        formatted_passages.append({
            "passage_id": pid,
            "doc_id": doc_id,
            "snapshot_date": snapshot_date,
            "text": text,
        })

    # 1. Static Heuristic Baseline
    t0 = time.time()
    static_h_res = _static_heuristic_verifier.verify(claim, formatted_passages)
    static_h_time = (time.time() - t0) * 1000.0

    # 2. Dynamic Heuristic Parallax
    t0 = time.time()
    dynamic_h_res = _dynamic_heuristic_verifier.verify(
        claim, formatted_passages, target_doc_id=req.target_doc_id
    )
    dynamic_h_time = (time.time() - t0) * 1000.0

    results: Dict[str, Any] = {
        "static_heuristic": {
            "name": "Conventional Static (Heuristic)",
            "verdict": static_h_res["verdict"],
            "explanation": static_h_res["explanation"],
            "inference_time_ms": round(static_h_time, 2),
            "supporting_passages": static_h_res.get("supporting_passages", []),
            "stage_a_results": static_h_res.get("passage_results", []),
        },
        "dynamic_heuristic": {
            "name": "Parallax Dynamic (Heuristic)",
            "verdict": dynamic_h_res["verdict"],
            "explanation": dynamic_h_res["explanation"],
            "inference_time_ms": round(dynamic_h_time, 2),
            "supporting_passages": dynamic_h_res.get("supporting_passages", []),
            "superseding_passage": dynamic_h_res.get("superseding_passage"),
            "contradicting_passage": dynamic_h_res.get("contradicting_passage"),
            "stage_a_results": dynamic_h_res.get("stage_a_results", []),
        },
    }

    # Optional Neural Verifiers
    if req.include_neural:
        try:
            static_n_v, dynamic_n_v = get_neural_verifiers()

            t0 = time.time()
            static_n_res = static_n_v.verify(claim, formatted_passages)
            static_n_time = (time.time() - t0) * 1000.0

            t0 = time.time()
            dynamic_n_res = dynamic_n_v.verify(claim, formatted_passages, target_doc_id=req.target_doc_id)
            dynamic_n_time = (time.time() - t0) * 1000.0

            results["static_neural"] = {
                "name": "Conventional Static (Zero-Shot DistilRoBERTa)",
                "verdict": static_n_res["verdict"],
                "explanation": static_n_res["explanation"],
                "inference_time_ms": round(static_n_time, 2),
                "supporting_passages": static_n_res.get("supporting_passages", []),
                "stage_a_results": static_n_res.get("passage_results", []),
            }
            results["dynamic_neural"] = {
                "name": "Parallax Dynamic (Zero-Shot DistilRoBERTa)",
                "verdict": dynamic_n_res["verdict"],
                "explanation": dynamic_n_res["explanation"],
                "inference_time_ms": round(dynamic_n_time, 2),
                "supporting_passages": dynamic_n_res.get("supporting_passages", []),
                "superseding_passage": dynamic_n_res.get("superseding_passage"),
                "contradicting_passage": dynamic_n_res.get("contradicting_passage"),
                "stage_a_results": dynamic_n_res.get("stage_a_results", []),
            }
        except Exception as e:
            results["neural_error"] = str(e)

    return {
        "claim": claim,
        "passages_count": len(formatted_passages),
        "results": results,
    }


# Serve static web frontend
static_dir = os.path.join(_cur_dir, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
