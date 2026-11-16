"""
Parallax Temporal Arbitration Verifier
Implements the two-stage verification architecture:
Stage 1: Linguistic Entailment / Fact Verification (Claim vs. Evidence text)
Stage 2: Temporal Arbitration (Valid interval and Supersession vs. Reference Time)
"""

from typing import Dict, Any, List, Optional
try:
    from verifier.static_baseline import HeuristicNLI
except ImportError:
    try:
        from src.verifier.static_baseline import HeuristicNLI
    except ImportError:
        from .static_baseline import HeuristicNLI


class ParallaxTemporalVerifier:
    """
    Two-stage verifier that audits answer claims against both their retrieved evidence
    and the temporal validity interval of that evidence relative to reference_time.
    """
    def __init__(self, nli_model: Optional[Any] = None):
        self.nli = nli_model or HeuristicNLI()
        
    def verify_example(self, row: Dict[str, Any]) -> str:
        claim = row["answer_claim"]
        evidence = row["evidence_text"]
        ans = row["answer"]
        
        # Stage 1: Linguistic Entailment Check
        linguistic_rel = self.nli.predict(claim, evidence, ans)
        
        # Stage 2: Temporal Validity Arbitration
        ref_time = str(row.get("reference_time", "2024-12-01"))
        valid_to = str(row.get("valid_to", "present"))
        superseded_by = row.get("superseded_by")
        
        if linguistic_rel == "ENTAILMENT":
            # Check if evidence was superseded or expired prior to reference_time
            is_superseded = False
            
            if valid_to != "present" and valid_to <= ref_time:
                is_superseded = True
            elif superseded_by is not None and str(superseded_by) <= ref_time:
                is_superseded = True
                
            if is_superseded:
                return "OUTDATED"
            else:
                return "SUPPORTED_CURRENT"
                
        elif linguistic_rel == "CONTRADICTION":
            return "CONTRADICTED"
            
        else: # NEUTRAL / UNRELATED
            return "UNSUPPORTED"
            
    def predict_batch(self, df: Any) -> List[str]:
        preds = []
        for _, row in df.iterrows():
            preds.append(self.verify_example(row.to_dict()))
        return preds
