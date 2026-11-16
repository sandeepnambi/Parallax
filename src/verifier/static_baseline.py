"""
Parallax Static Baseline Verifier
Implements standard static claim verification (linguistic entailment) without
temporal awareness. Maps Entailment -> SUPPORTED_CURRENT, Refutation -> CONTRADICTED,
and Neutral -> UNSUPPORTED.

Demonstrates the core failure mode: when superseded evidence entails an outdated claim,
static detectors falsely classify it as SUPPORTED_CURRENT.
"""

import re
from typing import Dict, Any, List, Optional


class HeuristicNLI:
    """
    High-precision linguistic alignment & entity matching engine.
    Detects entailment, contradiction, and neutral relationships
    between claim and evidence text.
    """
    
    @staticmethod
    def clean_tokens(text: str) -> List[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        stopwords = {
            "the", "a", "an", "is", "was", "are", "were", "in", "at", "on", "for",
            "to", "of", "and", "or", "that", "this", "it", "with", "as", "by", "from"
        }
        return [w for w in words if w not in stopwords]
    
    def predict(
        self,
        claim: str = "",
        evidence: str = "",
        answer_entity: str = "",
        premise: Optional[str] = None,
        hypothesis: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        actual_claim = claim if claim else (hypothesis or "")
        actual_evidence = evidence if evidence else (premise or "")
        claim_clean = actual_claim.lower()
        ev_clean = actual_evidence.lower()
        ans_clean = answer_entity.lower().strip()
        
        # 1. Check overlap of core topic / subject tokens
        claim_tokens = set(self.clean_tokens(actual_claim))
        ev_tokens = set(self.clean_tokens(actual_evidence))
        overlap = claim_tokens.intersection(ev_tokens)
        
        # If very low token overlap (< 20% of content tokens), it is an unrelated distractor
        if len(claim_tokens) == 0 or (len(overlap) / len(claim_tokens) < 0.20):
            return "NEUTRAL"
            
        # 2. Check explicit answer entity presence if supplied
        if ans_clean:
            if ans_clean in ev_clean:
                return "ENTAILMENT"
            elif len(overlap) >= 3:
                return "CONTRADICTION"
            else:
                return "NEUTRAL"
                
        # 3. Arbitrary claim evaluation against evidence
        # Check numbers and dates
        claim_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", claim_clean))
        ev_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", ev_clean))
        
        # If claim asserts numbers/dates that conflict with evidence numbers
        if claim_nums and ev_nums and not claim_nums.issubset(ev_nums):
            # If overlapping topic words exist, it is a direct numerical contradiction
            if len(overlap) >= 2:
                return "CONTRADICTION"
            return "NEUTRAL"
            
        # Check non-overlapping content words
        unsupported_tokens = claim_tokens - ev_tokens
        unsupported_ratio = len(unsupported_tokens) / len(claim_tokens)
        
        if unsupported_ratio > 0.35:
            # Too much new ungrounded information introduced
            return "NEUTRAL"
            
        return "ENTAILMENT"


class StaticBaselineVerifier:
    """
    Evaluates claims against evidence using standard static NLI.
    Blind to timestamps and version validity windows.
    """
    def __init__(self, nli_model: Optional[Any] = None):
        self.nli = nli_model or HeuristicNLI()
        
    def verify_example(self, row: Dict[str, Any]) -> str:
        claim = row["answer_claim"]
        evidence = row["evidence_text"]
        ans = row["answer"]
        
        rel = self.nli.predict(claim, evidence, ans)
        
        # Static baseline taxonomy mapping
        if rel == "ENTAILMENT":
            return "SUPPORTED_CURRENT"
        elif rel == "CONTRADICTION":
            return "CONTRADICTED"
        else:
            return "UNSUPPORTED"
            
    def predict_batch(self, df: Any) -> List[str]:
        preds = []
        for _, row in df.iterrows():
            preds.append(self.verify_example(row.to_dict()))
        return preds
