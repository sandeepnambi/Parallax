"""
Parallax End-to-End Auditor
Full multi-claim RAG auditing pipeline:
Decomposition -> Evidence Sentence Matching -> Neural/Heuristic NLI -> Temporal Arbitration -> Audit Report
"""

from typing import List, Dict, Any, Optional, Union

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from pipeline.decomposer import ClaimDecomposer
    from pipeline.matcher import EvidenceMatcher
    from verifier.static_baseline import HeuristicNLI
except ImportError:
    from src.pipeline.decomposer import ClaimDecomposer
    from src.pipeline.matcher import EvidenceMatcher
    from src.verifier.static_baseline import HeuristicNLI


class NeuralCrossEncoderNLI:
    """
    Wrapper around HuggingFace cross-encoder for deep natural language inference.
    """
    tokenizer: Any
    model: Any
    id2label: Dict[Any, Any]
    device: str

    def __init__(self, model_name: str = "cross-encoder/nli-distilroberta-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id2label = getattr(self.model.config, "id2label", {})
        
    def predict(
        self,
        premise: str = "",
        hypothesis: str = "",
        claim: Optional[str] = None,
        evidence: Optional[str] = None,
        answer_entity: str = "",
        **kwargs: Any
    ) -> str:
        actual_premise = premise if premise else (evidence or "")
        actual_hypothesis = hypothesis if hypothesis else (claim or "")
        if not actual_premise or not actual_hypothesis or self.tokenizer is None:
            return "NEUTRAL"
        tokenizer: Any = self.tokenizer
        inputs = tokenizer(
            actual_premise, actual_hypothesis,
            padding=True, truncation=True, max_length=256,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            pred_id = int(logits.argmax(dim=-1).item())
        label = str(self.id2label.get(pred_id, pred_id)).upper() # ENTAILMENT, CONTRADICTION, NEUTRAL
        return label


class ParallaxAuditor:
    """
    End-to-End Auditor for multi-sentence RAG responses.
    """
    def __init__(
        self,
        use_neural_nli: bool = False,
        default_ref_time: str = "2024-12-01"
    ):
        self.decomposer = ClaimDecomposer()
        self.matcher = EvidenceMatcher()
        self.default_ref_time = default_ref_time
        
        self.nli: Union[NeuralCrossEncoderNLI, HeuristicNLI]
        if use_neural_nli:
            print("Initializing Neural Cross-Encoder NLI (distilroberta-base)...")
            self.nli = NeuralCrossEncoderNLI()
            self.nli_type = "neural"
        else:
            self.nli = HeuristicNLI()
            self.nli_type = "heuristic"

    def audit_response(
        self,
        query: str,
        answer_text: str,
        retrieved_docs: List[Dict[str, Any]],
        reference_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end verification of an answer paragraph.
        """
        ref_time = reference_time or self.default_ref_time
        
        # 1. Decompose answer into atomic claims
        claims = self.decomposer.decompose(answer_text)
        
        if not claims:
            return {
                "query": query,
                "answer_text": answer_text,
                "overall_status": "EMPTY_OR_UNPARSEABLE",
                "truthfulness_score": 0.0,
                "total_claims": 0,
                "claim_breakdown": {},
                "claim_audits": [],
            }
            
        claim_audits = []
        counts = {
            "SUPPORTED_CURRENT": 0,
            "OUTDATED": 0,
            "CONTRADICTED": 0,
            "UNSUPPORTED": 0,
        }
        
        # 2. Audit each claim
        for idx, claim in enumerate(claims):
            # Align with retrieved evidence
            match = self.matcher.match_claim(claim, retrieved_docs)
            
            if not match:
                # No relevant evidence found in retrieved context
                verdict = "UNSUPPORTED"
                counts[verdict] += 1
                claim_audits.append({
                    "claim_index": idx + 1,
                    "claim_text": claim,
                    "verdict": verdict,
                    "evidence_matched": None,
                    "evidence_source": None,
                    "match_score": 0.0,
                    "temporal_status": "NO_EVIDENCE",
                    "explanation": "No relevant evidence sentence found in retrieved context passages.",
                })
                continue
                
            matched_sent = match["matched_sentence"]
            doc_title = match.get("doc_title", "Document")
            doc_id = match.get("doc_id", "unknown")
            doc_info = f"{doc_title} (id: {doc_id})"
            
            valid_from = str(match.get("valid_from", "unknown"))
            valid_to_raw = match.get("valid_to")
            valid_to = str(valid_to_raw).strip() if valid_to_raw is not None else "present"
            if valid_to.lower() in ("none", "null", ""):
                valid_to = "present"
                
            superseded_raw = match.get("superseded_by")
            superseded_by = str(superseded_raw).strip() if superseded_raw is not None else None
            if superseded_by and superseded_by.lower() in ("none", "null", ""):
                superseded_by = None
            
            # 3. Linguistic NLI verification
            contextualized_premise = f"{doc_title}: {matched_sent}"
            if isinstance(self.nli, NeuralCrossEncoderNLI):
                nli_rel = self.nli.predict(premise=contextualized_premise, hypothesis=claim)
            else:
                nli_rel = self.nli.predict(claim=claim, evidence=contextualized_premise, answer_entity="")
                
            # 4. Temporal Arbitration
            if nli_rel == "ENTAILMENT":
                is_stale = False
                if valid_to != "present" and valid_to <= ref_time:
                    is_stale = True
                elif superseded_by is not None and superseded_by <= ref_time:
                    is_stale = True
                    
                if is_stale:
                    verdict = "OUTDATED"
                    temp_status = f"SUPERSEDED on {valid_to} (expired before ref_time {ref_time})"
                    expl = "Evidence strictly entails claim, but that evidence was superseded and is obsolete."
                else:
                    verdict = "SUPPORTED_CURRENT"
                    temp_status = "ACTIVE / VALID"
                    expl = "Evidence entails claim and is valid at evaluation reference time."
                    
            elif nli_rel == "CONTRADICTION":
                verdict = "CONTRADICTED"
                temp_status = f"CONFLICTING (evidence valid {valid_from} to {valid_to})"
                expl = "Evidence directly refutes or contradicts the claim."
                
            else: # NEUTRAL
                verdict = "UNSUPPORTED"
                temp_status = "INSUFFICIENT"
                expl = "Retrieved passage does not contain sufficient facts to establish claim."
                
            counts[verdict] += 1
            claim_audits.append({
                "claim_index": idx + 1,
                "claim_text": claim,
                "verdict": verdict,
                "evidence_matched": matched_sent,
                "evidence_source": doc_info,
                "match_score": match["match_score"],
                "temporal_status": temp_status,
                "explanation": expl,
            })
            
        # 5. Overall Answer Synthesis
        total = len(claims)
        score = round(counts["SUPPORTED_CURRENT"] / total, 3) if total > 0 else 0.0
        
        if counts["SUPPORTED_CURRENT"] == total:
            status = "TRUSTWORTHY_CURRENT"
        elif counts["OUTDATED"] > 0 and (counts["CONTRADICTED"] == 0 and counts["UNSUPPORTED"] == 0):
            status = "DEGRADED_BY_STALE_EVIDENCE"
        elif counts["OUTDATED"] == 0 and (counts["CONTRADICTED"] > 0 or counts["UNSUPPORTED"] > 0):
            status = "HALLUCINATED_RESPONSE"
        else:
            status = "MIXED_STALE_AND_HALLUCINATED"
            
        return {
            "query": query,
            "answer_text": answer_text,
            "reference_time": ref_time,
            "total_claims": total,
            "overall_status": status,
            "truthfulness_score": score,
            "claim_breakdown": counts,
            "claim_audits": claim_audits,
        }

    def format_report(self, audit: Dict[str, Any]) -> str:
        """Generates a human-readable text audit report."""
        lines = [
            "============================================================",
            "PARALLAX END-TO-END RAG AUDIT REPORT",
            "============================================================",
            f"Query:          {audit['query']}",
            f"Reference Time: {audit['reference_time']}",
            f"Overall Status: {audit['overall_status']}",
            f"Trust Score:    {audit['truthfulness_score'] * 100:.1f}% ({audit['claim_breakdown']['SUPPORTED_CURRENT']}/{audit['total_claims']} claims verified current)",
            f"Breakdown:      Supported-Current: {audit['claim_breakdown']['SUPPORTED_CURRENT']} | Outdated: {audit['claim_breakdown']['OUTDATED']} | Contra: {audit['claim_breakdown']['CONTRADICTED']} | Unsupp: {audit['claim_breakdown']['UNSUPPORTED']}",
            "------------------------------------------------------------",
            "ANSWER PARAGRAPH AUDITED:",
            f"\"{audit['answer_text']}\"",
            "------------------------------------------------------------",
            "CLAIM-BY-CLAIM AUDIT DETAILS:",
        ]
        
        for c in audit["claim_audits"]:
            lines.append(f"\n[Claim {c['claim_index']}] [{c['verdict']}]")
            lines.append(f"  Claim:       \"{c['claim_text']}\"")
            if c["evidence_matched"]:
                lines.append(f"  Evidence:    \"{c['evidence_matched']}\"")
                lines.append(f"  Source:      {c['evidence_source']}")
                lines.append(f"  Alignment:   Match Score = {c['match_score']:.3f} | {c['temporal_status']}")
            lines.append(f"  Diagnosis:   {c['explanation']}")
            
        lines.append("============================================================")
        return "\n".join(lines)


if __name__ == "__main__":
    auditor = ParallaxAuditor(use_neural_nli=False)
    q = "Who is the Master of the Horse?"
    a = "The Master of the Horse is Lord de Mauley."
    docs = [{
        "doc_id": "100025",
        "doc_title": "Master of the Horse",
        "text": "The current Master of the Horse is Lord de Mauley.",
        "valid_from": "2024-06-01",
        "valid_to": "2024-07-01",
        "superseded_by": "2024-07-01",
    }]
    report = auditor.audit_response(q, a, docs)
    print(auditor.format_report(report))
