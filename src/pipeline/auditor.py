"""
Parallax End-to-End Auditor
Full multi-claim RAG auditing pipeline:
Decomposition -> Evidence Sentence Matching -> Neural/Heuristic NLI -> Temporal Arbitration -> Audit Report
"""

import os
import json
import html
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

try:
    from pipeline.decomposer import ClaimDecomposer
    from pipeline.matcher import EvidenceMatcher
    from pipeline.temporal_resolver import TemporalAnchorResolver
    from verifier.static_baseline import HeuristicNLI
except ImportError:
    from src.pipeline.decomposer import ClaimDecomposer
    from src.pipeline.matcher import EvidenceMatcher
    from src.pipeline.temporal_resolver import TemporalAnchorResolver
    from src.verifier.static_baseline import HeuristicNLI


class NeuralCrossEncoderNLI:
    """
    Wrapper around HuggingFace cross-encoder for deep natural language inference.
    """
    tokenizer: Any
    model: Any
    id2label: Dict[Any, Any]
    device: str

    def __init__(self, model_name: Optional[str] = None):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as exc:
            raise ImportError(
                "NeuralCrossEncoderNLI requires 'torch' and 'transformers'. "
                "Install them with `pip install .[neural]`"
            ) from exc
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_name is None:
            finetuned_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "parallax_nli_finetuned"))
            weights_file = os.path.join(finetuned_path, "model.safetensors")
            if os.path.exists(weights_file):
                model_name = finetuned_path
            else:
                model_name = "cross-encoder/nli-distilroberta-base"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        raw_id2label = getattr(self.model.config, "id2label", {})
        self.id2label = {}
        for k, v in raw_id2label.items():
            val = str(v).upper()
            try:
                self.id2label[int(k)] = val
            except ValueError:
                pass
            self.id2label[str(k)] = val
        
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
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            pred_id = int(logits.argmax(dim=-1).item())
        label = str(self.id2label.get(pred_id, pred_id)).upper() # ENTAILMENT, CONTRADICTION, NEUTRAL
        return label

    def predict_batch(
        self,
        premises: List[str],
        hypotheses: List[str],
        batch_size: int = 32
    ) -> List[str]:
        if not premises or not hypotheses or self.tokenizer is None:
            return ["NEUTRAL"] * len(premises)
        tokenizer: Any = self.tokenizer
        preds: List[str] = []
        total = len(premises)
        for i in range(0, total, batch_size):
            batch_prem = premises[i:i + batch_size]
            batch_hyp = hypotheses[i:i + batch_size]
            inputs = tokenizer(
                batch_prem, batch_hyp,
                padding=True, truncation=True, max_length=256,
                return_tensors="pt"
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
            for pid in pred_ids:
                label = str(self.id2label.get(pid, pid)).upper()
                preds.append(label)
        return preds


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
        self.temporal_resolver = TemporalAnchorResolver()
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
                
            # 4. Temporal Arbitration with Temporal Anchor Resolver
            anchor_info = self.temporal_resolver.extract_temporal_anchor(claim)
            
            if nli_rel == "ENTAILMENT":
                temp_eval = self.temporal_resolver.arbitrate_temporal_validity(
                    anchor_info=anchor_info,
                    valid_from=valid_from,
                    valid_to=valid_to,
                    superseded_by=superseded_by,
                    reference_time=ref_time
                )
                
                if temp_eval["is_stale"]:
                    verdict = "OUTDATED"
                    staleness_days = None
                    try:
                        ref_dt = datetime.strptime(ref_time[:10], "%Y-%m-%d")
                        vt_dt = datetime.strptime(valid_to[:10], "%Y-%m-%d")
                        staleness_days = max(0, (ref_dt - vt_dt).days)
                    except Exception:
                        pass
                    stale_str = f" ({staleness_days} days stale)" if staleness_days is not None else ""
                    if superseded_by:
                        temp_status = f"SUPERSEDED by {superseded_by} (expired before ref_time {ref_time}){stale_str}"
                    else:
                        temp_status = f"SUPERSEDED on {valid_to} (expired before ref_time {ref_time}){stale_str}"
                    expl = "Evidence strictly entails claim, but that evidence was superseded and is obsolete."
                elif temp_eval["verdict_modifier"] == "HISTORICALLY_VALID":
                    verdict = "SUPPORTED_CURRENT"
                    temp_status = "HISTORICALLY VALID (consistent with claim anchor)"
                    expl = temp_eval["reason"]
                elif temp_eval["verdict_modifier"] == "TEMPORALLY_INCOMPATIBLE":
                    verdict = "UNSUPPORTED"
                    temp_status = "TEMPORALLY INCOMPATIBLE (evidence does not cover claim period)"
                    expl = temp_eval["reason"]
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
                "temporal_anchor": anchor_info["anchor_type"],
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

    def format_html_report(self, audit: Dict[str, Any]) -> str:
        """Generates a standalone, beautifully styled HTML audit report with dynamic content escaped."""
        badge_colors = {
            "SUPPORTED_CURRENT": "#28a745",
            "OUTDATED": "#fd7e14",
            "CONTRADICTED": "#dc3545",
            "UNSUPPORTED": "#6c757d",
        }
        claims_html = []
        for c in audit["claim_audits"]:
            color = badge_colors.get(c.get("verdict"), "#333")
            claim_idx = html.escape(str(c.get("claim_index", "")))
            verdict_esc = html.escape(str(c.get("verdict", "")))
            claim_text_esc = html.escape(str(c.get("claim_text", "")))
            expl_esc = html.escape(str(c.get("explanation", "")))

            ev_html = ""
            if c.get("evidence_matched"):
                ev_matched_esc = html.escape(str(c.get("evidence_matched", "")))
                ev_source_esc = html.escape(str(c.get("evidence_source", "")))
                temp_status_esc = html.escape(str(c.get("temporal_status", "")))
                ev_html = f"""
                <div style="margin-top: 6px; padding: 8px; background: #f8f9fa; border-left: 3px solid {color}; border-radius: 4px;">
                    <div><strong>Evidence:</strong> "{ev_matched_esc}"</div>
                    <div style="font-size: 0.85em; color: #555; margin-top: 4px;"><strong>Source:</strong> {ev_source_esc} | <strong>Status:</strong> {temp_status_esc}</div>
                </div>
                """
            claims_html.append(f"""
            <div style="margin-bottom: 14px; padding: 12px; border: 1px solid #e9ecef; border-radius: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold;">Claim {claim_idx}</span>
                    <span style="background-color: {color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold;">{verdict_esc}</span>
                </div>
                <div style="margin: 8px 0; font-size: 1.05em;">"{claim_text_esc}"</div>
                {ev_html}
                <div style="margin-top: 6px; font-size: 0.85em; color: #444;"><em>Diagnosis:</em> {expl_esc}</div>
            </div>
            """)

        query_esc = html.escape(str(audit.get("query", "")))
        ref_time_esc = html.escape(str(audit.get("reference_time", "")))
        overall_status_esc = html.escape(str(audit.get("overall_status", "")))
        ans_text_esc = html.escape(str(audit.get("answer_text", "")))
        truth_score = audit.get("truthfulness_score", 0.0) * 100.0
        sc_count = audit.get("claim_breakdown", {}).get("SUPPORTED_CURRENT", 0)
        tot_claims = audit.get("total_claims", 0)

        html_out = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Parallax RAG Audit Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 30px auto; padding: 20px; color: #212529; }}
        .header {{ background: #212529; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .status-box {{ display: flex; gap: 15px; margin-top: 15px; }}
        .metric-card {{ background: rgba(255,255,255,0.1); padding: 10px 15px; border-radius: 6px; flex: 1; }}
        .section-title {{ font-size: 1.1em; font-weight: bold; margin: 20px 0 10px 0; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
        .answer-box {{ background: #eef2f7; padding: 15px; border-radius: 6px; font-style: italic; line-height: 1.5; }}
    </style>
</head>
<body>
    <div class="header">
        <h2 style="margin: 0 0 10px 0;">Parallax End-to-End RAG Audit Report</h2>
        <div><strong>Query:</strong> {query_esc}</div>
        <div style="font-size: 0.9em; opacity: 0.85;">Anchor Reference Time: {ref_time_esc}</div>
        <div class="status-box">
            <div class="metric-card">
                <div style="font-size: 0.8em; text-transform: uppercase;">Overall Status</div>
                <div style="font-size: 1.2em; font-weight: bold;">{overall_status_esc}</div>
            </div>
            <div class="metric-card">
                <div style="font-size: 0.8em; text-transform: uppercase;">Trust Score</div>
                <div style="font-size: 1.2em; font-weight: bold;">{truth_score:.1f}%</div>
            </div>
            <div class="metric-card">
                <div style="font-size: 0.8em; text-transform: uppercase;">Claims Current</div>
                <div style="font-size: 1.2em; font-weight: bold;">{sc_count} / {tot_claims}</div>
            </div>
        </div>
    </div>

    <div class="section-title">Audited Answer Paragraph</div>
    <div class="answer-box">"{ans_text_esc}"</div>

    <div class="section-title">Claim-by-Claim Breakdown</div>
    {''.join(claims_html)}
</body>
</html>
"""
        return html_out

    def export_json(self, audit: Dict[str, Any], filepath: str):
        """Exports the audit report to a structured JSON file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
        print(f"Exported audit report JSON to {filepath}")


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
