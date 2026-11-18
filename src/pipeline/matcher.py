"""
Parallax Evidence Matcher
Aligns individual atomic claims to relevant evidence sentences across
retrieved multi-document context passages, attaching temporal provenance.
"""

import re
from typing import List, Dict, Any, Optional

try:
    from pipeline.decomposer import ClaimDecomposer
except ImportError:
    from src.pipeline.decomposer import ClaimDecomposer


class EvidenceMatcher:
    """
    Finds the most relevant evidence sentence across retrieved documents for a given claim.
    """
    
    STOPWORDS = {
        "the", "a", "an", "is", "was", "are", "were", "in", "at", "on", "for",
        "to", "of", "and", "or", "that", "this", "it", "with", "as", "by", "from",
        "has", "have", "had", "been", "be", "do", "does", "did", "their", "its"
    }
    
    @classmethod
    def get_content_tokens(cls, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9$%'–—\.\-]+\b", text.lower())
        return [w for w in words if w not in cls.STOPWORDS and len(w) > 1]
    
    @classmethod
    def extract_salient_entities(cls, text: str) -> List[str]:
        """Extracts numbers, percentages, years, and capitalized proper nouns."""
        # Numbers, dates, percentages
        numerics = re.findall(r"\b(?:\$?\d+(?:,\d+)*(?:\.\d+)?%?|\d{4})\b", text)
        # Capitalized multi-word phrases
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        return list(set(numerics + [c.lower() for c in caps]))

    @classmethod
    def score_match(cls, claim: str, sentence: str) -> float:
        """
        Computes composite lexical and entity alignment score between a claim and an evidence candidate.
        """
        c_tokens = set(cls.get_content_tokens(claim))
        s_tokens = set(cls.get_content_tokens(sentence))
        
        if not c_tokens or not s_tokens:
            return 0.0
            
        overlap = c_tokens.intersection(s_tokens)
        jaccard = len(overlap) / len(c_tokens.union(s_tokens))
        coverage = len(overlap) / len(c_tokens)
        
        # Entity overlap bonus
        c_entities = set(cls.extract_salient_entities(claim))
        s_text_lower = sentence.lower()
        entity_matches = sum(1 for ent in c_entities if ent in s_text_lower)
        entity_bonus = (entity_matches / max(len(c_entities), 1)) * 0.3
        
        # Composite score
        score = (0.5 * coverage) + (0.2 * jaccard) + entity_bonus
        return float(score)

    def match_claim(
        self,
        claim: str,
        retrieved_docs: List[Dict[str, Any]],
        min_threshold: float = 0.15
    ) -> Optional[Dict[str, Any]]:
        """
        Matches a claim to the best evidence sentence across all retrieved documents.
        
        retrieved_docs schema:
        [
            {
                "doc_id": "...",
                "doc_title": "...",
                "text": "Full passage text with multiple sentences...",
                "valid_from": "2024-07-01",
                "valid_to": "2024-08-01" or "present",
                "superseded_by": "2024-08-01" or None,
            }, ...
        ]
        
        Returns:
            Dict containing best matched sentence, match score, and parent document metadata,
            or None if no sentence exceeds min_threshold.
        """
        best_match = None
        best_score = -1.0
        
        for doc in retrieved_docs:
            doc_text = doc.get("text", "")
            sentences = ClaimDecomposer.split_sentences(doc_text)
            
            for sent in sentences:
                score = self.score_match(claim, sent)
                if score > best_score:
                    best_score = score
                    best_match = {
                        "matched_sentence": sent,
                        "match_score": round(score, 4),
                        "doc_id": doc.get("doc_id"),
                        "doc_title": doc.get("doc_title"),
                        "valid_from": doc.get("valid_from"),
                        "valid_to": doc.get("valid_to", "present"),
                        "superseded_by": doc.get("superseded_by"),
                    }
                    
        if best_match and best_score >= min_threshold:
            return best_match
            
        return None

    def rank_matches(
        self,
        claim: str,
        retrieved_docs: List[Dict[str, Any]],
        top_k: int = 3,
        min_threshold: float = 0.10
    ) -> List[Dict[str, Any]]:
        """
        Ranks and returns the top-k matched evidence sentences across retrieved documents.
        """
        candidates = []
        for doc in retrieved_docs:
            doc_text = doc.get("text", "")
            sentences = ClaimDecomposer.split_sentences(doc_text)
            for sent in sentences:
                score = self.score_match(claim, sent)
                if score >= min_threshold:
                    candidates.append({
                        "matched_sentence": sent,
                        "match_score": round(score, 4),
                        "doc_id": doc.get("doc_id"),
                        "doc_title": doc.get("doc_title"),
                        "valid_from": doc.get("valid_from"),
                        "valid_to": doc.get("valid_to", "present"),
                        "superseded_by": doc.get("superseded_by"),
                    })
        candidates.sort(key=lambda x: x["match_score"], reverse=True)
        return candidates[:top_k]
