"""
Parallax Claim Decomposer
Splits multi-sentence RAG response paragraphs into atomic, verifiable claim units
and strips non-factual conversational boilerplate.
"""

import re
from typing import List


class ClaimDecomposer:
    """
    Decomposes an answer paragraph into individual declarative factual claims.
    """
    
    # Common conversational boilerplate to strip
    BOILERPLATE_PREFIXES = [
        r"^based on (the )?(provided |retrieved )?(documentation|documents?|context|information|text)\b,?\s*",
        r"^according to (the )?(provided |retrieved )?(documentation|documents?|context|text|wikipedia)\b,?\s*",
        r"^in summary\b,?\s*",
        r"^to answer your question\b,?\s*",
        r"^sure\b,?\s*(here is the information:?)?\s*",
        r"^certainly\b,?\s*",
        r"^in conclusion\b,?\s*",
        r"^as mentioned (above|earlier)\b,?\s*",
        r"^additionally\b,?\s*",
        r"^furthermore\b,?\s*",
    ]
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Splits text into sentences while protecting decimals, abbreviations, and initials.
        """
        s = text.strip()
        
        # Replace decimals like 3.14 with special token
        s = re.sub(r"(\d+)\.(\d+)", r"\1<DECIMAL>\2", s)
        
        # Protect abbreviations like U.S., Inc., Dr., etc. while preserving original case
        abbrevs = [r"\bu\.s\.", r"\bu\.k\.", r"\be\.g\.", r"\bi\.e\.", r"\bdr\.", r"\bmr\.", r"\bmrs\.", r"\bms\.", r"\binc\.", r"\bltd\.", r"\bco\.", r"\bvs\.", r"\bapprox\."]
        for pat in abbrevs:
            s = re.sub(pat, lambda m: m.group(0).replace(".", "<DOT>"), s, flags=re.IGNORECASE)
            
        # Split on sentence boundaries
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", s)
        
        clean_sentences = []
        for sent in raw_sentences:
            # Restore tokens
            restored = sent.replace("<DECIMAL>", ".").replace("<DOT>", ".")
            restored = restored.strip()
            if restored:
                clean_sentences.append(restored)
                
        return clean_sentences

    def clean_claim(self, sentence: str) -> str:
        """Strips conversational preambles, capitalizes, and normalizes punctuation."""
        c = sentence.strip()
        prev = None
        while prev != c:
            prev = c
            for pat in self.BOILERPLATE_PREFIXES:
                c = re.sub(pat, "", c, flags=re.IGNORECASE).strip()
            c = re.sub(r"^[,;:\s]+", "", c).strip()
            
        if c:
            c = c[0].upper() + c[1:]
        if c and not c.endswith((".", "!", "?")):
            c += "."
        return c

    def decompose(self, answer_text: str) -> List[str]:
        """
        Full decomposition pipeline:
        1. Splits paragraph into sentences.
        2. Cleans boilerplate.
        3. Filters out trivial non-factual utterances (< 3 words).
        """
        if not answer_text or not answer_text.strip():
            return []
            
        sentences = self.split_sentences(answer_text)
        claims = []
        
        for s in sentences:
            cleaned = self.clean_claim(s)
            words = cleaned.split()
            if len(words) >= 3:
                claims.append(cleaned)
                
        return claims

    @staticmethod
    def extract_temporal_cues(sentence: str) -> List[str]:
        """Extracts temporal cues and years present in a claim sentence."""
        patterns = [
            r"\b(currently|presently|as of \d{4}|in \d{4}|since \d{4}|formerly|previously|latest|recent)\b",
            r"\b(19\d{2}|20\d{2})\b"
        ]
        cues = []
        for pat in patterns:
            matches = re.findall(pat, sentence, re.IGNORECASE)
            cues.extend([m.lower() for m in matches])
        return sorted(list(set(cues)))

    def decompose_with_metadata(self, answer_text: str) -> List[dict]:
        """Decomposes answer paragraph into claims with attached linguistic and temporal metadata."""
        raw_claims = self.decompose(answer_text)
        metadata_claims = []
        for idx, claim in enumerate(raw_claims):
            cues = self.extract_temporal_cues(claim)
            metadata_claims.append({
                "claim_index": idx + 1,
                "claim_text": claim,
                "word_count": len(claim.split()),
                "temporal_cues": cues,
                "has_temporal_anchor": len(cues) > 0
            })
        return metadata_claims
