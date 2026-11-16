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
        
        # Protect abbreviations like U.S., Inc., Dr., etc.
        abbrevs = ["u.s.", "u.k.", "e.g.", "i.e.", "dr.", "mr.", "mrs.", "ms.", "inc.", "ltd.", "co.", "vs.", "approx."]
        for abb in abbrevs:
            pattern = re.compile(re.escape(abb), re.IGNORECASE)
            s = pattern.sub(abb.replace(".", "<DOT>"), s)
            
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
