"""
Parallax Verbalizer Module
Converts (question, answer) pairs into precise, declarative claims (answer_claim)
for claim-level verification without LLM hallucination artifacts.
"""

import re
from typing import Optional, Tuple, List, Callable

# Deterministic patterns mapping questions and entities to declarative assertions
PATTERNS: List[Tuple[str, Callable[[re.Match, str], str]]] = [
    # 1. "What/Who is/was/are/were the X?" -> "The X is/was/are/were ANS."
    (
        r"^(What|Who) (is|was|are|were) (the [^?]+)\??$",
        lambda m, ans: f"{m.group(3)[0].upper()}{m.group(3)[1:]} {m.group(2).lower()} {ans}."
    ),
    # 2. "Who is/was the X of/for Y?" -> "The X of/for Y is/was ANS."
    (
        r"^Who (is|was) (the [^?]+)\??$",
        lambda m, ans: f"{m.group(2)[0].upper()}{m.group(2)[1:]} {m.group(1).lower()} {ans}."
    ),
    # 3. "What percentage of X was/is Y?" -> "The percentage of X that was/is Y is/was ANS."
    (
        r"^What percentage of (.+) (is|was|are|were) (.+)\??$",
        lambda m, ans: f"The percentage of {m.group(1)} that {m.group(2)} {m.group(3)} {m.group(2)} {ans}."
    ),
    # 4. "What percentage of X ... ?" -> "The percentage of X is ANS."
    (
        r"^What percentage of ([^?]+)\??$",
        lambda m, ans: f"The percentage of {m.group(1)} is {ans}."
    ),
    # 5. "How many X are/were [there] in/at Y?" -> "There are/were ANS X in/at Y."
    (
        r"^How many ([^?]+) (are|were)(?: there)? (in|at|on) ([^?]+)\??$",
        lambda m, ans: f"There {m.group(2)} {ans} {m.group(1)} {m.group(3)} {m.group(4)}."
    ),
    # 6. "Which team/club does X play for?" -> "X plays for ANS."
    (
        r"^Which (team|club) does ([^?]+) play for\??$",
        lambda m, ans: f"{m.group(2).strip()} plays for {ans}."
    ),
    # 7. "What is/was X's Y?" -> "X's Y is/was ANS."
    (
        r"^What (is|was|are|were) (.+'s [^?]+)\??$",
        lambda m, ans: f"{m.group(2)[0].upper()}{m.group(2)[1:]} {m.group(1).lower()} {ans}."
    ),
    # 8. "In what year did X Y?" -> "X Y in ANS."
    (
        r"^In what year did (.+) (.+)\??$",
        lambda m, ans: f"{m.group(1).strip()} {m.group(2).strip()} in {ans}."
    ),
]


def verbalize_claim(question: str, answer: str, strict: bool = True) -> Optional[str]:
    """
    Verbalizes a (question, answer) pair into a declarative claim.
    
    Args:
        question: Natural language question.
        answer: Target answer entity or phrase.
        strict: If True, only returns claims that match verified linguistic patterns.
                If False, falls back to a generalized declarative template.
    Returns:
        Declarative sentence string, or None if strict=True and no pattern matches.
    """
    q = question.strip()
    ans = answer.strip()
    
    for pat, formatter in PATTERNS:
        match = re.match(pat, q, re.IGNORECASE)
        if match:
            return formatter(match, ans)
            
    if not strict:
        if q.endswith("?"):
            q = q[:-1].strip()
        return f"Regarding {q.lower()}, the answer is {ans}."
        
    return None
