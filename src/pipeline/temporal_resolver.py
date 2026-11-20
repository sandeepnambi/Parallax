"""
Parallax Temporal Anchor Resolver
=================================
Analyzes temporal expressions and anchors within natural language claims
to distinguish between:
1. Present-anchored claims ("Currently, X is Y", "X is the coach")
2. Historically-anchored claims ("In 2020, X was Y", "Between 2018 and 2022...")

Prevents false "OUTDATED" certification for claims that accurately describe historical facts.
"""

import re
from typing import Dict, Any, Optional, Tuple


class TemporalAnchorResolver:
    """
    Resolves temporal anchors and scopes within answer claims.
    """

    PRESENT_CUES = [
        r"\bcurrently\b",
        r"\bpresently\b",
        r"\bas of today\b",
        r"\bas of now\b",
        r"\bat present\b",
        r"\bcurrent\b",
        r"\bnow\b",
        r"\blatest\b",
        r"\bongoing\b",
    ]

    PAST_CUES = [
        r"\bformerly\b",
        r"\bpreviously\b",
        r"\bhistorically\b",
        r"\bin the past\b",
        r"\bused to be\b",
        r"\boriginally\b",
    ]

    YEAR_PATTERN = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
    DATE_PATTERN = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")

    def extract_temporal_anchor(self, claim_text: str) -> Dict[str, Any]:
        """
        Extracts temporal expressions and determines the temporal anchor type.
        
        Returns:
            Dict containing:
            - anchor_type: 'PRESENT', 'HISTORICAL', or 'IMPLICIT_PRESENT'
            - explicit_years: List of matched years
            - explicit_dates: List of matched ISO dates
            - temporal_cues: List of matched temporal keywords
            - target_interval: Tuple of (start_date, end_date) if bounded
        """
        text_lower = claim_text.lower()

        # Check explicit cues
        present_matches = [
            m.strip() for pat in self.PRESENT_CUES for m in re.findall(pat, text_lower)
        ]
        past_matches = [
            m.strip() for pat in self.PAST_CUES for m in re.findall(pat, text_lower)
        ]

        # Extract dates and years
        years = self.YEAR_PATTERN.findall(claim_text)
        dates = self.DATE_PATTERN.findall(claim_text)

        # Check for year intervals like "from 2018 to 2022" or "in 2020"
        interval_match = re.search(
            r"\b(from|between)\s+(19\d{2}|20\d{2})\s+(to|and|until)\s+(19\d{2}|20\d{2})\b",
            text_lower,
        )
        single_year_match = re.search(r"\b(in|during|for)\s+(19\d{2}|20\d{2})\b", text_lower)

        target_interval: Optional[Tuple[str, str]] = None
        anchor_type = "IMPLICIT_PRESENT"

        if interval_match:
            y_start = interval_match.group(2)
            y_end = interval_match.group(4)
            target_interval = (f"{y_start}-01-01", f"{y_end}-12-31")
            anchor_type = "HISTORICAL"
        elif single_year_match:
            y = single_year_match.group(2)
            target_interval = (f"{y}-01-01", f"{y}-12-31")
            anchor_type = "HISTORICAL"
        elif dates:
            d = dates[0]
            target_interval = (d, d)
            anchor_type = "HISTORICAL"
        elif present_matches:
            anchor_type = "PRESENT"
        elif past_matches:
            anchor_type = "HISTORICAL"
        else:
            # Check for present tense vs past tense verbs
            if re.search(r"\b(was|were|had|did|departed|resigned)\b", text_lower) and not re.search(
                r"\b(is|are|has|remains|continues)\b", text_lower
            ):
                anchor_type = "HISTORICAL"
            else:
                anchor_type = "IMPLICIT_PRESENT"

        return {
            "claim_text": claim_text,
            "anchor_type": anchor_type,
            "explicit_years": years,
            "explicit_dates": dates,
            "present_cues": present_matches,
            "past_cues": past_matches,
            "target_interval": target_interval,
        }

    def arbitrate_temporal_validity(
        self,
        anchor_info: Dict[str, Any],
        valid_from: Optional[str],
        valid_to: Optional[str],
        superseded_by: Optional[str] = None,
        reference_time: str = "2024-12-01",
        require_full_coverage: bool = True,
    ) -> Dict[str, Any]:
        """
        Arbitrates whether evidence validity satisfies the claim's temporal anchor.
        
        Args:
            anchor_info: Dict from extract_temporal_anchor.
            valid_from: Start of evidence validity.
            valid_to: End of evidence validity or 'present'.
            superseded_by: Date or identifier indicating when/by what evidence was superseded.
            reference_time: Evaluation anchor date (YYYY-MM-DD).
            require_full_coverage: If True, historical interval claims require evidence
                validity to completely cover the claimed interval rather than just overlapping it.
        """
        anchor_type = anchor_info["anchor_type"]
        v_from = str(valid_from)[:10] if valid_from and str(valid_from).lower() not in ("none", "null", "") else "1900-01-01"
        v_to = str(valid_to)[:10] if valid_to and str(valid_to).lower() not in ("none", "null", "present", "") else "9999-12-31"

        # Parse and normalize superseded_by metadata
        s_by: Optional[str] = None
        if superseded_by is not None:
            s_str = str(superseded_by).strip()
            if s_str.lower() not in ("none", "null", "false", ""):
                s_by = s_str

        # Determine if evidence is superseded or expired at reference_time
        is_stale = False
        stale_reason = ""
        if v_to != "9999-12-31" and v_to <= reference_time:
            is_stale = True
            stale_reason = f"Evidence expired on {v_to} prior to reference_time {reference_time}."
        elif s_by is not None:
            # Check if superseded_by contains an ISO date
            date_match = self.DATE_PATTERN.search(s_by)
            if date_match:
                s_date = date_match.group(1)
                if s_date <= reference_time:
                    is_stale = True
                    stale_reason = f"Evidence was explicitly superseded on {s_date} prior to reference_time {reference_time}."
            elif re.match(r"^\d{4}-\d{2}-\d{2}", s_by):
                if s_by[:10] <= reference_time:
                    is_stale = True
                    stale_reason = f"Evidence was explicitly superseded on {s_by[:10]} prior to reference_time {reference_time}."
            else:
                # Explicit supersession indicated by replacement identifier or flag
                is_stale = True
                stale_reason = f"Evidence was explicitly superseded by {s_by}."

        # Compute effective validity end for interval checks
        effective_v_to = v_to
        if s_by is not None:
            date_match = self.DATE_PATTERN.search(s_by) or re.match(r"^(\d{4}-\d{2}-\d{2})", s_by)
            if date_match:
                s_date = date_match.group(1)
                if s_date < effective_v_to:
                    effective_v_to = s_date

        if anchor_type == "HISTORICAL" and anchor_info.get("target_interval"):
            claim_start, claim_end = anchor_info["target_interval"]
            
            # Full coverage: evidence validity encompasses the entire claim interval
            covers = (v_from <= claim_start and effective_v_to >= claim_end)
            # Overlap: evidence validity shares some temporal interval with claim
            overlaps = max(v_from, claim_start) <= min(effective_v_to, claim_end)

            if covers:
                return {
                    "verdict_modifier": "HISTORICALLY_VALID",
                    "is_stale": False,
                    "reason": (
                        f"Claim targets historical interval [{claim_start}, {claim_end}], "
                        f"which is fully covered by evidence validity [{v_from}, {v_to}]."
                    ),
                }
            elif overlaps:
                if require_full_coverage:
                    return {
                        "verdict_modifier": "TEMPORALLY_INCOMPATIBLE",
                        "is_stale": False,
                        "reason": (
                            f"Claim targets historical interval [{claim_start}, {claim_end}], "
                            f"but evidence validity [{v_from}, {v_to}] only partially overlaps "
                            f"and does not cover the complete period."
                        ),
                    }
                else:
                    return {
                        "verdict_modifier": "HISTORICALLY_VALID",
                        "is_stale": False,
                        "reason": (
                            f"Claim targets historical interval [{claim_start}, {claim_end}], "
                            f"which partially overlaps evidence validity [{v_from}, {v_to}]."
                        ),
                    }
            else:
                return {
                    "verdict_modifier": "TEMPORALLY_INCOMPATIBLE",
                    "is_stale": False,
                    "reason": (
                        f"Claim targets historical interval [{claim_start}, {claim_end}], "
                        f"but evidence was only valid [{v_from}, {v_to}] (no temporal overlap)."
                    ),
                }

        if anchor_type == "HISTORICAL":
            # Historical claim without a specific interval (e.g. "Formerly, X was Y")
            if is_stale:
                return {
                    "verdict_modifier": "HISTORICALLY_VALID",
                    "is_stale": False,
                    "reason": (
                        f"Claim asserts a historical or past state, supported by past evidence "
                        f"(valid [{v_from}, {v_to}])."
                    ),
                }
            else:
                return {
                    "verdict_modifier": "ACTIVE",
                    "is_stale": False,
                    "reason": f"Claim asserts past fact and evidence remains active (valid [{v_from}, {v_to}]).",
                }

        # Present or Implicit Present
        return {
            "verdict_modifier": "OUTDATED" if is_stale else "ACTIVE",
            "is_stale": is_stale,
            "reason": (
                stale_reason
                if is_stale
                else "Evidence is active at reference_time."
            ),
        }
