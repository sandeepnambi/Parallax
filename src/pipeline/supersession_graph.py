"""
Parallax Supersession Graph
===========================
Maintains a temporal provenance DAG (Directed Acyclic Graph) of evolving facts
across document versions and snapshots. Traces the lineage of superseded evidence
and identifies modern replacement facts.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class FactNode:
    """Represents a factual assertion valid over a bounded temporal interval."""

    def __init__(
        self,
        node_id: str,
        doc_id: str,
        fact_text: str,
        answer_value: str,
        valid_from: str,
        valid_to: str = "present",
        superseded_by: Optional[str] = None,
    ):
        self.node_id = node_id
        self.doc_id = doc_id
        self.fact_text = fact_text
        self.answer_value = answer_value
        self.valid_from = valid_from
        self.valid_to = valid_to
        self.superseded_by = superseded_by
        self.superseded_by_node_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "doc_id": self.doc_id,
            "fact_text": self.fact_text,
            "answer_value": self.answer_value,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "superseded_by": self.superseded_by,
            "superseded_by_node_id": self.superseded_by_node_id,
            "is_active": self.valid_to == "present",
        }


class SupersessionGraph:
    """
    Graph of factual transitions over time.
    """

    def __init__(self):
        self.nodes: Dict[str, FactNode] = {}
        self.doc_chains: Dict[str, List[str]] = {}

    def add_fact_version(
        self,
        doc_id: str,
        fact_text: str,
        answer_value: str,
        valid_from: str,
        valid_to: str = "present",
        superseded_by: Optional[str] = None,
    ) -> FactNode:
        """Adds a fact version and links it into the document revision chain."""
        node_id = f"{doc_id}_{valid_from}"
        node = FactNode(
            node_id=node_id,
            doc_id=doc_id,
            fact_text=fact_text,
            answer_value=answer_value,
            valid_from=valid_from,
            valid_to=valid_to,
            superseded_by=superseded_by,
        )
        self.nodes[node_id] = node

        if doc_id not in self.doc_chains:
            self.doc_chains[doc_id] = []

        chain = self.doc_chains[doc_id]
        if chain:
            prev_node_id = chain[-1]
            prev_node = self.nodes[prev_node_id]
            if prev_node.valid_to != "present" and prev_node.superseded_by:
                prev_node.superseded_by_node_id = node_id

        chain.append(node_id)
        return node

    def trace_lineage(self, doc_id: str) -> List[Dict[str, Any]]:
        """Returns the full chronological transition history for a document topic."""
        node_ids = self.doc_chains.get(doc_id, [])
        return [self.nodes[nid].to_dict() for nid in node_ids]

    def get_active_replacement(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Finds the latest active (non-superseded) fact node for the document."""
        node_ids = self.doc_chains.get(doc_id, [])
        for nid in reversed(node_ids):
            node = self.nodes[nid]
            if node.valid_to == "present":
                return node.to_dict()
        return None

    def explain_supersession(self, doc_id: str, stale_claim_value: str) -> Dict[str, Any]:
        """
        Produces a rich diagnostic explanation of how and when a stale claim was superseded.
        """
        history = self.trace_lineage(doc_id)
        active = self.get_active_replacement(doc_id)

        matched_stale = None
        for item in history:
            if stale_claim_value.lower() in item["answer_value"].lower() or stale_claim_value.lower() in item["fact_text"].lower():
                matched_stale = item
                break

        return {
            "doc_id": doc_id,
            "stale_fact": matched_stale or (history[0] if history else None),
            "active_replacement": active,
            "history_depth": len(history),
            "timeline": history,
        }
