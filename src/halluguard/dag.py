"""
DependencyDAG — pipeline RAG modelise comme graphe networkx (DiGraph).

criticite(n) = nb_descendants(n) x (1 - confiance(n))
should_verify(n) : criticite >= seuil OU noeud suspect
propagate_invalidation(n) : marque n et tous ses descendants comme suspects
force_verify_suspicious() : retourne tous les noeuds suspects
build_from_langgraph() : construction depuis un dict d'aretes LangGraph

Seuils par type (CONTEXT.md) :
  tool_call  = 0.0  (toujours verifie)
  generation = 0.0  (toujours verifie)
  reasoning  = 0.4
  retrieval  = 0.6
"""

from __future__ import annotations
import networkx as nx
from typing import Dict, List, Optional, Set


# Confiance par defaut par type de noeud
DEFAULT_CONFIANCE: Dict[str, float] = {
    "retrieval": 0.6,
    "reasoning": 0.4,
    "tool_call": 0.0,
    "generation": 0.0,
}

# Seuil de criticite pour declencher la verification
VERIFY_THRESHOLDS: Dict[str, float] = {
    "tool_call": 0.0,    # toujours
    "generation": 0.0,   # toujours
    "reasoning": 0.4,
    "retrieval": 0.6,
}


class DependencyDAG:
    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------ #
    # Construction                                                          #
    # ------------------------------------------------------------------ #

    def add_node(
        self,
        node_id: str,
        node_type: Optional[str] = None,
        confiance: Optional[float] = None,
    ) -> None:
        """Ajoute un noeud. node_type determine seuil et confiance par defaut."""
        if node_id in self._g:
            return
        ntype = node_type if node_type is not None else node_id
        c = confiance if confiance is not None else DEFAULT_CONFIANCE.get(ntype, 1.0)
        self._g.add_node(node_id, node_type=ntype, confiance=c, suspicious=False)

    def add_edge(self, parent_id: str, child_id: str) -> None:
        """Arete dirigee parent -> child (child depend de parent)."""
        if parent_id not in self._g:
            self.add_node(parent_id)
        if child_id not in self._g:
            self.add_node(child_id)
        self._g.add_edge(parent_id, child_id)

    def set_confiance(self, node_id: str, confiance: float) -> None:
        if node_id in self._g:
            self._g.nodes[node_id]["confiance"] = max(0.0, min(1.0, confiance))

    # ------------------------------------------------------------------ #
    # Metriques                                                             #
    # ------------------------------------------------------------------ #

    def descendants(self, node_id: str) -> Set[str]:
        """Tous les descendants via nx.descendants (BFS/DFS networkx)."""
        if node_id not in self._g:
            return set()
        return nx.descendants(self._g, node_id)

    def criticite(self, node_id: str) -> float:
        """criticite(n) = nb_descendants(n) x (1 - confiance(n))."""
        if node_id not in self._g:
            return 0.0
        nb_desc = len(self.descendants(node_id))
        confiance = self._g.nodes[node_id]["confiance"]
        return nb_desc * (1.0 - confiance)

    # ------------------------------------------------------------------ #
    # Verification                                                          #
    # ------------------------------------------------------------------ #

    def should_verify(self, node_id: str) -> bool:
        """
        Retourne True si le noeud doit etre verifie :
          - noeud suspect (propagation d'invalidation), OU
          - criticite >= seuil du type de noeud
        """
        if node_id not in self._g:
            return False
        data = self._g.nodes[node_id]
        if data.get("suspicious", False):
            return True
        node_type = data["node_type"]
        threshold = VERIFY_THRESHOLDS.get(node_type, 1.0)
        return self.criticite(node_id) >= threshold

    # ------------------------------------------------------------------ #
    # Propagation d'invalidation                                            #
    # ------------------------------------------------------------------ #

    def propagate_invalidation(self, node_id: str) -> List[str]:
        """
        Marque node_id ET tous ses descendants comme suspects.
        Retourne la liste des noeuds marques.
        """
        if node_id not in self._g:
            return []
        affected = [node_id] + list(self.descendants(node_id))
        for nid in affected:
            self._g.nodes[nid]["suspicious"] = True
        return affected

    def force_verify_suspicious(self) -> List[str]:
        """Retourne tous les noeuds actuellement suspects."""
        return [
            n for n, d in self._g.nodes(data=True)
            if d.get("suspicious", False)
        ]

    def reset_suspicious(self, node_id: str) -> None:
        if node_id in self._g:
            self._g.nodes[node_id]["suspicious"] = False

    def reset_all_suspicious(self) -> None:
        for n in self._g.nodes():
            self._g.nodes[n]["suspicious"] = False

    # ------------------------------------------------------------------ #
    # Integration LangGraph                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def build_from_langgraph(cls, graph_edges: Dict[str, List[str]]) -> "DependencyDAG":
        """
        graph_edges = {
            "retrieval": ["reasoning"],
            "reasoning": ["tool_call", "generation"],
        }
        """
        dag = cls()
        for parent, children in graph_edges.items():
            for child in children:
                dag.add_edge(parent, child)
        return dag

    # ------------------------------------------------------------------ #
    # Utilitaires                                                           #
    # ------------------------------------------------------------------ #

    def all_nodes(self) -> List[str]:
        return list(self._g.nodes())

    def suspicious_nodes(self) -> List[str]:
        return self.force_verify_suspicious()

    def get_node_data(self, node_id: str) -> Optional[Dict]:
        if node_id not in self._g:
            return None
        return dict(self._g.nodes[node_id])

    def summary(self) -> Dict:
        return {
            n: {
                "node_type": d["node_type"],
                "confiance": round(d["confiance"], 3),
                "suspicious": d["suspicious"],
                "criticite": round(self.criticite(n), 3),
                "nb_descendants": len(self.descendants(n)),
                "should_verify": self.should_verify(n),
            }
            for n, d in self._g.nodes(data=True)
        }
