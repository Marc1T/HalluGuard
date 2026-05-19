"""
HalluGuardMiddleware — s'integre comme middleware dans le pipeline LangGraph.

Intercepte chaque output via des hooks pre/post-noeud.
Variante B : active uniquement M1 (LightweightVerifier - coherence inter-sources).
Variante C : active M1 + M2 (BeliefState) + memoire persistante.
"""

from __future__ import annotations
import time
from typing import Callable, Dict, List, Literal, Optional

from .dag import DependencyDAG
from .verifier import LightweightVerifier
from .policy import PolicyEngine

Variant = Literal["A", "B", "C"]

_MAX_CLAIM = 512   # troncature avant NLI pour eviter des latences excessives


class HalluGuardMiddleware:
    """
    Variant A : baseline RAG sans HalluGuard (pass-through).
    Variant B : M1 seul (coherence inter-sources NLI, pas de memoire inter-sessions).
    Variant C : M1 + M2 (BeliefState) + memoire persistante.
    """

    def __init__(
        self,
        variant: Variant = "C",
        policy_mode: str = "log",
        session_id: str = "default",
        mcp_client=None,
        llm_fn: Optional[Callable[[str], str]] = None,
        log_path: str = "results/logs/halluguard.log",
    ) -> None:
        self.variant = variant
        self.session_id = session_id
        self.mcp_client = mcp_client
        self.llm_fn = llm_fn

        # "log-only" est un alias de "log" (compat rag_agent.py)
        effective_mode = "log" if policy_mode in ("log", "log-only") else "auto-correct"

        if variant in ("B", "C"):
            self.dag = DependencyDAG.build_from_langgraph({
                "retrieval": ["reasoning"],
                "reasoning": ["tool_call", "generation"],
                "tool_call": ["generation"],
            })
            self.verifier = LightweightVerifier()
            self.policy = PolicyEngine(mode=effective_mode, log_path=log_path)

        # BeliefState uniquement pour C
        if variant == "C":
            from src.memory.persistent_memory import PersistentMemory
            self._mem = PersistentMemory(memory_dir="data/memory")
            self.belief = self._mem.load_or_create(session_id)
        else:
            self.belief = None
            self._mem = None

        self._overhead_ms: List[float] = []

    # ------------------------------------------------------------------ #
    # Hooks LangGraph                                                        #
    # ------------------------------------------------------------------ #

    def pre_node(self, _node_type: str, state: Dict) -> Dict:
        """Appele avant l'execution d'un noeud — pass-through pour l'instant."""
        return state

    def post_node(
        self,
        node_type: str,
        output: str,
        _state: Dict,
        evidences: Optional[List[str]] = None,
    ) -> Dict:
        """
        Appele apres l'execution d'un noeud.

        output    : texte produit par le noeud (response LLM, chunks, etc.)
        evidences : documents RAG associes (pour M1)

        Retourne : {"output": str, "action": str, "event": dict|None, "overhead_ms": float}
        """
        t0 = time.time()

        if self.variant == "A":
            return {"output": output, "action": "pass", "event": None, "overhead_ms": 0.0}

        claim = output[:_MAX_CLAIM]
        evs   = [e[:_MAX_CLAIM] for e in (evidences or [])]

        # ---- M1 : coherence inter-sources (NLI) ----
        result_m1 = self.verifier.verify(claim, evs, node_type=node_type)
        policy_out = self.policy.handle(
            result_m1,
            claim=claim,
            node_id=node_type,
            evidences=evs,
            reinvoke_fn=self.llm_fn if self.policy.mode == "auto-correct" else None,
        )

        final_output = policy_out["output"] or output

        # Propager l'invalidation dans le DAG si hallucination detectee
        if result_m1["label"] == "hallucinated":
            self.dag.propagate_invalidation(node_type)

        # ---- M2 : BeliefState (C uniquement) ----
        if self.variant == "C" and self.belief is not None:
            try:
                m2 = self.belief.check_temporal_coherence(
                    final_output[:_MAX_CLAIM], verifier=self.verifier
                )
                if m2.get("conflict"):
                    ev = policy_out.get("event") or {}
                    ev["m2_conflict"] = m2
                    policy_out["event"] = ev
                self.belief.add_fact(
                    content=final_output[:256],
                    confidence=result_m1["score"] if result_m1["label"] == "correct" else 0.5,
                    step=1,
                )
            except Exception:
                pass

        # ---- MCP : add_fact pour audit persistant (C uniquement) ----
        if self.variant == "C" and self.mcp_client is not None:
            try:
                self.mcp_client.call_tool("add_fact", {
                    "fact": final_output[:256],
                    "confidence": round(result_m1["score"], 3),
                    "step": 1,
                })
            except Exception:
                pass

        overhead = (time.time() - t0) * 1000
        self._overhead_ms.append(overhead)

        return {
            "output": final_output,
            "action": policy_out["action"],
            "event": policy_out.get("event"),
            "overhead_ms": round(overhead, 1),
        }

    # ------------------------------------------------------------------ #
    # Persistance session (C uniquement)                                    #
    # ------------------------------------------------------------------ #

    def save_session(self) -> None:
        if self.variant == "C" and self._mem is not None and self.belief is not None:
            self._mem.save(self.belief)

    # ------------------------------------------------------------------ #
    # Metriques overhead                                                    #
    # ------------------------------------------------------------------ #

    def overhead_stats(self) -> Dict:
        if not self._overhead_ms:
            return {"calls": 0, "avg_ms": None, "max_ms": None}
        return {
            "calls": len(self._overhead_ms),
            "avg_ms": round(sum(self._overhead_ms) / len(self._overhead_ms), 1),
            "max_ms": round(max(self._overhead_ms), 1),
        }
