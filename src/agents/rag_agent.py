"""
Pipeline RAG complet basé sur LangGraph.

Contient le StateGraph avec 4 nœuds : retrieval (ChromaDB), reasoning (Mistral),
tool_call (recherche complémentaire), generation (Mistral).
Intègre les hooks pre/post-nœud pour brancher HalluGuardMiddleware.
Utilisé comme baseline (Variante A) sans HalluGuard, et comme pipeline complet (Variante C).
"""

from __future__ import annotations
import time
from typing import Dict, List, TypedDict

import ollama
from langgraph.graph import StateGraph, END

# ------------------------------------------------------------------ #
# État partagé du pipeline                                              #
# ------------------------------------------------------------------ #

class RAGState(TypedDict):
    query: str
    documents: List[str]          # passages récupérés par ChromaDB
    reasoning: str                # sortie du nœud reasoning
    tool_output: str              # sortie du nœud tool_call
    answer: str                   # réponse finale
    halluguard_events: List[Dict] # événements de détection


# ------------------------------------------------------------------ #
# Fonctions utilitaires LLM / RAG                                       #
# ------------------------------------------------------------------ #

def _call_mistral(prompt: str, model: str = "mistral") -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def _retrieve_documents(query: str, collection_name: str = "halluguard_docs", n_results: int = 3) -> List[str]:
    """Récupère des passages depuis ChromaDB. Retourne [] si ChromaDB non initialisé."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path="data/chromadb")
        collection = client.get_collection(collection_name)
        results = collection.query(query_texts=[query], n_results=n_results)
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []


# ------------------------------------------------------------------ #
# Builder du pipeline                                                   #
# ------------------------------------------------------------------ #

def build_rag_pipeline(variant: str = "A", session_id: str = "default", policy_mode: str = "log-only"):
    """
    Construit et retourne un pipeline LangGraph compilé.

    variant      : "A" (baseline), "B" (M1 only), "C" (full HalluGuard)
    session_id   : identifiant de session pour le BeliefState
    policy_mode  : "log-only" | "auto-correct"
    """
    middleware = None
    if variant in ("B", "C"):
        from src.halluguard.middleware import HalluGuardMiddleware
        middleware = HalluGuardMiddleware(
            variant=variant,
            policy_mode=policy_mode,
            session_id=session_id,
            llm_fn=_call_mistral,
        )

    # ------------------------------------------------------------------ #
    # Nœuds du graphe                                                       #
    # ------------------------------------------------------------------ #

    def node_retrieval(state: RAGState) -> RAGState:
        query = state["query"]
        if middleware:
            middleware.pre_node("retrieval", state)

        docs = _retrieve_documents(query)

        if middleware:
            result = middleware.post_node("retrieval", " ".join(docs), state, evidences=docs)
            if result["event"]:
                state["halluguard_events"].append(result["event"])

        state["documents"] = docs
        return state

    def node_reasoning(state: RAGState) -> RAGState:
        query = state["query"]
        docs = state["documents"]
        if middleware:
            middleware.pre_node("reasoning", state)

        context = "\n".join(f"- {d}" for d in docs) if docs else "Aucun document disponible."
        prompt = (
            f"Question : {query}\n\n"
            f"Contexte :\n{context}\n\n"
            f"Analyse la question en te basant sur le contexte. Sois factuel et concis."
        )
        reasoning_text = _call_mistral(prompt)

        if middleware:
            result = middleware.post_node("reasoning", reasoning_text, state, evidences=docs)
            reasoning_text = result["output"] or reasoning_text
            if result["event"]:
                state["halluguard_events"].append(result["event"])

        state["reasoning"] = reasoning_text
        return state

    def node_tool_call(state: RAGState) -> RAGState:
        query = state["query"]
        if middleware:
            middleware.pre_node("tool_call", state)

        # Recherche complémentaire si les documents initiaux sont insuffisants
        supplementary = _retrieve_documents(f"{query} détails supplémentaires")
        tool_output = " ".join(supplementary) if supplementary else ""

        # post_node toujours appelé pour tool_call, même si output vide :
        # un outil silencieux (output="") peut indiquer une T5 non interceptée.
        if middleware:
            tc_output = tool_output or ""
            result = middleware.post_node(
                "tool_call", tc_output, state, evidences=supplementary or []
            )
            tool_output = result["output"] or tool_output
            if result["event"]:
                state["halluguard_events"].append(result["event"])

        state["tool_output"] = tool_output
        return state

    def node_generation(state: RAGState) -> RAGState:
        query = state["query"]
        docs = state["documents"]
        reasoning = state["reasoning"]
        tool_output = state["tool_output"]
        if middleware:
            middleware.pre_node("generation", state)

        context_parts = [f"- {d}" for d in docs]
        if tool_output:
            context_parts.append(f"- [outil] {tool_output}")

        context = "\n".join(context_parts) if context_parts else "Aucun contexte disponible."
        prompt = (
            f"Question : {query}\n\n"
            f"Analyse préliminaire : {reasoning}\n\n"
            f"Contexte complet :\n{context}\n\n"
            f"Génère une réponse finale complète, précise et sourcée."
        )
        answer = _call_mistral(prompt)

        if middleware:
            all_evidences = docs + ([tool_output] if tool_output else [])
            result = middleware.post_node("generation", answer, state, evidences=all_evidences)
            answer = result["output"] or answer
            if result["event"]:
                state["halluguard_events"].append(result["event"])
            middleware.save_session()

        state["answer"] = answer
        return state

    # ------------------------------------------------------------------ #
    # Construction du graphe                                                #
    # ------------------------------------------------------------------ #

    graph = StateGraph(RAGState)
    graph.add_node("retrieval", node_retrieval)
    graph.add_node("reasoning", node_reasoning)
    graph.add_node("tool_call", node_tool_call)
    graph.add_node("generation", node_generation)

    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "reasoning")
    graph.add_edge("reasoning", "tool_call")
    graph.add_edge("tool_call", "generation")
    graph.add_edge("generation", END)

    return graph.compile(), middleware


# ------------------------------------------------------------------ #
# Point d'entrée CLI                                                    #
# ------------------------------------------------------------------ #

def run_query(query: str, variant: str = "C", session_id: str = "default") -> Dict:
    pipeline, middleware = build_rag_pipeline(variant=variant, session_id=session_id)

    initial_state: RAGState = {
        "query": query,
        "documents": [],
        "reasoning": "",
        "tool_output": "",
        "answer": "",
        "halluguard_events": [],
    }

    t0 = time.time()
    final_state = pipeline.invoke(initial_state)
    elapsed = time.time() - t0

    result = {
        "query": query,
        "answer": final_state["answer"],
        "elapsed_seconds": round(elapsed, 2),
        "variant": variant,
        "halluguard_events": final_state.get("halluguard_events", []),
    }

    if middleware:
        result["overhead_stats"] = middleware.overhead_stats()
        if variant in ("B", "C"):
            result["policy_stats"] = middleware.policy.stats()

    return result


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Qu'est-ce qu'une hallucination dans un LLM ?"
    variant = "C"
    print(f"\nVariante : {variant}")
    print(f"Question : {query}\n")
    result = run_query(query, variant=variant)
    print(f"Réponse  : {result['answer']}")
    print(f"Temps    : {result['elapsed_seconds']}s")
    if result.get("halluguard_events"):
        print(f"Événements HalluGuard : {len(result['halluguard_events'])}")
