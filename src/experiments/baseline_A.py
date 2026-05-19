"""
Expérimentation Baseline — Variante A (pipeline RAG sans HalluGuard).

Étapes :
  1. Indexe data/documents/*.txt dans ChromaDB (collection halluguard_docs)
  2. Lance le pipeline LangGraph 4 noeuds sur 3 questions via Ollama/Mistral
  3. Mesure la latence totale par question
  4. Sauvegarde dans results/baseline_latency.json

Lancer : python src/experiments/baseline_A.py
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DOCS_DIR     = ROOT / "data" / "documents"
CHROMA_PATH  = ROOT / "data" / "chromadb"
COLLECTION   = "halluguard_docs"
RESULTS_FILE = ROOT / "results" / "baseline_latency.json"
SEPARATOR    = "-" * 60

QUESTIONS = [
    "Qu'est-ce qu'un LLM ?",
    "Comment fonctionne le RAG ?",
    "Quelles sont les limites des agents IA ?",
]


# ------------------------------------------------------------------ #
# Étape 1 — Indexation ChromaDB                                        #
# ------------------------------------------------------------------ #

def index_documents() -> int:
    """Indexe les .txt dans ChromaDB. Retourne le nb de chunks indexés."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION)

    docs, ids, metas = [], [], []
    chunk_id = 0
    for txt_file in sorted(DOCS_DIR.glob("*.txt")):
        for line in txt_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                docs.append(line)
                ids.append(f"chunk_{chunk_id:04d}")
                metas.append({"source": txt_file.name})
                chunk_id += 1

    if docs:
        collection.add(documents=docs, ids=ids, metadatas=metas)

    return len(docs)


# ------------------------------------------------------------------ #
# Étape 2 — Pipeline Variante A                                        #
# ------------------------------------------------------------------ #

def run_baseline() -> list[dict]:
    """Exécute le pipeline Variante A sur les 3 questions."""
    from src.agents.rag_agent import build_rag_pipeline

    pipeline, _ = build_rag_pipeline(variant="A")

    results = []
    for q in QUESTIONS:
        state = {
            "query": q,
            "documents": [],
            "reasoning": "",
            "tool_output": "",
            "answer": "",
            "halluguard_events": [],
        }

        t0 = time.time()
        final = pipeline.invoke(state)
        latency = round(time.time() - t0, 2)

        results.append({
            "query": q,
            "answer": final["answer"],
            "latency_seconds": latency,
            "documents_used": len(final.get("documents", [])),
            "variant": "A",
        })

    return results


# ------------------------------------------------------------------ #
# Main                                                                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 60)
    print("  HalluGuard - Baseline Variante A (sans HalluGuard)")
    print("  Pipeline : retrieval -> reasoning -> tool_call -> generation")
    print("=" * 60)

    # 1. Indexation
    print("\n[1/2] Indexation des documents dans ChromaDB...")
    print(SEPARATOR)
    nb = index_documents()
    print(f"  Chunks indexes  : {nb}")
    print(f"  Collection      : {COLLECTION}")
    print(f"  ChromaDB path   : {CHROMA_PATH}")

    # 2. Pipeline
    print(f"\n[2/2] Pipeline sur {len(QUESTIONS)} questions (Mistral x2 par question)...")
    print(SEPARATOR)

    results = run_baseline()

    for i, r in enumerate(results, 1):
        print(f"\nQ{i} : {r['query']}")
        preview = r["answer"][:250] + ("..." if len(r["answer"]) > 250 else "")
        print(f"  Reponse  : {preview}")
        print(f"  Latence  : {r['latency_seconds']}s")
        print(f"  Docs RAG : {r['documents_used']}")

    # 3. Sauvegarde
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    latencies = [r["latency_seconds"] for r in results]
    payload = {
        "variant": "A",
        "description": "Baseline sans HalluGuard — pipeline LangGraph 4 noeuds",
        "questions": results,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 2),
        "min_latency_seconds": min(latencies),
        "max_latency_seconds": max(latencies),
    }
    RESULTS_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{'='*60}")
    print(f"  Resultats sauvegardes : {RESULTS_FILE.name}")
    print(f"  Latence moyenne       : {payload['avg_latency_seconds']}s")
    print(f"  Latence min/max       : {payload['min_latency_seconds']}s / {payload['max_latency_seconds']}s")
    print(f"{'='*60}")
