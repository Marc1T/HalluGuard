# RAG Tools — indexation ChromaDB + recherche, exposés via MCP et appelés par l'agent LangGraph.
# Construit la collection "halluguard_docs" à partir de data/documents/*.txt et
# fournit rag_search(query) (point d'entrée de la délégation A2A agent RAG -> HalluGuard).
#
# Réindexer (1re installation / ChromaDB vide) :
#   venv\Scripts\python.exe src/tools/rag_tools.py
from __future__ import annotations

import re
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "data" / "documents"
CHROMA_PATH = str(ROOT / "data" / "chromadb")
COLLECTION = "halluguard_docs"


def _chunk_text(text: str, max_chars: int = 180) -> List[str]:
    """Découpe en chunks : par paragraphes, puis par phrases si trop long."""
    chunks: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        # paragraphe long : regrouper les phrases jusqu'à max_chars
        buf = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if len(buf) + len(sent) + 1 > max_chars and buf:
                chunks.append(buf.strip())
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf.strip():
            chunks.append(buf.strip())
    return chunks


def _client():
    import chromadb
    return chromadb.PersistentClient(path=CHROMA_PATH)


def build_index(reset: bool = True) -> int:
    """(Re)construit la collection halluguard_docs depuis data/documents/. Retourne le nb de chunks."""
    client = _client()
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    coll = client.get_or_create_collection(COLLECTION)

    docs, ids, metas = [], [], []
    for txt in sorted(DOCS_DIR.glob("*.txt")):
        content = txt.read_text(encoding="utf-8")
        for i, chunk in enumerate(_chunk_text(content)):
            docs.append(chunk)
            ids.append(f"{txt.stem}_{i:03d}")
            metas.append({"source": txt.name})

    if docs:
        coll.add(documents=docs, ids=ids, metadatas=metas)
    print(f"[rag_tools] Index '{COLLECTION}' construit : {len(docs)} chunks "
          f"depuis {len(list(DOCS_DIR.glob('*.txt')))} fichiers ({CHROMA_PATH})")
    return len(docs)


def rag_search(query: str, n: int = 3) -> List[str]:
    """Recherche les n chunks les plus pertinents (utilisé par l'agent / MCP)."""
    try:
        coll = _client().get_collection(COLLECTION)
        res = coll.query(query_texts=[query], n_results=n)
        return res["documents"][0] if res["documents"] else []
    except Exception:
        return []


if __name__ == "__main__":
    n = build_index(reset=True)
    if n:
        sample = rag_search("Quelle architecture utilisent les LLM ?", n=2)
        print(f"[rag_tools] Test requête -> {len(sample)} résultats. Exemple : {sample[0][:80] if sample else '(vide)'}...")
