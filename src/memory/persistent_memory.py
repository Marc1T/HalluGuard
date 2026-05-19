"""
PersistentMemory — gestion de la memoire inter-sessions.

Sauvegarde le BeliefState au format JSON sur disque (data/memory/session_{id}.json).
Recharge automatiquement le BeliefState au demarrage si une session precedente existe.
Indexe les embeddings des faits dans ChromaDB pour la recherche semantique persistante.
"""

from __future__ import annotations
from pathlib import Path

DEFAULT_MEMORY_DIR = "data/memory"


class PersistentMemory:
    """
    Utilitaire de persistance pour BeliefState.
    Delegue save/load a BeliefState mais centralise la gestion des chemins.
    """

    def __init__(self, memory_dir: str = DEFAULT_MEMORY_DIR) -> None:
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Chemins                                                               #
    # ------------------------------------------------------------------ #

    def path_for(self, session_id: str) -> Path:
        return self.memory_dir / f"session_{session_id}.json"

    def session_exists(self, session_id: str) -> bool:
        return self.path_for(session_id).exists()

    def list_sessions(self):
        return [p.stem.replace("session_", "") for p in self.memory_dir.glob("session_*.json")]

    # ------------------------------------------------------------------ #
    # Save / Load via BeliefState                                           #
    # ------------------------------------------------------------------ #

    def save(self, belief_state) -> Path:
        """Sauvegarde un BeliefState dans memory_dir/session_{id}.json."""
        path = self.path_for(belief_state.session_id)
        return belief_state.save_to_disk(str(path))

    def load(self, session_id: str, embedding_model=None):
        """
        Charge un BeliefState depuis le disque.
        Retourne un BeliefState vide si la session n'existe pas.
        """
        from src.halluguard.belief_state import BeliefState
        path = self.path_for(session_id)
        return BeliefState.load_from_disk(str(path), embedding_model=embedding_model)

    def load_or_create(self, session_id: str, embedding_model=None):
        """Charge si existe, sinon cree un nouveau BeliefState."""
        from src.halluguard.belief_state import BeliefState
        if self.session_exists(session_id):
            return self.load(session_id, embedding_model=embedding_model)
        return BeliefState(session_id=session_id, embedding_model=embedding_model)

    def delete(self, session_id: str) -> bool:
        """Supprime le fichier de session. Retourne True si supprime."""
        path = self.path_for(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    # ------------------------------------------------------------------ #
    # ChromaDB (persistance semantique des embeddings)                      #
    # ------------------------------------------------------------------ #

    def index_to_chromadb(self, belief_state, collection_name: str = "belief_state") -> None:
        """
        Indexe les faits actifs dans ChromaDB pour recherche inter-sessions.
        Necessite chromadb installe dans le venv.
        """
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.memory_dir / "chromadb"))
            collection = client.get_or_create_collection(collection_name)

            facts = belief_state.get_active_facts()
            if not facts:
                return

            ids = [f.fact_id for f in facts]
            documents = [f.content for f in facts]
            metadatas = [
                {"session_id": belief_state.session_id, "confidence": f.confidence, "step": f.step}
                for f in facts
            ]

            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        except ImportError:
            pass  # ChromaDB optionnel

    def search_chromadb(self, query: str, collection_name: str = "belief_state", n: int = 5):
        """Recherche semantique dans les faits persistes (ChromaDB)."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.memory_dir / "chromadb"))
            collection = client.get_collection(collection_name)
            results = collection.query(query_texts=[query], n_results=n)
            return results
        except Exception:
            return {"documents": [], "metadatas": [], "distances": []}
