"""
BeliefState — memoire des faits etablis dans la session courante.

Interface JSON (CONTEXT.md Interface 3) :
  {"session_id": str, "facts": [{"fact_id", "content", "confidence", "step", "status"}]}

- max 50 faits actifs, expulsion par score = confidence x recentness
- Recherche cosinus top-K=5 (avec embedding) ou iteration complete (sans)
- check_temporal_coherence(claim, verifier) : detection de contradiction
- save_to_disk(path) / load_from_disk(path) : persistance JSON
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MAX_FACTS = 50
TOP_K = 5


@dataclass
class Fact:
    fact_id: str
    content: str
    confidence: float         # [0, 1]
    step: int                 # etape du pipeline (1, 2, ...)
    status: str               # "active" | "invalidated"
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def to_dict(self) -> Dict:
        return {
            "fact_id": self.fact_id,
            "content": self.content,
            "confidence": self.confidence,
            "step": self.step,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Fact":
        return cls(
            fact_id=d["fact_id"],
            content=d["content"],
            confidence=d["confidence"],
            step=d["step"],
            status=d["status"],
            timestamp=d.get("timestamp", time.time()),
            embedding=None,
        )


class BeliefState:
    def __init__(self, session_id: str = "default", embedding_model=None) -> None:
        self.session_id = session_id
        self._facts: List[Fact] = []
        self._counter: int = 0
        self._embedding_model = embedding_model

    # ------------------------------------------------------------------ #
    # Ajout de faits                                                        #
    # ------------------------------------------------------------------ #

    def add_fact(self, content: str, confidence: float, step: int) -> str:
        """
        Ajoute un fait. Expulse le moins pertinent si MAX_FACTS actifs atteint.
        Retourne le fact_id genere.
        """
        if len(self) >= MAX_FACTS:
            self._evict()

        self._counter += 1
        fact_id = f"{self.session_id}_{self._counter:04d}"
        embedding = self._embed(content)

        fact = Fact(
            fact_id=fact_id,
            content=content,
            confidence=confidence,
            step=step,
            status="active",
            embedding=embedding,
        )
        self._facts.append(fact)
        return fact_id

    def _evict(self) -> Fact:
        """
        Marque comme 'invalidated' le fait actif avec le score le plus bas.
        Le fait reste dans self._facts (visible dans le JSON avec status='invalidated').
        Score d'expulsion = confidence x recentness.
        """
        now = time.time()
        active = [f for f in self._facts if f.status == "active"]
        scored = [
            (f, f.confidence * (1.0 / (1.0 + now - f.timestamp)))
            for f in active
        ]
        scored.sort(key=lambda x: x[1])
        victim = scored[0][0]
        victim.status = "invalidated"
        return victim

    # ------------------------------------------------------------------ #
    # Recherche semantique top-K                                            #
    # ------------------------------------------------------------------ #

    def top_k_similar(self, query: str, k: int = TOP_K) -> List[Tuple[Fact, float]]:
        """
        Retourne les k faits actifs les plus proches par similarite cosinus.
        Si pas d'embedding disponible, retourne tous les faits avec score 0.
        """
        active = [f for f in self._facts if f.status == "active"]
        if not active:
            return []

        q_emb = self._embed(query)
        if q_emb is None or not any(f.embedding for f in active):
            return [(f, 0.0) for f in active]

        scored = []
        for f in active:
            if f.embedding is not None:
                sim = self._cosine(q_emb, f.embedding)
            else:
                sim = 0.0
            scored.append((f, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------ #
    # Coherence temporelle (M2)                                             #
    # ------------------------------------------------------------------ #

    def check_temporal_coherence(self, claim: str, verifier=None) -> Dict:
        """
        Detecte si claim contredit un fait existant.

        Sans verifier : seuil cosinus > 0.9 (approximation)
        Avec verifier : top-K cosinus puis NLI contradiction

        Retourne : {"conflict": bool, "fact_id": str|None, "score": float}
        """
        active = [f for f in self._facts if f.status == "active"]
        if not active:
            return {"conflict": False, "fact_id": None, "score": 0.0}

        # Candidats : top-K par cosinus, ou tous si pas d'embedding
        candidates = self.top_k_similar(claim, k=TOP_K)
        if not candidates:
            return {"conflict": False, "fact_id": None, "score": 0.0}

        if verifier is None:
            best_fact, best_sim = candidates[0]
            conflict = best_sim > 0.9
            return {
                "conflict": conflict,
                "fact_id": best_fact.fact_id if conflict else None,
                "score": round(best_sim, 3),
            }

        # Avec verifier NLI
        max_score = 0.0
        conflict_fact: Optional[Fact] = None

        for fact, sim in candidates:
            # Pre-filtre lexical quand pas d'embedding (sim==0.0) :
            # evite les faux positifs NLI sur des phrases sans rapport.
            if sim == 0.0 and self._jaccard(claim, fact.content) < 0.1:
                continue

            try:
                result = verifier.nli(claim, fact.content)
            except Exception:
                # Verifier indisponible : on ignore ce candidat sans crasher
                continue

            if result["label"] == "contradiction" and result["score"] > max_score:
                max_score = result["score"]
                conflict_fact = fact

        return {
            "conflict": max_score > 0.5,
            "fact_id": conflict_fact.fact_id if conflict_fact else None,
            "score": round(max_score, 3),
        }

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        """
        Similarite Jaccard sur les mots de contenu uniquement (stopwords exclus).
        Pre-filtre rapide avant NLI pour eviter les faux positifs sur phrases sans rapport.
        """
        _SW = {
            "is", "are", "was", "were", "be", "been", "being",
            "the", "a", "an", "this", "that", "these", "those",
            "in", "on", "at", "to", "of", "and", "or", "not",
            "it", "its", "by", "for", "with", "as", "do", "does",
            "have", "has", "had", "will", "can", "may", "all",
        }
        import re as _re
        _tok = lambda s: set(_re.findall(r"\b[a-záàâéèêîïôùûç]+\b", s.lower())) - _SW
        sa = _tok(a)
        sb = _tok(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # ------------------------------------------------------------------ #
    # Persistance JSON                                                       #
    # ------------------------------------------------------------------ #

    def save_to_disk(self, path: str) -> Path:
        """
        Sauvegarde au format Interface 3 (CONTEXT.md).
        Cree les dossiers parents si necessaire.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.session_id,
            "facts": [f.to_dict() for f in self._facts],
            "_counter": self._counter,
        }
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    @classmethod
    def load_from_disk(cls, path: str, embedding_model=None) -> "BeliefState":
        """
        Charge depuis le JSON. Recompute les embeddings si embedding_model fourni.
        Retourne un BeliefState vide si le fichier n'existe pas.
        """
        p = Path(path)
        if not p.exists():
            session_id = p.stem
            return cls(session_id=session_id, embedding_model=embedding_model)

        data = json.loads(p.read_text(encoding="utf-8"))
        bs = cls(session_id=data["session_id"], embedding_model=embedding_model)
        bs._counter = data.get("_counter", 0)

        for fd in data.get("facts", []):
            fact = Fact.from_dict(fd)
            if embedding_model is not None:
                fact.embedding = bs._embed(fact.content)
            bs._facts.append(fact)

        return bs

    # ------------------------------------------------------------------ #
    # Accesseurs                                                             #
    # ------------------------------------------------------------------ #

    def get_active_facts(self) -> List[Fact]:
        return [f for f in self._facts if f.status == "active"]

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        for f in self._facts:
            if f.fact_id == fact_id:
                return f
        return None

    def __len__(self) -> int:
        return len(self.get_active_facts())

    # ------------------------------------------------------------------ #
    # Embedding / cosinus                                                    #
    # ------------------------------------------------------------------ #

    def _embed(self, text: str) -> Optional[List[float]]:
        if self._embedding_model is None:
            return None
        vec = self._embedding_model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        va, vb = np.array(a), np.array(b)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def summary(self) -> Dict:
        return {
            "session_id": self.session_id,
            "active_facts": len(self),
            "total_in_memory": len(self._facts),
            "max_facts": MAX_FACTS,
            "counter": self._counter,
        }
