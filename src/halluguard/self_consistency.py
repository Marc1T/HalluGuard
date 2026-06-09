"""
self_consistency.py — Détection d'incertitude par self-consistency (axe Sujet 15).

Principe : un LLM sûr de lui produit des réponses cohérentes lorsqu'on l'échantillonne
plusieurs fois (température > 0)~; face à une question piégée (fausse prémisse, fait
obscur), il fabule et ses réponses divergent. On échantillonne k fois, puis on mesure
l'accord par la similarité cosinus moyenne des embeddings des réponses (score de
self-consistency dans [0, 1]). Un score bas signale une hallucination potentielle.

Back-ends LLM :
  - "mistral_api"  : API Mistral (clé dans .env : MISTRAL_API_KEY) — rapide
  - "ollama"       : Mistral local via Ollama — conforme à la contrainte "LLM local"

Aucune clé n'est codée en dur (lecture via .env, gitignoré).
"""
from __future__ import annotations
import os
from itertools import combinations
from typing import List

_EMB_MODEL = None  # singleton sentence-transformers


def _embedder():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMB_MODEL


# ------------------------------------------------------------------ #
# Échantillonnage du LLM                                              #
# ------------------------------------------------------------------ #

def sample_answers(query: str, k: int = 5, temperature: float = 0.7,
                   backend: str = "mistral_api", model: str | None = None,
                   max_tokens: int = 60) -> List[str]:
    """Échantillonne k réponses du LLM à la même question."""
    if backend == "mistral_api":
        return _sample_mistral_api(query, k, temperature, model or "open-mistral-7b", max_tokens)
    if backend == "ollama":
        return _sample_ollama(query, k, temperature, model or "mistral", max_tokens)
    raise ValueError(f"backend inconnu : {backend}")


def _sample_mistral_api(query, k, temperature, model, max_tokens) -> List[str]:
    import requests
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY absent (.env).")
    out = []
    for _ in range(k):
        r = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user", "content": query}],
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=60,
        )
        r.raise_for_status()
        out.append(r.json()["choices"][0]["message"]["content"].strip())
    return out


def _sample_ollama(query, k, temperature, model, max_tokens) -> List[str]:
    import ollama
    out = []
    for _ in range(k):
        resp = ollama.chat(model=model, messages=[{"role": "user", "content": query}],
                           options={"temperature": temperature, "num_predict": max_tokens})
        out.append(resp["message"]["content"].strip())
    return out


# ------------------------------------------------------------------ #
# Score de self-consistency                                          #
# ------------------------------------------------------------------ #

def consistency_score(answers: List[str]) -> float:
    """Similarité cosinus moyenne entre toutes les paires de réponses, dans [0, 1]."""
    answers = [a for a in answers if a.strip()]
    if len(answers) < 2:
        return 1.0
    import numpy as np
    emb = _embedder().encode(answers, normalize_embeddings=True)
    sims = [float(np.dot(emb[i], emb[j])) for i, j in combinations(range(len(emb)), 2)]
    return round(float(np.mean(sims)), 4)


def is_uncertain(answers: List[str], threshold: float = 0.7) -> bool:
    """True si le LLM est incohérent (self-consistency < seuil) → hallucination probable."""
    return consistency_score(answers) < threshold
