"""
LightweightVerifier — vérificateur NLI adapté CPU, sans entraînement.

Modèle : cross-encoder/nli-MiniLM2-L6-H768 (117M params, pré-entraîné MNLI).
Téléchargement automatique depuis HuggingFace au premier appel.

Interface API (CONTEXT.md §Interface 2) :
  Input  : {"claim": str, "evidences": [str]}
  Output : {"score": float, "label": str, "confidence": float, "hallucination_type": str}

Labels verify() : "correct" | "hallucinated"
Labels nli()    : "entailment" | "neutral" | "contradiction"
Fallback si evidences vide : score=0.5, insufficient_evidence=True.
"""

from __future__ import annotations
import re
import time
from typing import Dict, List, Optional

# Ordre des labels pour cross-encoder/nli-MiniLM2-L6-H768 (MNLI training)
_DEFAULT_LABELS = ["contradiction", "entailment", "neutral"]

# Heuristiques T1-T5
_TEMPORAL_KW = re.compile(
    r"\b(hier|aujourd'hui|demain|récemment|actuellement|en \d{4}|depuis|avant|après)\b",
    re.IGNORECASE,
)
_CAUSAL_KW = re.compile(r"\b(donc|parce que|car|ainsi|entraîne|provoque|cause)\b", re.IGNORECASE)
_DELEGATION_KW = re.compile(
    r"\b(outil|tool|api|appel|call|délègue|recherche|retrieval)\b", re.IGNORECASE
)


class LightweightVerifier:
    def __init__(self, model_name: str = "cross-encoder/nli-MiniLM2-L6-H768") -> None:
        self.model_name = model_name
        self._model = None
        self._label_names: List[str] = _DEFAULT_LABELS

    # ------------------------------------------------------------------ #
    # Chargement paresseux                                                  #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers.cross_encoder import CrossEncoder
        self._model = CrossEncoder(self.model_name)
        # Lire l'ordre réel des labels depuis la config du modèle
        try:
            id2label = self._model.config.id2label
            self._label_names = [id2label[i] for i in sorted(id2label.keys())]
        except Exception:
            self._label_names = _DEFAULT_LABELS

    # ------------------------------------------------------------------ #
    # API principale — Interface 2 (CONTEXT.md)                             #
    # ------------------------------------------------------------------ #

    def verify(self, claim: str, evidences: List[str], node_type: str = "generation",
               threshold: Optional[float] = None) -> Dict:
        """
        Vérifie si claim est supporté par les evidences.

        Règle de décision :
          - threshold is None (défaut) : label="hallucinated" si P(contradiction) > P(entailment)
            (argmax entre contradiction et entailment — règle historique).
          - threshold fixé (ex. calibré par conformal prediction) : label="hallucinated"
            si P(contradiction) > threshold.

        Returns:
            score              : probabilité du label gagnant [0, 1]
            label              : "correct" | "hallucinated"
            confidence         : même valeur que score (compatibilité Interface 2)
            p_contradiction    : P(contradiction) — axe de détection cohérent (calibration/conformal)
            p_entailment       : P(entailment)
            p_neutral          : P(neutral)
            hallucination_type : "T1"–"T5" | None
            insufficient_evidence : True si evidences vide
            latency_ms         : temps d'inférence en ms
        """
        if not evidences:
            # Pas d'evidence → impossible de conclure : on renvoie "insufficient_evidence"
            # (non détecté, score neutre 0.5) — distinct de "correct" pour ne pas masquer
            # les cas où aucun document n'a été récupéré (T1 silencieux).
            return {
                "score": 0.5,
                "label": "insufficient_evidence",
                "confidence": 0.5,
                "hallucination_type": None,
                "insufficient_evidence": True,
                "latency_ms": 0.0,
            }

        self._load()
        import numpy as np

        t0 = time.time()
        pairs = [(claim, ev) for ev in evidences]
        raw = self._model.predict(pairs)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        probs = self._softmax(raw)  # (n, 3)

        contra_idx = self._idx("contradiction")
        entail_idx = self._idx("entailment")

        neutral_idx = self._idx("neutral")
        max_contra = float(probs[:, contra_idx].max())
        max_entail = float(probs[:, entail_idx].max())
        max_neutral = float(probs[:, neutral_idx].max())

        if threshold is None:
            is_hallu = max_contra > max_entail
        else:
            is_hallu = max_contra > threshold
        label = "hallucinated" if is_hallu else "correct"
        score = max_contra if is_hallu else max_entail

        latency = (time.time() - t0) * 1000
        h_type = self._classify_type(claim, node_type, label)

        return {
            "score": round(score, 3),
            "label": label,
            "confidence": round(max(max_contra, max_entail), 3),
            "p_contradiction": round(max_contra, 4),
            "p_entailment": round(max_entail, 4),
            "p_neutral": round(max_neutral, 4),
            "hallucination_type": h_type,
            "insufficient_evidence": False,
            "latency_ms": round(latency, 1),
        }

    # ------------------------------------------------------------------ #
    # NLI brut premise vs hypothesis                                        #
    # ------------------------------------------------------------------ #

    def nli(self, premise: str, hypothesis: str) -> Dict:
        """
        NLI direct entre deux phrases.

        Returns:
            label      : "entailment" | "neutral" | "contradiction"
            score      : probabilité du label gagnant [0, 1]
            latency_ms : temps d'inférence en ms
        """
        self._load()
        import numpy as np

        t0 = time.time()
        raw = self._model.predict([(premise, hypothesis)])
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        probs = self._softmax(raw)[0]

        best_idx = int(np.argmax(probs))
        latency = (time.time() - t0) * 1000

        return {
            "label": self._label_names[best_idx],
            "score": round(float(probs[best_idx]), 3),
            "latency_ms": round(latency, 1),
        }

    # ------------------------------------------------------------------ #
    # Classification T1-T5                                                  #
    # ------------------------------------------------------------------ #

    def _classify_type(self, claim: str, node_type: str, label: str) -> Optional[str]:
        if label != "hallucinated":
            return None
        if node_type == "retrieval":
            return "T1"
        if node_type == "tool_call" or _DELEGATION_KW.search(claim):
            return "T5"
        if _TEMPORAL_KW.search(claim):
            return "T2"
        if _CAUSAL_KW.search(claim):
            return "T4"
        if node_type == "reasoning":
            return "T3"
        return "T1"

    # ------------------------------------------------------------------ #
    # Utilitaires                                                           #
    # ------------------------------------------------------------------ #

    def _idx(self, name: str) -> int:
        for i, lbl in enumerate(self._label_names):
            if name.lower() in lbl.lower():
                return i
        return 0

    @staticmethod
    def _softmax(scores):
        import numpy as np
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def is_loaded(self) -> bool:
        return self._model is not None
