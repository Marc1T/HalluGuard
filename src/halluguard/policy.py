"""
PolicyEngine — repond quand une hallucination est detectee.

Mode "log"          : journalise dans results/logs/halluguard.log (JSONL), ne bloque pas.
Mode "auto-correct" : appelle Mistral via Ollama pour corriger, mesure le TTR.

Champs du log : timestamp, node_id, hallucination_type, score, claim_preview,
                corrected, ttr_seconds, correction_preview.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

PolicyMode = Literal["log", "auto-correct"]

DEFAULT_LOG_PATH = "results/logs/halluguard.log"


@dataclass
class HallucinationEvent:
    event_id: str
    timestamp: str            # ISO-8601 lisible
    node_id: str              # identifiant du noeud (retrieval, reasoning, tool_call, generation)
    hallucination_type: Optional[str]   # T1-T5
    score: float
    claim_preview: str        # 120 premiers caracteres du claim
    corrected: bool = False
    ttr_seconds: Optional[float] = None
    correction_preview: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class PolicyEngine:
    def __init__(
        self,
        mode: PolicyMode = "log",
        log_path: str = DEFAULT_LOG_PATH,
        llm_model: str = "mistral",
    ) -> None:
        self.mode = mode
        self.log_path = Path(log_path)
        self.llm_model = llm_model
        self._events: List[HallucinationEvent] = []
        self._event_counter = 0
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Point d'entree principal                                              #
    # ------------------------------------------------------------------ #

    def handle(
        self,
        verification_result: Dict,
        claim: str,
        node_id: str,
        evidences: Optional[List[str]] = None,
        reinvoke_fn=None,
    ) -> Dict:
        """
        Traite un resultat de verification.

        verification_result : sortie de LightweightVerifier.verify()
          label attendu : "hallucinated" (ou "contradiction" pour compat)
        claim       : texte verifie
        node_id     : noeud source ("retrieval", "reasoning", "tool_call", "generation")
        evidences   : documents RAG (pour auto-correct)
        reinvoke_fn : callable(prompt: str) -> str  (Mistral via Ollama)

        Retourne : {"action": "pass"|"logged"|"corrected", "output": str|None, "event": dict|None}
        """
        label = verification_result.get("label", "correct")
        score = verification_result.get("score", 0.0)
        h_type = verification_result.get("hallucination_type")

        # Accepte "hallucinated" (interface Prompt 5) et "contradiction" (compat)
        is_hallucination = label in ("hallucinated", "contradiction") and score > 0.5

        if not is_hallucination:
            return {"action": "pass", "output": None, "event": None}

        event = self._log_event(claim, node_id, score, h_type)

        if self.mode == "log" or reinvoke_fn is None:
            return {"action": "logged", "output": None, "event": event.to_dict()}

        # Mode auto-correct
        correction, ttr = self._auto_correct(claim, evidences or [], reinvoke_fn)
        event.corrected = True
        event.ttr_seconds = round(ttr, 2)
        event.correction_preview = correction[:120] if correction else None
        self._update_last_log_entry(event)

        return {"action": "corrected", "output": correction, "event": event.to_dict()}

    # ------------------------------------------------------------------ #
    # Logging JSONL                                                          #
    # ------------------------------------------------------------------ #

    def _log_event(
        self,
        claim: str,
        node_id: str,
        score: float,
        h_type: Optional[str],
    ) -> HallucinationEvent:
        self._event_counter += 1
        event = HallucinationEvent(
            event_id=f"evt_{self._event_counter:04d}",
            timestamp=datetime.now().isoformat(timespec="seconds"),
            node_id=node_id,
            hallucination_type=h_type,
            score=round(score, 3),
            claim_preview=claim[:120],
        )
        self._events.append(event)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return event

    def _update_last_log_entry(self, event: HallucinationEvent) -> None:
        """Remplace la derniere ligne du log par l'evenement mis a jour (correction ajoutee)."""
        text = self.log_path.read_text(encoding="utf-8")
        lines = text.strip().split("\n")
        lines[-1] = json.dumps(event.to_dict(), ensure_ascii=False)
        self.log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Auto-correction via Mistral                                            #
    # ------------------------------------------------------------------ #

    def _auto_correct(
        self, claim: str, evidences: List[str], reinvoke_fn
    ) -> tuple[Optional[str], float]:
        """
        Appelle reinvoke_fn (Mistral) avec un prompt enrichi.
        Retourne (texte_corrige, ttr_secondes).
        """
        evidence_block = (
            "\n".join(f"- {e}" for e in evidences[:3])
            if evidences else "(aucune source disponible)"
        )
        prompt = (
            f"The following answer was flagged as potentially incorrect.\n\n"
            f"Original answer: {claim}\n\n"
            f"Available sources:\n{evidence_block}\n\n"
            f"Provide a corrected, factual answer based strictly on the sources above. "
            f"Be concise."
        )
        t0 = time.time()
        try:
            correction = reinvoke_fn(prompt)
        except Exception:
            correction = None
        ttr = time.time() - t0
        return correction, ttr

    # ------------------------------------------------------------------ #
    # Statistiques                                                           #
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict:
        total = len(self._events)
        corrected = sum(1 for e in self._events if e.corrected)
        ttrs = [e.ttr_seconds for e in self._events if e.ttr_seconds is not None]
        by_type: Dict[str, int] = {}
        for e in self._events:
            k = e.hallucination_type or "unknown"
            by_type[k] = by_type.get(k, 0) + 1

        return {
            "total_events": total,
            "corrected": corrected,
            "avg_ttr_seconds": round(sum(ttrs) / len(ttrs), 2) if ttrs else None,
            "by_hallucination_type": by_type,
            "mode": self.mode,
        }
