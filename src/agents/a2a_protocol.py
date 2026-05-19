"""
a2a_protocol.py — Communication A2A formalisée entre l'agent RAG et HalluGuard.

Inspiré du protocole Agent-to-Agent (A2A) de Google DeepMind (2024).
Références : https://github.com/google-deepmind/a2a

Concepts implémentés :
  - AgentCard  : carte d'identité d'un agent (capacités, skills, endpoint)
  - TaskObject : unité de travail échangée entre agents (states : submitted→working→completed|failed)
  - Message / Part : contenu structuré d'un échange
  - Artifact   : résultat produit par l'agent destinataire
  - delegate_verification() : l'agent RAG délègue la vérification à HalluGuard via MCP

Journal des échanges : results/logs/a2a_exchanges.log (JSONL)
"""

from __future__ import annotations

import importlib
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# ------------------------------------------------------------------ #
# Chemins + sys.path                                                    #
# ------------------------------------------------------------------ #

_ROOT    = Path(__file__).resolve().parents[2]
_A2A_LOG = _ROOT / "results" / "logs" / "a2a_exchanges.log"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_A2A_LOG.parent.mkdir(parents=True, exist_ok=True)


# ================================================================== #
# SECTION 1 — AgentCard (carte d'identité A2A)                        #
# ================================================================== #

@dataclass
class Skill:
    """
    Capacité atomique exposée par un agent A2A.
    Correspond à un outil MCP dans l'implémentation HalluGuard.
    """
    id: str
    name: str
    description: str
    input_modes: List[str]   # ex. ["text", "json"]
    output_modes: List[str]  # ex. ["json"]
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AgentCapabilities:
    """
    Flags de capacités A2A standard.
    Voir : https://google.github.io/A2A/#/documentation?id=agentcapabilities
    """
    streaming: bool          = False  # SSE streaming non implémenté (CPU-only)
    push_notifications: bool = False  # pas de webhook sortant
    state_transition_history: bool = True  # historique des états de tâche conservé

    def to_dict(self) -> Dict:
        return {
            "streaming":               self.streaming,
            "pushNotifications":       self.push_notifications,
            "stateTransitionHistory":  self.state_transition_history,
        }


@dataclass
class AgentCard:
    """
    Carte d'identité d'un agent A2A.

    Dans le protocole A2A, cette carte est publiée à /.well-known/agent.json
    et permet à d'autres agents de découvrir les capacités de l'agent.
    """
    name: str
    description: str
    version: str
    provider: str
    url: str                      # endpoint (ici : MCP stdio)
    protocol: str                 # "MCP/stdio" | "HTTP/SSE"
    capabilities: AgentCapabilities
    skills: List[Skill]
    default_input_modes: List[str]  = field(default_factory=lambda: ["text", "json"])
    default_output_modes: List[str] = field(default_factory=lambda: ["json"])

    def to_dict(self) -> Dict:
        return {
            "name":               self.name,
            "description":        self.description,
            "version":            self.version,
            "provider":           self.provider,
            "url":                self.url,
            "protocol":           self.protocol,
            "capabilities":       self.capabilities.to_dict(),
            "defaultInputModes":  self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "skills":             [s.to_dict() for s in self.skills],
        }

    def pretty(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ------------------------------------------------------------------ #
# AgentCard HalluGuard (singleton)                                     #
# ------------------------------------------------------------------ #

HALLUGUARD_AGENT_CARD = AgentCard(
    name        = "HalluGuard",
    description = (
        "Agent de détection et mitigation des hallucinations dans les pipelines RAG agentiques. "
        "Expose 4 outils MCP : vérification NLI inter-sources (M1), cohérence temporelle "
        "via BeliefState (M2), audit de faits, et inspection de la mémoire de session."
    ),
    version     = "1.0.0",
    provider    = "ENSAM Meknès — Projet HalluGuard (Sujet 15)",
    url         = "mcp://halluguard/stdio",
    protocol    = "MCP/stdio",
    capabilities = AgentCapabilities(
        streaming=False,
        push_notifications=False,
        state_transition_history=True,
    ),
    skills = [
        Skill(
            id           = "verify_claim",
            name         = "Vérification NLI inter-sources (M1)",
            description  = (
                "Vérifie si un claim est cohérent avec des evidences via NLI "
                "(cross-encoder/nli-MiniLM2-L6-H768). Retourne score, label, "
                "type d'hallucination (T1-T5)."
            ),
            input_modes  = ["json"],
            output_modes = ["json"],
            tags         = ["nli", "hallucination", "verification", "T1", "T3", "T4", "T5"],
        ),
        Skill(
            id           = "check_temporal_coherence",
            name         = "Cohérence temporelle BeliefState (M2)",
            description  = (
                "Détecte les contradictions entre un claim et les faits mémorisés "
                "dans le BeliefState de la session (mémoire épisodique)."
            ),
            input_modes  = ["json"],
            output_modes = ["json"],
            tags         = ["belief_state", "temporal", "memory", "T2"],
        ),
        Skill(
            id           = "add_fact",
            name         = "Ajout de fait dans le BeliefState",
            description  = (
                "Ajoute un fait validé dans le BeliefState persistant (max 50 faits, "
                "politique d'expulsion par confiance × récence)."
            ),
            input_modes  = ["json"],
            output_modes = ["json"],
            tags         = ["belief_state", "memory", "persistence"],
        ),
        Skill(
            id           = "get_belief_state",
            name         = "Inspection du BeliefState",
            description  = (
                "Retourne le BeliefState complet de la session MCP courante "
                "(session_id, faits actifs, max_facts)."
            ),
            input_modes  = ["json"],
            output_modes = ["json"],
            tags         = ["belief_state", "inspection", "audit"],
        ),
    ],
)


RAG_AGENT_CARD = AgentCard(
    name        = "RAGAgent",
    description = "Agent RAG LangGraph 4 nœuds qui délègue la vérification à HalluGuard via A2A/MCP.",
    version     = "1.0.0",
    provider    = "ENSAM Meknès — Projet HalluGuard",
    url         = "internal://rag_agent",
    protocol    = "internal",
    capabilities = AgentCapabilities(streaming=False, push_notifications=False),
    skills       = [],
)


# ================================================================== #
# SECTION 2 — TaskObject (unité de travail A2A)                        #
# ================================================================== #

TaskState = Literal["submitted", "working", "completed", "failed"]


@dataclass
class TextPart:
    """Partie textuelle d'un message A2A."""
    type: str = "text"
    text: str = ""

    def to_dict(self) -> Dict:
        return {"type": self.type, "text": self.text}


@dataclass
class DataPart:
    """Partie structurée (JSON) d'un message A2A."""
    type: str = "data"
    data: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"type": self.type, "data": self.data}


@dataclass
class Message:
    """Message échangé entre deux agents (A2A Message)."""
    role: str           # "user" (initiateur) | "agent" (destinataire)
    parts: List[Any]    # list[TextPart | DataPart]

    def to_dict(self) -> Dict:
        return {
            "role":  self.role,
            "parts": [p.to_dict() for p in self.parts],
        }


@dataclass
class Artifact:
    """Résultat produit par l'agent destinataire (A2A Artifact)."""
    artifact_id: str
    name: str
    parts: List[Any]
    created_at: str = field(default_factory=lambda: _now_iso())

    def to_dict(self) -> Dict:
        return {
            "artifactId": self.artifact_id,
            "name":       self.name,
            "parts":      [p.to_dict() for p in self.parts],
            "createdAt":  self.created_at,
        }


@dataclass
class TaskObject:
    """
    Unité de travail A2A (Task).

    Cycle de vie : submitted → working → completed | failed
    """
    id: str
    from_agent: str             # nom de l'agent émetteur
    to_agent: str               # nom de l'agent destinataire
    skill_id: str               # skill demandé (correspond à un tool MCP)
    status: TaskState           # état courant
    input_message: Message      # message entrant (request)
    output_artifacts: List[Artifact] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)   # historique des transitions
    metadata: Dict = field(default_factory=dict)
    created_at: str  = field(default_factory=lambda: _now_iso())
    updated_at: str  = field(default_factory=lambda: _now_iso())
    completed_at: Optional[str] = None

    def transition(self, new_state: TaskState) -> None:
        """Enregistre une transition d'état."""
        old_state = self.status
        self.status = new_state
        self.updated_at = _now_iso()
        self.history.append({
            "from": old_state,
            "to":   new_state,
            "at":   self.updated_at,
        })
        if new_state in ("completed", "failed"):
            self.completed_at = self.updated_at

    def add_artifact(self, name: str, result: Dict) -> None:
        """Ajoute un artifact de résultat."""
        art = Artifact(
            artifact_id = f"art_{uuid.uuid4().hex[:8]}",
            name        = name,
            parts       = [DataPart(data=result)],
        )
        self.output_artifacts.append(art)

    def to_dict(self) -> Dict:
        return {
            "id":              self.id,
            "fromAgent":       self.from_agent,
            "toAgent":         self.to_agent,
            "skillId":         self.skill_id,
            "status":          self.status,
            "inputMessage":    self.input_message.to_dict(),
            "outputArtifacts": [a.to_dict() for a in self.output_artifacts],
            "history":         self.history,
            "metadata":        self.metadata,
            "createdAt":       self.created_at,
            "updatedAt":       self.updated_at,
            "completedAt":     self.completed_at,
        }


# ================================================================== #
# SECTION 3 — A2ALogger                                                #
# ================================================================== #

class A2ALogger:
    """Journal des échanges inter-agents (JSONL)."""

    def __init__(self, log_path: Path = _A2A_LOG) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_exchange(
        self,
        task: TaskObject,
        duration_ms: float,
        verification_result: Optional[Dict] = None,
    ) -> None:
        entry = {
            "timestamp":           _now_iso(),
            "exchange_id":         f"ex_{uuid.uuid4().hex[:8]}",
            "task_id":             task.id,
            "from_agent":          task.from_agent,
            "to_agent":            task.to_agent,
            "skill_id":            task.skill_id,
            "task_status":         task.status,
            "duration_ms":         round(duration_ms, 1),
            "request_summary":     task.metadata.get("request_summary", ""),
            "verification_result": verification_result,
            "history":             task.history,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def tail(self, n: int = 5) -> List[Dict]:
        """Retourne les n derniers échanges loggés."""
        if not self._path.exists():
            return []
        lines = [l for l in self._path.read_text(encoding="utf-8").strip().splitlines() if l]
        return [json.loads(l) for l in lines[-n:]]


# Instance globale du logger
_logger = A2ALogger()


# ================================================================== #
# SECTION 4 — delegate_verification (cœur du protocole A2A)           #
# ================================================================== #

class _MCPBridge:
    """
    Pont vers le serveur MCP HalluGuard (appels directs sans stdio).
    Partage le verifier NLI déjà chargé pour éviter le double chargement.
    """

    _instance: Optional["_MCPBridge"] = None

    def __new__(cls) -> "_MCPBridge":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._mod = importlib.import_module("src.mcp_servers.halluguard_mcp")
            cls._instance = obj
        return cls._instance

    def call(self, tool_name: str, **kwargs) -> Dict:
        fn = getattr(self._mod, tool_name, None)
        if fn is None:
            raise ValueError(f"Tool MCP inconnu : {tool_name}")
        return fn(**kwargs)


def delegate_verification(
    claim: str,
    evidences: List[str],
    node_type: str = "generation",
    from_agent: str = "RAGAgent",
) -> Dict:
    """
    L'agent RAG délègue la vérification d'un claim à HalluGuard via A2A/MCP.

    Protocole :
      1. RAGAgent crée un TaskObject (submitted)
      2. HalluGuard accepte la tâche (working)
      3. HalluGuard appelle verify_claim via MCP
      4. HalluGuard optionnellement appelle check_temporal_coherence (M2)
      5. La tâche passe en completed (ou failed)
      6. L'échange est journalisé dans a2a_exchanges.log

    Returns:
        dict avec clés : task_id, label, score, hallucination_type,
                         m2_conflict, duration_ms, task_status
    """
    t_start = time.time()

    # ---- Étape 1 : Création de la tâche (RAGAgent → HalluGuard) ---- #
    task = TaskObject(
        id          = f"task_{uuid.uuid4().hex[:12]}",
        from_agent  = from_agent,
        to_agent    = "HalluGuard",
        skill_id    = "verify_claim",
        status      = "submitted",
        input_message = Message(
            role  = "user",
            parts = [
                TextPart(text=claim),
                DataPart(data={
                    "evidences": evidences,
                    "node_type": node_type,
                }),
            ],
        ),
        metadata = {
            "request_summary": f"Vérification: «{claim[:80]}»",
            "n_evidences":     len(evidences),
            "node_type":       node_type,
        },
    )

    bridge = _MCPBridge()
    verification_result: Dict = {}

    try:
        # ---- Étape 2 : HalluGuard accepte (working) ---- #
        task.transition("working")

        # ---- Étape 3 : M1 — NLI inter-sources ---- #
        m1_result = bridge.call(
            "verify_claim",
            claim     = claim,
            evidences = evidences,
            node_type = node_type,
        )

        # ---- Étape 4 : M2 — Cohérence temporelle (toujours tentée) ---- #
        try:
            m2_result = bridge.call("check_temporal_coherence", claim=claim)
            m2_conflict = bool(m2_result.get("conflict", False))
        except Exception:
            m2_result   = {"conflict": False, "fact_id": None, "score": 0.0}
            m2_conflict = False

        # ---- Artifact de résultat ---- #
        combined = {
            "m1": m1_result,
            "m2": m2_result,
            "detected": m1_result.get("label") == "hallucinated" or m2_conflict,
        }
        task.add_artifact(name="verification_result", result=combined)

        # ---- Étape 5 : Tâche complétée ---- #
        task.transition("completed")

        verification_result = {
            "task_id":           task.id,
            "label":             m1_result.get("label", "unknown"),
            "score":             m1_result.get("score", 0.0),
            "hallucination_type": m1_result.get("hallucination_type"),
            "m2_conflict":       m2_conflict,
            "m2_fact_id":        m2_result.get("fact_id"),
            "detected":          combined["detected"],
            "duration_ms":       round((time.time() - t_start) * 1000, 1),
            "task_status":       task.status,
        }

    except Exception as exc:
        task.transition("failed")
        task.metadata["error"] = str(exc)
        verification_result = {
            "task_id":    task.id,
            "label":      "error",
            "error":      str(exc),
            "duration_ms": round((time.time() - t_start) * 1000, 1),
            "task_status": task.status,
        }

    finally:
        duration_ms = (time.time() - t_start) * 1000
        _logger.log_exchange(task, duration_ms, verification_result)

    return verification_result


# ================================================================== #
# Utilitaire                                                           #
# ================================================================== #

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def show_agent_cards() -> None:
    """Affiche les AgentCards des deux agents."""
    print("\n=== AgentCard : RAGAgent ===")
    print(RAG_AGENT_CARD.pretty())
    print("\n=== AgentCard : HalluGuard ===")
    print(HALLUGUARD_AGENT_CARD.pretty())


# ================================================================== #
# Point d'entrée — démonstration A2A                                  #
# ================================================================== #

if __name__ == "__main__":
    SEPARATOR = "-" * 60

    print("=" * 60)
    print("  HalluGuard — Simulation A2A explicite")
    print("  RAGAgent  ->  HalluGuard  (via MCP/stdio bridge)")
    print("=" * 60)

    # ---- AgentCards ---- #
    print("\n[1/4] AgentCards")
    print(SEPARATOR)
    show_agent_cards()

    # ---- Cas 1 : hallucination claire ---- #
    print("\n[2/4] Délégation A2A — Cas 1 : affirmation hallucinée")
    print(SEPARATOR)

    claim_bad   = "GPT-3 a été publié en 2015 avec 13 milliards de paramètres."
    evidences_1 = [
        "GPT-3 est publié par OpenAI en mai 2020 avec 175 milliards de paramètres.",
        "Le coût d'entraînement de GPT-3 est estimé à 4.6 millions de dollars.",
    ]
    print(f"  Claim     : {claim_bad}")
    print(f"  Evidences : {len(evidences_1)} document(s)")
    print(f"  Délégation en cours (NLI + BeliefState)...")

    result_1 = delegate_verification(claim_bad, evidences_1, node_type="generation")

    print(f"\n  Résultat A2A :")
    print(f"    task_id          : {result_1['task_id']}")
    print(f"    label (M1)       : {result_1['label']}")
    print(f"    score (M1)       : {result_1['score']}")
    print(f"    type hallucination: {result_1['hallucination_type']}")
    print(f"    m2_conflict      : {result_1['m2_conflict']}")
    print(f"    detected         : {result_1['detected']}")
    print(f"    duration_ms      : {result_1['duration_ms']} ms")
    print(f"    task_status      : {result_1['task_status']}")

    # ---- Cas 2 : affirmation correcte ---- #
    print("\n[3/4] Délégation A2A — Cas 2 : affirmation correcte")
    print(SEPARATOR)

    claim_ok    = "Mistral 7B a été publié en octobre 2023 sous licence Apache 2.0."
    evidences_2 = [
        "Mistral 7B est publié en octobre 2023 par Mistral AI avec 7.3 milliards de paramètres sous licence Apache 2.0.",
    ]
    print(f"  Claim     : {claim_ok}")
    print(f"  Evidences : {len(evidences_2)} document(s)")

    result_2 = delegate_verification(claim_ok, evidences_2, node_type="retrieval")

    print(f"\n  Résultat A2A :")
    print(f"    label (M1)       : {result_2['label']}")
    print(f"    score            : {result_2['score']}")
    print(f"    detected         : {result_2['detected']}")
    print(f"    task_status      : {result_2['task_status']}")

    # ---- Cas 3 : temporal (T2) ---- #
    print("\n[4/4] Délégation A2A — Cas 3 : hallucination temporelle (T2)")
    print(SEPARATOR)

    claim_t2    = "Le Transformer était déjà utilisé avant 2010 sous le nom de réseau d'attention."
    evidences_3 = [
        "L'architecture Transformer a été introduite en 2017 dans l'article 'Attention is All You Need' par Vaswani et al.",
    ]
    print(f"  Claim     : {claim_t2}")

    result_3 = delegate_verification(claim_t2, evidences_3, node_type="reasoning")

    print(f"\n  Résultat A2A :")
    print(f"    label (M1)       : {result_3['label']}")
    print(f"    type hallucination: {result_3['hallucination_type']}")
    print(f"    m2_conflict      : {result_3['m2_conflict']}")
    print(f"    detected         : {result_3['detected']}")

    # ---- Journal A2A ---- #
    print(f"\n{'='*60}")
    print(f"  Journal A2A : {_A2A_LOG}")
    print(f"{'='*60}")
    exchanges = _logger.tail(n=3)
    print(f"  Derniers {len(exchanges)} échanges :\n")
    for ex in exchanges:
        status_icon = "✓" if ex["task_status"] == "completed" else "✗"
        vr = ex.get("verification_result") or {}
        label = vr.get("label", "?")
        detected = vr.get("detected", "?")
        print(f"  {status_icon} [{ex['timestamp']}]")
        print(f"    {ex['from_agent']} -> {ex['to_agent']} | skill={ex['skill_id']}")
        print(f"    {ex['request_summary']}")
        print(f"    label={label} | detected={detected} | {ex['duration_ms']} ms")
        print(f"    transitions: {' -> '.join(h['to'] for h in ex['history'])}")
        print()

    print(f"  Total échanges dans le log : "
          f"{len([l for l in _A2A_LOG.read_text(encoding='utf-8').strip().splitlines() if l])}")
    print("=" * 60)
