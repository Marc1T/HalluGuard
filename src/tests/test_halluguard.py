"""
test_halluguard.py — Suite de tests complète HalluGuard (pytest).

Composants testés :
  1. LightweightVerifier  — NLI inter-sources (4 tests)
  2. DependencyDAG        — criticité, propagation (4 tests)
  3. BeliefState          — faits, expulsion, persistance, cohérence (4 tests)
  4. PolicyEngine         — log, auto-correct, TTR, format (4 tests)
  5. MCP Server           — 4 tools via appels directs sans stdio (4 tests)
  6. Middleware           — variantes A / B / C end-to-end (3 tests)

Lancer :
  python -m pytest src/tests/test_halluguard.py -v --tb=short

Le modèle NLI est chargé une seule fois grâce à la fixture session-scoped.
"""

from __future__ import annotations
import importlib
import json
import sys
import time
from pathlib import Path

import pytest

# ------------------------------------------------------------------ #
# Chemin racine projet                                                  #
# ------------------------------------------------------------------ #

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ================================================================== #
# FIXTURES                                                             #
# ================================================================== #

@pytest.fixture(scope="session")
def verifier():
    """Charge le modèle NLI une seule fois pour toute la session pytest."""
    from src.halluguard.verifier import LightweightVerifier
    v = LightweightVerifier()
    v._load()
    return v


@pytest.fixture(scope="session")
def mcp(verifier):
    """
    Importe le module MCP et injecte le verifier déjà chargé.
    BeliefState frais pour les tests MCP.
    """
    mod = importlib.import_module("src.mcp_servers.halluguard_mcp")
    from src.halluguard.belief_state import BeliefState
    mod._verifier = verifier
    mod._belief   = BeliefState(session_id="pytest_mcp_session")
    return mod


# ================================================================== #
# SECTION 1 — LightweightVerifier                                      #
# ================================================================== #

class TestLightweightVerifier:
    """4 tests : correct, hallucinated, evidence vide, NLI brut."""

    def test_correct_claim_supported_by_evidence(self, verifier):
        """Un claim vrai doit retourner label='correct'."""
        r = verifier.verify(
            "La Tour Eiffel est située à Paris.",
            ["La Tour Eiffel est un monument emblématique de Paris, capitale de la France."],
        )
        assert r["label"] == "correct", f"attendu 'correct', obtenu '{r['label']}'"
        assert r["insufficient_evidence"] is False
        assert 0.0 <= r["score"] <= 1.0
        assert r["latency_ms"] >= 0

    def test_hallucinated_claim_contradicts_evidence(self, verifier):
        """Un claim contradictoire doit retourner label='hallucinated'."""
        r = verifier.verify(
            "La Tour Eiffel est située à Londres.",
            ["La Tour Eiffel est un monument emblématique de Paris, capitale de la France."],
        )
        assert r["label"] == "hallucinated", f"attendu 'hallucinated', obtenu '{r['label']}'"
        assert r["insufficient_evidence"] is False
        assert r["score"] > 0.5

    def test_empty_evidence_returns_insufficient(self, verifier):
        """Evidence vide → insufficient_evidence=True, score=0.5, label='correct'."""
        r = verifier.verify("N'importe quel claim sans contexte.", [])
        assert r["insufficient_evidence"] is True, "insufficient_evidence devrait être True"
        assert r["score"] == 0.5, f"score devrait être 0.5, obtenu {r['score']}"
        assert r["latency_ms"] == 0.0

    def test_nli_raw_contradiction(self, verifier):
        """NLI brut entre deux phrases opposées → label='contradiction'."""
        r = verifier.nli("All dogs are mammals.", "No dogs are mammals.")
        assert r["label"] == "contradiction", \
            f"attendu 'contradiction', obtenu '{r['label']}'"
        assert r["score"] > 0.5


# ================================================================== #
# SECTION 2 — DependencyDAG                                            #
# ================================================================== #

class TestDependencyDAG:
    """4 tests : criticités, seuil tool_call, propagation, nœud suspect."""

    def test_criticite_linear_pipeline(self):
        """Pipeline 4 nœuds : vérifie toutes les criticités calculées."""
        from src.halluguard.dag import DependencyDAG, DEFAULT_CONFIANCE

        dag = DependencyDAG.build_from_langgraph({
            "retrieval":  ["reasoning"],
            "reasoning":  ["tool_call"],
            "tool_call":  ["generation"],
        })

        # retrieval a 3 descendants, reasoning 2, tool_call 1, generation 0
        assert abs(dag.criticite("retrieval")  - 3 * (1 - DEFAULT_CONFIANCE["retrieval"]))  < 1e-9
        assert abs(dag.criticite("reasoning")  - 2 * (1 - DEFAULT_CONFIANCE["reasoning"]))  < 1e-9
        assert abs(dag.criticite("tool_call")  - 1 * (1 - DEFAULT_CONFIANCE["tool_call"]))  < 1e-9
        assert abs(dag.criticite("generation") - 0.0)                                        < 1e-9

        assert len(dag.all_nodes()) == 4
        assert "generation" in dag.descendants("retrieval")

    def test_tool_call_always_verified(self):
        """tool_call a un seuil 0.0 → should_verify() = True même isolé."""
        from src.halluguard.dag import DependencyDAG, VERIFY_THRESHOLDS

        dag = DependencyDAG()
        dag.add_node("tool_call")

        assert VERIFY_THRESHOLDS["tool_call"] == 0.0
        assert dag.should_verify("tool_call") is True

        # generation aussi toujours vérifié
        dag.add_node("generation")
        assert dag.should_verify("generation") is True

        # retrieval isolé (criticité=0 < seuil=0.6) → NON vérifié
        dag.add_node("retrieval")
        assert dag.should_verify("retrieval") is False

    def test_propagate_invalidation_marks_descendants(self):
        """Invalider node2 doit marquer node2, node3, node4 mais PAS node1."""
        from src.halluguard.dag import DependencyDAG

        dag = DependencyDAG()
        for name, ntype in [("n1","retrieval"),("n2","reasoning"),("n3","tool_call"),("n4","generation")]:
            dag.add_node(name, node_type=ntype)
        dag.add_edge("n1", "n2")
        dag.add_edge("n2", "n3")
        dag.add_edge("n3", "n4")

        dag.propagate_invalidation("n2")
        suspicious = dag.force_verify_suspicious()

        assert "n1" not in suspicious, f"n1 (amont) ne doit pas être suspect : {suspicious}"
        assert "n2" in suspicious, "n2 (invalidé) doit être suspect"
        assert "n3" in suspicious, "n3 (descendant direct) doit être suspect"
        assert "n4" in suspicious, "n4 (descendant indirect) doit être suspect"

    def test_suspicious_flag_forces_verification(self):
        """Un nœud marqué suspect doit être vérifié même si criticité < seuil."""
        from src.halluguard.dag import DependencyDAG, VERIFY_THRESHOLDS

        dag = DependencyDAG()
        dag.add_node("retrieval")

        # Sans suspect : criticité=0 < seuil=0.6 → False
        assert dag.should_verify("retrieval") is False
        assert dag.criticite("retrieval") < VERIFY_THRESHOLDS["retrieval"]

        # Avec suspect : → True
        dag._g.nodes["retrieval"]["suspicious"] = True
        assert dag.should_verify("retrieval") is True
        assert "retrieval" in dag.force_verify_suspicious()


# ================================================================== #
# SECTION 3 — BeliefState + PersistentMemory                           #
# ================================================================== #

class TestBeliefState:
    """4 tests : add, eviction, save/load, cohérence temporelle."""

    def test_add_three_facts_all_active(self):
        """Ajouter 3 faits → 3 actifs."""
        from src.halluguard.belief_state import BeliefState

        bs = BeliefState(session_id="pytest_bs1")
        id1 = bs.add_fact("Fait A.", confidence=0.9,  step=1)
        id2 = bs.add_fact("Fait B.", confidence=0.85, step=2)
        id3 = bs.add_fact("Fait C.", confidence=0.95, step=3)

        active = bs.get_active_facts()
        assert len(active) == 3
        assert all(f.status == "active" for f in active)
        ids_present = {f.fact_id for f in active}
        assert id1 in ids_present and id2 in ids_present and id3 in ids_present

    def test_eviction_at_max_capacity(self):
        """Dépasser MAX_FACTS → exactement 1 expulsé, MAX_FACTS actifs."""
        from src.halluguard.belief_state import BeliefState, MAX_FACTS

        bs = BeliefState(session_id="pytest_bs2")
        for i in range(MAX_FACTS + 1):
            bs.add_fact(f"Fait {i:03d}.", confidence=0.8, step=i)

        active = bs.get_active_facts()
        evicted = [f for f in bs._facts if f.status == "invalidated"]

        assert len(active)  == MAX_FACTS, f"attendu {MAX_FACTS} actifs, obtenu {len(active)}"
        assert len(evicted) == 1,         f"attendu 1 expulsé, obtenu {len(evicted)}"

    def test_save_and_reload_preserve_facts(self, tmp_path):
        """save(bs) puis load(session_id) → faits identiques (contenu + confiance)."""
        from src.halluguard.belief_state import BeliefState
        from src.memory.persistent_memory import PersistentMemory

        bs = BeliefState(session_id="pytest_bs3")
        bs.add_fact("La Terre orbite autour du Soleil.", confidence=0.99, step=1)
        bs.add_fact("L'eau bout à 100°C.",               confidence=0.98, step=2)
        bs.add_fact("HalluGuard détecte les hallucinations.", confidence=0.9, step=3)

        mem = PersistentMemory(memory_dir=str(tmp_path))
        mem.save(bs)

        # Vérifier le fichier JSON (Interface 3)
        json_file = mem.path_for("pytest_bs3")
        assert json_file.exists(), f"Fichier JSON non créé : {json_file}"
        raw = json.loads(json_file.read_text(encoding="utf-8"))
        assert raw["session_id"] == "pytest_bs3"
        required_keys = {"fact_id", "content", "confidence", "step", "status"}
        for fd in raw["facts"]:
            assert required_keys <= set(fd.keys()), f"Champs manquants dans {fd}"
            assert isinstance(fd["step"], int)

        # Rechargement
        bs2 = mem.load("pytest_bs3")
        assert bs2.session_id == bs.session_id
        assert len(bs2.get_active_facts()) == 3

        orig   = {f.content: f.confidence for f in bs.get_active_facts()}
        reload = {f.content: f.confidence for f in bs2.get_active_facts()}
        assert orig == reload, "Contenu/confiance différents après rechargement"

    def test_temporal_coherence_detects_contradiction(self, verifier):
        """Fait stocké 'Paris' → claim 'London' doit déclencher conflict=True."""
        from src.halluguard.belief_state import BeliefState

        bs = BeliefState(session_id="pytest_bs4")
        bs.add_fact("The Eiffel Tower is located in Paris.", confidence=0.99, step=1)
        bs.add_fact("Water boils at 100 degrees Celsius.",   confidence=0.98, step=2)

        # Claim contradictoire
        r_bad = bs.check_temporal_coherence(
            "The Eiffel Tower is located in London.", verifier=verifier
        )
        assert r_bad["conflict"] is True,  f"conflict attendu True, obtenu {r_bad}"
        assert r_bad["fact_id"] is not None
        assert r_bad["score"] > 0.5

        # Claim cohérent (sans mots communs → filtre Jaccard élimine)
        r_ok = bs.check_temporal_coherence(
            "Mount Everest is the tallest mountain on Earth.", verifier=verifier
        )
        assert r_ok["conflict"] is False, f"conflict attendu False, obtenu {r_ok}"


# ================================================================== #
# SECTION 4 — PolicyEngine                                             #
# ================================================================== #

class TestPolicyEngine:
    """4 tests : log, auto-correct (mock LLM), TTR, format champs."""

    def _fake_hallu(self, score=0.9, h_type="T1"):
        return {"label": "hallucinated", "score": score, "hallucination_type": h_type}

    def test_log_mode_writes_jsonl_entry(self, tmp_path):
        """Mode log → fichier JSONL créé, champs corrects."""
        from src.halluguard.policy import PolicyEngine

        log_file = str(tmp_path / "policy_test.log")
        pe = PolicyEngine(mode="log", log_path=log_file)

        result = pe.handle(
            self._fake_hallu(score=0.9, h_type="T1"),
            claim="The Eiffel Tower is in London.",
            node_id="generation",
            evidences=["The Eiffel Tower is in Paris, France."],
        )

        assert result["action"] == "logged", f"attendu 'logged', obtenu '{result['action']}'"
        assert result["event"] is not None

        log_path = Path(log_file)
        assert log_path.exists()
        entry = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert entry["node_id"] == "generation"
        assert entry["hallucination_type"] == "T1"
        assert 0.0 <= entry["score"] <= 1.0
        assert "timestamp" in entry

    def test_auto_correct_with_mock_llm(self, tmp_path):
        """Mode auto-correct avec mock LLM → action='corrected', output contient la correction."""
        from src.halluguard.policy import PolicyEngine

        log_file = str(tmp_path / "autocorrect.log")
        pe = PolicyEngine(mode="auto-correct", log_path=log_file)

        mock_output = "The Eiffel Tower is located in Paris, France."

        result = pe.handle(
            self._fake_hallu(score=0.9, h_type="T1"),
            claim="The Eiffel Tower is in London.",
            node_id="generation",
            evidences=["The Eiffel Tower is in Paris, France."],
            reinvoke_fn=lambda _: mock_output,
        )

        assert result["action"] == "corrected", f"attendu 'corrected', obtenu '{result['action']}'"
        assert result["output"] == mock_output
        assert "Paris" in result["output"]

    def test_ttr_measured_and_positive(self, tmp_path):
        """TTR doit être mesuré et > 0 après correction (même avec LLM simulé)."""
        from src.halluguard.policy import PolicyEngine

        log_file = str(tmp_path / "ttr.log")
        pe = PolicyEngine(mode="auto-correct", log_path=log_file)

        calls = []
        def slow_llm(prompt: str) -> str:
            time.sleep(0.05)
            calls.append(prompt)
            return "The Eiffel Tower is in Paris."

        result = pe.handle(
            self._fake_hallu(score=0.85, h_type="T4"),
            claim="The Eiffel Tower is in Berlin.",
            node_id="reasoning",
            evidences=["The Eiffel Tower is in Paris."],
            reinvoke_fn=slow_llm,
        )

        assert result["action"] == "corrected"
        ttr = result["event"].get("ttr_seconds")
        assert ttr is not None, "ttr_seconds ne doit pas être None"
        assert ttr > 0,         f"TTR doit être > 0, obtenu {ttr}"
        assert len(calls) == 1, "le LLM doit être appelé exactement 1 fois"

    def test_log_format_required_fields(self, tmp_path):
        """Trois événements dans le log : tous les champs obligatoires présents."""
        from src.halluguard.policy import PolicyEngine

        log_file = str(tmp_path / "format.log")
        pe = PolicyEngine(mode="log", log_path=log_file)

        cases = [
            ("retrieval", "T1", 0.88),
            ("reasoning", "T3", 0.76),
            ("tool_call", "T5", 0.92),
        ]
        for nid, htype, score in cases:
            pe.handle(
                {"label": "hallucinated", "score": score, "hallucination_type": htype},
                claim=f"Faux claim depuis {nid}.",
                node_id=nid,
            )

        lines = [l for l in Path(log_file).read_text(encoding="utf-8").strip().splitlines() if l]
        assert len(lines) == 3, f"attendu 3 entrées log, obtenu {len(lines)}"

        required = {"timestamp", "node_id", "hallucination_type", "score"}
        for line in lines:
            entry = json.loads(line)
            missing = required - set(entry.keys())
            assert not missing, f"Champs manquants : {missing} dans {entry}"
            assert 0.0 <= entry["score"] <= 1.0


# ================================================================== #
# SECTION 5 — MCP Server (appels directs, sans stdio)                  #
# ================================================================== #

class TestMCPServer:
    """
    4 tools testés via appels directs (sans sous-process stdio).
    L'ordre compte : add_fact doit précéder get_belief_state et check_temporal.
    """

    def test_mcp_add_fact(self, mcp):
        """add_fact → dict avec status et fact_id."""
        result = mcp.add_fact(
            fact="The Eiffel Tower is located in Paris, France.",
            confidence=0.99,
            step=1,
        )
        assert "status"  in result, "champ 'status' manquant"
        assert "fact_id" in result, "champ 'fact_id' manquant"
        assert result["status"] in ("added", "evicted_oldest"), \
            f"status inconnu : {result['status']}"

    def test_mcp_get_belief_state(self, mcp):
        """get_belief_state → dict complet, au moins 1 fait (ajouté ci-dessus)."""
        result = mcp.get_belief_state()
        assert "session_id"   in result, "champ 'session_id' manquant"
        assert "active_facts" in result, "champ 'active_facts' manquant"
        assert "facts"        in result, "champ 'facts' manquant"
        assert result["active_facts"] >= 1, \
            f"au moins 1 fait attendu (ajouté dans test précédent), obtenu {result['active_facts']}"

    def test_mcp_verify_claim(self, mcp):
        """verify_claim avec claim contradictoire → label='hallucinated'."""
        result = mcp.verify_claim(
            claim="The Eiffel Tower is located in London.",
            evidences=["The Eiffel Tower is in Paris, France."],
            node_type="generation",
        )
        assert "score"              in result, "champ 'score' manquant"
        assert "label"              in result, "champ 'label' manquant"
        assert "hallucination_type" in result, "champ 'hallucination_type' manquant"
        assert result["label"] == "hallucinated", \
            f"attendu 'hallucinated', obtenu '{result['label']}'"
        assert result["score"] > 0.5, f"score trop faible : {result['score']}"

    def test_mcp_check_temporal_coherence(self, mcp):
        """check_temporal_coherence → conflict=True car Paris ≠ London."""
        # Le fait 'Paris' est dans le BeliefState depuis test_mcp_add_fact
        result = mcp.check_temporal_coherence(
            claim="The Eiffel Tower is located in London."
        )
        assert "conflict" in result, "champ 'conflict' manquant"
        assert "fact_id"  in result, "champ 'fact_id' manquant"
        assert "score"    in result, "champ 'score' manquant"
        assert result["conflict"] is True, \
            f"conflict attendu True (Paris ≠ London), obtenu {result['conflict']}"


# ================================================================== #
# SECTION 6 — HalluGuardMiddleware (intégration A / B / C)            #
# ================================================================== #

class TestMiddlewareIntegration:
    """3 tests d'intégration end-to-end des 3 variantes du middleware."""

    def test_variant_a_passthrough(self):
        """Variante A : post_node retourne toujours action='pass', overhead=0."""
        from src.halluguard.middleware import HalluGuardMiddleware

        mw = HalluGuardMiddleware(variant="A")
        result = mw.post_node("generation", "N'importe quel output.", {})

        assert result["action"]      == "pass"
        assert result["overhead_ms"] == 0.0
        assert result["event"]       is None

    def test_variant_b_detects_hallucination(self, verifier, tmp_path):
        """Variante B : claim contradictoire → action='logged', event non null."""
        from src.halluguard.middleware import HalluGuardMiddleware

        log_file = str(tmp_path / "mw_b.log")
        mw = HalluGuardMiddleware(variant="B", policy_mode="log",
                                  session_id="pytest_mw_b", log_path=log_file)
        mw.verifier = verifier  # injecter le verifier déjà chargé

        result = mw.post_node(
            "retrieval",
            "The Eiffel Tower is located in London.",
            {},
            evidences=["The Eiffel Tower is located in Paris, France."],
        )

        assert result["action"] == "logged", \
            f"attendu 'logged', obtenu '{result['action']}' (event={result['event']})"
        assert result["event"] is not None
        assert result["event"]["hallucination_type"] == "T1"
        assert result["overhead_ms"] > 0

    def test_variant_c_with_belief_state(self, verifier, tmp_path):
        """Variante C : pipeline complet (NLI + BeliefState), pas d'exception."""
        from src.halluguard.middleware import HalluGuardMiddleware

        log_file = str(tmp_path / "mw_c.log")
        mw = HalluGuardMiddleware(variant="C", policy_mode="log",
                                  session_id="pytest_mw_c", log_path=log_file)
        mw.verifier = verifier  # injecter le verifier déjà chargé

        result = mw.post_node(
            "generation",
            "GPT-3 a été publié en 2015 avec seulement 13 milliards de paramètres.",
            {},
            evidences=["GPT-3 est publié par OpenAI en mai 2020 avec 175 milliards de paramètres."],
        )

        # La détection peut varier (hallucinated ou non selon le score NLI)
        # mais le pipeline doit s'exécuter sans erreur et retourner les bons champs
        assert "action"      in result
        assert "overhead_ms" in result
        assert result["action"] in ("pass", "logged", "corrected")
        assert result["overhead_ms"] >= 0

        # Sur un claim aussi contradictoire, on s'attend à la détection
        assert result["action"] == "logged", \
            f"attendu 'logged' pour claim très contradictoire, obtenu '{result['action']}'"
