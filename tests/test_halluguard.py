"""
Tests unitaires de tous les composants HalluGuard.
Couvre : DependencyDAG, BeliefState, LightweightVerifier, PolicyEngine, HalluGuardMiddleware.
Lancer avec : python -m pytest tests/test_halluguard.py -v
"""

from src.halluguard.dag import DependencyDAG
from src.halluguard.belief_state import BeliefState
from src.halluguard.verifier import LightweightVerifier
from src.halluguard.policy import PolicyEngine
from src.halluguard.middleware import HalluGuardMiddleware


# ================================================================== #
# DependencyDAG                                                         #
# ================================================================== #

class TestDependencyDAG:

    def test_build_from_langgraph(self):
        dag = DependencyDAG.build_from_langgraph({
            "retrieval": ["reasoning"],
            "reasoning": ["generation"],
        })
        assert "retrieval" in dag.all_nodes()
        assert "generation" in dag.all_nodes()

    def test_descendants(self):
        dag = DependencyDAG.build_from_langgraph({
            "retrieval": ["reasoning"],
            "reasoning": ["tool_call", "generation"],
        })
        desc = dag.descendants("retrieval")
        assert "reasoning" in desc
        assert "generation" in desc
        assert "tool_call" in desc

    def test_criticite_formula(self):
        dag = DependencyDAG()
        dag.add_node("retrieval", confiance=0.6)
        dag.add_edge("retrieval", "reasoning")
        dag.add_edge("retrieval", "generation")
        # criticite = 2 descendants * (1 - 0.6) = 0.8
        assert abs(dag.criticite("retrieval") - 0.8) < 1e-6

    def test_propagate_invalidation(self):
        dag = DependencyDAG.build_from_langgraph({
            "retrieval": ["reasoning"],
            "reasoning": ["generation"],
        })
        invalidated = dag.propagate_invalidation("retrieval")
        assert "retrieval" in invalidated
        assert "reasoning" in invalidated
        assert "generation" in invalidated
        assert dag.get_node("generation").stale is True

    def test_no_descendants_for_leaf(self):
        dag = DependencyDAG()
        dag.add_node("generation")
        assert len(dag.descendants("generation")) == 0
        assert dag.criticite("generation") == 0.0


# ================================================================== #
# BeliefState                                                           #
# ================================================================== #

class TestBeliefState:

    def test_add_fact(self):
        bs = BeliefState(session_id="test")
        fact = bs.add_fact("La Terre orbite autour du Soleil.", confiance=0.95, source="retrieval")
        assert fact.fact_id.startswith("test_")
        assert len(bs) == 1

    def test_max_facts_eviction(self):
        from src.halluguard.belief_state import MAX_FACTS
        bs = BeliefState(session_id="test_evict")
        for i in range(MAX_FACTS + 5):
            bs.add_fact(f"Fait numéro {i}", confiance=0.8, source="test")
        assert len(bs) == MAX_FACTS
        assert len(bs._archived) == 5

    def test_check_contradiction_no_embedding(self):
        bs = BeliefState(session_id="test_contra")
        bs.add_fact("Le ciel est bleu.", confiance=0.9, source="test")
        result = bs.check_contradiction("Le ciel est rouge.")
        assert "contradiction" in result
        assert "score" in result

    def test_save_load(self, tmp_path):
        bs = BeliefState(session_id="persist_test")
        bs.add_fact("Fait persistant.", confiance=0.9, source="test")
        path = bs.save(directory=str(tmp_path))
        assert path.exists()

        loaded = BeliefState.load("persist_test", directory=str(tmp_path))
        assert len(loaded) == 1
        assert loaded._facts[0].content == "Fait persistant."

    def test_summary(self):
        bs = BeliefState(session_id="test_summary")
        bs.add_fact("Un fait.", confiance=0.8, source="test")
        s = bs.summary()
        assert s["active_facts"] == 1
        assert s["session_id"] == "test_summary"


# ================================================================== #
# LightweightVerifier (sans modèle NLI — mode heuristique)             #
# ================================================================== #

class TestLightweightVerifier:

    def test_verify_m1_no_evidence(self):
        v = LightweightVerifier()
        result = v.verify_m1("Une assertion quelconque.", evidences=[])
        assert result["insufficient_evidence"] is True
        assert result["label"] == "neutral"
        assert result["score"] == 0.5

    def test_classify_t1_retrieval(self):
        v = LightweightVerifier()
        t = v._classify("document récupéré", "retrieval", "contradiction")
        assert t == "T1"

    def test_classify_t5_tool(self):
        v = LightweightVerifier()
        t = v._classify("résultat de l'outil API", "tool_call", "contradiction")
        assert t == "T5"

    def test_classify_t2_temporal(self):
        v = LightweightVerifier()
        t = v._classify("récemment publié en 2023", "reasoning", "contradiction")
        assert t == "T2"

    def test_classify_no_contradiction(self):
        v = LightweightVerifier()
        t = v._classify("quelque chose", "retrieval", "entailment")
        assert t is None

    def test_verify_m2_no_embedding(self):
        v = LightweightVerifier()
        bs = BeliefState(session_id="m2_test")
        bs.add_fact("Le soleil se lève à l'est.", confiance=0.9, source="test")
        result = v.verify_m2("Le soleil se lève à l'ouest.", bs)
        assert "contradiction" in result


# ================================================================== #
# PolicyEngine                                                          #
# ================================================================== #

class TestPolicyEngine:

    def test_pass_when_no_hallucination(self, tmp_path):
        pe = PolicyEngine(mode="log-only", log_path=str(tmp_path / "events.jsonl"))
        result = pe.handle(
            {"label": "entailment", "score": 0.9, "hallucination_type": None},
            claim="Claim correct.", node_type="generation"
        )
        assert result["action"] == "pass"

    def test_log_when_hallucination(self, tmp_path):
        pe = PolicyEngine(mode="log-only", log_path=str(tmp_path / "events.jsonl"))
        result = pe.handle(
            {"label": "contradiction", "score": 0.85, "hallucination_type": "T1"},
            claim="Claim incorrect.", node_type="retrieval"
        )
        assert result["action"] == "logged"
        assert result["event"] is not None
        assert pe.stats()["total_events"] == 1

    def test_auto_correct_calls_reinvoke(self, tmp_path):
        corrections = []
        def fake_llm(prompt: str) -> str:
            corrections.append(prompt)
            return "Correction générée."

        pe = PolicyEngine(mode="auto-correct", log_path=str(tmp_path / "events.jsonl"))
        result = pe.handle(
            {"label": "contradiction", "score": 0.9, "hallucination_type": "T3"},
            claim="Claim erroné.", node_type="reasoning",
            evidences=["Source fiable 1."],
            reinvoke_fn=fake_llm,
        )
        assert result["action"] == "corrected"
        assert result["output"] == "Correction générée."
        assert len(corrections) == 1

    def test_stats_by_type(self, tmp_path):
        pe = PolicyEngine(mode="log-only", log_path=str(tmp_path / "events.jsonl"))
        pe.handle({"label": "contradiction", "score": 0.8, "hallucination_type": "T1"}, "c1", "retrieval")
        pe.handle({"label": "contradiction", "score": 0.7, "hallucination_type": "T1"}, "c2", "retrieval")
        pe.handle({"label": "contradiction", "score": 0.9, "hallucination_type": "T3"}, "c3", "reasoning")
        stats = pe.stats()
        assert stats["by_hallucination_type"]["T1"] == 2
        assert stats["by_hallucination_type"]["T3"] == 1


# ================================================================== #
# HalluGuardMiddleware                                                  #
# ================================================================== #

class TestHalluGuardMiddleware:

    def test_variant_a_passthrough(self):
        mw = HalluGuardMiddleware(variant="A")
        result = mw.post_node("generation", "Texte de sortie.", {}, evidences=[])
        assert result["action"] == "pass"
        assert result["output"] == "Texte de sortie."

    def test_variant_b_no_belief(self):
        mw = HalluGuardMiddleware(variant="B")
        assert mw.belief is None

    def test_variant_c_has_belief(self):
        mw = HalluGuardMiddleware(variant="C", session_id="test_c")
        assert mw.belief is not None
        assert isinstance(mw.belief, BeliefState)

    def test_pre_node_no_stale(self):
        mw = HalluGuardMiddleware(variant="B")
        state = {"query": "test", "documents": [], "reasoning": "", "tool_output": "", "answer": "", "halluguard_events": []}
        result = mw.pre_node("retrieval", state)
        assert "_halluguard_stale_upstream" not in result or result["_halluguard_stale_upstream"] == []

    def test_overhead_stats_empty(self):
        mw = HalluGuardMiddleware(variant="B")
        stats = mw.overhead_stats()
        assert stats["calls"] == 0

    def test_dag_built_correctly(self):
        mw = HalluGuardMiddleware(variant="B")
        nodes = mw.dag.all_nodes()
        assert "retrieval" in nodes
        assert "generation" in nodes
