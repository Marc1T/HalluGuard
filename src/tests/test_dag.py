"""
Tests du DependencyDAG networkx - 4 tests avec resultats reels.
Lancer avec : python src/tests/test_dag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.halluguard.dag import DependencyDAG, DEFAULT_CONFIANCE, VERIFY_THRESHOLDS

SEPARATOR = "-" * 60


def run_test(num, name, fn):
    print(f"\n[Test {num}] {name}")
    print(SEPARATOR)
    try:
        fn()
        print(f"  --> PASSED")
        return True
    except AssertionError as e:
        print(f"  --> FAILED : {e}")
        return False


# ------------------------------------------------------------------ #
# Test 1 — Pipeline 4 noeuds en ligne, verifier les criticites         #
# ------------------------------------------------------------------ #
# retrieval -> reasoning -> tool_call -> generation
#
# criticite(retrieval)  = 3 x (1 - 0.6) = 1.2
# criticite(reasoning)  = 2 x (1 - 0.4) = 1.2
# criticite(tool_call)  = 1 x (1 - 0.0) = 1.0
# criticite(generation) = 0 x (1 - 0.0) = 0.0

def test_1():
    dag = DependencyDAG.build_from_langgraph({
        "retrieval": ["reasoning"],
        "reasoning": ["tool_call"],
        "tool_call": ["generation"],
    })

    expected = {
        "retrieval":  3 * (1 - DEFAULT_CONFIANCE["retrieval"]),   # 1.2
        "reasoning":  2 * (1 - DEFAULT_CONFIANCE["reasoning"]),   # 1.2
        "tool_call":  1 * (1 - DEFAULT_CONFIANCE["tool_call"]),   # 1.0
        "generation": 0 * (1 - DEFAULT_CONFIANCE["generation"]),  # 0.0
    }

    print(f"  Pipeline : retrieval -> reasoning -> tool_call -> generation")
    print(f"  {'Noeud':<12} {'Descendants':>12} {'Confiance':>10} {'Criticite':>10} {'Attendu':>10}")
    print(f"  {'-'*56}")

    for node_id, exp in expected.items():
        nb_desc = len(dag.descendants(node_id))
        conf = dag.get_node_data(node_id)["confiance"]
        crit = dag.criticite(node_id)
        match = "OK" if abs(crit - exp) < 1e-9 else "ERR"
        print(f"  {node_id:<12} {nb_desc:>12} {conf:>10.1f} {crit:>10.2f} {exp:>10.2f}  {match}")
        assert abs(crit - exp) < 1e-9, \
            f"criticite({node_id}) = {crit:.4f}, attendu {exp:.4f}"

    # Verifier aussi la structure networkx
    assert len(dag.all_nodes()) == 4, "le DAG doit avoir 4 noeuds"
    assert "generation" in dag.descendants("retrieval"), "generation doit etre descendant de retrieval"


# ------------------------------------------------------------------ #
# Test 2 — tool_call toujours verifie (seuil 0.0)                     #
# ------------------------------------------------------------------ #

def test_2():
    # Noeud tool_call SANS descendants -> criticite = 0
    dag = DependencyDAG()
    dag.add_node("tool_call")

    crit = dag.criticite("tool_call")
    threshold = VERIFY_THRESHOLDS["tool_call"]
    result = dag.should_verify("tool_call")

    print(f"  Noeud     : tool_call (isole, sans descendants)")
    print(f"  Confiance : {dag.get_node_data('tool_call')['confiance']}")
    print(f"  Criticite : {crit}  (= 0 car 0 descendants)")
    print(f"  Seuil     : {threshold}  (= 0.0 -> toujours verifie)")
    print(f"  should_verify() : {result}  (attendu : True)")

    assert threshold == 0.0, \
        f"seuil tool_call doit etre 0.0, obtenu {threshold}"
    assert result is True, \
        f"tool_call doit toujours etre verifie, obtenu {result}"

    # Verifier aussi generation (meme logique)
    dag.add_node("generation")
    assert dag.should_verify("generation") is True, \
        "generation doit aussi etre toujours verifie (seuil=0.0)"
    print(f"  should_verify('generation') : True  (confirme, seuil=0.0)")

    # Verifier que retrieval isolé (criticite=0 < 0.6) n'est PAS verifie
    dag.add_node("retrieval")
    assert dag.should_verify("retrieval") is False, \
        "retrieval isole (criticite=0 < 0.6) ne doit pas etre verifie"
    print(f"  should_verify('retrieval') isole : False  (confirme, criticite=0 < seuil=0.6)")


# ------------------------------------------------------------------ #
# Test 3 — propagate_invalidation sur noeud 2 marque noeuds 3 et 4   #
# ------------------------------------------------------------------ #

def test_3():
    # Pipeline en ligne : node1 -> node2 -> node3 -> node4
    dag = DependencyDAG()
    dag.add_node("node1", node_type="retrieval")
    dag.add_node("node2", node_type="reasoning")
    dag.add_node("node3", node_type="tool_call")
    dag.add_node("node4", node_type="generation")
    dag.add_edge("node1", "node2")
    dag.add_edge("node2", "node3")
    dag.add_edge("node3", "node4")

    print(f"  Pipeline : node1 -> node2 -> node3 -> node4")
    print(f"  Action   : propagate_invalidation('node2')")

    marked = dag.propagate_invalidation("node2")
    suspicious = dag.force_verify_suspicious()

    print(f"  Marques  : {marked}")
    print(f"  Suspects : {suspicious}")

    # node1 ne doit PAS etre marque (il est en amont)
    assert "node1" not in suspicious, \
        f"node1 (amont) ne doit pas etre suspect, suspects={suspicious}"

    # node2, node3, node4 doivent etre marques
    assert "node2" in suspicious, \
        f"node2 (invalide) doit etre suspect"
    assert "node3" in suspicious, \
        f"node3 (descendant) doit etre suspect"
    assert "node4" in suspicious, \
        f"node4 (descendant) doit etre suspect"

    print(f"  node1 non suspect : OK (amont, non affecte)")
    print(f"  node2 suspect     : OK (noeud invalide)")
    print(f"  node3 suspect     : OK (descendant direct)")
    print(f"  node4 suspect     : OK (descendant indirect)")


# ------------------------------------------------------------------ #
# Test 4 — Noeud suspect verifie meme si criticite < seuil            #
# ------------------------------------------------------------------ #

def test_4():
    # retrieval isole : criticite=0 < seuil=0.6 -> normalement NON verifie
    dag = DependencyDAG()
    dag.add_node("retrieval")

    crit_before = dag.criticite("retrieval")
    verify_before = dag.should_verify("retrieval")

    print(f"  Noeud    : retrieval (isole)")
    print(f"  Criticite: {crit_before}  Seuil: {VERIFY_THRESHOLDS['retrieval']}")
    print(f"  should_verify avant suspect : {verify_before}  (attendu : False)")

    assert verify_before is False, \
        f"retrieval isole ne doit pas etre verifie (criticite={crit_before} < {VERIFY_THRESHOLDS['retrieval']})"

    # Marquer comme suspect
    dag._g.nodes["retrieval"]["suspicious"] = True
    verify_after = dag.should_verify("retrieval")

    print(f"  Action   : marquer 'retrieval' comme suspect")
    print(f"  should_verify apres suspect : {verify_after}  (attendu : True)")
    print(f"  force_verify_suspicious()   : {dag.force_verify_suspicious()}")

    assert verify_after is True, \
        f"noeud suspect doit etre verifie meme si criticite < seuil"
    assert "retrieval" in dag.force_verify_suspicious(), \
        "force_verify_suspicious() doit retourner 'retrieval'"


# ------------------------------------------------------------------ #
# Runner                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 60)
    print("  HalluGuard - DependencyDAG Tests (networkx)")
    print("=" * 60)

    tests = [
        (1, "Pipeline 4 noeuds, verification criticites",           test_1),
        (2, "tool_call toujours verifie (seuil=0.0)",               test_2),
        (3, "propagate_invalidation -> noeuds 3 et 4 suspects",     test_3),
        (4, "Noeud suspect verifie meme si criticite < seuil",      test_4),
    ]

    passed = 0
    for num, name, fn in tests:
        ok = run_test(num, name, fn)
        if ok:
            passed += 1
        else:
            print(f"\nARRET : Test {num} echoue. Correction requise.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(f"  Resultat final : {passed}/4 tests passes")
    print("=" * 60)
