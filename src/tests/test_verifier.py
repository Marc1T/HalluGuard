"""
Tests du LightweightVerifier — 4 tests avec résultats réels.
Modèle : cross-encoder/nli-MiniLM2-L6-H768 (téléchargé auto depuis HuggingFace).
Lancer avec : python src/tests/test_verifier.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.halluguard.verifier import LightweightVerifier

SEPARATOR = "-" * 60

v = LightweightVerifier()


def run_test(num: int, name: str, fn):
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
# Test 1 — Claim correct supporté par l'evidence                       #
# ------------------------------------------------------------------ #

def test_1():
    claim = "La Tour Eiffel est située à Paris."
    evidence = "La Tour Eiffel est un monument emblématique de Paris, capitale de la France."

    result = v.verify(claim, [evidence])

    print(f"  Claim    : {claim}")
    print(f"  Evidence : {evidence}")
    print(f"  Label    : {result['label']}  (attendu : 'correct')")
    print(f"  Score    : {result['score']}")
    print(f"  Confiance: {result['confidence']}")
    print(f"  Temps    : {result['latency_ms']} ms")

    assert result["label"] == "correct", \
        f"attendu 'correct', obtenu '{result['label']}'"
    assert result["insufficient_evidence"] is False


# ------------------------------------------------------------------ #
# Test 2 — Claim qui contredit l'evidence                              #
# ------------------------------------------------------------------ #

def test_2():
    claim = "La Tour Eiffel est située à Londres."
    evidence = "La Tour Eiffel est un monument emblématique de Paris, capitale de la France."

    result = v.verify(claim, [evidence])

    print(f"  Claim    : {claim}")
    print(f"  Evidence : {evidence}")
    print(f"  Label    : {result['label']}  (attendu : 'hallucinated')")
    print(f"  Score    : {result['score']}")
    print(f"  Confiance: {result['confidence']}")
    print(f"  Temps    : {result['latency_ms']} ms")

    assert result["label"] == "hallucinated", \
        f"attendu 'hallucinated', obtenu '{result['label']}'"
    assert result["insufficient_evidence"] is False


# ------------------------------------------------------------------ #
# Test 3 — Evidence vide                                               #
# ------------------------------------------------------------------ #

def test_3():
    claim = "N'importe quelle assertion sans contexte."

    result = v.verify(claim, [])

    print(f"  Claim               : {claim}")
    print(f"  Evidences           : []")
    print(f"  insufficient_evidence: {result['insufficient_evidence']}  (attendu : True)")
    print(f"  Score               : {result['score']}  (attendu : 0.5)")
    print(f"  Label               : {result['label']}")
    print(f"  Temps               : {result['latency_ms']} ms")

    assert result["insufficient_evidence"] is True, \
        "insufficient_evidence devrait être True"
    assert result["score"] == 0.5, \
        f"score devrait être 0.5, obtenu {result['score']}"


# ------------------------------------------------------------------ #
# Test 4 — NLI brut : premise et hypothesis contradictoires            #
# ------------------------------------------------------------------ #

def test_4():
    premise    = "All dogs are mammals."
    hypothesis = "No dogs are mammals."

    result = v.nli(premise, hypothesis)

    print(f"  Premise    : {premise}")
    print(f"  Hypothesis : {hypothesis}")
    print(f"  Label      : {result['label']}  (attendu : 'contradiction')")
    print(f"  Score      : {result['score']}")
    print(f"  Temps      : {result['latency_ms']} ms")

    assert result["label"] == "contradiction", \
        f"attendu 'contradiction', obtenu '{result['label']}'"


# ------------------------------------------------------------------ #
# Runner                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 60)
    print("  HalluGuard - LightweightVerifier Tests")
    print("  Modele : cross-encoder/nli-MiniLM2-L6-H768")
    print("=" * 60)
    print("\nChargement du modèle NLI...")
    print("(premier lancement = telechargement HuggingFace ~400 Mo)")

    # Préchargement + mesure du temps de chargement
    t_load = time.time()
    v._load()
    load_time = time.time() - t_load
    print(f"Modele charge en {load_time:.1f}s\n")

    tests = [
        (1, "Claim correct vs evidence supportante  -> label 'correct'",      test_1),
        (2, "Claim contradictoire vs evidence       -> label 'hallucinated'",  test_2),
        (3, "Evidence vide                          -> insufficient_evidence", test_3),
        (4, "NLI brut : premise vs hypothesis       -> label 'contradiction'", test_4),
    ]

    passed = 0
    for num, name, fn in tests:
        ok = run_test(num, name, fn)
        if ok:
            passed += 1
        else:
            print("\nARRET : un test a échoué. Correction requise avant de continuer.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(f"  Résultat final : {passed}/4 tests passés")
    print("=" * 60)
