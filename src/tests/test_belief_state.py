"""
Tests du BeliefState et PersistentMemory - 4 tests avec resultats reels.
Lancer avec : python src/tests/test_belief_state.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.halluguard.belief_state import BeliefState, MAX_FACTS
from src.memory.persistent_memory import PersistentMemory

SEPARATOR = "-" * 60
MEMORY_DIR = "data/memory"


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
# Test 1 — Ajouter 3 faits et verifier qu'ils sont tous actifs         #
# ------------------------------------------------------------------ #

def test_1():
    bs = BeliefState(session_id="test1")

    id1 = bs.add_fact("La Tour Eiffel est a Paris.", confidence=0.9, step=1)
    id2 = bs.add_fact("Mistral est un LLM open-source.", confidence=0.85, step=2)
    id3 = bs.add_fact("Python est un langage de programmation.", confidence=0.95, step=3)

    active = bs.get_active_facts()

    print(f"  Faits ajoutes : 3")
    print(f"  Faits actifs  : {len(active)}")
    for f in active:
        print(f"    - [{f.fact_id}] step={f.step} conf={f.confidence} : {f.content[:50]}")

    assert len(active) == 3, f"attendu 3 faits actifs, obtenu {len(active)}"
    assert all(f.status == "active" for f in active), "tous les faits doivent etre 'active'"

    ids = {f.fact_id for f in active}
    assert id1 in ids and id2 in ids and id3 in ids, "les 3 fact_ids doivent etre presents"


# ------------------------------------------------------------------ #
# Test 2 — Ajouter 51 faits et verifier que le 51e declenche           #
#          l'expulsion (MAX_FACTS=50)                                   #
# ------------------------------------------------------------------ #

def test_2():
    bs = BeliefState(session_id="test2")

    for i in range(MAX_FACTS + 1):
        bs.add_fact(f"Fait numero {i:03d}.", confidence=0.8, step=i)

    active = bs.get_active_facts()
    total_in_memory = len(bs._facts)

    print(f"  Faits inseres      : {MAX_FACTS + 1}")
    print(f"  MAX_FACTS          : {MAX_FACTS}")
    print(f"  Faits actifs apres : {len(active)}")
    print(f"  Faits en memoire   : {total_in_memory}")
    print(f"  Expulsion          : {MAX_FACTS + 1 - len(active)} fait(s) expulse(s)")

    assert len(active) == MAX_FACTS, \
        f"attendu {MAX_FACTS} actifs, obtenu {len(active)}"

    expelled = [f for f in bs._facts if f.status == "invalidated"]
    assert len(expelled) == 1, \
        f"attendu 1 expulse, obtenu {len(expelled)}"

    print(f"  Fait expulse       : [{expelled[0].fact_id}] '{expelled[0].content}'")


# ------------------------------------------------------------------ #
# Test 3 — Sauvegarder, recharger, verifier identite des faits         #
# ------------------------------------------------------------------ #

def test_3():
    bs = BeliefState(session_id="test3")
    bs.add_fact("La Terre orbite autour du Soleil.", confidence=0.99, step=1)
    bs.add_fact("L'eau bout a 100 degres Celsius.", confidence=0.98, step=2)
    bs.add_fact("HalluGuard detecte les hallucinations.", confidence=0.9, step=3)

    # Sauvegarde via PersistentMemory (teste la classe du module memory/)
    mem = PersistentMemory(memory_dir=MEMORY_DIR)
    mem.save(bs)
    path = str(mem.path_for("test3"))

    print(f"  Sauvegarde dans : {path}")

    # Verifier que le fichier existe et est valide
    p = Path(path)
    assert p.exists(), f"fichier JSON non cree : {path}"

    raw = json.loads(p.read_text(encoding="utf-8"))
    print(f"  session_id dans JSON : {raw['session_id']}")
    print(f"  Nombre de faits dans JSON : {len(raw['facts'])}")

    # Verifier le format Interface 3
    for fd in raw["facts"]:
        assert "fact_id" in fd, "champ 'fact_id' manquant"
        assert "content" in fd, "champ 'content' manquant"
        assert "confidence" in fd, "champ 'confidence' manquant"
        assert "step" in fd, "champ 'step' manquant"
        assert "status" in fd, "champ 'status' manquant"
        assert isinstance(fd["step"], int), f"'step' doit etre int, obtenu {type(fd['step'])}"

    # Rechargement via PersistentMemory
    bs2 = mem.load("test3")
    active2 = bs2.get_active_facts()

    print(f"  Faits recharges  : {len(active2)}")
    for f in active2:
        print(f"    - [{f.fact_id}] step={f.step} conf={f.confidence} : {f.content[:50]}")

    assert bs2.session_id == bs.session_id, "session_id different apres rechargement"
    assert len(active2) == 3, f"attendu 3 faits apres rechargement, obtenu {len(active2)}"

    original_contents = {f.content for f in bs.get_active_facts()}
    reloaded_contents = {f.content for f in active2}
    assert original_contents == reloaded_contents, "contenu des faits different apres rechargement"

    original_confidences = {f.fact_id: f.confidence for f in bs.get_active_facts()}
    reloaded_confidences = {f.fact_id: f.confidence for f in active2}
    assert original_confidences == reloaded_confidences, "confidences differentes apres rechargement"

    print(f"  Fichier JSON cree : {path}")


# ------------------------------------------------------------------ #
# Test 4 — Fait contradictoire detecte par check_temporal_coherence   #
# ------------------------------------------------------------------ #

def test_4():
    from src.halluguard.verifier import LightweightVerifier

    bs = BeliefState(session_id="test4")
    bs.add_fact("The Eiffel Tower is located in Paris.", confidence=0.99, step=1)
    bs.add_fact("Water boils at 100 degrees Celsius.", confidence=0.98, step=2)

    verifier = LightweightVerifier()

    # Claim contradictoire
    contradictory_claim = "The Eiffel Tower is located in London."

    print(f"  Fait etabli    : 'The Eiffel Tower is located in Paris.'")
    print(f"  Claim a tester : '{contradictory_claim}'")

    result = bs.check_temporal_coherence(contradictory_claim, verifier=verifier)

    print(f"  conflict  : {result['conflict']}  (attendu : True)")
    print(f"  fact_id   : {result['fact_id']}")
    print(f"  score NLI : {result['score']}")

    assert result["conflict"] is True, \
        f"attendu conflict=True, obtenu {result['conflict']} (score={result['score']})"
    assert result["fact_id"] is not None, "fact_id doit etre present quand conflict=True"
    assert result["score"] > 0.5, \
        f"score de contradiction trop faible : {result['score']}"

    # Claim coherent : aucun mot commun avec les faits stockes -> filtre Jaccard
    coherent_claim = "Mount Everest is the tallest mountain on Earth."
    result2 = bs.check_temporal_coherence(coherent_claim, verifier=verifier)

    print(f"\n  Claim coherent : '{coherent_claim}'")
    print(f"  conflict  : {result2['conflict']}  (attendu : False)")
    print(f"  score NLI : {result2['score']}")

    assert result2["conflict"] is False, \
        f"attendu conflict=False pour claim coherent, obtenu {result2}"


# ------------------------------------------------------------------ #
# Runner                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 60)
    print("  HalluGuard - BeliefState + PersistentMemory Tests")
    print(f"  MAX_FACTS = {MAX_FACTS}")
    print("=" * 60)

    tests = [
        (1, "Ajouter 3 faits -> tous actifs",                          test_1),
        (2, f"Ajouter {MAX_FACTS+1} faits -> expulsion du moins confiant", test_2),
        (3, "save_to_disk / load_from_disk -> faits identiques",        test_3),
        (4, "check_temporal_coherence -> detecte contradiction NLI",    test_4),
    ]

    passed = 0
    for num, name, fn in tests:
        ok = run_test(num, name, fn)
        if ok:
            passed += 1
        else:
            print(f"\nARRET : Test {num} echoue. Correction requise avant de continuer.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(f"  Resultat final : {passed}/4 tests passes")
    print(f"  Fichier JSON   : {MEMORY_DIR}/session_test3.json")
    print("=" * 60)
