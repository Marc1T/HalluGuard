"""
Tests du PolicyEngine - 4 tests avec resultats reels.
Test 2 appelle Mistral via Ollama (Ollama doit etre actif).
Lancer avec : python src/tests/test_policy.py
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.halluguard.policy import PolicyEngine, DEFAULT_LOG_PATH

SEPARATOR = "-" * 60
LOG_PATH = "results/logs/halluguard_test.log"   # log dedie aux tests


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


def _fake_hallucination(score=0.9, h_type="T1", node_id="generation"):
    """Construit un faux resultat de verification hallucine."""
    return {
        "label": "hallucinated",
        "score": score,
        "hallucination_type": h_type,
    }


# ------------------------------------------------------------------ #
# Test 1 — Mode log : detection ecrite dans le fichier log            #
# ------------------------------------------------------------------ #

def test_1():
    # Nettoyer le log de test au demarrage
    p = Path(LOG_PATH)
    if p.exists():
        p.unlink()

    pe = PolicyEngine(mode="log", log_path=LOG_PATH)

    claim = "The Eiffel Tower is located in London."
    result = pe.handle(
        _fake_hallucination(score=0.9, h_type="T1", node_id="generation"),
        claim=claim,
        node_id="generation",
        evidences=["The Eiffel Tower is in Paris, France."],
    )

    print(f"  Mode         : log")
    print(f"  Claim        : {claim}")
    print(f"  Action       : {result['action']}  (attendu : 'logged')")
    print(f"  Fichier log  : {LOG_PATH}")

    assert result["action"] == "logged", \
        f"attendu 'logged', obtenu '{result['action']}'"
    assert p.exists(), f"fichier log non cree : {LOG_PATH}"

    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 1, "le log doit contenir au moins une entree"

    entry = json.loads(lines[-1])
    print(f"  Entree log   : {json.dumps(entry, indent=2, ensure_ascii=False)}")

    assert "timestamp" in entry, "champ 'timestamp' manquant"
    assert "node_id" in entry, "champ 'node_id' manquant"
    assert "hallucination_type" in entry, "champ 'hallucination_type' manquant"
    assert "score" in entry, "champ 'score' manquant"
    assert entry["node_id"] == "generation", f"node_id incorrect : {entry['node_id']}"
    assert entry["hallucination_type"] == "T1", \
        f"hallucination_type incorrect : {entry['hallucination_type']}"


# ------------------------------------------------------------------ #
# Test 2 — Mode auto-correct : Mistral appele, correction retournee   #
# ------------------------------------------------------------------ #

def test_2():
    import ollama

    def call_mistral(prompt: str) -> str:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    pe = PolicyEngine(mode="auto-correct", log_path=LOG_PATH)

    claim = "The Eiffel Tower is located in London."
    evidences = ["The Eiffel Tower is a famous monument located in Paris, France."]

    print(f"  Mode         : auto-correct")
    print(f"  Claim        : {claim}")
    print(f"  Appel Mistral en cours (peut prendre ~60s sur CPU)...")

    t0 = time.time()
    result = pe.handle(
        _fake_hallucination(score=0.9, h_type="T1", node_id="generation"),
        claim=claim,
        node_id="generation",
        evidences=evidences,
        reinvoke_fn=call_mistral,
    )
    elapsed = time.time() - t0

    print(f"  Action       : {result['action']}  (attendu : 'corrected')")
    print(f"  Temps total  : {elapsed:.1f}s")
    print(f"  Correction   : {result['output'][:150] if result['output'] else 'None'}...")

    assert result["action"] == "corrected", \
        f"attendu 'corrected', obtenu '{result['action']}'"
    assert result["output"] is not None, "la correction ne doit pas etre None"
    assert len(result["output"]) > 0, "la correction ne doit pas etre vide"
    assert "Paris" in result["output"] or "paris" in result["output"].lower(), \
        f"la correction devrait mentionner 'Paris', obtenu : {result['output'][:200]}"


# ------------------------------------------------------------------ #
# Test 3 — TTR mesure et non nul                                       #
# ------------------------------------------------------------------ #

def test_3():
    corrections_done = []

    def fake_slow_llm(prompt: str) -> str:
        time.sleep(0.1)   # simulation d'une latence minimale
        corrections_done.append(prompt)
        return "The Eiffel Tower is located in Paris, France."

    pe = PolicyEngine(mode="auto-correct", log_path=LOG_PATH)
    result = pe.handle(
        _fake_hallucination(score=0.85, h_type="T4", node_id="reasoning"),
        claim="The Eiffel Tower is in Berlin.",
        node_id="reasoning",
        evidences=["The Eiffel Tower is in Paris."],
        reinvoke_fn=fake_slow_llm,
    )

    event = result["event"]
    ttr = event.get("ttr_seconds")

    print(f"  Action       : {result['action']}")
    print(f"  TTR mesure   : {ttr}s  (attendu : > 0)")
    print(f"  Correction   : {result['output']}")
    print(f"  LLM appele   : {len(corrections_done)} fois")

    assert result["action"] == "corrected", \
        f"attendu 'corrected', obtenu '{result['action']}'"
    assert ttr is not None, "ttr_seconds ne doit pas etre None"
    assert ttr > 0, f"TTR doit etre > 0, obtenu {ttr}"
    assert len(corrections_done) == 1, "le LLM doit etre appele exactement 1 fois"

    print(f"  TTR = {ttr:.3f}s >= 0.1s (sleep simule) : OK")


# ------------------------------------------------------------------ #
# Test 4 — Log contient timestamp, hallucination_type, score, node_id #
# ------------------------------------------------------------------ #

def test_4():
    pe = PolicyEngine(mode="log", log_path=LOG_PATH)

    # Generer 3 evenements de types differents
    cases = [
        ("retrieval", "T1", 0.88),
        ("reasoning", "T3", 0.76),
        ("tool_call", "T5", 0.92),
    ]

    for nid, htype, score in cases:
        pe.handle(
            _fake_hallucination(score=score, h_type=htype, node_id=nid),
            claim=f"Faux claim depuis noeud {nid}.",
            node_id=nid,
        )

    # Lire et valider toutes les entrees du log
    p = Path(LOG_PATH)
    assert p.exists(), f"fichier log absent : {LOG_PATH}"
    lines = [l for l in p.read_text(encoding="utf-8").strip().split("\n") if l.strip()]

    print(f"  Fichier log  : {LOG_PATH}")
    print(f"  Entrees lues : {len(lines)}")

    REQUIRED_FIELDS = {"timestamp", "hallucination_type", "score", "node_id"}

    for i, line in enumerate(lines):
        entry = json.loads(line)
        missing = REQUIRED_FIELDS - set(entry.keys())
        assert not missing, f"Ligne {i+1} : champs manquants {missing}"

    # Verifier les 3 derniers evenements generes dans ce test
    last_3 = [json.loads(l) for l in lines[-3:]]
    print(f"\n  Les 3 derniers evenements du log :")
    for entry in last_3:
        print(f"    timestamp={entry['timestamp']}  node_id={entry['node_id']}"
              f"  type={entry['hallucination_type']}  score={entry['score']}")
        assert entry["timestamp"] != "", "timestamp ne doit pas etre vide"
        assert entry["node_id"] in ("retrieval", "reasoning", "tool_call", "generation"), \
            f"node_id inconnu : {entry['node_id']}"
        assert 0 <= entry["score"] <= 1, f"score hors [0,1] : {entry['score']}"


# ------------------------------------------------------------------ #
# Affichage du fichier log final                                        #
# ------------------------------------------------------------------ #

def show_log():
    p = Path(LOG_PATH)
    if not p.exists():
        print("\n  (aucun fichier log genere)")
        return
    print(f"\n{'='*60}")
    print(f"  Contenu de {LOG_PATH} :")
    print(f"{'='*60}")
    lines = [l for l in p.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    for line in lines:
        entry = json.loads(line)
        print(f"  {entry.get('timestamp')} | {entry.get('node_id'):<12} "
              f"| type={entry.get('hallucination_type')} "
              f"| score={entry.get('score')} "
              f"| corrige={entry.get('corrected')}"
              + (f" | TTR={entry.get('ttr_seconds')}s" if entry.get('ttr_seconds') else ""))


# ------------------------------------------------------------------ #
# Runner                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 60)
    print("  HalluGuard - PolicyEngine Tests")
    print("  (Test 2 : appel Mistral reel via Ollama)")
    print("=" * 60)

    tests = [
        (1, "Mode log -> detection ecrite dans halluguard.log",   test_1),
        (2, "Mode auto-correct -> Mistral corrige la reponse",    test_2),
        (3, "TTR mesure et non nul apres correction",             test_3),
        (4, "Log contient timestamp, hallucination_type, score, node_id", test_4),
    ]

    passed = 0
    for num, name, fn in tests:
        ok = run_test(num, name, fn)
        if ok:
            passed += 1
        else:
            print(f"\nARRET : Test {num} echoue.")
            show_log()
            sys.exit(1)

    show_log()

    print("\n" + "=" * 60)
    print(f"  Resultat final : {passed}/4 tests passes")
    print("=" * 60)
