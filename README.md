# HalluGuard — Détection et Mitigation des Hallucinations dans les Pipelines Agentiques RAG

Projet 15 — ENSAM Meknès 4ème Année — Prof. Hajji Tarik — 2025-2026 — Étudiants : Souleymane Diallo & Marc Thierry Nankouli

HalluGuard est un middleware de détection d'hallucinations pour pipelines RAG agentiques (LangGraph). Il combine un vérificateur NLI inter-sources (M1, cross-encoder 117 M params), un BeliefState temporel (M2, 50 faits max), un graphe de dépendances (DependencyDAG), une politique de correction (PolicyEngine) et un serveur MCP exposant 4 outils. Sur 90 scénarios HaluEval-Agentic (60 hallucinés + 30 corrects), la configuration recommandée **Variante B** (M1 seul) atteint **F1 = 82,9 %** (rappel 76,7 %, précision 90,2 %) pour un overhead médian de **~40 ms sur CPU** (variable selon la charge, toujours ≪ 300 ms) (contre 0 % de détection pour le baseline). Le vérificateur est en outre analysé en calibration (ECE) et son seuil calibré par conformal prediction.

> **Chiffres officiels** : tous les résultats ci-dessous proviennent de `results/SUMMARY.json`, régénéré par `python scripts/run_all.py`. Ne pas citer d'autres sources.

---

## Installation (5 commandes)

```powershell
git clone https://github.com/Marc1T/HalluGuard.git
cd HalluGuard
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> Pour le pipeline RAG complet avec Mistral : `ollama pull mistral` (optionnel — le benchmark et la démo directe n'en ont pas besoin).

---

## Lancer le serveur MCP

Le serveur MCP expose 4 tools via protocole stdio (FastMCP) et journalise chaque appel dans `results/logs/mcp_calls.log`.

```powershell
# Terminal dédié — protocole MCP/stdio
venv\Scripts\python.exe src/mcp_servers/halluguard_mcp.py
```

Tools exposés : `verify_claim`, `check_temporal_coherence`, `add_fact`, `get_belief_state`.

---

## Lancer le pipeline RAG

```powershell
# Variante C — pipeline complet (M1 + M2 + MCP) — nécessite Ollama + Mistral
venv\Scripts\python.exe src/agents/rag_agent.py "Qu'est-ce qu'une hallucination dans un LLM ?"

# Mode interactif pour la soutenance (avec ou sans Mistral)
venv\Scripts\python.exe demo_interactive.py

# Démo automatique 3 questions (sans Mistral)
venv\Scripts\python.exe demo_pipeline.py
```

---

## Lancer le benchmark

```powershell
# 10 scénarios — test rapide (~2 min)
venv\Scripts\python.exe src/tests/benchmark_runner.py --n 10

# 60 scénarios complets — résultats sauvegardés dans results/resultats_comparatifs.json
venv\Scripts\python.exe src/tests/benchmark_runner.py --all

# Suite de tests unitaires (23 tests, 5 composants)
venv\Scripts\python.exe -m pytest src/tests/test_halluguard.py -v

# Démo de bout en bout Windows (active venv, MCP, pipeline, logs, métriques)
demo.bat
```

---

## Description des fichiers

### Composants HalluGuard (`src/halluguard/`)

| Fichier | Description |
|---|---|
| `verifier.py` | **M1** — Vérificateur NLI inter-sources. Modèle : `cross-encoder/nli-MiniLM2-L6-H768` (117 M params, MNLI 433 K paires). Classifie chaque paire (claim, evidence) en contradiction / neutral / entailment. **Règle de décision** : `hallucinated` si `P(contradiction) > P(entailment)` (argmax), ou seuil calibré par conformal prediction. Les seuils par nœud (`tool_call`=0.0, …) servent au *gating* de criticité du `DependencyDAG`, pas à la décision de détection. |
| `belief_state.py` | **M2** — BeliefState temporel. 50 faits max par session, éviction par score `confiance × récence`. Détecte les contradictions temporelles via NLI contre les faits stockés. |
| `dag.py` | **DependencyDAG** — Graphe de dépendances des nœuds LangGraph. Criticité : `criticite(n) = nb_descendants(n) × (1 − confiance(n))`. Propage l'invalidation aux nœuds descendants. |
| `policy.py` | **PolicyEngine** — Mode `log-only` (JSONL) ou `auto-correct` (réinvocation LLM). Mesure le TTR (Time-To-Recover). |
| `middleware.py` | **HalluGuardMiddleware** — Hooks `pre_node` / `post_node` pour LangGraph. Variante B : M1 seul. Variante C : M1 + M2 + mémoire persistante. |

### Agents et protocoles (`src/agents/`)

| Fichier | Description |
|---|---|
| `rag_agent.py` | Pipeline RAG LangGraph 4 nœuds : `retrieval → reasoning → tool_call → generation`. Construit un `StateGraph` compilé, supporte les variantes A/B/C. |
| `a2a_protocol.py` | Protocole A2A (Google DeepMind 2024). Implémente `AgentCard`, `TaskObject` (états : `submitted → working → completed/failed`), `delegate_verification()`. Journal : `results/logs/a2a_exchanges.log`. |

### Serveur MCP (`src/mcp_servers/`)

| Fichier | Description |
|---|---|
| `halluguard_mcp.py` | Serveur FastMCP — 4 tools exposés via stdio. `verify_claim` (M1 NLI), `check_temporal_coherence` (M2 BeliefState), `add_fact`, `get_belief_state`. Journal : `results/logs/mcp_calls.log`. |

### Mémoire et expériences (`src/memory/`, `src/experiments/`)

| Fichier | Description |
|---|---|
| `memory/persistent_memory.py` | Sérialisation JSON du BeliefState entre sessions. Répertoire : `data/memory/`. |
| `experiments/baseline_A.py` | Variante A — baseline sans HalluGuard (détection 0 %). |
| `experiments/variantB.py` | Variante B — M1 NLI seul (76.7 %). |
| `experiments/variantC.py` | Variante C — M1 + M2 + MCP (85.0 %). |

### Tests et benchmark (`src/tests/`)

| Fichier | Description |
|---|---|
| `test_halluguard.py` | Suite pytest consolidée — 23 tests, 5 composants (Verifier, DAG, BeliefState, Policy, MCP, Middleware). |
| `benchmark_runner.py` | Runner HaluEval-Agentic — variantes A/B/C, métriques PBR@1 et latence. |
| `conftest.py` | Fixtures pytest session-scoped (NLI chargé une seule fois). |

### Données (`data/`)

| Répertoire / Fichier | Description |
|---|---|
| `data/documents/` | 5 fichiers de faits précis et chiffrés (histoire IA, LLM, RAG, frameworks, hallucinations) — 70 chunks indexés dans ChromaDB. |
| `data/chromadb/` | Base vectorielle HNSW persistante (`halluguard_docs`, indexation O(log n), rappel >95 %). |
| `data/halueval/scenarios.jsonl` | 60 scénarios hallucinés T1-T5 (longueurs 3/7/12 nœuds, distribution équilibrée). |
| `data/memory/` | Sessions BeliefState JSON persistantes inter-sessions. |

### Résultats et logs (`results/`)

| Fichier | Description |
|---|---|
| `resultats_comparatifs.json` | Résultats benchmark complets — variantes A/B/C, métriques par type T1-T5. |
| `logs/halluguard.log` | Événements de détection JSONL (policy engine). |
| `logs/a2a_exchanges.log` | Échanges inter-agents A2A (task_id, from/to, label, score, duration_ms). |
| `logs/mcp_calls.log` | Appels outils MCP (tool, args tronqués, résultat, timestamp). |

### Scripts de démo

| Fichier | Description |
|---|---|
| `demo.bat` | Script Windows de bout en bout : venv, MCP background, 3 questions, logs, métriques. |
| `demo_pipeline.py` | Démo automatique 3 questions via A2A (sans Mistral, ~15 s). |
| `demo_interactive.py` | Démo interactive pour la soutenance : question manuelle, BeliefState, mode log/auto-correct. |

---

## Résultats obtenus

### Benchmark HaluEval-Agentic (90 scénarios : 60 hallucinés + 30 corrects)

| Variante | Description | Rappel | Précision | F1 | FPR | Overhead médian |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **A** | Baseline RAG sans HalluGuard | 0.0 % | — | 0.0 % | 0.0 % | 0 ms |
| **B** ⭐ | HalluGuard M1 (NLI seul) — **recommandée** | 76.7 % | **90.2 %** | **82.9 %** | 16.7 % | **~40 ms** |
| **C** | HalluGuard M1 + M2 (BeliefState) | **86.7 %** | 68.4 % | 76.5 % | 80.0 % | ~194 ms |

> **Lecture honnête** : M1 (Variante B) porte tout le gain net de détection. L'ajout de M2 (Variante C) augmente le rappel (+10 pts) mais **effondre la précision** (90,2 → 68,4 %, FPR 16,7 → 80 %) : sur un benchmark *mono-tour*, M2 sur-déclenche sur des faits de session sans rapport. M2 vise les contradictions *inter-tours* (sessions conversationnelles), un régime non couvert par ce benchmark. **La Variante B est donc recommandée.**

### Détection par type d'hallucination (12 scénarios chacun)

| Type | Description | Variante B | Variante C |
|---|---|:---:|:---:|
| **T1** | Perception / Retrieval | 91.7 % | 91.7 % |
| **T2** | Mémoire temporelle | 83.3 % | 100.0 % |
| **T3** | Planification / Reasoning | 50.0 % | 66.7 % |
| **T4** | Causale / Generation | 83.3 % | 83.3 % |
| **T5** | Délégation / Tool call | 75.0 % | 91.7 % |

> ⚠️ Les gains de rappel de la Variante C par type **ne doivent pas être lus isolément** : ils s'accompagnent d'une chute de précision globale (FPR 80 %). Voir la matrice de confusion dans `results/SUMMARY.json`.

### Validation de la taxonomie & calibration

- **Accord inter-annotateurs** (2 auteurs, annotation réelle à l'aveugle) : **κ = 0,94** (global), κ = 0,75 sur la distinction ambiguë T2/T3.
- **Calibration** : le score NLI brut est mal calibré (ECE = 0,24) ; Platt scaling le réduit à 0,16.
- **Conformal prediction** : seuil de détection avec garantie de couverture (rappel cible respecté ; cf. `results/conformal_calibration.json`).

### Tests unitaires

```
23 passed in 13.58s — Python 3.13.11, pytest-9.0.3, Windows 11
Composants : LightweightVerifier (4) · DependencyDAG (4) · BeliefState (4) · PolicyEngine (4) · MCPServer (4) · Middleware (3)
```

### Protocole A2A

- 3 échanges inter-agents loggés lors de la démo (`results/logs/a2a_exchanges.log`)
- États de tâche : `submitted → working → completed` tracés avec timestamps ISO 8601
- Overhead A2A (TaskObject + MCP bridge) : 100–300 ms selon la charge NLI

---

## Architecture

```
Question
   │
   ▼
[retrieval] ──► ChromaDB HNSW (70 chunks)
   │                │
   │         ◄──────┘  documents
   │
   ▼
[reasoning] ──► Mistral 7B (Ollama)
   │                │
   │    ┌───────────┘  texte intermédiaire
   │    │
   │    ▼
   │  HalluGuardMiddleware
   │    ├── M1 : NLI cross-encoder (claim vs evidences)
   │    ├── M2 : BeliefState (cohérence temporelle)
   │    ├── DependencyDAG (propagation invalidation)
   │    └── PolicyEngine (log / auto-correct)
   │
   ▼
[tool_call] ──► ChromaDB recherche complémentaire
   │
   ▼
[generation] ──► Mistral 7B → réponse finale
   │
   ▼
 Réponse vérifiée + événements HalluGuard
```

---

## Reproduire exactement nos résultats

### Dépendances exactes (versions testées)

```
Python          3.13.11
torch           2.12.0
transformers    5.8.1
sentence-transformers 5.5.0
chromadb        1.5.9
langchain       1.3.1
langgraph       1.2.0
mcp             1.27.1
fastmcp         3.3.1
networkx        3.6.1
pytest          9.0.3
ollama          0.6.2   (optionnel — pipeline RAG complet uniquement)
```

### Commandes dans l'ordre

```powershell
# 1. Installer l'environnement
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt

# 2. Réindexer ChromaDB (si première installation)
venv\Scripts\python.exe src\tools\rag_tools.py

# 3. Lancer la suite de tests (23 tests, ~14s)
venv\Scripts\python.exe -m pytest src/tests/test_halluguard.py -v

# 4. Lancer le benchmark complet (90 scénarios : 60 hallucinés + 30 corrects)
venv\Scripts\python.exe src/tests/benchmark_runner.py --all

# 5. Régénérer TOUS les chiffres officiels -> results/SUMMARY.json
#    (scores NLI, calibration, conformal, stats McNemar, kappa réel)
venv\Scripts\python.exe scripts/run_all.py
```

#### Expériences optionnelles (nécessitent un LLM — Ollama local OU clé API dans `.env`)

```powershell
# Self-consistency (signal d'incertitude) — 16 questions, k=5 échantillons
venv\Scripts\python.exe src/experiments/self_consistency_eval.py --backend mistral_api

# Baseline LLM-juge (Mistral 7B) comparée à M1 sur les 90 scénarios
venv\Scripts\python.exe src/experiments/baseline_llm_judge.py
```

> Pour ces deux scripts, créez un fichier `.env` (gitignoré) contenant `MISTRAL_API_KEY=...`, ou utilisez `--backend ollama` pour rester 100 % local.

### Temps d'exécution (machine de référence — CPU Intel, 16 Go RAM)

| Opération | Temps |
|---|---|
| Tests unitaires (23 tests) | 13.58 s |
| Benchmark Variante B (60 scénarios, NLI seul) | ~7 s |
| Benchmark Variante C (60 scénarios, NLI + BeliefState) | ~17 s |
| Chargement modèle NLI (première fois) | ~5 s |
| Demo pipeline 3 questions | ~25 s |

### Reproductibilité des résultats NLI

Le modèle `cross-encoder/nli-MiniLM2-L6-H768` est déterministe pour une entrée donnée (pas d'échantillonnage stochastique). Les résultats du benchmark sont **reproductibles à 100 %** à architecture matérielle identique. Pas de seed aléatoire requis.

---

## Configuration requise

| Composant | Version / Détail |
|---|---|
| Python | 3.10+ (testé sur 3.13.11) |
| Mémoire RAM | 2 Go minimum (NLI CPU) · 6 Go pour Mistral Q4 |
| GPU | Non requis (inférence CPU uniquement) |
| Ollama + Mistral | Optionnel — nécessaire pour pipeline RAG complet |
| pytest | 9.0.3 |
| ChromaDB | PersistentClient, HNSW |
