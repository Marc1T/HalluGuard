# Guide d'annotation — Taxonomie des hallucinations agentiques (T1–T5)

Guide de codage pour l'étude d'accord inter-annotateurs (κ de Cohen).
À lire **avant** d'annoter `data/taxonomy_annotation_sheet.csv`.

## Objectif

Mesurer si deux annotateurs humains, appliquant **indépendamment** ces définitions,
classent les mêmes scénarios de la même façon. Le κ qui en résulte valide (ou non)
que la taxonomie est **reproductible** et non arbitraire.

## Règles d'intégrité (obligatoires)

1. **Indépendance** : chacun remplit **uniquement sa colonne** (`ann_marc` ou `ann_souleymane`).
2. **Aveugle** : ne regardez **pas** la colonne de l'autre, ni le champ `hallucination_type`
   du fichier `data/halueval/scenarios.jsonl` (c'est le « gold », il ne doit pas vous influencer).
3. **Pas de concertation** pendant l'annotation. La discussion des désaccords se fait
   **après** le calcul du κ (et ne modifie pas les annotations déjà posées).
4. Une seule étiquette par ligne, parmi `T1, T2, T3, T4, T5`.

## Les 5 catégories

Pour chaque ligne, vous disposez de : `query` (question), `claim` (affirmation
potentiellement hallucinée), `evidence` (document de référence), `node_type`
(étape du pipeline où le claim apparaît).

| Type | Nom | Indice principal | Définition opérationnelle |
|---|---|---|---|
| **T1** | Perception | nœud `retrieval` | Le claim s'appuie sur des documents récupérés inexacts / hors-sujet : l'erreur vient de la **récupération**. |
| **T2** | Mémoire temporelle | nœud `reasoning` + marqueur temporel | Le claim contredit un fait antérieur ou contient un **marqueur temporel** (« avant », « récemment », « depuis la mise à jour », « en 20XX », « désormais »). L'erreur porte sur la **chronologie / un état mémorisé**. |
| **T3** | Planification | nœud `reasoning`, sans marqueur temporel | **Enchaînement logique faux** : prémisses correctes mais conclusion qui ne suit pas, raisonnement « en étapes » (« d'abord… ensuite… donc »). L'erreur est **logique**, pas temporelle. |
| **T4** | Causale | nœud `generation` | Le claim **contredit directement** le document de référence (contradiction factuelle frontale, souvent un chiffre/fait faux). |
| **T5** | Délégation | nœud `tool_call` | La sortie attribuée à un **outil / API / appel** diverge de ce qui est attendu. Mentionne souvent un outil, une recherche, un appel. |

## Procédure de décision conseillée

1. Le `node_type` oriente fortement : `retrieval`→T1, `generation`→T4, `tool_call`→T5.
   **Mais vérifiez toujours le contenu** : un claim au nœud `generation` qui repose en
   réalité sur une mauvaise récupération peut être T1.
2. Si `node_type = reasoning`, c'est **T2 ou T3** — c'est là que se joue le vrai jugement :
   - marqueur temporel / contradiction d'un fait établi → **T2** ;
   - chaîne logique fausse sans dimension temporelle → **T3**.
3. En cas d'hésitation, choisissez la catégorie dont la **définition opérationnelle**
   colle le mieux, et ne revenez pas en arrière après avoir vu la suite.

## Cas limites documentés (à connaître, pas à harmoniser pendant l'annotation)

- **T2 vs T3** : un raisonnement séquentiel (« à chaque étape… ») peut sembler temporel (T2)
  alors qu'il relève de la planification (T3). C'est la principale source de désaccord
  attendue — et c'est un résultat scientifique légitime à rapporter.
- **T4 vs T5** : une sortie d'outil qui contredit un document peut hésiter entre causale (T4)
  et délégation (T5) ; trancher selon l'**origine** de l'erreur (document → T4, outil → T5).

## Combien de scénarios ?

- **Recommandé : les 60** (12 par type) pour un κ robuste.
- Minimum accepté par le script : 20 (mais 60 est nettement préférable pour la publication).

## Après l'annotation

```
venv\Scripts\python.exe scripts\run_all.py
```
→ calcule le κ réel (`results/cohen_kappa_result.json`) et régénère `results/SUMMARY.json`.
Analysez ensuite les désaccords listés (champ `desaccords`) dans la section Résultats de l'article.
