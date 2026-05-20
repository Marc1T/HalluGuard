"""
create_correct_scenarios.py
Appends 30 correct (non-hallucinated) scenarios to data/halueval/scenarios.jsonl.
These scenarios have expected_label="correct" and hallucination_type=null.
Claims paraphrase the evidence so NLI returns entailment, not contradiction.
Distribution: 6 retrieval, 6 reasoning, 6 generation, 6 tool_call, 6 mixed.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_FILE = ROOT / "data" / "halueval" / "scenarios.jsonl"

correct_scenarios = [
    # ── s061-s066 : retrieval (T1 zone, expected_label=correct) ──────────────
    {
        "id": "s061",
        "length": 6,
        "node_type": "retrieval",
        "hallucination_type": None,
        "query": "Quelle architecture utilise le modele Transformer ?",
        "claim": "Le modele Transformer repose sur un mecanisme d'attention qui etablit des relations entre tous les tokens d'une sequence.",
        "evidences": [
            "Le Transformer utilise un mecanisme d'auto-attention permettant de modeliser les dependances entre tous les tokens d'une sequence en parallele."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s062",
        "length": 6,
        "node_type": "retrieval",
        "hallucination_type": None,
        "query": "Comment ChromaDB stocke-t-il les embeddings ?",
        "claim": "ChromaDB est une base de donnees vectorielle qui indexe les embeddings pour permettre des recherches par similarite semantique.",
        "evidences": [
            "ChromaDB est une base de donnees vectorielle open-source qui stocke et indexe des embeddings afin de permettre des recherches par similarite semantique efficaces."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s063",
        "length": 6,
        "node_type": "retrieval",
        "hallucination_type": None,
        "query": "Qu'est-ce que le RAG ?",
        "claim": "Le RAG combine la recherche de documents pertinents avec la generation de texte pour produire des reponses ancrees dans des sources exterieures.",
        "evidences": [
            "Le Retrieval-Augmented Generation (RAG) associe un module de recherche documentaire a un generateur de texte, ancrant ainsi les reponses dans des documents recuperes."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s064",
        "length": 6,
        "node_type": "retrieval",
        "hallucination_type": None,
        "query": "Mistral est-il un modele de langage leger ?",
        "claim": "Mistral 7B est un modele de langage leger optimise pouvant fonctionner localement sur du materiel grand public.",
        "evidences": [
            "Mistral 7B est un modele de langage leger et efficace concu pour tourner localement sur du materiel accessible, y compris des CPU standards."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s065",
        "length": 6,
        "node_type": "retrieval",
        "hallucination_type": None,
        "query": "LangGraph permet-il de definir des pipelines a plusieurs noeuds ?",
        "claim": "LangGraph est un framework qui permet de construire des pipelines agentiques multi-noeuds avec des transitions entre etats.",
        "evidences": [
            "LangGraph est un framework Python base sur LangChain permettant de definir des graphes d'agents avec des noeuds et des transitions d'etat pour des pipelines complexes."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s066",
        "length": 6,
        "node_type": "retrieval",
        "hallucination_type": None,
        "query": "A quoi servent les embeddings dans un systeme RAG ?",
        "claim": "Les embeddings encodent le sens semantique des textes sous forme de vecteurs numeriques pour mesurer leur similarite.",
        "evidences": [
            "Dans un systeme RAG, les embeddings representent les textes sous forme de vecteurs numeriques denses capturant leur semantique, permettant de mesurer la proximite entre une requete et les documents."
        ],
        "expected_label": "correct",
    },
    # ── s067-s072 : reasoning (T2/T3 zone, expected_label=correct) ────────────
    {
        "id": "s067",
        "length": 8,
        "node_type": "reasoning",
        "hallucination_type": None,
        "query": "Le RAG reduit-il les hallucinations ?",
        "claim": "Le RAG reduit les hallucinations en ancrant les reponses dans des documents recuperes plutot qu'en s'appuyant uniquement sur la memoire parametrique du modele.",
        "evidences": [
            "Le RAG diminue les hallucinations en conditionnant la generation sur des documents externes pertinents, limitant ainsi la dependance a la connaissance interne du LLM."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s068",
        "length": 8,
        "node_type": "reasoning",
        "hallucination_type": None,
        "query": "En quoi le Transformer differe-t-il des RNN ?",
        "claim": "Contrairement aux RNN qui traitent les sequences de facon sequentielle, le Transformer traite tous les tokens en parallele grace a l'attention.",
        "evidences": [
            "Le Transformer se distingue des RNN par son traitement parallele des sequences via le mecanisme d'attention, eliminarant la dependance sequentielle des architectures recurrentes."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s069",
        "length": 8,
        "node_type": "reasoning",
        "hallucination_type": None,
        "query": "GPT-2 utilise-t-il un encodeur et un decodeur ?",
        "claim": "GPT-2 est un modele de langage base uniquement sur la partie decodeur du Transformer, sans encodeur.",
        "evidences": [
            "GPT-2 est une architecture Transformer decoder-only : il ne comprend pas d'encodeur, contrairement aux modeles sequence-to-sequence comme T5."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s070",
        "length": 8,
        "node_type": "reasoning",
        "hallucination_type": None,
        "query": "Qu'est-ce que le fine-tuning d'un LLM ?",
        "claim": "Le fine-tuning adapte un LLM pre-entraine a une tache specifique en continuant l'entrainement sur un jeu de donnees cible.",
        "evidences": [
            "Le fine-tuning consiste a reprendre un modele pre-entraine et a poursuivre son entrainement sur des donnees specifiques a une tache, ajustant ses parametres pour ameliorer ses performances."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s071",
        "length": 8,
        "node_type": "reasoning",
        "hallucination_type": None,
        "query": "Les agents LLM peuvent-ils halluciner leurs propres capacites ?",
        "claim": "Les agents bases sur des LLM peuvent signaler des capacites qu'ils ne possedent pas reellement, un phenomene appele hallucination d'agence.",
        "evidences": [
            "Les agents LLM peuvent produire des hallucinations sur leurs propres aptitudes, declarant pouvoir effectuer des actions indisponibles, ce qu'on appelle hallucination d'agence."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s072",
        "length": 8,
        "node_type": "reasoning",
        "hallucination_type": None,
        "query": "Quand a ete introduit le modele Transformer ?",
        "claim": "L'architecture Transformer a ete presentee en 2017 dans l'article Attention is All You Need de Vaswani et al.",
        "evidences": [
            "Le Transformer a ete propose par Vaswani et al. dans l'article Attention is All You Need publie en 2017, revolutionnant le traitement du langage naturel."
        ],
        "expected_label": "correct",
    },
    # ── s073-s078 : generation (T3/T4 zone, expected_label=correct) ───────────
    {
        "id": "s073",
        "length": 7,
        "node_type": "generation",
        "hallucination_type": None,
        "query": "La taille d'un LLM elimine-t-elle les hallucinations ?",
        "claim": "Augmenter la taille d'un LLM ne suffit pas a eliminer les hallucinations ; des techniques complementaires comme le RAG restent necessaires.",
        "evidences": [
            "Meme les tres grands LLM produisent des hallucinations : la taille seule ne resout pas le probleme et des approches comme le RAG ou la verification externe sont indispensables."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s074",
        "length": 7,
        "node_type": "generation",
        "hallucination_type": None,
        "query": "Le RAG ameliore-t-il la qualite des reponses generees ?",
        "claim": "Le RAG ameliore la qualite des reponses en fournissant au generateur des informations recentes et pertinentes issues de la base documentaire.",
        "evidences": [
            "En enrichissant le contexte du generateur avec des documents pertinents recuperes, le RAG produit des reponses plus precises, actuelles et factuellement fondees."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s075",
        "length": 7,
        "node_type": "generation",
        "hallucination_type": None,
        "query": "Comment la temperature influence-t-elle la generation de texte ?",
        "claim": "Une temperature proche de zero rend la generation deterministe, favorisant la reponse la plus probable a chaque etape.",
        "evidences": [
            "Dans les LLM, une temperature basse (proche de 0) reduit l'aleatoire de l'echantillonnage et produit des sorties quasi-deterministes en selectionnant le token le plus probable."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s076",
        "length": 7,
        "node_type": "generation",
        "hallucination_type": None,
        "query": "ChromaDB utilise-t-il HNSW pour l'indexation ?",
        "claim": "ChromaDB utilise un index HNSW pour effectuer des recherches approximatives de voisins les plus proches de maniere efficace.",
        "evidences": [
            "ChromaDB s'appuie sur l'algorithme HNSW (Hierarchical Navigable Small World) pour indexer les embeddings et effectuer des recherches ANN rapides et precisees."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s077",
        "length": 7,
        "node_type": "generation",
        "hallucination_type": None,
        "query": "La qualite des embeddings influence-t-elle le RAG ?",
        "claim": "La qualite des embeddings conditionne directement la pertinence des documents recuperes et donc la qualite des reponses du systeme RAG.",
        "evidences": [
            "Dans un pipeline RAG, des embeddings de haute qualite capturant bien la semantique ameliorent la precision du module de recuperation, ce qui se repercute directement sur la qualite des reponses generees."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s078",
        "length": 7,
        "node_type": "generation",
        "hallucination_type": None,
        "query": "Les modeles NLI compacts sont-ils utilisables en production ?",
        "claim": "Des modeles NLI compacts comme MiniLM peuvent etre deployes en production pour la verification de faits en temps reel.",
        "evidences": [
            "Les modeles NLI de petite taille tels que cross-encoder/nli-MiniLM2 offrent un bon compromis vitesse/precision et sont adaptes aux pipelines de verification factuelle en production."
        ],
        "expected_label": "correct",
    },
    # ── s079-s084 : tool_call (T5 zone, expected_label=correct) ───────────────
    {
        "id": "s079",
        "length": 6,
        "node_type": "tool_call",
        "hallucination_type": None,
        "query": "GPT-4 est-il open source ?",
        "claim": "L'outil de recherche a confirme que GPT-4 est un modele proprietaire d'OpenAI dont les poids ne sont pas publiquement disponibles.",
        "evidences": [
            "GPT-4 est un modele ferme et proprietaire developpe par OpenAI ; ses poids et son architecture ne sont pas rendus publics."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s080",
        "length": 6,
        "node_type": "tool_call",
        "hallucination_type": None,
        "query": "ChromaDB peut-il fonctionner localement ?",
        "claim": "L'outil de configuration a indique que ChromaDB peut etre deploye en mode local sans necessiter de connexion a un service cloud.",
        "evidences": [
            "ChromaDB est disponible en mode embarque (in-process) et peut fonctionner entierement en local sans dependance a un service cloud externe."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s081",
        "length": 6,
        "node_type": "tool_call",
        "hallucination_type": None,
        "query": "Mistral peut-il tourner sur CPU ?",
        "claim": "L'outil de diagnostic systeme a verifie que Mistral 7B peut s'executer sur CPU via Ollama, bien que plus lentement qu'avec un GPU.",
        "evidences": [
            "Mistral 7B est utilisable sur CPU via des outils comme Ollama ou llama.cpp, mais les performances sont inferieures a celles obtenues avec un GPU dedie."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s082",
        "length": 6,
        "node_type": "tool_call",
        "hallucination_type": None,
        "query": "LangGraph supporte-t-il l'execution parallele de noeuds ?",
        "claim": "L'outil d'analyse de pipeline a confirme que LangGraph supporte l'execution parallele de plusieurs noeuds dans un meme graphe.",
        "evidences": [
            "LangGraph permet l'execution parallele de noeuds independants dans un graphe d'agents, ameliorant les performances des pipelines multi-etapes."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s083",
        "length": 6,
        "node_type": "tool_call",
        "hallucination_type": None,
        "query": "La complexite de l'attention est-elle quadratique ?",
        "claim": "L'outil de profilage a rapporte que le mecanisme d'attention standard a une complexite quadratique en memoire et en temps par rapport a la longueur de sequence.",
        "evidences": [
            "Le mecanisme d'attention du Transformer a une complexite en O(n^2) en temps et en memoire par rapport a la longueur de sequence n, ce qui limite son passage a l'echelle."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s084",
        "length": 6,
        "node_type": "tool_call",
        "hallucination_type": None,
        "query": "OpenAI utilise-t-il le RLHF pour aligner ses modeles ?",
        "claim": "L'outil de recherche a confirme qu'OpenAI utilise le RLHF pour aligner ses modeles sur les preferences humaines.",
        "evidences": [
            "OpenAI emploie le Reinforcement Learning from Human Feedback (RLHF) pour affiner ses modeles GPT afin de les rendre plus utiles, inoffensifs et honnetes."
        ],
        "expected_label": "correct",
    },
    # ── s085-s090 : mixed node types (expected_label=correct) ─────────────────
    {
        "id": "s085",
        "length": 7,
        "node_type": "retrieval",
        "hallucination_type": None,
        "query": "Sentence-transformers produit-il des embeddings semantiques ?",
        "claim": "Sentence-transformers genere des embeddings semantiques denses en encodant les phrases avec un modele Transformer fine-tune sur des paires de phrases.",
        "evidences": [
            "La bibliotheque sentence-transformers produit des embeddings de phrases denses et semantiquement riches en utilisant des modeles Transformer (BERT, RoBERTa) fine-tunes sur des donnees de similarite semantique."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s086",
        "length": 7,
        "node_type": "reasoning",
        "hallucination_type": None,
        "query": "La similarite cosinus est-elle adaptee aux embeddings ?",
        "claim": "La similarite cosinus est une mesure standard et efficace pour comparer des embeddings vectoriels, y compris en haute dimension.",
        "evidences": [
            "La similarite cosinus est la metrique de reference pour comparer des embeddings de grande dimension car elle mesure l'angle entre vecteurs independamment de leur norme."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s087",
        "length": 7,
        "node_type": "generation",
        "hallucination_type": None,
        "query": "Combien de faits le BeliefState peut-il stocker ?",
        "claim": "Le BeliefState de HalluGuard peut maintenir jusqu'a 50 faits actifs et expulse les faits de faible confiance en priorite.",
        "evidences": [
            "Le composant BeliefState de HalluGuard maintient jusqu'a 50 faits simultanement et applique une politique d'expulsion basee sur la confiance et la recence des faits."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s088",
        "length": 7,
        "node_type": "generation",
        "hallucination_type": None,
        "query": "Les LLM peuvent-ils generer du code de qualite ?",
        "claim": "Les LLM modernes sont capables de generer du code fonctionnel dans de nombreux langages de programmation mais peuvent introduire des erreurs subtiles.",
        "evidences": [
            "Les grands modeles de langage peuvent produire du code syntaxiquement et fonctionnellement correct dans de multiples langages, bien qu'ils restent sujets a des erreurs logiques et de securite."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s089",
        "length": 7,
        "node_type": "reasoning",
        "hallucination_type": None,
        "query": "LangGraph supporte-t-il les transitions conditionnelles ?",
        "claim": "LangGraph permet de definir des transitions conditionnelles entre noeuds, rendant les pipelines adaptatifs selon l'etat courant.",
        "evidences": [
            "LangGraph supporte les edges conditionnels entre noeuds permettant de router dynamiquement l'execution selon des conditions evaluees sur l'etat du graphe."
        ],
        "expected_label": "correct",
    },
    {
        "id": "s090",
        "length": 7,
        "node_type": "tool_call",
        "hallucination_type": None,
        "query": "Les premiers LLM produisaient-ils des hallucinations frequentes ?",
        "claim": "L'outil de synthese historique a confirme que les premiers LLM produisaient des hallucinations frequentes faute de mecanismes de verification.",
        "evidences": [
            "Les LLM de premiere generation etaient particulierement sujets aux hallucinations en l'absence de modules de verification factuelle ou de RAG, generant des affirmations plausibles mais incorrectes."
        ],
        "expected_label": "correct",
    },
]

# Append to scenarios.jsonl
count = 0
with open(SCENARIOS_FILE, "a", encoding="utf-8") as f:
    for s in correct_scenarios:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
        count += 1

print(f"Appended {count} correct scenarios (s061-s090) to {SCENARIOS_FILE}")

# Verify total count
total = 0
with open(SCENARIOS_FILE, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            total += 1
print(f"Total scenarios in file: {total}")
