"""conftest.py — Configuration pytest pour src/tests/."""
import os
import sys
from pathlib import Path

# ── Contournement crash PyTorch/transformers sur Python 3.13 Windows ────────
# Le chargement multi-threadé du modèle provoque une access violation dans
# torch.storage sur Python 3.13 (bug connu transformers >= 4.46).
# Désactiver le parallélisme avant tout import torch/transformers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Ajoute la racine du projet au sys.path pour tous les tests
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
