"""conftest.py — Configuration pytest pour src/tests/."""
import sys
from pathlib import Path

# Ajoute la racine du projet au sys.path pour tous les tests
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
