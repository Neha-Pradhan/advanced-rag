from pathlib import Path

# Project root = one level up from this file (src/config.py -> advanced-rag/)
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FIGURES = DATA / "figures"

# Embedding model + its hard token limit (drives chunk size)
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_MAX_TOKENS = 256

GEMINI_MODEL = "gemini-3.5-flash-lite"

from sentence_transformers import SentenceTransformer

# Single shared model instance — import this everywhere, never re-instantiate.
_embed_model = None


def get_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model