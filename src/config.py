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