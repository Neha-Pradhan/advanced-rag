import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct
from src.chunk import build_all_chunks

from src.config import EMBED_MODEL

load_dotenv()

COLLECTION = "ncert_science"

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


def create_collection():
    if client.collection_exists(COLLECTION):
        print(f"Collection '{COLLECTION}' already exists — skipping create")
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION}' created: 384-dim, cosine")

model = SentenceTransformer(EMBED_MODEL)


def embed_and_load():
    chunks = build_all_chunks()
    print(f"Embedding {len(chunks)} chunks...")

    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    points = [
        PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i])
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Upserted {len(points)} points into '{COLLECTION}'")

if __name__ == "__main__":
    create_collection()
    embed_and_load()