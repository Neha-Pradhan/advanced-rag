from src.config import EMBED_MODEL, get_model
from src.embed import client, COLLECTION
from qdrant_client.models import Filter, FieldCondition, Range

def search(query: str, user_grade: int, top_k: int = 5):
    qvec = get_model().encode(query, normalize_embeddings=True).tolist()
    grade_filter = Filter(
        must=[FieldCondition(key="grade", range=Range(lte=user_grade))]
    )
    hits = client.query_points(
        collection_name=COLLECTION,
        query=qvec,
        query_filter=grade_filter,
        limit=top_k,
    ).points
    return hits


if __name__ == "__main__":
    query = "what is a cell made of"
    print("--- grade 7 (should exclude grade 8 hits) ---")
    for h in search(query, user_grade=7):
        print(f"{h.score:.3f}  g{h.payload['grade']}  [{h.payload['id']}]")
    print("--- grade 8 (includes both) ---")
    for h in search(query, user_grade=8):
        print(f"{h.score:.3f}  g{h.payload['grade']}  [{h.payload['id']}]")