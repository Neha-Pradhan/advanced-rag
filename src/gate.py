from src.retrieve import search
import os
from dotenv import load_dotenv
from google import genai
from src.config import GEMINI_MODEL

load_dotenv()
_llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Below this, top hit is almost certainly irrelevant (measured: out-of-corpus ~0.25,
# in-corpus ~0.5). Cheap first-stage reject before spending an LLM call.
SCORE_FLOOR = 0.35


def score_gate(hits):
    if not hits:
        return False
    return hits[0].score >= SCORE_FLOOR

def llm_relevance_check(query: str, hits) -> bool:
    context = "\n\n".join(h.payload["text"] for h in hits)
    prompt = (
        "You are checking whether retrieved textbook passages can answer a student's "
        "question. Answer with ONLY 'yes' or 'no'.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved passages:\n{context}\n\n"
        "Can these passages answer the question? Answer yes or no:"
    )
    resp = _llm.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return resp.text.strip().lower().startswith("yes")

def gate(query: str, hits) -> bool:
    # Stage 1: cheap score floor. Fails fast, no LLM call.
    if not score_gate(hits):
        return False
    # Stage 2: LLM judges actual relevance.
    return llm_relevance_check(query, hits)

if __name__ == "__main__":
    for q in ["what is a cell made of", "who won the cricket world cup", "how do black holes form"]:
        hits = search(q, user_grade=8)
        result = gate(q, hits)
        print(f"{'PASS' if result else 'REJECT'}  {q!r}")