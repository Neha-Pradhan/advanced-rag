import os
from dotenv import load_dotenv
from google import genai
from src.config import GEMINI_MODEL
from src.retrieve import search
from src.gate import gate
import time
from google.genai.errors import ServerError

load_dotenv()
_llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_answer(query: str, hits, user_grade: int) -> str:
    context = "\n\n".join(
        f"[Chapter {h.payload['chapter']}, Section {h.payload['section']}]\n{h.payload['text']}"
        for h in hits
    )
    prompt = (
        f"You are a friendly science tutor for a Grade {user_grade} student. "
        "Answer the question using ONLY the textbook passages below. "
        "Do not add facts from outside the passages. "
        "Explain simply, at the student's grade level. "
        "If the passages mention a figure (like 'Fig. 2.2'), tell the student to look at it.\n\n"
        f"Question: {query}\n\n"
        f"Textbook passages:\n{context}\n\n"
        "Answer:"
    )
    for attempt in range(5):
        try:
            resp = _llm.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return resp.text.strip()
        except ServerError:
            wait = 2 ** attempt
            print(f"  503, retry in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Gemini unavailable after retries")

def answer_query(query: str, user_grade: int) -> dict:
    hits = search(query, user_grade=user_grade)

    # Gate says no → consent fallback, don't answer.
    if not gate(query, hits):
        return {
            "status": "not_found",
            "message": (
                f"I couldn't find this in your Grade {user_grade} science books. "
                "It might be covered in a higher grade. "
                "Want me to look there instead?"
            ),
            "answer": None,
        }

    # Gate says yes → generate grounded answer.
    answer = generate_answer(query, hits, user_grade)
    sources = sorted({
        f"Ch {h.payload['chapter']} §{h.payload['section']}" for h in hits
    })
    return {"status": "answered", "message": None, "answer": answer, "sources": sources}


if __name__ == "__main__":
    for q in ["what is a cell made of", "how do black holes form"]:
        print(f"\n=== {q!r} ===")
        result = answer_query(q, user_grade=8)
        if result["status"] == "answered":
            print(result["answer"])
            print("\nSources:", ", ".join(result["sources"]))
        else:
            print(result["message"])