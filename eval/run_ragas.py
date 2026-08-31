import json
from src.retrieve import search
from src.answer import generate_answer
from src.config import ROOT
import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import GEMINI_MODEL, EMBED_MODEL

load_dotenv()

REFS = ROOT / "eval" / "references.json"
DATASET = ROOT / "eval" / "captured.json"

def build_dataset():
    done = {}
    if DATASET.exists():
        for row in json.loads(DATASET.read_text()):
            done[row["question"]] = row

    rows = json.loads(REFS.read_text())
    data = []
    for r in rows:
        if r["expect"] != "answered":
            continue
        if r["q"] in done:
            data.append(done[r["q"]])
            print(f"skip (done): {r['q']}")
            continue
        hits = search(r["q"], user_grade=r["grade"])
        contexts = [h.payload["text"] for h in hits]
        answer = generate_answer(r["q"], hits, r["grade"])
        data.append({
            "question": r["q"],
            "contexts": contexts,
            "answer": answer,
            "ground_truth": r["reference"],
        })
        DATASET.write_text(json.dumps(data, indent=2), encoding="utf-8")  # save each
        print(f"captured: {r['q']}")
    return data

def run_evaluation():
    data = build_dataset()
    dataset = Dataset.from_list(data)

    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=os.getenv("GEMINI_API_KEY"))
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    print("\n=== RAGAS baseline ===")
    print(result)
    return result

if __name__ == "__main__":
    run_evaluation()