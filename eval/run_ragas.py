import os
import json
from dotenv import load_dotenv
from datasets import Dataset
import mlflow

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import llm_factory
# from ragas.embeddings import HuggingfaceEmbeddings
from ragas.run_config import RunConfig
from google import genai

from src.retrieve import search
from src.answer import generate_answer
from src.config import ROOT, GEMINI_MODEL, EMBED_MODEL

load_dotenv()

REFS = ROOT / "eval" / "references.json"


def build_dataset(top_k=5):
    dataset_file = ROOT / "eval" / f"captured_k{top_k}.json"
    done = {}
    if dataset_file.exists():
        for row in json.loads(dataset_file.read_text()):
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
        hits = search(r["q"], user_grade=r["grade"], top_k=top_k)
        contexts = [h.payload["text"] for h in hits]
        answer = generate_answer(r["q"], hits, r["grade"])
        data.append({
            "question": r["q"],
            "contexts": contexts,
            "answer": answer,
            "ground_truth": r["reference"],
        })
        dataset_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"captured: {r['q']}")
    return data


def run_evaluation(top_k=5):
    data = build_dataset(top_k=top_k)
    dataset = Dataset.from_list(data)

    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    judge_llm = llm_factory(GEMINI_MODEL, provider="google", client=gemini_client)
    # judge_embeddings = HuggingfaceEmbeddings(model_name=EMBED_MODEL)

    mlflow.set_experiment("advanced-rag-retrieval")
    with mlflow.start_run(run_name=f"top_k={top_k}"):
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("embed_model", EMBED_MODEL)
        mlflow.log_param("judge_model", GEMINI_MODEL)
        mlflow.log_param("n_questions", len(data))

        result = evaluate(
            dataset,
            metrics=[context_precision, context_recall],
            llm=judge_llm,
            run_config=RunConfig(max_workers=1, timeout=120),
        )

        scores = {k: float(v) for k, v in result._repr_dict.items()}
        mlflow.log_metrics(scores)
        print(f"\n=== RAGAS (top_k={top_k}) ===")
        print(scores)
    return result


if __name__ == "__main__":
    run_evaluation(top_k=5)