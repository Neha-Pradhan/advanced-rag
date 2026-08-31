import json
from src.retrieve import search
from src.answer import generate_answer
from eval.eval_set import EVAL
from src.config import ROOT

OUT = ROOT / "eval" / "references.json"

import time
from google.genai.errors import ServerError


def gen_with_retry(q, hits, grade, tries=4):
    for attempt in range(tries):
        try:
            return generate_answer(q, hits, grade)
        except ServerError:
            wait = 2 ** attempt  # 1, 2, 4, 8 seconds
            print(f"  503, retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {tries} tries: {q}")


def build():
    existing = {}
    if OUT.exists():
        for r in json.loads(OUT.read_text()):
            if r.get("reference"):
                existing[r["q"]] = r["reference"]

    rows = []
    for e in EVAL:
        if e["expect"] != "answered":
            rows.append({**e, "reference": None})
        elif e["q"] in existing:
            rows.append({**e, "reference": existing[e["q"]]})
            print(f"skip (done): {e['q']}")
        else:
            hits = search(e["q"], user_grade=e["grade"])
            ref = gen_with_retry(e["q"], hits, e["grade"])
            rows.append({**e, "reference": ref})
            print(f"done: {e['q']}")
        OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")  # save after EVERY row

    print(f"\nWrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    build()