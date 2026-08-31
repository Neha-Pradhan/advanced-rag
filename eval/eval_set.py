# Eval set: questions + expected behavior.
# in_corpus: should answer. out_of_corpus: should hit consent fallback.
EVAL = [
    {"q": "What is a cell made of?", "grade": 8, "expect": "answered"},
    {"q": "What is the function of the nucleus in a cell?", "grade": 8, "expect": "answered"},
    {"q": "How does yeast help in making bread?", "grade": 8, "expect": "answered"},
    {"q": "What are microorganisms?", "grade": 8, "expect": "answered"},
    {"q": "How do we see the phases of the moon?", "grade": 8, "expect": "answered"},
    {"q": "What is photosynthesis?", "grade": 7, "expect": "answered"},
    {"q": "How does light travel?", "grade": 7, "expect": "answered"},
    {"q": "What is an eclipse?", "grade": 7, "expect": "answered"},
    {"q": "Who won the cricket world cup?", "grade": 8, "expect": "not_found"},
    {"q": "How do black holes form?", "grade": 8, "expect": "not_found"},
]

if __name__ == "__main__":
    print(f"{len(EVAL)} eval questions")
    for e in EVAL:
        print(f"  [{e['expect']:10}] g{e['grade']}  {e['q']}")