import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src.config import PROCESSED, EMBED_MODEL, EMBED_MAX_TOKENS

# NCERT section headings look like "2.1 What Is a Cell?" at line start.
# Pattern: digits.digits, a space, then a Title-Case word.
# Heading: "2.1 Title", title may wrap to a second line.
HEADING_RE = re.compile(r"^(\d+\.\d+)\s+([A-Z].*)$", re.MULTILINE)

# Load model once; we only need its tokenizer for length counting here.
_model = SentenceTransformer(EMBED_MODEL)

SAFETY_MARGIN = 10   # small buffer for tokenizer edge cases
CHUNK_OVERLAP = 30   # tokens carried between forced sub-splits

def token_len(text: str) -> int:
    # Length in MiniLM tokens, not characters — this is what the 256 budget means.
    return len(_model.tokenizer.encode(text, add_special_tokens=False))

def find_headings(text: str):
    # now also returns the start position of each heading in the text
    return [
        {"num": m.group(1), "title": m.group(2).strip(), "start": m.start()}
        for m in HEADING_RE.finditer(text)
    ]

def split_sections(text: str):
    headings = find_headings(text)
    sections = []

    # Intro: everything before the first heading
    if headings and headings[0]["start"] > 0:
        intro = text[: headings[0]["start"]].strip()
        if intro:
            sections.append({"num": "intro", "title": "", "text": intro})

    # Each section: from its heading to the next heading's start
    for i, h in enumerate(headings):
        end = headings[i + 1]["start"] if i + 1 < len(headings) else len(text)
        body = text[h["start"] : end].strip()
        sections.append({"num": h["num"], "title": h["title"], "text": body})

    return sections

def chunk_sections(sections):
    chunks = []
    for s in sections:
        heading = f"{s['num']} {s['title']}".strip()
        budget = EMBED_MAX_TOKENS - token_len(heading) - SAFETY_MARGIN

        # If the whole section already fits, keep it as one chunk.
        if token_len(s["text"]) <= EMBED_MAX_TOKENS:
            chunks.append({"section": s["num"], "title": s["title"], "text": s["text"]})
            continue

        # Oversized: sub-split the body under this section's budget.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=token_len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        for piece in splitter.split_text(s["text"]):
            # Re-inject the heading so every sub-chunk carries its section context.
            body = f"{heading}\n{piece}" if not piece.startswith(s["num"]) else piece
            chunks.append({"section": s["num"], "title": s["title"], "text": body})

    return chunks

if __name__ == "__main__":
    text = (PROCESSED / "grade8_ch02.txt").read_text(encoding="utf-8")
    sections = split_sections(text)
    chunks = chunk_sections(sections)
    print(f"{len(sections)} sections -> {len(chunks)} chunks\n")
    for c in chunks:
        print(f"[{c['section']}] {token_len(c['text'])} tok :: {c['text'][:70]!r}")