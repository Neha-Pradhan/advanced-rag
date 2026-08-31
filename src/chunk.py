import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import PROCESSED, EMBED_MODEL, EMBED_MAX_TOKENS, get_model

# NCERT section headings look like "2.1 What Is a Cell?" at line start.
# Pattern: digits.digits, a space, then a Title-Case word.
# Heading: "2.1 Title", title may wrap to a second line.
HEADING_RE = re.compile(r"^(\d+\.\d+)\s+([A-Z].*)$", re.MULTILINE)

# Load model once; we only need its tokenizer for length counting here.
# _model = SentenceTransformer(EMBED_MODEL)

SAFETY_MARGIN = 10   # small buffer for tokenizer edge cases
CHUNK_OVERLAP = 30   # tokens carried between forced sub-splits

def token_len(text: str) -> int:
    # Length in MiniLM tokens, not characters — this is what the 256 budget means.
    return len(get_model().tokenizer.encode(text, add_special_tokens=False))

def find_headings(text: str):
    seen = set()
    out = []
    for m in HEADING_RE.finditer(text):
        num = m.group(1)
        if num in seen:
            continue          # repeat = running header, skip
        seen.add(num)
        out.append({"num": num, "title": m.group(2).strip(), "start": m.start()})
    return out

FIG_RE = re.compile(r"Fig\.?\s*\d+\.\d+")

def find_figures(text: str):
    text = text.replace("\xa0", " ")
    return sorted(set(FIG_RE.findall(text)))

def split_sections(text: str):
    headings = find_headings(text)
    sections = []

    # No headings at all: whole chapter becomes one "intro" section.
    if not headings:
        body = text.strip()
        return [{"num": "intro", "title": "", "text": body}] if body else []

    # Intro: everything before the first heading
    if headings[0]["start"] > 0:
        intro = text[: headings[0]["start"]].strip()
        if intro:
            sections.append({"num": "intro", "title": "", "text": intro})


    # Each section: from its heading to the next heading's start
    for i, h in enumerate(headings):
        end = headings[i + 1]["start"] if i + 1 < len(headings) else len(text)
        body = text[h["start"] : end].strip()
        sections.append({"num": h["num"], "title": h["title"], "text": body})

    return sections

def chunk_sections(sections, meta):
    chunks = []
    for s in sections:
        heading = f"{s['num']} {s['title']}".strip()
        budget = EMBED_MAX_TOKENS - token_len(heading) - SAFETY_MARGIN

        if token_len(s["text"]) <= EMBED_MAX_TOKENS:
            pieces = [s["text"]]
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=budget,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=token_len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            pieces = [
                p if p.startswith(s["num"]) else f"{heading}\n{p}"
                for p in splitter.split_text(s["text"])
            ]

        for i, text in enumerate(pieces):
            chunks.append({
                "id": f"grade{meta['grade']}_ch{meta['chapter']:02d}_sec{s['num']}_{i}",
                "text": text,
                "grade": meta["grade"],
                "subject": "science",
                "chapter": meta["chapter"],
                "section": s["num"],
                "section_title": s["title"],
                "figures": find_figures(text),
                "figure_paths": [],
                "source_pdf": meta["source_pdf"],
            })
    return chunks

def build_all_chunks():
    all_chunks = []
    for txt in sorted(PROCESSED.glob("grade*_ch*.txt")):
        # filename: grade8_ch02.txt
        grade = int(txt.stem.split("_")[0].replace("grade", ""))
        chapter = int(txt.stem.split("_")[1].replace("ch", ""))
        meta = {"grade": grade, "chapter": chapter, "source_pdf": txt.name}

        text = txt.read_text(encoding="utf-8")
        sections = split_sections(text)
        chunks = chunk_sections(sections, meta)
        all_chunks.extend(chunks)
        print(f"{txt.name}: {len(sections)} sections -> {len(chunks)} chunks")
    return all_chunks

if __name__ == "__main__":
    chunks = build_all_chunks()
    print(f"\nTOTAL: {len(chunks)} chunks across all chapters")