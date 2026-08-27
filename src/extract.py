import re
from pypdf import PdfReader

from src.config import RAW, PROCESSED


def clean_page(text: str) -> str:
    # Footer/header junk
    text = re.sub(r"Chapter \d+\.indd.*", "", text)
    text = re.sub(r"Reprint \d{4}-\d{2}", "", text)
    text = re.sub(r"Curiosity\s*—\s*Textbook of Science for Grade \d+", "", text)
    text = re.sub(r"Chapter \d+\s*—\s*[A-Z][^\n]*", "", text)  # running chapter-title header
    # Worksheet fill-in dots (3+ in a row)
    text = re.sub(r"\.{3,}", "", text)
    # Bare page-number lines
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    # Fix soft line-breaks
    text = re.sub(r"(?<=[a-z,])\n(?=[a-z])", " ", text)
    return text


def extract_pdf(pdf_path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [clean_page(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages)

def parse_filename(stem: str) -> dict:
    # "hecu102" -> grade 8, chapter 2 ; "gecu107" -> grade 7, chapter 7
    grade = {"h": 8, "g": 7}[stem[0]]
    chapter = int(stem[-2:])
    return {"grade": grade, "chapter": chapter}

def process_all():
    PROCESSED.mkdir(exist_ok=True)
    for grade_dir in sorted(RAW.iterdir()):
        if not grade_dir.is_dir():
            continue
        for pdf in sorted(grade_dir.glob("*.pdf")):
            if "ps" in pdf.stem:  # skip prelims (hecu1ps, gecu1ps)
                continue
            meta = parse_filename(pdf.stem)
            text = extract_pdf(pdf)
            out_name = f"grade{meta['grade']}_ch{meta['chapter']:02d}.txt"
            (PROCESSED / out_name).write_text(text, encoding="utf-8")
            print(f"{pdf.name} -> {out_name}  ({len(text)} chars)")

if __name__ == "__main__":
    process_all()