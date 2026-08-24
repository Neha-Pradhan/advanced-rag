import re
from src.config import PROCESSED

# NCERT section headings look like "2.1 What Is a Cell?" at line start.
# Pattern: digits.digits, a space, then a Title-Case word.
# Heading: "2.1 Title", title may wrap to a second line.
HEADING_RE = re.compile(r"^(\d+\.\d+)\s+([A-Z].*)$", re.MULTILINE)

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

if __name__ == "__main__":
    text = (PROCESSED / "grade8_ch02.txt").read_text(encoding="utf-8")
    for s in split_sections(text):
        print(f"[{s['num']}] {s['title']}  —  {len(s['text'])} chars")