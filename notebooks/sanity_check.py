from pypdf import PdfReader

pdf_path = "data/raw/grade8/hecu102.pdf"   # try a few chapters


reader = PdfReader(pdf_path)
print(f"Pages: {len(reader.pages)}")

page = reader.pages[13]    # a middle-ish page, past the chapter title art
print(page.extract_text())