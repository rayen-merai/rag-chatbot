from pathlib import Path
from pypdf import PdfReader

PDF_DIR = Path("data/pdfs")
OUT_DIR = Path("data/text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for pdf_path in PDF_DIR.glob("*.pdf"):
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    out_file = OUT_DIR / f"{pdf_path.stem}.txt"
    out_file.write_text("\f".join(pages), encoding="utf-8")
    print(f"{pdf_path.name}: {len(pages)} pages -> {out_file}")