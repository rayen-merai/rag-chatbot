from pathlib import Path
from pypdf import PdfReader
from langdetect import detect_langs, DetectorFactory
import pandas as pd
import re

DetectorFactory.seed = 0                      # same result on every run

PDF_DIR, RAW_DIR, CLEAN_DIR = Path("data/pdfs"), Path("data/text"), Path("data/clean")
MIN_CHARS = 50                                # below this a page counts as empty
MIN_LANG_CHARS = 80                           # below this, language detection is unreliable
MIN_CONF = 0.80                               # below this, the language is "uncertain"
EXPECTED_LANGS = {"fr", "ar", "en"}           # languages your documents should be in

rows, problems = [], 0


def warn(msg):
    global problems
    problems += 1
    print("  WARNING:", msg)


def page_language(text):
    """Returns (language code, confidence) using a trained language model."""
    if len(text) < MIN_LANG_CHARS:
        return "none", 0.0
    try:
        best = detect_langs(text)[0]
        return best.lang, round(best.prob, 2)
    except Exception:
        return "unknown", 0.0


for pdf in sorted(PDF_DIR.glob("*.pdf")):
    print(f"\n=== {pdf.name} ===")
    n_pdf = len(PdfReader(pdf).pages)
    raw_f, clean_f = RAW_DIR / f"{pdf.stem}.txt", CLEAN_DIR / f"{pdf.stem}.txt"
    if not raw_f.exists() or not clean_f.exists():
        warn("missing text or clean file, run extract.py and clean.py")
        continue

    raw = raw_f.read_text(encoding="utf-8").split("\f")
    clean = clean_f.read_text(encoding="utf-8").split("\f")
    print(f"  PDF pages={n_pdf}  raw pages={len(raw)}  clean pages={len(clean)}")

    # 1. COMPLETENESS
    if len(raw) != n_pdf:
        warn("raw pages != PDF pages (extract.py must join pages with \\f)")
    if len(clean) != n_pdf:
        warn("clean pages != PDF pages: page numbers will be wrong (clean.py must not drop empty pages)")
    empty = [i + 1 for i, p in enumerate(clean) if len(p) < MIN_CHARS]
    if empty:
        print(f"  Empty or very short pages: {empty}")
    if len(empty) > 0.2 * max(len(clean), 1):
        warn("more than 20% empty pages: probably a scanned PDF (needs OCR)")

    lens = [len(p) for p in clean]
    mean_len = sum(lens) / max(len(lens), 1)
    seen = {}
    file_rows = []

    for i, text in enumerate(clean, start=1):
        r = raw[i - 1] if i - 1 < len(raw) else ""
        letters = sum(c.isalpha() for c in text)
        lang, conf = page_language(text)
        issues = []

        # 2. ACCURACY
        if "\ufffd" in text or "(cid:" in text:
            issues.append("broken characters")
        if re.search(r"(?:\b\w\s){6,}", text):
            issues.append("spaced letters")
        if len(re.findall(r"\w-\s\w", text)) > 3:
            issues.append("broken words")
        if lang not in ("none", "unknown") and lang not in EXPECTED_LANGS:
            issues.append(f"unexpected language '{lang}' (garbage text?)")

        # 3. CONSISTENCY
        if 0 < len(text) < 0.2 * mean_len:
            issues.append("much shorter than average")
        if lang not in ("none", "unknown") and conf < MIN_CONF:
            issues.append(f"uncertain or mixed language ({lang} {conf})")

        # 4. RELEVANCE
        if text and (text.count("..") > 10 or letters / len(text) < 0.5):
            issues.append("table of contents or numbers only")
        if len(r) > 0 and len(text) < 0.6 * len(r):
            issues.append("cleaning removed >40%")
        if len(text) >= MIN_CHARS:
            if text in seen:
                issues.append(f"duplicate of page {seen[text]}")
            seen[text] = i

        file_rows.append({"file": pdf.name, "page": i, "raw_len": len(r), "clean_len": len(text),
                          "lang": lang, "lang_conf": conf, "issues": "; ".join(issues)})

    rows.extend(file_rows)

    # language summary for the file
    langs = pd.Series([r["lang"] for r in file_rows if r["lang"] not in ("none", "unknown")])
    if len(langs):
        share = (langs.value_counts(normalize=True) * 100).round().astype(int).to_dict()
        print("  Languages (% of pages):", share)

    flagged = [r for r in file_rows if r["issues"]]
    print(f"  Pages with issues: {len(flagged)}")
    for r in flagged[:10]:
        print(f"    p.{r['page']}: {r['issues']}")
    if flagged:
        problems += 1

report = pd.DataFrame(rows)
report.to_csv("data/quality_report.csv", index=False, encoding="utf-8-sig")
print("\nPages checked:", len(report), "| problems found:", problems)
print("Report saved: data/quality_report.csv")