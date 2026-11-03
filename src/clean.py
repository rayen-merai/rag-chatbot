from pathlib import Path
from collections import Counter
import re

IN_DIR = Path("data/text")
OUT_DIR = Path("data/clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lines like "12", "Page 3", "3/20", "page 3 sur 20"
PAGE_NUM = re.compile(r"(page\s*)?\d+(\s*(/|sur|of)\s*\d+)?", re.I)


def find_repeated_lines(pages):
    """Short lines that repeat a lot are usually headers or footers."""
    counts = Counter()
    if len(pages) > 1:
        for page in pages:
            counts.update({l.strip() for l in page.split("\n") if l.strip()})
        threshold = max(3, int(0.3 * len(pages)))
    else:
        for l in pages[0].split("\n"):
            if l.strip():
                counts[l.strip()] += 1
        threshold = 5
    return {l for l, c in counts.items() if c >= threshold and len(l) < 100}


def clean_page(page, repeated):
    lines = [l.strip() for l in page.split("\n")]
    lines = [l for l in lines if l and l not in repeated]
    lines = [l for l in lines if not PAGE_NUM.fullmatch(l)]
    text = "\n".join(lines)
    # words broken at line end
    text = re.sub(r"(?<=[A-Za-zà-ÿ])-\n(?=[a-zà-ÿ])", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


for txt_path in sorted(IN_DIR.glob("*.txt")):
    raw = txt_path.read_text(encoding="utf-8")
    pages = raw.split("\f")

    repeated = find_repeated_lines(pages)
    cleaned = [clean_page(p, repeated) for p in pages]   # keep empty pages

    out_file = OUT_DIR / txt_path.name
    out_file.write_text("\f".join(cleaned), encoding="utf-8")

    non_empty = sum(1 for p in cleaned if p)
    print(f"\n{txt_path.name}: {len(raw)} -> {sum(len(p) for p in cleaned)} characters")
    print(f"  pages: {len(cleaned)} ({non_empty} non-empty)")
    print(f"  repeated lines removed: {sorted(repeated)[:10]}")
    