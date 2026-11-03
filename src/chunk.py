from pathlib import Path
import pandas as pd

IN_DIR = Path("data/clean")
CHUNK_SIZE = 500
OVERLAP = 80


def split_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):                      # try to cut at a space, not in a word
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


rows = []
for path in sorted(IN_DIR.glob("*.txt")):
    pages = path.read_text(encoding="utf-8").split("\f")
    for page_num, page in enumerate(pages, start=1):
        for i, chunk in enumerate(split_text(page)):
            rows.append({
                "id": f"{path.stem}_p{page_num}_c{i}",
                "source": path.name,
                "page": page_num,
                "text": chunk,
            })

df = pd.DataFrame(rows)
df.to_csv("data/chunks.csv", index=False)

print("Chunks:", len(df))
print(df.groupby("source").size())
print("Length min/mean/max:", df.text.str.len().min(), int(df.text.str.len().mean()), df.text.str.len().max())
print("\nSample chunk:\n", df.sample(1).iloc[0].text)