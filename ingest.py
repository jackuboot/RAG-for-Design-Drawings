import json, pdfplumber, os
from typing import Dict, Any, List
from utils import find_sheet_id, guess_section, bbox_to_str, load_config
import sys

def extract_lines_with_bbox(pdf_path: str):
    docs = []
    import pdfplumber, sys
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for pageno, page in enumerate(pdf.pages, start=1):
            print(f"[ingest] start page {pageno}/{total}", flush=True)
            try:
                # try the precise path with word boxes
                words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
                # guard against pathological pages with extreme word counts
                if words and len(words) > 20000:
                    print(f"[ingest] page {pageno}: too many words ({len(words)}). Fallback to extract_text.", flush=True)
                    words = None
            except Exception as e:
                print(f"[ingest] page {pageno}: word extraction failed: {e}. Fallback to extract_text.", flush=True)
                words = None

            if words:
                by_line = {}
                for w in words:
                    y = round(w["top"]/2)*2
                    by_line.setdefault(y, []).append(w)
                for y, ws in sorted(by_line.items()):
                    x0 = min(w["x0"] for w in ws); x1 = max(w["x1"] for w in ws)
                    top = min(w["top"] for w in ws); bottom = max(w["bottom"] for w in ws)
                    text = " ".join(w["text"] for w in sorted(ws, key=lambda w: w["x0"]))
                    if text.strip():
                        docs.append({
                            "page": pageno,
                            "text": text.strip(),
                            "bbox": [float(x0), float(top), float(x1), float(bottom)]
                        })
            else:
                # quick fallback without boxes
                page_text = page.extract_text() or ""
                if page_text.strip():
                    for line in page_text.split("\n"):
                        if line.strip():
                            docs.append({"page": pageno, "text": line.strip(), "bbox": None})

            print(f"[ingest] done  page {pageno}/{total}", flush=True)
    return docs


def chunk_docs(lines, max_chars=800, overlap=120):
    """
    Fast O(n) chunker.
    - Buckets by page.
    - Builds chunks via list-join (no repeated string concat).
    - Uses a small line overlap in COUNT not chars, to avoid quadratic behavior.
    """
    from collections import defaultdict

    # 1) bucket lines by page
    by_page = defaultdict(list)
    for rec in lines:
        by_page[rec["page"]].append(rec)

    chunks = []
    # heuristics: about how many lines fit in max_chars
    avg_line_len = max(20, min(120, int(sum(len(r["text"]) for r in lines) / max(1, len(lines)))))
    approx_lines_per_chunk = max(4, max_chars // avg_line_len)
    line_overlap = 1 if overlap > 0 else 0  # small, fixed overlap

    for page in sorted(by_page.keys()):
        page_lines = by_page[page]
        n = len(page_lines)
        i = 0
        while i < n:
            # collect up to approx_lines_per_chunk lines, respecting max_chars
            acc = []
            boxes = []
            char_count = 0
            j = i
            while j < n:
                t = page_lines[j]["text"]
                if char_count and (char_count + 1 + len(t)) > max_chars:
                    break
                acc.append(t)
                boxes.append(page_lines[j].get("bbox"))
                char_count += (len(t) + 1)
                j += 1
                if len(acc) >= approx_lines_per_chunk:
                    # try to keep chunks reasonably sized; next loop will continue
                    break

            text = "\n".join(acc).strip()
            # coarse bbox
            bxs = [b for b in boxes if b]
            bbox = None
            if bxs:
                x0 = min(b[0] for b in bxs); y0 = min(b[1] for b in bxs)
                x1 = max(b[2] for b in bxs); y1 = max(b[3] for b in bxs)
                bbox = [x0, y0, x1, y1]

            # metadata
            # only peek at a prefix for section detection to stay fast
            from utils import find_sheet_id, guess_section
            sheet_id = find_sheet_id(text)
            section = guess_section(text[:1200])

            if text:
                chunks.append({
                    "page": page,
                    "sheet_id": sheet_id,
                    "section": section,
                    "text": text,
                    "bbox": bbox
                })

            # advance with tiny fixed overlap (by lines), never step back more than needed
            if j >= n:
                i = n
            else:
                i = max(i + max(1, len(acc) - line_overlap), i + 1)

    return chunks

def ingest(pdf_path: str, out_path: str, max_chars=800, overlap=120) -> int:
    print("[ingest] extracting lines...", flush=True)
    lines = extract_lines_with_bbox(pdf_path)
    print(f"[ingest] extracted {len(lines)} lines", flush=True)

    print("[ingest] chunking...", flush=True)
    chunks = chunk_docs(lines, max_chars=max_chars, overlap=overlap)
    print(f"[ingest] created {len(chunks)} chunks", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[ingest] writing {out_path}...", flush=True)
    with open(out_path, "w") as f:
        for ch in chunks:
            f.write(json.dumps(ch) + "\n")
    print("[ingest] write complete", flush=True)
    return len(chunks)

def main():
    cfg = load_config()
    pdf_path = cfg["pdf_path"]
    print(f"Loading PDF: {pdf_path}", flush=True)
    out = "./data/chunks.jsonl"
    n = ingest(pdf_path, out, cfg["chunk"]["max_chars"], cfg["chunk"]["overlap"])
    print(f"Ingested {n} chunks -> {out}", flush=True)

if __name__ == "__main__":
    main()
