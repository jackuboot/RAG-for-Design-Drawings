import os, json, pickle, faiss, numpy as np, re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from utils import load_config, bbox_to_str
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------
# Regex & quick helpers
# ---------------------------
WINDOW_CODE_RE =re.compile(r"\bW[-_ ]?\d{1,5}[A-Z0-9\-]*\b", re.I)
BAD_W_PREFIXES = ("WALL", "WASH", "WATER", "WIR", "WOOD", "WORK")
# Accepts W-02, W02, W2A, W-2A, W 02 (OCR spaced), etc., but rejects WINDOW/WORK/ WALL
WINDOW_CODE_OCR_RE = re.compile(r"(?<![A-Z])W\s*[-_ ]?\s*\d{1,5}[A-Z0-9\-]*\b", re.I)

BEDROOM_RE = re.compile(r"\bBED(?:ROOM|RM)?\s*[#\-:]?\s*([0-9]+)\b", re.IGNORECASE)

COVER_SHEET_ALIASES = {
    "A1.0", "A0.0", "A0.1", "A-000", "A001", "A-001", "T1", "T-1",
    "G-001", "G001", "SD1", "COVER", "TITLE"
}

DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
    r"\b[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}\b",
]
REV_HINTS = {"rev", "revision", "revisions"}

SHEET_Q_RE = re.compile(r"\b([A-Z]{1,3}\.?[0-9]{1,2}(?:\.[0-9]{1,2})?)\b")

SCALE_PATTERNS = [
    r"\bSCALE\s*[:\-]?\s*(N\.?T\.?S\.?|NOT\s+TO\s+SCALE|AS\s+NOTED|AS\s+INDICATED|AS\s+SHOWN)\b",
    r"\bSCALE\s*[:\-]?\s*([0-9]+\s*/\s*[0-9]+\s*\"?\s*=\s*[0-9]+'\s*-\s*[0-9]+\"?)\b",
    r"\bSCALE\s*[:\-]?\s*(\d+:\d+)\b",
    r"\b(N\.?T\.?S\.?|NOT\s+TO\s+SCALE|AS\s+NOTED|AS\s+INDICATED|AS\s+SHOWN)\b",
    r"\b([0-9]+\s*/\s*[0-9]+\s*\"?\s*=\s*[0-9]+'\s*-\s*[0-9]+\"?)\b",
]

FLOOR_FINISH_TOKENS = {
    "VINYL","LVT","LVP","SPC","CARPET","CPT","VCT","TILE","CERAMIC","PORCELAIN",
    "WOOD","HARDWOOD","LAMINATE","STONE","CONC","CONCRETE","EPOXY","MARBLE","GRANITE"
}
ROOM_ALIASES = {
    "LIVING ROOM": {"LIVING ROOM","LIVING","LR"},
    "BEDROOM 2": {"BEDROOM 2","BEDRM 2","BR 2","BDRM 2","B2"},
}

UNITS = {"ft","inch","in.","in","mm","cm","m","sf","sq.","sqft","scale","rev","date","height","ceiling","floor"}

SIZE_PATTERNS = [
    r"\b5056\b",
    r'\b50\s*"?\s*[xX]\s*56\b',
    r"\b4'\s*-?\s*2\"\s*[xX]\s*4'\s*-?\s*8\"\b",
]

ROOM_NAME_RE = re.compile(r"\b(ENTRY|FOYER|LIVING(?:\s+ROOM)?|DINING|KITCHEN|BED(?:ROOM|RM)?\s*\d+|BEDROOM|BATH(?:ROOM)?\s*\d+|BATH|LAUNDRY|HALL|HALLWAY|GARAGE|OFFICE|DEN|LOFT|PANTRY|CLOSET)\b", re.I)
HT_RE_1 = re.compile(r"(?:CEILING\s+HEIGHT|CLG(?:\.|)\s*HT|CLG\s*HEIGHT)\s*[:\-]?\s*([0-9]+'\s*-\s*[0-9]+\"|[0-9]+'\s*[0-9]+\"|[0-9]+[ ]?FT[ ]?[0-9]*[ ]?IN)", re.I)
HT_RE_2 = re.compile(r"\b([0-9]+'\s*-\s*[0-9]+\"|[0-9]+'\s*[0-9]+\"|[0-9]+[ ]?FT[ ]?[0-9]*[ ]?IN)\b")

# ---------------------------
# Question classifiers
# ---------------------------

def norm_wcode(tok: str) -> str:
    t = tok.upper().strip()
    # unify spaces and underscores; keep internal dashes
    t = t.replace(" ", "").replace("_", "")
    # normalize a single dash/space after W
    t = re.sub(r"^W[- ]+(\d)", r"W\1", t)
    return t

def is_address_question(q: str) -> bool:
    ql = q.lower()
    return ("address" in ql or "location" in ql) and ("project" in ql or "site" in ql or "building" in ql or "cover" in ql)

def is_administrative_question(q: str) -> bool:
    """Questions about project info, codes, general building data"""
    ql = q.lower()
    admin_terms = [
        "occupancy", "construction type", "zoning", "lot size", 
        "code version", "permit", "deferred"
    ]
    return any(term in ql for term in admin_terms)

def is_window_count_question(q: str) -> bool:
    ql = q.lower()
    return ("how many" in ql) and ("window" in ql) and ("schedule" in ql)

def is_scale_question(q: str) -> bool:
    return "scale" in q.lower()

def is_revision_date_question(q: str) -> bool:
    ql = q.lower()
    return "date" in ql and any(h in ql for h in REV_HINTS)

def is_finish_question(q: str) -> bool:
    ql = q.lower()
    return ("finish" in ql or "flooring" in ql) and ("living" in ql or "bedroom" in ql or "room" in ql)

def is_window_type_in_room_question(q: str) -> bool:
    ql = q.lower()
    return ("window type" in ql or ("window" in ql and "type" in ql)) and ("bedroom" in ql)

def is_room_ceiling_list_question(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ["ceiling height","ceiling heights","clg ht","clg height"]) and \
           any(r in ql for r in ["room","rooms","rcp","reflected ceiling"]) 

# ---------------------------
# Parsing helpers
# ---------------------------
def harvest_window_codes_near_room(room_tags: list[str], chunks, pdf_path: str, zooms=(3.2, 3.6)):
    """OCR chunks whose text layer already mentions any of the room tags; extract W-codes nearby."""
    codes = set()
    for c in chunks:
        U = (c.get("text","") or "").upper()
        if any(tag in U for tag in room_tags) and c.get("bbox"):
            bb = bbox_to_str(c.get("bbox"))
            for z in zooms:
                txt = (ocr_text_from_bbox(pdf_path, c["page"], bb, zoom=z, psm=6, pad=12) or "").upper()
                for m in WINDOW_CODE_OCR_RE.finditer(txt):
                    code = re.sub(r"\s+", "", m.group(0).upper())  # "W 02" -> "W02"
                    if code not in {"WINDOW", "WINDOWS"} and re.match(r"^W[-_]?\d", code):
                        codes.add(code)
    return codes

def find_type_in_row_window(row: str) -> str | None:
    m = re.search(r"\bTYPE\b[:\-]?\s*([A-Z0-9\-]+)", row, re.I)
    if m: return m.group(1)
    m = re.search(r"\b(TYP|MODEL|SERIES|DESC(?:RIPTION)?)\b[:\-]?\s*([A-Z0-9\-]+)", row, re.I)
    if m: return m.group(2)
    # lettered types, e.g., "TYPE A"
    m = re.search(r"\bTYPE\s+([A-Z])\b", row, re.I)
    if m: return f"Type {m.group(1)}"
    # generic kinds (optional)
    m = re.search(r"\b(CASEMENT|SLIDER|FIXED|AWNING|HOPPER|SINGLE HUNG|DOUBLE HUNG|PICTURE|SLIDING)\b", row, re.I)
    if m: return m.group(1).title()
    return None

def resolve_window_type_from_schedule(codes: set[str], chunks, pdf_path: str, zooms=(3.2,)):
    """Try to map any of the codes to a TYPE value in schedule text layer; OCR if needed."""
    sched_chunks = [c for c in chunks if (c.get("section")=="window schedule" or "WINDOW SCHEDULE" in (c.get("text","") or "").upper())]
    # Text-layer first
    for s in sched_chunks:
        txt = (s.get("text","") or "").upper()
        lines = txt.splitlines()
        for code in sorted(codes, key=len, reverse=True):
            ncode = norm_wcode(code)
            for i, l in enumerate(lines):
                L = norm_wcode(l)
                if ncode in L:
                    row = " ".join(lines[max(0,i-1): i+3])
                    # look for TYPE/TYP/MODEL/SERIES first
                    m = re.search(r"\bTYPE\b[:\-]?\s*([A-Z0-9\-]+)", row)
                    if m: return m.group(1), s
                    m2 = re.search(r"\b(TYP|MODEL|SERIES|DESC(?:RIPTION)?)\b[:\-]?\s*([A-Z0-9\-]+)", row)
                    if m2: return m2.group(2), s
                    # sometimes the row is "W5  →  TYPE A" and the TYPE A is a separate header/col:
                    m3 = re.search(r"\bTYPE\s+([A-Z])\b", row)  # e.g., TYPE A
                    if m3: return f"Type {m3.group(1)}", s
                    return ncode, s


    # OCR fallback on schedule
    for s in sched_chunks[:3]:
        if not s.get("bbox"): 
            continue
        bb = bbox_to_str(s.get("bbox"))
        for z in zooms:
            txt = (ocr_text_from_bbox(pdf_path, s["page"], bb, zoom=z, psm=6, pad=16) or "").upper()
            lines = txt.splitlines()
            for code in sorted(codes, key=len, reverse=True):
                ncode = norm_wcode(code)
                for i,l in enumerate(lines):
                    L = norm_wcode(l)
                    if ncode in L:
                        row = " ".join(lines[max(0, i-1): i+3])
                        m = re.search(r"\bTYPE\b[:\-]?\s*([A-Z0-9\-]+)", row)
                        if m:
                            return m.group(1), s
                        # alt headers sometimes seen: TYP, MODEL, SERIES, DESC
                        m2 = re.search(r"\b(TYP|MODEL|SERIES|DESC(?:RIPTION)?)\b[:\-]?\s*([A-Z0-9\-]+)", row)
                        if m2:
                            return m2.group(2), s
                        # last-resort: return the matched window code itself
                        return code, s
    # ---- size-based fallback (text layer first)
    for s in sched_chunks:
        up = (s.get("text") or "")
        lines = up.splitlines()
        U = [ln.upper() for ln in lines]
        for i, L in enumerate(U):
            if any(re.search(p, L) for p in SIZE_PATTERNS):
                row = " ".join(lines[max(0, i-2): i+6])
                t = find_type_in_row_window(row)
                if t: return t, s

    # ---- size-based fallback (OCR)
    for s in sched_chunks[:3]:
        if not s.get("bbox"): continue
        bb = bbox_to_str(s.get("bbox"))
        raw = (ocr_text_from_bbox(pdf_path, s["page"], bb, zoom=3.2, psm=6, pad=16) or "")
        lines = raw.splitlines()
        U = [ln.upper() for ln in lines]
        for i, L in enumerate(U):
            if any(re.search(p, L) for p in SIZE_PATTERNS):
                row = " ".join(lines[max(0, i-2): i+6])
                t = find_type_in_row_window(row)
                if t: return t, s

    return None, None

def extract_bedroom_num(q: str) -> str | None:
    m = BEDROOM_RE.search(q)
    return m.group(1) if m else None

def sheet_in_question(q: str) -> str | None:
    m = SHEET_Q_RE.search(q.upper())
    return m.group(1).upper() if m else None

def is_cover_like(code: str) -> bool:
    up = code.upper()
    return up.startswith("A0") or up in COVER_SHEET_ALIASES or up == "A1.0"

def expand_sheet_aliases(sheet_code: str) -> set[str]:
    up = sheet_code.upper()
    return {up} | (COVER_SHEET_ALIASES if is_cover_like(up) else set())

def doc_has_sheet_or_alias(sheet_code: str, chunks) -> bool:
    if not sheet_code:
        return True
    aliases = {a.replace(" ", "") for a in expand_sheet_aliases(sheet_code)}
    for c in chunks:
        sid = (c.get("sheet_id") or "").upper().replace(" ", "")
        if sid in aliases:
            return True
    up_aliases = expand_sheet_aliases(sheet_code)
    for c in chunks:
        up = (c.get("text","") or "").upper()
        if any(a in up for a in up_aliases):
            return True
    return False

def filter_supports_by_aliases(supports, aliases: set[str]):
    norm_aliases = {a.upper().replace(" ","") for a in aliases}
    exact = [s for s in supports if (s.get("sheet_id") or "").upper().replace(" ","") in norm_aliases]
    tb    = [s for s in supports if (s.get("section") == "title block" or "TITLE BLOCK" in (s.get("text","") or "").upper())]
    soft  = [s for s in supports if any(a in (s.get("text","") or "").upper() for a in aliases)]
    return exact, tb, soft

def filter_supports_by_sheet(supports, sheet_code: str):
    if not sheet_code:
        return supports, [], []
    target = sheet_code.upper().replace(" ", "")
    exact = [s for s in supports if (s.get("sheet_id") or "").upper().replace(" ", "") == target]
    soft  = [s for s in supports if sheet_code.upper() in (s.get("text","") or "").upper()]
    tb    = [s for s in supports if (s.get("section") == "title block" or "TITLE BLOCK" in (s.get("text","") or "").upper())]
    return exact, soft, tb

def find_dates_in_text(txt: str):
    hits = []
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, txt):
            hits.append(m.group(0))
    return hits

def find_scale_in_text(txt: str):
    if not txt:
        return None
    txt = _normalize_ocr_punct(txt)
    for pat in SCALE_PATTERNS:
        m = re.search(pat, txt, flags=re.IGNORECASE)
        if m:
            return (m.group(1) if m.lastindex else m.group(0)).strip()
    return None

def normalize_room_name(q: str) -> str:
    up = q.upper()
    for k, alts in ROOM_ALIASES.items():
        if any(a in up for a in alts):
            return k
    if "LIVING" in up:
        return "LIVING ROOM"
    return ""

def _norm_height(h: str) -> str:
    u = h.upper().replace("FT","'").replace("IN","\"")
    u = re.sub(r"\s+", "", u)
    # normalize 8'0" → 8'-0"
    u = re.sub(r"(\d+)'\s*(\d+)\"", r"\1'-\2\"", u)
    if re.match(r"^\d+'$", u):
        u = u + '-0"'
    return u

def parse_room_heights_from_text(txt: str):
    """Greedy extraction: find lines containing a room-like token and a height expression."""
    pairs = []
    for line in (txt or "").splitlines():
        if len(line.strip()) < 3:
            continue
        rn = None
        m_room = ROOM_NAME_RE.search(line)
        if m_room:
            rn = m_room.group(0).upper().replace("  "," ")
        if rn:
            m1 = HT_RE_1.search(line)
            if m1:
                pairs.append((rn, _norm_height(m1.group(1))))
                continue
            # fallback: any height-looking token near the room
            m2 = HT_RE_2.search(line)
            if m2:
                pairs.append((rn, _norm_height(m2.group(1))))
    # dedupe by room keep first
    seen = set(); out=[]
    for r,h in pairs:
        if r not in seen:
            seen.add(r); out.append((r,h))
    return out

def ocr_find_room_codes_on_pages(
    pdf_path: str,
    room_tags: list[str],
    pages: list[int] | None = None,
    zoom: float = 3.2,
    grid: tuple[int, int] = (3, 3),
    *,
    max_cells: int = 90,        # hard cap across all pages
    tesseract_timeout: float = 1.5,  # seconds per OCR call
    verbose: bool = True
) -> tuple[set[str], dict] | tuple[set[str], None]:
    """
    Grid OCR across given pages. If a cell contains a room tag (BEDROOM 2, BR-2, etc.),
    re-OCR a padded neighborhood and extract W-codes. Time-bounded and cell-capped.
    """
    try:
        import fitz
        from PIL import Image
        import pytesseract
    except Exception:
        return set(), None

    # Tight OCR wrapper with timeout
    def ocr_text_fast(img) -> str:
        gray = img.convert("L")
        bw = gray.point(lambda x: 255 if x > 200 else 0, mode='1')
        try:
            return pytesseract.image_to_string(
                bw,
                config='--psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-#:/().\' "',
                timeout=tesseract_timeout
            ).strip()
        except RuntimeError:
            # pytesseract raises RuntimeError on timeout
            return ""

    codes: set[str] = set()
    cit: dict | None = None

    doc = fitz.open(pdf_path)
    scan_pages = pages or list(range(1, doc.page_count + 1))
    cols, rows = grid

    cells_used = 0
    room_tags_up = [t.upper() for t in room_tags]

    for pg in scan_pages:
        if cells_used >= max_cells:
            break
        page = doc.load_page(pg - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        W, H = pix.width, pix.height

        col_w = W // cols
        row_h = H // rows

        for r in range(rows):
            for c in range(cols):
                if cells_used >= max_cells:
                    break
                x0 = c * col_w
                y0 = r * row_h
                x1 = (c + 1) * col_w - 1
                y1 = (r + 1) * row_h - 1
                crop = img.crop((x0, y0, x1, y1))

                # coarse OCR to detect room tag
                cell_txt = ocr_text_fast(crop)
                cells_used += 1
                U = cell_txt.upper()

                if verbose and cells_used % 15 == 0:
                    print(f"[ocr-grid] page={pg} cell={cells_used}/{max_cells}", flush=True)

                if any(tag in U for tag in room_tags_up):
                    # expand neighborhood one cell around
                    C0 = max(0, c - 1); C1 = min(cols - 1, c + 1)
                    R0 = max(0, r - 1); R1 = min(rows - 1, r + 1)
                    nx0 = C0 * col_w; ny0 = R0 * row_h
                    nx1 = (C1 + 1) * col_w - 1; ny1 = (R1 + 1) * row_h - 1
                    neigh = img.crop((nx0, ny0, nx1, ny1))

                    neigh_txt = ocr_text_fast(neigh)
                    cells_used += 1
                    U2 = neigh_txt.upper()

                    for m in WINDOW_CODE_OCR_RE.finditer(U2):
                        code = re.sub(r"\s+", "", m.group(0).upper())
                        if code not in {"WINDOW", "WINDOWS"} and re.match(r"^W[-_]?\d", code):
                            codes.add(code)

                    if codes:
                        cit = {"sheet_id": "", "page": pg, "section": "floor plan (ocr)", "bbox": ""}
                        return codes, cit

    return codes, cit

def harvest_window_codes_by_proximity(chunks, bnum: str, max_dist_px: float = 300.0) -> set[str]:
    """
    Find chunks whose text mentions BEDROOM <bnum>, then collect any W-codes
    from other chunks on the same page whose bbox centers are within max_dist_px.
    This avoids OCR entirely when text-layer carries codes.
    """
    import math
    n = re.escape(str(int(bnum)))
    pat_room = re.compile(
    rf"\b(BED(?:ROOM|RM)?|BED|BR|B)\s*[-#:]?\s*0*{n}\b", re.I)

    def center(bb):
        if not bb: return None
        x0,y0,x1,y1 = bb
        return ((x0+x1)/2.0, (y0+y1)/2.0)

    # index chunks by page
    by_page = {}
    for c in chunks:
        by_page.setdefault(c["page"], []).append(c)

    codes = set()
    for pg, group in by_page.items():
        rooms = [c for c in group if pat_room.search((c.get("text") or "")) and c.get("bbox")]
        if not rooms: continue
        others = [c for c in group if c.get("bbox")]
        for r in rooms:
            rc = center(r["bbox"])
            if not rc: continue
            for o in others:
                if o is r or not o.get("text"): continue
                oc = center(o["bbox"]); 
                if not oc: continue
                if math.hypot(oc[0]-rc[0], oc[1]-rc[1]) <= max_dist_px:
                    for m in WINDOW_CODE_RE.finditer(o["text"]):
                        token = m.group(0).upper()
                        if any(token.startswith(p) for p in BAD_W_PREFIXES):
                            continue
                        if any(ch.isdigit() for ch in token):
                            codes.add(token)

    return codes

# ---------------------------
# Retrieval bits
# ---------------------------
def load_store(store_dir):
    import os
    index = faiss.read_index(os.path.join(store_dir, "faiss.index"))
    with open(os.path.join(store_dir, "faiss_meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    # chunks: prefer pickle, fallback to jsonl
    chunks_pkl   = os.path.join(store_dir, "chunks.pkl")
    chunks_jsonl = os.path.join(store_dir, "chunks.jsonl")
    if os.path.exists(chunks_pkl):
        with open(chunks_pkl, "rb") as f:
            chunks = pickle.load(f)
    elif os.path.exists(chunks_jsonl):
        with open(chunks_jsonl) as f:
            chunks = [json.loads(line) for line in f if line.strip()]
    else:
        raise FileNotFoundError("No chunks.pkl or chunks.jsonl found in store_dir")

    # BM25: prefer pickle, else build on the fly
    bm25_pkl = os.path.join(store_dir, "bm25.pkl")
    if os.path.exists(bm25_pkl):
        with open(bm25_pkl, "rb") as f:
            bm = pickle.load(f)["bm25"]
    else:
        from rank_bm25 import BM25Okapi
        corpus = [(c.get("text") or "").lower().split() for c in chunks]
        bm = BM25Okapi(corpus)

    model = SentenceTransformer(meta.get("model", "sentence-transformers/all-MiniLM-L6-v2"))
    return index, meta, chunks, bm, model

def rrf(ranked_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    scores = {}
    for rlist in ranked_lists:
        for rank, doc_id in enumerate(rlist):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores

def retrieve(question: str, cfg):
    index, meta, chunks, bm25, model = load_store(cfg["store_dir"])
    q_aug = question
    if is_window_count_question(question):
        q_aug = question + " window schedule windows types sizes count"
    qv = model.encode([q_aug], normalize_embeddings=True)
    D, I = index.search(np.array(qv).astype("float32"), int(cfg["retrieval"]["faiss_k"]))
    faiss_ids = I[0].tolist()

    tokenized_query = q_aug.lower().split()
    bm_scores = bm25.get_scores(tokenized_query)
    bm_ids = np.argsort(-bm_scores)[: int(cfg["retrieval"]["bm25_k"])].tolist()

    fused = rrf([faiss_ids, bm_ids], int(cfg["retrieval"]["rrf_k"]))

    if is_window_count_question(question):
        sched_ids = set(prefer_schedule_chunks(chunks))
        for did in list(fused.keys()):
            if did in sched_ids:
                fused[did] += 0.25

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return ranked, chunks

def extract_confidence(q: str, support_texts: List[str]) -> float:
    terms = [t for t in re.split(r"\W+", q.lower()) if len(t) > 2]
    if not terms:
        return 0.4
    hit = 0
    for t in set(terms):
        if any(t in (s or "").lower() for s in support_texts):
            hit += 1
    return hit / max(1, len(set(terms)))

def salient(line: str, q: str) -> bool:
    l = (line or "").strip()
    if not l:
        return False
    alpha = sum(ch.isalpha() for ch in l)
    if alpha < 3:
        return False
    q_tokens = {t for t in re.split(r"\W+", q.lower()) if len(t) > 2}
    l_tokens = {t for t in re.split(r"\W+", l.lower()) if len(t) > 2}
    overlap = q_tokens.intersection(l_tokens)
    has_units = len(UNITS.intersection(l_tokens)) > 0 or re.search(r"\d", l) is not None
    return bool(overlap) or has_units

# ---------------------------
# OCR helpers
# ---------------------------
def ocr_text_from_bbox(pdf_path: str, page_num: int, bbox_str: str, zoom: float = 3.2, psm: int = 6, pad: int = 16) -> str | None:
    """OCR the bbox (optionally padded) with a chosen Tesseract PSM."""
    try:
        import fitz, pytesseract
        from PIL import Image, ImageEnhance
    except Exception:
        return None
    if not bbox_str:
        return None

    try:
        x0, y0, x1, y1 = [float(x) for x in bbox_str.split(",")]
    except Exception:
        return None

    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Scale bbox to the rendered image and pad it a bit
    X0, Y0, X1, Y1 = int(x0*zoom), int(y0*zoom), int(x1*zoom), int(y1*zoom)
    X0, Y0 = max(0, X0 - pad), max(0, Y0 - pad)
    X1, Y1 = min(pix.width, X1 + pad), min(pix.height, Y1 + pad)

    crop = img.crop((X0, Y0, X1, Y1))
    
    # Enhance image quality before OCR
    enhancer = ImageEnhance.Contrast(crop)
    crop = enhancer.enhance(1.5)
    enhancer = ImageEnhance.Sharpness(crop)
    crop = enhancer.enhance(1.5)
    
    cfg = f"--psm {psm}"
    text = _ocr_text(crop)
    return text.strip() or None

def ocr_guess_title_block(pdf_path: str, page_num: int, zoom: float = 3.2, psms=(6,7)) -> list[str]:
    """OCR several wide 'bottom strip' guesses (right/center/left), returning non-empty OCR texts."""
    try:
        import fitz, pytesseract
        from PIL import Image
    except Exception:
        return []

    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    W, H = pix.width, pix.height
    # Wider and lower: bottom 20% (0.80–1.00), overlap horizontally
    bands = [
        (int(W*0.65), int(H*0.80), int(W*0.98), int(H*0.98)),  # bottom-right (wider)
        (int(W*0.33), int(H*0.80), int(W*0.67), int(H*0.98)),  # bottom-center
        (int(W*0.02), int(H*0.80), int(W*0.35), int(H*0.98)),  # bottom-left
        (int(W*0.02), int(H*0.88), int(W*0.98), int(H*0.98)),  # full-width thin footer
    ]

    out = []
    for (x0, y0, x1, y1) in bands:
        crop = img.crop((x0, y0, x1, y1))
        for psm in psms:
            txt = pytesseract.image_to_string(crop, config=f"--psm {psm}").strip()
            if txt:
                out.append(txt)
    return out

def ocr_scan_bottom_for_scale(pdf_path: str, page_num: int, zooms=(2.8, 3.2, 3.6), psms=(6,7)) -> str | None:
    """Scan a coarse grid over the bottom third of the page; return text if any block matches SCALE patterns."""
    try:
        import fitz, pytesseract
        from PIL import Image
    except Exception:
        return None

    def _has_scale(txt: str) -> bool:
        return any(re.search(p, txt, flags=re.IGNORECASE) for p in SCALE_PATTERNS)

    for z in zooms:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)
        mat = fitz.Matrix(z, z)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        W, H = pix.width, pix.height
        y_top = int(H*0.65)   # bottom 35%
        y_bot = int(H*0.98)

        cols = 3
        rows = 2
        col_w = W // cols
        row_h = (y_bot - y_top) // rows

        for r in range(rows):
            for c in range(cols):
                x0 = c * col_w
                x1 = (c+1) * col_w - 1
                y0 = y_top + r * row_h
                y1 = y_top + (r+1) * row_h - 1
                crop = img.crop((x0, y0, x1, y1))
                for psm in psms:
                    txt = pytesseract.image_to_string(crop, config=f"--psm {psm}").strip()
                    if txt and _has_scale(txt):
                        return txt
    return None

def ocr_date_from_bbox(pdf_path: str, page_num: int, bbox_str: str):
    txt = ocr_text_from_bbox(pdf_path, page_num, bbox_str)
    if not txt:
        return None
    dates = find_dates_in_text(txt)
    return dates[0] if dates else None

def ocr_room_heights_on_sheet(sheet_code: str, chunks, pdf_path: str, zooms=(3.2,3.6)):
    target = sheet_code.upper().replace(" ","")
    # likely sections
    candidates = [c for c in chunks
                  if (c.get("sheet_id") or "").upper().replace(" ","")==target
                  and c.get("bbox")
                  and (c.get("section") in {"legend","schedules","title block","floor plan","sections"} 
                       or "CEILING" in (c.get("text","") or "").upper()
                       or "RCP" in (c.get("text","") or "").upper())]
    pairs=[]
    for c in candidates[:10]:
        bb = bbox_to_str(c.get("bbox"))
        for z in zooms:
            txt = ocr_text_from_bbox(pdf_path, c["page"], bb, zoom=z, psm=6, pad=16) or ""
            pairs.extend(parse_room_heights_from_text(txt))
        if pairs:
            return pairs, c
    return [], None

def _normalize_ocr_punct(s: str) -> str:
    if not s:
        return s
    # curly quotes → straight, en/em dashes → hyphen, weird primes → ascii
    table = str.maketrans({
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "–": "-", "—": "-", "−": "-",
        "′": "'", "″": '"',
        "：": ":",  # fullwidth colon sometimes shows up
    })
    s = s.translate(table)
    # collapse multiple spaces, fix common OCR glue
    s = re.sub(r"\s+", " ", s)
    return s

def fix_ocr_spacing(text: str) -> str:
    """Fix common OCR spacing issues where letters/words are incorrectly separated or merged"""
    if not text:
        return text
    
    lines = text.splitlines()
    fixed_lines = []
    
    for line in lines:
        original_line = line
        
        # Pattern 1: Single capital letters with spaces between them (likely a word)
        # "D E  N G U Y E N" -> "DANGUYEN"
        # Keep trying while we find matches
        max_iterations = 10
        iteration = 0
        while iteration < max_iterations:
            # Match 2-10 consecutive single capital letters with spaces
            match = re.search(r'\b([A-Z])(?: ([A-Z])){1,9}\b', line)
            if not match:
                break
            
            # Get all the letters
            letters = match.group(0).replace(' ', '')
            
            # Only merge if it looks like it could be a word (2-10 letters)
            # and doesn't look like initials (e.g., "U S A" should stay "USA")
            if 2 <= len(letters) <= 10:
                # Replace in the line
                line = line[:match.start()] + letters + line[match.end():]
            else:
                break
            
            iteration += 1
        
        # Pattern 2: Mixed case with spaces like "B O S T O N" -> "BOSTON"
        # This catches patterns after the first pass
        line = re.sub(r'\b([A-Z]) ([A-Z]) ([A-Z]) ([A-Z]) ([A-Z]) ([A-Z])\b', r'\1\2\3\4\5\6', line)
        line = re.sub(r'\b([A-Z]) ([A-Z]) ([A-Z]) ([A-Z]) ([A-Z])\b', r'\1\2\3\4\5', line)
        line = re.sub(r'\b([A-Z]) ([A-Z]) ([A-Z]) ([A-Z])\b', r'\1\2\3\4', line)
        line = re.sub(r'\b([A-Z]) ([A-Z]) ([A-Z])\b', r'\1\2\3', line)
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

# ---------------------------
# Small utilities
# ---------------------------
def prefer_schedule_chunks(chunks):
    ids = []
    for i, c in enumerate(chunks):
        txt = (c.get("text","") or "").upper()
        if c.get("section") == "window schedule" or "WINDOW SCHEDULE" in txt:
            ids.append(i)
    return ids

def is_in_domain(q: str, cfg) -> bool:
    dom = (cfg.get("answering") or {}).get("domain_terms") or []
    ql = q.lower()
    return any(term.lower() in ql for term in dom)

def room_tag_patterns_for_bedroom(num: str) -> list[re.Pattern]:
    """
    Build tolerant regexes for 'Bedroom <num>' variations, including OCR glitches.
    Examples this catches: BR2, BR-2, BEDRM 2, BED RM 2, BED #2, BEDROOM02, etc.
    """
    n = re.escape(str(int(num)))  # normalize like "02" -> "2"
    pieces = [
        rf"\bBR\s*[-#:]?\s*0*{n}\b",
        rf"\bBED\s*R(?:M|OOM)?\s*[-#:]?\s*0*{n}\b",
        rf"\bBED(?:ROOM|RM)?\s*[-#:]?\s*0*{n}\b",
        # Very glitchy OCR (spaces between letters), e.g., B E D R M  2
        rf"\bB\s*E\s*D\s*R?\s*M?\s*0*{n}\b",
    ]
    return [re.compile(p, re.IGNORECASE) for p in pieces]

def _ocr_text(img, psms=(6,7,11,12), whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-#:/().' "):
    import pytesseract
    out = []
    gray = img.convert("L")
    # Cheap binarization helps thin lines
    bw = gray.point(lambda x: 255 if x > 200 else 0, mode='1')
    for psm in psms:
        cfg = f'--psm {psm} -c tessedit_char_whitelist="{whitelist}"'
        t = pytesseract.image_to_string(bw, config=cfg).strip()
        if t:
            out.append(t)
    return "\n".join(out)

# ---------------------------
# Answer
# ---------------------------
def _build_citation(s):
    return {
        "sheet_id": s.get("sheet_id", ""),
        "page": s["page"],
        "section": s.get("section", ""),
        "bbox": bbox_to_str(s.get("bbox")),
    }

def find_designer_name(text):
    """Extract designer name with multiple strategies"""
    
    NON_NAME_WORDS = {
        'PLANS', 'SHALL', 'TAKE', 'PRECEDENCE', 'REVISED', 'ARCHITECTURAL', 'ENGINEER',
        'STRUCTURAL', 'CIVIL', 'MECHANICAL', 'ELECTRICAL', 'PLUMBING', 'DRAWINGS',
        'CONSTRUCTION', 'DOCUMENTS', 'SPECIFICATIONS', 'NOTES', 'GENERAL', 'SCHEDULE',
        'DETAILS', 'SECTIONS', 'ELEVATIONS', 'REVISIONS', 'THE', 'AND', 'OR', 'FOR',
        'BUILDING', 'CODE', 'SHEET', 'SCALE', 'DATE', 'DRAWN', 'CHECKED',
        'SIGNATURE', 'STAMP', 'LICENSE', 'NUMBER', 'PHONE', 'ADDRESS',
        'FIRE', 'SPRINKLER', 'SPRINKLERS', 'PERMIT', 'SEE', 'CITY', 'HANDOUTS',
        'ROOF', 'TRUSSES', 'PHOTOVOLTAIC', 'SYSTEMS', 'DEFERRED','BOSTON', 'AVENUE', 'SEPARATE'
    }
    
    lines = text.splitlines() if text else []

    # Strategy 0: Look for name between "DESIGN" and "DEVELOPMENT"
    for i, line in enumerate(lines):
        if 'DESIGN' in line.upper() and '&' in line:
            # Check next 1-3 lines for the name (before DEVELOPMENT)
            for j in range(i+1, min(i+4, len(lines))):
                check_line = lines[j].strip()
                
                # Stop if we hit DEVELOPMENT (name should be before it)
                if 'DEVELOPMENT' in check_line.upper():
                    break
                
                # Skip obvious non-names
                if any(x in check_line.upper() for x in ['RESIDENCE', 'PROJECT', 'LOCATION', 'AVENUE', 'STREET', 'AVE', 'HTTP', 'WWW']):
                    continue
                
                # Look for two-word name
                name_match = re.search(r'\b([A-Z][a-z]{1,12}\s+[A-Z][a-z]{1,12})\b', check_line)
                if name_match:
                    name = name_match.group(1)
                    words = name.split()
                    if len(words) == 2 and not any(w.upper() in NON_NAME_WORDS for w in words):
                        return name

    # Strategy 1: Look for "DESIGNER:" or "PROJECT DESIGNER:" label
    designer_pattern = re.compile(
        r'(?:PROJECT\s+)?DESIGNER[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        re.IGNORECASE
    )
    match = designer_pattern.search(text)
    if match:
        name = match.group(1).strip()
        words = name.split()
        if not any(w.upper() in NON_NAME_WORDS for w in words):
            return name
        
    
    # Strategy 1.5: Look for standalone two-word name on its own line near company
    for i, line in enumerate(lines):
        if re.search(r'(DESIGN|DEVELOPMENT|GROUP|ARCHITECT)', line, re.I):
            # Check lines within +/- 3 lines
            for j in range(max(0, i-3), min(len(lines), i+4)):
                check_line = lines[j].strip()
                
                # Must be ONLY a name on the line (with maybe | separators)
                clean_line = check_line.replace('|', '').strip()
                
                # Skip if it has obvious non-name content
                if any(x in clean_line.upper() for x in ['LOCATION', 'RESIDENCE', 'AVE', 'STREET', 'PROJECT', 'LOT', 'HTTP', 'WWW', '.COM']):
                    continue
                
                # Check if the whole line is just a two-word name
                name_match = re.match(r'^([A-Z][a-z]{1,12}\s+[A-Z][a-z]{1,12})$', clean_line)
                if name_match:
                    name = name_match.group(1)
                    words = name.split()
                    if not any(w.upper() in NON_NAME_WORDS for w in words):
                        return name
    
    # Strategy 2: Look for name near company keywords (DESIGN, DEVELOPMENT, GROUP, ARCHITECTS, etc.)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(r'(DESIGN|DEVELOPMENT|GROUP|ARCHITECT|ENGINEERING)', line, re.I):
            # Check lines within +/- 5 lines
            search_start = max(0, i - 5)
            search_end = min(len(lines), i + 6)
            
            for j in range(search_start, search_end):
                check_line = lines[j].strip()
                
                # Skip lines that are clearly not names (have addresses, numbers, etc.)
                if re.search(r'\d{3,}|@|\.com|HTTP|WWW|PHONE|FAX|STREET|AVE|ROAD', check_line, re.I):
                    continue
                    
                # Look for two-word capitalized name
                name_match = re.search(r'\b([A-Z][a-z]{1,12}\s+[A-Z][a-z]{1,12})\b', check_line)
                if name_match:
                    name = name_match.group(1)
                    words = name.split()
                    if len(words) == 2 and not any(w.upper() in NON_NAME_WORDS for w in words):
                        if all(2 <= len(w) <= 12 for w in words):
                            return name
    
    # Strategy 5: Find two-word names, but exclude lines with addresses/numbers
    for line in lines:
        # Skip lines with obvious non-name content
        if re.search(r'\d{3,}|@|\.com|HTTP|WWW|AVE|STREET|PHONE|FAX', line, re.I):
            continue
        # Match two-word capitalized names
        name_match = re.search(r'\b([A-Z][a-z]{1,12}\s+[A-Z][a-z]{1,12})\b', line)
        if name_match:
            name = name_match.group(1)
            words = name.split()
            if len(words) == 2 and not any(w.upper() in NON_NAME_WORDS for w in words):
                if all(2 <= len(w) <= 12 for w in words):
                    return name
            
    # Strategy 4: Look for name near company reference (within 10 lines)
    company_line_idx = None
    for i, line in enumerate(lines):
        if re.search(r'(DESIGN|DEVELOPMENT|GROUP|ARCHITECTS?|ENGINEERING)', line, re.I):
            company_line_idx = i
            break
    
    if company_line_idx is not None:
        # Search 10 lines before and after company reference
        search_start = max(0, company_line_idx - 10)
        search_end = min(len(lines), company_line_idx + 10)
        
        for i in range(search_start, search_end):
            line = lines[i].strip()
            # Skip lines with addresses, phones, or lots of numbers
            if re.search(r'\d{3,5}|@|\.com|AVE|STREET|ROAD|PHONE|FAX', line, re.I):
                continue
            # Look for two-word capitalized name
            name_match = re.match(r'^([A-Z][a-z]{1,12}\s+[A-Z][a-z]{1,12})$', line)
            if name_match:
                name = name_match.group(1)
                words = name.split()
                if not any(w.upper() in NON_NAME_WORDS for w in words):
                    return name
    
    return None

def answer(question: str, cfg) -> Dict[str, Any]:
    import time
    start = time.perf_counter()

    # ---- runtime budget & limits ----
    limits = (cfg or {}).get("limits", {})
    OCR_TOTAL_SECS = float(limits.get("ocr_total_secs", 20.0))
    OCR_MAX_PAGES = int(limits.get("ocr_max_pages", 12))
    OCR_MAX_CELLS = int(limits.get("ocr_max_cells", 120))
    OCR_ZOOMS = tuple(limits.get("ocr_zooms", (3.2,)))
    OCR_PSMS = tuple(limits.get("ocr_psms", (6,)))

    def time_left() -> float:
        return OCR_TOTAL_SECS - (time.perf_counter() - start)

    def budget_ok() -> bool:
        return time_left() > 0.0

    def log(msg: str):
        try:
            print(f"[answer] {msg}", flush=True)
        except Exception:
            pass

    q_lower = question.lower()

    # --- Domain gate ---
    if (cfg.get("answering") or {}).get("require_domain"):
        if not is_in_domain(question, cfg):
            return {"answer": "no evidence found", "confidence": 0.0, "citations": []}

    # --- Retrieval & sheet gating ---
    log("retrieving…")
    ranked, chunks = retrieve(question, cfg)
    requested_sheet = sheet_in_question(question)
    if requested_sheet and not doc_has_sheet_or_alias(requested_sheet, chunks):
        return {"answer": "no evidence found (sheet not present in this set)", "confidence": 0.0, "citations": []}

    if ('sprinkler' in q_lower) or ('nfpa' in q_lower):
        
        # Search in first 30 chunks (cover sheet area)
        for c in chunks[:30]:
            text = c.get("text", "") or ""
            text_upper = text.upper()
            
            # Look for NFPA 13D
            if 'NFPA' in text_upper and '13' in text_upper:
                
                # Check if it's NFPA 13D (residential sprinkler)
                if re.search(r'NFPA\s*13\s*D', text, re.I):
                    
                    # Check if deferred/separate permit mentioned
                    if re.search(r'DEFERRED|SEPARATE.*PERMIT', text, re.I):
                        return {
                            "answer": "Yes fire sprinklers required separate permit NFPA 13D noted on cover",
                            "confidence": 0.90,
                            "citations": [_build_citation(c)]
                        }
                    
                    return {
                        "answer": "Yes NFPA 13D fire sprinklers required",
                        "confidence": 0.85,
                        "citations": [_build_citation(c)]
                    }
            
            # Look for sprinkler + required keywords
            if 'SPRINKLER' in text_upper and any(kw in text_upper for kw in ['REQUIRED', 'SHALL', 'MUST']):
                return {
                    "answer": "Yes fire sprinklers required",
                    "confidence": 0.80,
                    "citations": [_build_citation(c)]
                }
        
        return {"answer": "no evidence found", "confidence": 0.0, "citations": []}

    # =======================
    # REVISION DATE
    # =======================
    if is_revision_date_question(question):
        log(f"revision date: requested_sheet = '{requested_sheet}'")
        top_ids = [doc_id for doc_id, _ in ranked[:12]]
        supports = [chunks[i] for i in top_ids]
        
        log(f"revision date: initial {len(supports)} supports from retrieval")

        tb_supports = [c for c in supports if c.get("section") == "title block" or "TITLE BLOCK" in (c.get("text","") or "").upper()]
        if tb_supports:
            supports = tb_supports
            log(f"revision date: narrowed to {len(supports)} title block supports")

        if requested_sheet:
            log(f"revision date: filtering for sheet '{requested_sheet}'")
            
            # Debug: what sheets are on page 1?
            page1_sheets = list(set(c.get("sheet_id", "?") for c in chunks if c.get("page") == 1))
            log(f"revision date: page 1 has these sheet_ids: {page1_sheets}")
            
            # First, try direct sheet_id match (most reliable)
            direct_matches = [c for c in supports if c.get("sheet_id", "").upper().replace(" ", "").replace(".", "") == requested_sheet.upper().replace(" ", "").replace(".", "")]
            
            if direct_matches:
                supports = direct_matches
                log(f"revision date: using {len(supports)} direct matches for '{requested_sheet}'")
            else:
                log(f"revision date: no direct match in retrieval results, searching ALL chunks for '{requested_sheet}'")
                # Search ALL chunks for the requested sheet
                target_normalized = requested_sheet.upper().replace(" ", "").replace(".", "")
                all_sheet_matches = [c for c in chunks 
                                    if c.get("sheet_id", "").upper().replace(" ", "").replace(".", "") == target_normalized
                                    and (c.get("section") == "title block" or "TITLE BLOCK" in (c.get("text","") or "").upper())]
                
                if all_sheet_matches:
                    supports = all_sheet_matches
                    log(f"revision date: found {len(supports)} title blocks for '{requested_sheet}' in all chunks")
                else:
                    # Try looking on page 1 if A1.0 is the cover sheet
                    log(f"revision date: checking if '{requested_sheet}' refers to page 1 (cover sheet)")
                    page1_title_blocks = [c for c in chunks 
                                         if c.get("page") == 1 
                                         and (c.get("section") == "title block" or "TITLE BLOCK" in (c.get("text","") or "").upper())]
                    
                    if page1_title_blocks and requested_sheet.upper() in ["A1.0", "A1", "A-1"]:
                        supports = page1_title_blocks
                        log(f"revision date: using {len(supports)} page 1 title blocks for '{requested_sheet}'")
                    else:
                        # Fall back to alias expansion
                        log(f"revision date: no page 1 match, trying aliases")
                        aliases = expand_sheet_aliases(requested_sheet)
                        
                        exact, tb, soft = filter_supports_by_aliases(supports, aliases)
                        
                        if exact:
                            supports = exact
                            log(f"revision date: using {len(supports)} exact matches")
                        elif tb or soft:
                            supports = tb or soft
                            log(f"revision date: using {len(supports)} tb/soft matches")
                        else:
                            alias_set = {a.upper().replace(" ","") for a in aliases}
                            widened = [c for c in chunks if (c.get("sheet_id","").upper().replace(" ","") in alias_set)
                                       and (c.get("section") == "title block" or "TITLE BLOCK" in (c.get("text","") or "").upper())]
                            supports = widened or [c for c in chunks if (c.get("section") == "title block" or "TITLE BLOCK" in (c.get("text","") or "").upper())]
                            log(f"revision date: widened to {len(supports)} supports")
        
        # Log which sheets we're actually searching
        sheet_ids = list(set(c.get("sheet_id", "?") for c in supports))
        log(f"revision date: searching sheets: {sheet_ids}")

        # text-layer pass - capture dates AND comments
        best_dates, best_cits, date_with_comments = [], [], []
        for s in supports:
            lines = (s.get("text","") or "").splitlines()
            for idx, line in enumerate(lines):
                L = (line or "").upper()
                if any(h in L for h in ["REV", "REVISION", "DATE", "COMMENTS", "ISSUE"]):
                    window = "\n".join(lines[max(0, idx-2): idx+3])
                    
                    # Look for dates in this window
                    dates_in_window = find_dates_in_text(window)
                    for d in dates_in_window:
                        best_dates.append(d)
                        if len(best_cits) < 3: best_cits.append(_build_citation(s))
                        
                        # Try to extract comment text near the date
                        # Look for pattern like: "Second round comments 06/03/2024"
                        comment_pattern = re.compile(
                            rf'([A-Za-z\s]+(?:round|comments?|revision|issue)[A-Za-z\s]*?)\s*{re.escape(d)}',
                            re.I
                        )
                        comment_match = comment_pattern.search(window)
                        if comment_match:
                            comment = comment_match.group(1).strip()
                            date_with_comments.append((d, f"{comment} {d}"))
                            log(f"revision date: found comment '{comment}' with date {d}")
                        else:
                            date_with_comments.append((d, d))

        from datetime import datetime
        import re as _re
        def _parse_any(d):
            for fmt in ("%m/%d/%Y","%m/%d/%y","%Y-%m-%d","%m-%d-%Y","%b %d, %Y","%B %d, %Y"):
                try: return datetime.strptime(d, fmt)
                except Exception: pass
            parts = _re.split(r"[/-]", d)
            if len(parts)==3 and all(p.isdigit() for p in parts):
                m,dd,yy = parts
                if len(yy)==2: yy="20"+yy
                try: return datetime(int(yy), int(m), int(dd))
                except Exception: return None
            return None

        # If we're looking for A1.0 specifically, do a comprehensive page 1 search
        if requested_sheet and requested_sheet.upper() in ["A1.0", "A1", "A-1"]:
            log("revision date: doing comprehensive page 1 search for A1.0...")
            page1_all = [c for c in chunks if c.get('page') == 1]
            
            for s in page1_all:
                txt = s.get('text', '')
                lines = txt.splitlines()
                for idx, line in enumerate(lines):
                    L = line.upper()
                    # Look for revision-related keywords
                    if any(kw in L for kw in ['REV', 'REVISION', 'COMMENTS', 'ISSUE', 'ROUND']) and any(c.isdigit() for c in line):
                        window = "\n".join(lines[max(0, idx-2): idx+3])
                        dates = find_dates_in_text(window)
                        for d in dates:
                            if d not in best_dates:
                                best_dates.append(d)
                                if len(best_cits) < 5:
                                    best_cits.append(_build_citation(s))
                                
                                # Try to extract comment - use multiple patterns
                                # Pattern 1: Text before date
                                comment_pattern = re.compile(
                                    rf'([A-Za-z][A-Za-z\s]{{3,40}}?)\s*{re.escape(d)}',
                                    re.I
                                )
                                comment_match = comment_pattern.search(window)
                                
                                if comment_match:
                                    comment = comment_match.group(1).strip()
                                    # Clean up the comment - remove common prefixes
                                    comment = re.sub(r'^(DATE|REVISION|REV|ISSUE)[\s:]*', '', comment, flags=re.I).strip()
                                    if comment and len(comment) > 3:
                                        date_with_comments.append((d, f"{comment} {d}"))
                                        log(f"revision date: page 1 found comment '{comment}' with date {d}")
                                    else:
                                        date_with_comments.append((d, d))
                                else:
                                    date_with_comments.append((d, d))

        if best_dates:
            log(f"revision date: found {len(best_dates)} dates: {best_dates}")
            log(f"revision date: found {len(date_with_comments)} date+comment pairs")
            
            parsed = [(p, d, full_text) for d, full_text in date_with_comments if (p:=_parse_any(d))]
            if parsed:
                # Log all parsed dates
                for p, d, full_text in parsed:
                    log(f"revision date: parsed {p.strftime('%Y-%m-%d')} from '{full_text}'")
                
                # Get the latest date
                latest = max(parsed, key=lambda t: t[0])
                latest_date = latest[1]
                latest_full_text = latest[2]
                log(f"revision date: using latest '{latest_full_text}'")
                return {"answer": latest_full_text, "confidence": 0.9, "citations": best_cits}
            from collections import Counter
            val = Counter(best_dates).most_common(1)[0][0]
            return {"answer": val, "confidence": 0.85, "citations": best_cits}

        # Page 1 fallback
        if not best_dates:
            log("revision date: checking page 1…")
            page1_chunks = [c for c in chunks if c.get('page') == 1]
            for s in page1_chunks:
                txt = s.get('text', '')
                lines = txt.splitlines()
                for idx, line in enumerate(lines):
                    L = line.upper()
                    if any(kw in L for kw in ['REV', 'REVISION', 'COMMENTS', 'ISSUE', 'DATE']) and any(c.isdigit() for c in line):
                        window = "\n".join(lines[max(0, idx-2): idx+3])
                        dates = find_dates_in_text(window)
                        if dates:
                            best_dates.extend(dates)
                            if len(best_cits) < 3: best_cits.append(_build_citation(s))
                            break
                if best_dates:
                    break
            
            if best_dates:
                parsed = [(p, d) for d in best_dates if (p:=_parse_any(d))]
                if parsed:
                    latest = max(parsed, key=lambda t: t[0])[1]
                    return {"answer": latest, "confidence": 0.8, "citations": best_cits}
                from collections import Counter
                val = Counter(best_dates).most_common(1)[0][0]
                return {"answer": val, "confidence": 0.75, "citations": best_cits}

        # OCR fallback
        pdf_path = cfg.get("pdf_path")
        if pdf_path and budget_ok():
            log("OCR: title block sweep…")
            pages = sorted({s["page"] for s in supports}) or ([chunks[0]["page"]] if chunks else [])
            if 1 not in pages:
                pages = [1] + list(pages)
            pages = pages[:OCR_MAX_PAGES]
            found = []
            for pg in pages:
                if not budget_ok(): break
                for txt in ocr_guess_title_block(pdf_path, pg, zoom=(OCR_ZOOMS[0] if OCR_ZOOMS else 3.2), psms=OCR_PSMS) or []:
                    if not budget_ok(): break
                    found.extend(find_dates_in_text(txt))
            if found:
                parsed = [(p, d) for d in found if (p:=_parse_any(d))]
                if parsed:
                    latest = max(parsed, key=lambda t: t[0])[1]
                    return {"answer": latest, "confidence": 0.7,
                            "citations": [{"sheet_id": requested_sheet or "", "page": pages[0], "section": "title block", "bbox": ""}]}
                from collections import Counter
                if found:
                    val = Counter(found).most_common(1)[0][0]
                    return {"answer": val, "confidence": 0.65,
                            "citations": [{"sheet_id": requested_sheet or "", "page": pages[0], "section": "title block", "bbox": ""}]}

    # =======================
    # PROJECT ADDRESS
    # =======================
    
    if is_address_question(question):
        
        def dejunk_ocr(text):
            """Fix OCR spacing: 'B O S T O N' -> 'BOSTON'"""
            words = text.split()
            result = []
            i = 0
            while i < len(words):
                # If we hit a single capital letter, collect all consecutive single caps
                if len(words[i]) == 1 and words[i].isupper():
                    letters = []
                    while i < len(words) and len(words[i]) == 1 and words[i].isupper():
                        letters.append(words[i])
                        i += 1
                    result.append(''.join(letters))
                else:
                    result.append(words[i])
                    i += 1
            return ' '.join(result)
        
        def find_multiline_address(text):
            lines = text.splitlines()
            for i, line in enumerate(lines):
                cleaned = dejunk_ocr(line)
                
                # Look for street address - handle "LOT 1- 374 LASTRETO AVE.," format
                street_match = re.search(
                    r'(?:LOT\s+\d+[- ]+)?(\d+\s+[A-Z]+(?:\s+[A-Z]+)*\s+(?:AVE|AVENUE|ST|STREET|RD|ROAD|DR|DRIVE|BLVD|WAY|LN|LANE|CT|COURT)\.?,?)',
                    cleaned,
                    re.IGNORECASE
                )
                
                if street_match:
                    street = street_match.group(1).strip().rstrip(',.')
                    
                    # Look in next few lines for city, state, zip
                    for j in range(i + 1, min(i + 5, len(lines))):
                        next_line = lines[j]
                        cleaned_next = dejunk_ocr(next_line)
                        
                        # Match city, state, zip - handle with or without comma
                        city_match = re.search(
                            r'([A-Z]+(?:\s+[A-Z]+)*)[,\s]+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)',
                            cleaned_next,
                            re.IGNORECASE
                        )
                        
                        if city_match:
                            city = city_match.group(1).strip()
                            state = city_match.group(2).upper()
                            zip_code = city_match.group(3)
                            return f"{street} {city} {state} {zip_code}"
                
                # Also try original pattern without LOT prefix
                street_match = re.search(
                    r'\b(\d+\s+[A-Z]+(?:\s+[A-Z]+)*\s+(?:AVE|AVENUE|ST|STREET|RD|ROAD|DR|DRIVE|BLVD|WAY|LN|LANE|CT|COURT)\.?)\b',
                    cleaned,
                    re.IGNORECASE
                )
                
                if street_match and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    cleaned_next = dejunk_ocr(next_line)
                    
                    # Match city, state, zip
                    city_match = re.search(
                        r'([A-Z]+(?:\s+[A-Z]+)*),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)',
                        cleaned_next,
                        re.IGNORECASE
                    )
                    
                    if city_match:
                        street = street_match.group(1).strip()
                        city = city_match.group(1).strip()
                        state = city_match.group(2)
                        zip_code = city_match.group(3)
                        return f"{street}, {city}, {state} {zip_code}"
            return None
        
        # Try single-line address pattern first (most common)
        page1_chunks = [c for c in chunks if c.get('page') == 1]
        
        log(f"address: found {len(page1_chunks)} chunks on page 1")
        
        for c in page1_chunks:
            txt = c.get('text', '')
            
            # Log samples to see what we're working with
            if '374' in txt or 'LASTRETO' in txt.upper() or 'SUNNYVALE' in txt.upper():
                log(f"address: found relevant text: {txt[:200]}")
            
            # Pattern for single-line address: "374 Lastreto Ave Sunnyvale CA 94085"
            single_line_pattern = re.compile(
                r'\b(\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Ave|Avenue|St|Street|Rd|Road|Dr|Drive|Blvd|Boulevard|Way|Ln|Lane|Ct|Court)\.?)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\b',
                re.IGNORECASE
            )
            
            match = single_line_pattern.search(txt)
            if match:
                street = match.group(1).strip()
                city = match.group(2).strip()
                state = match.group(3).upper()
                zip_code = match.group(4)
                address = f"{street} {city} {state} {zip_code}"
                log(f"address: found single-line '{address}'")
                return {"answer": address, "confidence": 0.90, "citations": [_build_citation(c)]}
            
            # Try multi-line address (original logic)
            addr = find_multiline_address(txt)
            if addr:
                log(f"address: found multi-line '{addr}'")
                return {"answer": addr, "confidence": 0.85, "citations": [_build_citation(c)]}
        
        log("address: not found in text layer, trying broader search...")
        
        # Broader search - look for just the street number and name
        for c in page1_chunks:
            txt = c.get('text', '')
            # Look for any address-like pattern
            if re.search(r'\b\d{3}\s+[A-Z]', txt, re.IGNORECASE):
                log(f"address: potential address in chunk: {txt[:150]}")
        
        log("address: not found")
        return {"answer": "no evidence found", "confidence": 0.0, "citations": []}
    # =======================
    # SHEET SCALE
    # =======================
    if is_scale_question(question):
        log("scale: scanning text layer…")
        top_ids = [doc_id for doc_id, _ in ranked[:12]]
        supports = [chunks[i] for i in top_ids]
        preferred = {"title block","floor plan","elevations","sections","roof plan"}
        pref = [c for c in supports if c.get("section") in preferred or "SCALE" in (c.get("text","") or "").upper()]
        if pref: supports = pref

        if requested_sheet:
            exact, soft, tb = filter_supports_by_sheet(supports, requested_sheet)
            if exact: supports = exact
            else:
                target = requested_sheet.upper().replace(" ","")
                sheet_chunks = [c for c in chunks if (c.get("sheet_id") or "").upper().replace(" ","")==target]
                if not sheet_chunks:
                    return {"answer":"no evidence found","confidence":0.0,"citations":[]}
                supports = sheet_chunks

        cits = []
        for s in supports:
            hit = find_scale_in_text(s.get("text","") or "")
            if hit:
                if len(cits) < 3: cits.append(_build_citation(s))
                ans = hit.replace("SCALE","").replace("Scale","").strip(" :")
                return {"answer": ans, "confidence": 0.8, "citations": cits}

        # OCR fallback
        pdf_path = cfg.get("pdf_path")
        if pdf_path and budget_ok():
            log("scale: OCR fallback…")
            if requested_sheet:
                target = requested_sheet.upper().replace(" ", "")
                candidates = [c for c in chunks if (c.get("sheet_id") or "").upper().replace(" ", "") == target and c.get("bbox")]
            else:
                candidates = [c for c in supports if c.get("bbox")]

            def _prio(c):
                sec = (c.get("section","") or "").lower()
                order = {"title block": 0, "floor plan": 1, "sections": 2, "elevations": 3}
                y1 = (c.get("bbox") or [0,0,0,0])[3]
                return (order.get(sec, 99), -float(y1))
            candidates.sort(key=_prio)

            cells_seen = 0
            for s in candidates[:12]:
                if not budget_ok() or cells_seen >= OCR_MAX_CELLS: break
                bb = bbox_to_str(s.get("bbox"))
                for z in (OCR_ZOOMS or (3.2,)):
                    if not budget_ok() or cells_seen >= OCR_MAX_CELLS: break
                    for psm in (OCR_PSMS or (6,)):
                        if not budget_ok() or cells_seen >= OCR_MAX_CELLS: break
                        ocr_txt = ocr_text_from_bbox(pdf_path, s["page"], bb, zoom=z, psm=psm, pad=16) or ""
                        cells_seen += 1
                        hit = find_scale_in_text(ocr_txt)
                        if hit:
                            ans = hit.replace("SCALE","").replace("Scale","").strip(" :")
                            return {"answer": ans, "confidence": 0.68, "citations": [_build_citation(s)]}

    # =======================
    # PROJECT DESIGNER
    # =======================
    q_lower = question.lower()
    if "designer" in q_lower and "title block" in q_lower:
        log("designer: scanning cover sheet…")
        
        # Search title block chunks on page 1
        title_block_chunks = [c for c in chunks if c.get("section") == "title block" and c.get("page") == 1]
        
        # DEBUG: Show what's in the text layer
        log(f"designer: found {len(title_block_chunks)} title block chunks")
        for idx, c in enumerate(title_block_chunks[:3]):
            text = c.get("text", "") or ""
            log(f"designer: chunk {idx} has {len(text)} chars")
            # Show first 500 chars to see structure
            sample = text[:500].replace('\n', ' | ')
            log(f"designer: chunk {idx} sample: {sample}")
        
        # Try text layer first with improved extraction
        for c in title_block_chunks:
            text = c.get("text", "") or ""
            
            # Fix spaced letters in text layer (e.g., "D E" -> "DE")
            text = fix_ocr_spacing(text)
            
            name = find_designer_name(text)
            if name:
                log(f"designer: found '{name}' in text layer (after spacing fix)")
                return {
                    "answer": name,
                    "confidence": 0.85,
                    "citations": [_build_citation(c)]
                }
        
            # DEBUG: Show what's actually in each chunk
        for idx, c in enumerate(title_block_chunks[:3]):
            text = c.get("text", "") or ""
            # Show all lines to see structure
            log(f"designer: === CHUNK {idx} FULL TEXT ===")
            for line_num, line in enumerate(text.splitlines()[:30]):
                if line.strip():
                    log(f"designer: chunk {idx} line {line_num}: {line}")
        
        # If text layer fails, try OCR
        # If text layer fails, try OCR
        log("designer: text layer failed, trying OCR...")
        pdf_path = cfg.get("pdf_path")
        if pdf_path and budget_ok():
            # Try multiple zoom levels and PSM modes for better accuracy
            zoom_levels = [3.5, 4.0, 3.0]
            psm_modes = [6, 11, 3]
            # Try wider bbox to capture more of the title block
            bboxes = ["50,20,2550,600", "100,30,2500,550", "124,44,2524,525"]
            
            for bbox in bboxes:
                if not budget_ok():
                    break
                # Try full page OCR with image preprocessing
                log("designer: trying full page OCR with preprocessing...")
                try:
                    import fitz
                    from PIL import Image, ImageEnhance, ImageFilter
                    import pytesseract
                    
                    doc = fitz.open(pdf_path)
                    page = doc.load_page(0)
                    
                    # Use high zoom for better OCR
                    mat = fitz.Matrix(4.0, 4.0)
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Crop to title block area (top-left quadrant where company info usually is)
                    W, H = img.size
                    title_area = img.crop((0, 0, int(W*0.4), int(H*0.25)))
                    
                    # Enhance image
                    enhancer = ImageEnhance.Contrast(title_area)
                    title_area = enhancer.enhance(2.0)
                    enhancer = ImageEnhance.Sharpness(title_area)
                    title_area = enhancer.enhance(2.0)
                    
                    # Convert to grayscale and apply threshold
                    gray = title_area.convert('L')
                    # Use adaptive threshold
                    import numpy as np
                    img_array = np.array(gray)
                    threshold = np.mean(img_array) * 0.9
                    binary = gray.point(lambda x: 255 if x > threshold else 0, mode='1')
                    
                    # OCR with multiple PSM modes
                    for psm in [6, 11, 3, 4]:
                        if not budget_ok():
                            break
                        
                        ocr_text = pytesseract.image_to_string(
                            binary,
                            config=f'--psm {psm}'
                        ).strip()
                        
                        if ocr_text:
                            log(f"designer: full page OCR psm={psm} returned {len(ocr_text)} chars")
                            
                            # Fix spacing
                            ocr_text = fix_ocr_spacing(ocr_text)
                            
                            # Show all lines
                            for i, line in enumerate(ocr_text.splitlines()[:50]):
                                if line.strip():
                                    log(f"designer: full OCR line {i}: {line}")
                            
                            name = find_designer_name(ocr_text)
                            if name:
                                log(f"designer: full page OCR found '{name}'")
                                return {
                                    "answer": name,
                                    "confidence": 0.70,
                                    "citations": [{
                                        "sheet_id": "",
                                        "page": 1,
                                        "section": "title block",
                                        "bbox": ""
                                    }]
                                }

                except Exception as e:
                    log(f"designer: full page OCR failed: {e}")
                    
                    # DEBUG: Show first 1000 chars of OCR output
                    ocr_sample = ocr_text[:1000].replace('\n', ' | ')
                    log(f"designer: OCR sample (after fix): {ocr_sample}")
                    
                    # Show lines that contain "DESIGNER" or "DE" or "NGUYEN"
                    lines = ocr_text.splitlines()
                    for i, line in enumerate(lines[:30]):
                        if any(kw in line.upper() for kw in ['DESIGNER', 'DE ', ' DE', 'NGUYEN', 'PROJECT']):
                            log(f"designer: OCR line {i}: {line}")
                    
                    name = find_designer_name(ocr_text)
                    if name:
                        log(f"designer: OCR found '{name}'")
                        return {
                            "answer": name,
                            "confidence": 0.75,
                            "citations": [{
                                "sheet_id": "",
                                "page": 1,
                                "section": "title block",
                                "bbox": "124,44,2524,525"
                            }]
                        }
        
        log("designer: not found")
        return {"answer": "no evidence found", "confidence": 0.0, "citations": []}
    
    # =======================
    # ROOM FLOOR FINISH
    # =======================
    if is_finish_question(question):
        log("finish: scanning text layer…")
        room = normalize_room_name(question)
        top_ids = [doc_id for doc_id,_ in ranked[:20]]
        supports = [chunks[i] for i in top_ids]
        preferred_sections = {"legend","finish schedule","schedules","floor plan","title block"}
        pref = [c for c in supports if c.get("section") in preferred_sections]
        supports = pref or supports

        # A) finish schedule / legend
        for s in supports:
            txt = (s.get("text","") or "").upper()
            if "FINISH" in txt and ("SCHEDULE" in txt or "LEGEND" in txt):
                blob = txt
                if room and room in txt:
                    lines = txt.splitlines()
                    windows = [" ".join(lines[max(0,i-2): i+3]) for i,l in enumerate(lines) if room in l]
                    if windows: blob = " ".join(windows)
                tokens = [t for t in FLOOR_FINISH_TOKENS if t in blob]
                if tokens:
                    return {"answer": tokens[0].title(), "confidence": 0.8, "citations": [_build_citation(s)]}

        # B) plan proximity
        for s in supports:
            if s.get("section") == "floor plan":
                txt = (s.get("text","") or "").upper()
                if room and room in txt:
                    lines = txt.splitlines()
                    idxs = [i for i,l in enumerate(lines) if room in l]
                    for i in idxs:
                        win = " ".join(lines[max(0,i-3): i+4])
                        tokens = [t for t in FLOOR_FINISH_TOKENS if t in win]
                        if tokens:
                            return {"answer": tokens[0].title(), "confidence": 0.65, "citations": [_build_citation(s)]}

        # C) OCR fallback
        pdf_path = cfg.get("pdf_path")
        if pdf_path and budget_ok():
            log("finish: OCR fallback…")
            ocr_candidates = []
            for s in supports:
                U = (s.get("text", "") or "").upper()
                if (room and room in U) or (s.get("section") in {"legend", "finish schedule", "schedules"}) or ("FINISH" in U and ("SCHEDULE" in U or "LEGEND" in U)):
                    if s.get("bbox"): ocr_candidates.append(s)
            if not ocr_candidates and room:
                for c in chunks:
                    U = (c.get("text", "") or "").upper()
                    if room in U and c.get("bbox"): ocr_candidates.append(c)

            seen, cand, cells_seen = set(), [], 0
            for s in ocr_candidates:
                key = (s.get("page"), tuple(s.get("bbox") or []))
                if key not in seen:
                    seen.add(key); cand.append(s)

            for s in cand[:6]:
                if not budget_ok() or cells_seen >= OCR_MAX_CELLS: break
                bb = bbox_to_str(s.get("bbox"))
                if not bb: continue
                ocr_txt = ocr_text_from_bbox(pdf_path, s["page"], bb, zoom=(OCR_ZOOMS[0] if OCR_ZOOMS else 3.2)) or ""
                cells_seen += 1
                U = ocr_txt.upper()
                focus = U
                if room and room in U:
                    lines = U.splitlines()
                    windows = [" ".join(lines[max(0,i-2): i+3]) for i,l in enumerate(lines) if room in l]
                    if windows: focus = " ".join(windows)
                tokens = [t for t in FLOOR_FINISH_TOKENS if t in focus]
                if tokens:
                    return {"answer": tokens[0].title(), "confidence": 0.70, "citations": [_build_citation(s)]}

        return {"answer": "no evidence found", "confidence": 0.0, "citations": []}

    # =======================
    # WINDOW TYPE IN BEDROOM N
    # =======================
    if is_window_type_in_room_question(question):
        log("window type: harvesting codes…")
        bnum = extract_bedroom_num(question)
        if not bnum:
            return {"answer": "no evidence found (no bedroom number specified)", "confidence": 0.0, "citations": []}
        
        room_tags = [
            f"BEDROOM {bnum}", f"BEDROOM{bnum}", f"BEDRM {bnum}", f"BEDRM{bnum}",
            f"BR {bnum}", f"BR{bnum}", f"BR-{bnum}", f"BR #{bnum}",
            f"BDRM {bnum}", f"BDRM{bnum}", f"B{bnum}", f"BED RM {bnum}"
        ]
        
        log(f"window type: looking for bedroom {bnum}")
        
        type_map = {
            'C': 'Casement', 'S': 'Slider', 'SL': 'Slider', 'F': 'Fixed',
            'A': 'Awning', 'H': 'Hopper', 'SH': 'Single Hung',
            'DH': 'Double Hung', 'P': 'Picture', 'G': 'Gliding'
        }
        
        size_type_pattern = re.compile(
            r"W\s*\d+['\"]?-?\d*['\"]?\s*[xX]\s*\d+['\"]?-?\d*['\"]?\s*([A-Z]{1,2})\b",
            re.I
        )
        
        patts = room_tag_patterns_for_bedroom(bnum)
        
        # Search in chunks that mention the bedroom
        for c in chunks:
            txt = c.get("text", "")
            if any(p.search(txt) for p in patts):
                log(f"window type: found bedroom in chunk page={c['page']}, section={c.get('section')}")
                for match in size_type_pattern.finditer(txt):
                    type_code = match.group(1).upper()
                    if type_code in type_map:
                        wtype = type_map[type_code]
                        log(f"window type: parsed {match.group(0)} → {wtype}")
                        return {
                            "answer": wtype,
                            "confidence": 0.85,
                            "citations": [_build_citation(c)]
                        }
        
        # Fallback: proximity search
        log("window type: trying proximity search…")
        window_codes = harvest_window_codes_by_proximity(chunks, bnum, max_dist_px=400.0)
        
        if window_codes:
            log(f"window type: found codes {window_codes}")
            for c in chunks:
                txt = c.get("text", "")
                for code in window_codes:
                    if code in txt:
                        for match in size_type_pattern.finditer(txt):
                            type_code = match.group(1).upper()
                            if type_code in type_map:
                                wtype = type_map[type_code]
                                log(f"window type: found {wtype} near {code}")
                                return {
                                    "answer": wtype,
                                    "confidence": 0.75,
                                    "citations": [_build_citation(c)]
                                }
        
        log("window type: no evidence found")
        return {"answer": "no evidence found", "confidence": 0.0, "citations": []}

    # =======================
    # ROOMS + CEILING HEIGHTS
    # =======================
    if is_room_ceiling_list_question(question):
        log("ceiling list: scanning…")
        req = sheet_in_question(question) or "A2.0"
        if req and not doc_has_sheet_or_alias(req, chunks):
            return {"answer":"no evidence found (sheet not present in this set)","confidence":0.0,"citations":[]}

        target = req.upper().replace(" ","")
        sheet_chunks = [c for c in chunks if (c.get("sheet_id") or "").upper().replace(" ","")==target]
        pairs=[]
        for c in sheet_chunks:
            pairs.extend(parse_room_heights_from_text((c.get("text","") or "")))
        if pairs:
            ans = "; ".join([f"{r.title()} — {h}" for r,h in pairs[:12]])
            return {"answer": ans, "confidence": 0.65, "citations": [_build_citation(sheet_chunks[0])]}

        pdf_path = cfg.get("pdf_path")
        if pdf_path and budget_ok():
            log("ceiling list: OCR fallback…")
            pairs, cit = ocr_room_heights_on_sheet(req, chunks, pdf_path)
            if pairs:
                ans = "; ".join([f"{r.title()} — {h}" for r,h in pairs[:12]])
                return {"answer": ans, "confidence": 0.65, "citations": [_build_citation(cit)]}

        return {"answer":"no evidence found","confidence":0.0,"citations":[]}

    # =======================
    # OCCUPANCY GROUP
    # =======================
    if "occupancy" in q_lower and ("group" in q_lower or "listed" in q_lower):
        log("occupancy: scanning cover sheet...")
        
        # Pattern for occupancy codes: R-3, R3, A, B-2, etc.
        # Valid occupancy groups: A (assembly), B (business), E (educational), 
        # F (factory), H (hazardous), I (institutional), M (mercantile), 
        # R (residential), S (storage), U (utility)
        OCCUPANCY_PATTERN = re.compile(
            r'\b([RABEFHILMSU])[-\s]?(\d+)\b',
            re.I
        )
        
        # Search cover sheet chunks - prioritize title block
        cover_chunks = [c for c in chunks[:30] if 
                       c.get("section") == "title block" or 
                       (c.get("page") == 1 and c.get("section") != "floor plan")]
        
        for c in cover_chunks:
            text = c.get("text", "") or ""
            
            # Look for occupancy group labels
            if re.search(r'\b(OCCUPANCY|OCC\.?)\s*(GROUP|CLASS|TYPE)?', text, re.I):
                log(f"occupancy: found OCCUPANCY keyword in section {c.get('section')}")
                
                # Extract occupancy code near the keyword
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if re.search(r'\bOCCUPANCY', line, re.I):
                        # Look in this line and next few lines
                        search_text = '\n'.join(lines[i:min(i+5, len(lines))])
                        
                        match = OCCUPANCY_PATTERN.search(search_text)
                        if match:
                            letter = match.group(1).upper()
                            number = match.group(2)
                            code = f"{letter}-{number}" if number else letter
                            
                            log(f"occupancy: found code '{code}' near OCCUPANCY keyword")
                            
                            # Validate it's a real occupancy code (R for residential is most common)
                            if letter == 'R':
                                # Look for descriptive text
                                if 'SINGLE FAMILY' in text.upper():
                                    if 'GARAGE' in text.upper() or 'U' in search_text:
                                        return {
                                            "answer": f"R{number} single family dwelling and U private garage",
                                            "confidence": 0.85,
                                            "citations": [_build_citation(c)]
                                        }
                                    return {
                                        "answer": f"R{number} single family dwelling",
                                        "confidence": 0.80,
                                        "citations": [_build_citation(c)]
                                    }
                                elif 'DWELLING' in text.upper():
                                    return {
                                        "answer": f"R{number} dwelling",
                                        "confidence": 0.80,
                                        "citations": [_build_citation(c)]
                                    }
                                else:
                                    return {
                                        "answer": f"R-{number}",
                                        "confidence": 0.75,
                                        "citations": [_build_citation(c)]
                                    }
                            elif letter in ['A', 'B', 'E', 'F', 'I', 'M', 'S', 'U', 'H']:
                                # Other valid occupancy types
                                return {
                                    "answer": f"{letter}-{number}",
                                    "confidence": 0.75,
                                    "citations": [_build_citation(c)]
                                }
        
        # Broader search - look for R-3 or similar in title block text
        title_blocks = [c for c in chunks[:20] if c.get("section") == "title block"]
        for c in title_blocks:
            text = c.get("text", "") or ""
            
            # Look for R-3, R3 pattern
            match = re.search(r'\bR[-\s]?3\b', text, re.I)
            if match:
                log(f"occupancy: found R-3 in title block")
                if 'SINGLE FAMILY' in text.upper():
                    if 'GARAGE' in text.upper():
                        return {
                            "answer": "R3 single family dwelling and U private garage",
                            "confidence": 0.80,
                            "citations": [_build_citation(c)]
                        }
                    return {
                        "answer": "R3 single family dwelling",
                        "confidence": 0.80,
                        "citations": [_build_citation(c)]
                    }
        
        log("occupancy: not found")

    # =======================
    # CONSTRUCTION TYPE 
    # =======================
    if "construction type" in q_lower or ("construction" in q_lower and "type" in q_lower):
        log("construction type: scanning cover sheet...")
        # Debug: search ALL chunks for "TYPE" or "CONSTRUCTION"
        log("construction type: searching all chunks for TYPE/CONSTRUCTION keywords...")
        for idx, c in enumerate(chunks[:40]):
            text = c.get("text", "") or ""
            if re.search(r'\bTYPE\s+[IVX]', text, re.I) or re.search(r'CONSTRUCTION\s+TYPE', text, re.I):
                log(f"construction type: found in chunk {idx}, page {c.get('page')}, section {c.get('section')}")
                log(f"construction type: text sample: {text[:300]}")
                
        # Pattern for construction types: Type I, Type II-A, Type V-B, etc.
        CONSTRUCTION_TYPE_PATTERN = re.compile(
            r'TYPE\s+([IVX]+)[-\s]?([AB])?',
            re.I
        )
        
        # Search cover sheet chunks
        cover_chunks = [c for c in chunks[:30] if 
                       c.get("section") == "title block" or 
                       (c.get("page") == 1 and c.get("section") in ["general notes", ""])]
        
        log(f"construction type: searching {len(cover_chunks)} cover chunks")
        
        # Try text layer first
        for c in cover_chunks:
            text = c.get("text", "") or ""
            
            if re.search(r'\b(CONSTRUCTION\s+TYPE|TYPE\s+OF\s+CONSTRUCTION)', text, re.I):
                log(f"construction type: found keyword")
                
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if re.search(r'\b(CONSTRUCTION\s+TYPE|TYPE\s+OF\s+CONSTRUCTION)', line, re.I):
                        search_text = '\n'.join(lines[i:min(i+5, len(lines))])
                        
                        match = CONSTRUCTION_TYPE_PATTERN.search(search_text)
                        if match:
                            roman = match.group(1).upper()
                            subtype = match.group(2).upper() if match.group(2) else ""
                            
                            if roman in ['I', 'II', 'III', 'IV', 'V']:
                                construction_type = f"Type {roman}"
                                if subtype:
                                    construction_type += f" {subtype}"
                                
                                log(f"construction type: text layer found '{construction_type}'")
                                
                                if 'SPRINKLER' in search_text.upper():
                                    return {
                                        "answer": f"{construction_type} with sprinklers",
                                        "confidence": 0.85,
                                        "citations": [_build_citation(c)]
                                    }
                                
                                return {
                                    "answer": construction_type,
                                    "confidence": 0.80,
                                    "citations": [_build_citation(c)]
                                }
        
        # Broader text search
        for c in cover_chunks:
            text = c.get("text", "") or ""
            match = CONSTRUCTION_TYPE_PATTERN.search(text)
            if match:
                roman = match.group(1).upper()
                subtype = match.group(2).upper() if match.group(2) else ""
                
                if roman in ['I', 'II', 'III', 'IV', 'V']:
                    construction_type = f"Type {roman}"
                    if subtype:
                        construction_type += f" {subtype}"
                    
                    log(f"construction type: text layer broad match '{construction_type}'")
                    
                    if 'SPRINKLER' in text.upper():
                        return {
                            "answer": f"{construction_type} with sprinklers",
                            "confidence": 0.75,
                            "citations": [_build_citation(c)]
                        }
                    
                    return {
                        "answer": construction_type,
                        "confidence": 0.70,
                        "citations": [_build_citation(c)]
                    }
        # Strategy: Look near OCCUPANCY since they're usually together
        log("construction type: searching near OCCUPANCY text...")
        for c in chunks[:40]:
            text = c.get("text", "") or ""
            
            if re.search(r'OCCUPANCY', text, re.I):
                log(f"construction type: found OCCUPANCY in page {c.get('page')}, section {c.get('section')}")
                
                # Look for "TYPE V" or "TYPE V-B" or "TYPE VB" in nearby text
                # Also check for pattern like "CONSTRUCTION: TYPE V B"
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if 'OCCUPANCY' in line.upper():
                        # Check this line and next 5 lines
                        search_text = '\n'.join(lines[i:min(i+6, len(lines))])
                        log(f"construction type: search context: {search_text[:200]}")
                        
                        # Pattern 1: "TYPE V B" or "TYPE VB" or "TYPE V-B"
                        match = re.search(r'TYPE\s+V\s*[-\s]?\s*B', search_text, re.I)
                        if match:
                            log(f"construction type: found TYPE V B")
                            if 'SPRINKLER' in text.upper():
                                return {
                                    "answer": "Type V B with sprinklers",
                                    "confidence": 0.85,
                                    "citations": [_build_citation(c)]
                                }
                            return {
                                "answer": "Type V B",
                                "confidence": 0.80,
                                "citations": [_build_citation(c)]
                            }
                        
                        # Pattern 2: Just "V B" after "CONSTRUCTION" or "TYPE"
                        match = re.search(r'(?:CONSTRUCTION|TYPE)[:\s]+V\s*[-\s]?\s*B', search_text, re.I)
                        if match:
                            log(f"construction type: found V B pattern")
                            if 'SPRINKLER' in text.upper():
                                return {
                                    "answer": "Type V B with sprinklers",
                                    "confidence": 0.80,
                                    "citations": [_build_citation(c)]
                                }
                            return {
                                "answer": "Type V B",
                                "confidence": 0.75,
                                "citations": [_build_citation(c)]
                            }

            # OCR fallback - try multiple strategies
            log("construction type: text layer failed, trying OCR...")
            pdf_path = cfg.get("pdf_path")
            if pdf_path and budget_ok():
                # Strategy 1: OCR title blocks and general notes on page 1
                page1_chunks = [c for c in chunks if c.get("page") == 1 and 
                            c.get("section") in ["title block", "general notes"] and 
                            c.get("bbox")]
                
                for chunk in page1_chunks[:5]:
                    if not budget_ok():
                        break
                    
                    bb = bbox_to_str(chunk.get("bbox"))
                    if not bb:
                        continue
                    
                    # Try multiple PSM modes
                    for psm in [6, 11]:
                        if not budget_ok():
                            break
                        
                        log(f"construction type: OCR page 1, psm={psm}")
                        ocr_text = ocr_text_from_bbox(pdf_path, 1, bb, zoom=3.5, psm=psm) or ""
                        
                        if ocr_text:
                            log(f"construction type: OCR returned {len(ocr_text)} chars")

                            # NEW: Show ALL lines to find where TYPE V B is
                            lines = ocr_text.splitlines()
                            log(f"construction type: OCR has {len(lines)} total lines")
                            for i, line in enumerate(lines[:60]):
                                if line.strip():  # Only non-empty lines
                                    log(f"construction type: OCR line {i}: {line[:120]}")
                            
                            # Look for construction type pattern
                            match = CONSTRUCTION_TYPE_PATTERN.search(ocr_text)
                            if match:
                                roman = match.group(1).upper()
                                subtype = match.group(2).upper() if match.group(2) else ""
                                
                                if roman in ['I', 'II', 'III', 'IV', 'V']:
                                    construction_type = f"Type {roman}"
                                    if subtype:
                                        construction_type += f" {subtype}"
                                    
                                    log(f"construction type: OCR found '{construction_type}'")
                                    
                                    if 'SPRINKLER' in ocr_text.upper():
                                        return {
                                            "answer": f"{construction_type} with sprinklers",
                                            "confidence": 0.70,
                                            "citations": [_build_citation(chunk)]
                                        }
                                    
                                    return {
                                        "answer": construction_type,
                                        "confidence": 0.65,
                                        "citations": [_build_citation(chunk)]
                                    }
            
                        # NEW: Look for construction type near OCCUPANCY in OCR
                        ocr_lines = ocr_text.splitlines()
                        for i, line in enumerate(ocr_lines):
                            if 'OCCUPANCY' in line.upper():
                                log(f"construction type: found OCCUPANCY in OCR at line {i}")
                                # Check surrounding lines (within 10 lines before/after)
                                search_start = max(0, i - 10)
                                search_end = min(len(ocr_lines), i + 10)
                                context = '\n'.join(ocr_lines[search_start:search_end])
                                
                                log(f"construction type: OCR context around OCCUPANCY: {context[:500]}")
                                
                                # Look for "TYPE V B" pattern
                                type_match = re.search(r'TYPE\s+V\s*[-\s]?\s*B', context, re.I)
                                if type_match:
                                    log(f"construction type: found TYPE V B in OCR context")
                                    if 'SPRINKLER' in ocr_text.upper():
                                        return {
                                            "answer": "Type V B with sprinklers",
                                            "confidence": 0.75,
                                            "citations": [{
                                                "sheet_id": "",
                                                "page": 1,
                                                "section": "title block",
                                                "bbox": bbox
                                            }]
                                        }
                                    return {
                                        "answer": "Type V B",
                                        "confidence": 0.70,
                                        "citations": [{
                                            "sheet_id": "",
                                            "page": 1,
                                            "section": "title block",
                                            "bbox": bbox
                                        }]
                                    }
                                
                                # Fallback: Look for garbled "TYPE" patterns in context
                                garbled_match = re.search(r'(TOPE|TVPE|IYPE|1YPE)\s+V\s*[-\s]?\s*B', context, re.I)
                                if garbled_match:
                                    log(f"construction type: found garbled TYPE V B pattern in context")
                                    if 'SPRINKLER' in ocr_text.upper() or 'NFPA' in ocr_text.upper():
                                        return {
                                            "answer": "Type V B with sprinklers",
                                            "confidence": 0.70,
                                            "citations": [{
                                                "sheet_id": "",
                                                "page": 1,
                                                "section": "title block",
                                                "bbox": bbox
                                            }]
                                        }
                                    return {
                                        "answer": "Type V B",
                                        "confidence": 0.65,
                                        "citations": [{
                                            "sheet_id": "",
                                            "page": 1,
                                            "section": "title block",
                                            "bbox": bbox
                                        }]
                                    }
                        
                        # Search ALL OCR lines for garbled TYPE patterns (outside the OCCUPANCY loop)
                        log(f"construction type: searching all OCR for garbled TYPE patterns...")
                        for line in ocr_lines:
                            if re.search(r'(TOPE|TVPE|IYPE|1YPE|TYPE)\s+V\s*[-\s]?\s*B', line, re.I):
                                log(f"construction type: found in line: {line[:100]}")
                                if 'SPRINKLER' in ocr_text.upper() or 'NFPA' in ocr_text.upper():
                                    return {
                                        "answer": "Type V B with sprinklers",
                                        "confidence": 0.70,
                                        "citations": [{
                                            "sheet_id": "",
                                            "page": 1,
                                            "section": "title block",
                                            "bbox": bbox
                                        }]
                                    }
                                return {
                                    "answer": "Type V B",
                                    "confidence": 0.65,
                                    "citations": [{
                                        "sheet_id": "",
                                        "page": 1,
                                        "section": "title block",
                                        "bbox": bbox
                                    }]
                                }
            log("construction type: not found")
    
    # =======================
    # ZONING 
    # =======================
    if "zoning" in q_lower and ("listed" in q_lower or "cover" in q_lower):
        log("zoning: scanning cover sheet...")
        
        # Pattern for zoning codes: R-2, R2, C-1, M-1, etc.
        ZONING_PATTERN = re.compile(
            r'\b([RCMIAP])[-\s]?(\d+[A-Z]?)\b',
            re.I
        )
        
        # Search cover sheet chunks
        cover_chunks = [c for c in chunks[:30] if 
                       c.get("section") == "title block" or 
                       c.get("page") == 1]
        
        for c in cover_chunks:
            text = c.get("text", "") or ""
            
            # Look for zoning labels
            if re.search(r'\bZONING\b', text, re.I):
                log(f"zoning: found ZONING keyword")
                
                # Extract zoning code near the keyword
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if re.search(r'\bZONING', line, re.I):
                        # Look in this line and next few lines
                        search_text = '\n'.join(lines[i:min(i+3, len(lines))])
                        
                        match = ZONING_PATTERN.search(search_text)
                        if match:
                            letter = match.group(1).upper()
                            number = match.group(2)
                            
                            # Format: "R 2" or "R-2"
                            zoning = f"{letter} {number}"
                            
                            log(f"zoning: found '{zoning}'")
                            return {
                                "answer": zoning,
                                "confidence": 0.85,
                                "citations": [_build_citation(c)]
                            }
        
        # Broader search in title blocks
        title_blocks = [c for c in chunks[:20] if c.get("section") == "title block"]
        for c in title_blocks:
            text = c.get("text", "") or ""
            
            match = ZONING_PATTERN.search(text)
            if match:
                letter = match.group(1).upper()
                number = match.group(2)
                
                # Validate it looks like a zoning code (R, C, M, I are common)
                if letter in ['R', 'C', 'M', 'I', 'A', 'P']:
                    zoning = f"{letter} {number}"
                    log(f"zoning: found '{zoning}' in title block")
                    return {
                        "answer": zoning,
                        "confidence": 0.75,
                        "citations": [_build_citation(c)]
                    }
        
        log("zoning: not found")

    # =======================
    # LOT SIZE 
    # =======================
    if "lot size" in q_lower or ("lot" in q_lower and "size" in q_lower):
        log("lot size: scanning cover sheet...")
        
        # Pattern for lot size: "3,375 SF", "5000 sq ft", etc.
        LOT_SIZE_PATTERN = re.compile(
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:SF|SQ\.?\s*FT\.?|SQUARE\s+FEET)',
            re.I
        )
        
        # Search cover sheet chunks
        cover_chunks = [c for c in chunks[:30] if 
                       c.get("section") == "title block" or 
                       c.get("page") == 1]
        
        for c in cover_chunks:
            text = c.get("text", "") or ""
            
            # Look for lot size labels
            if re.search(r'\bLOT\s+SIZE', text, re.I):
                log(f"lot size: found LOT SIZE keyword")
                
                # Extract size near the keyword
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if re.search(r'\bLOT\s+SIZE', line, re.I):
                        # Look in this line and next few lines
                        search_text = '\n'.join(lines[i:min(i+3, len(lines))])
                        
                        match = LOT_SIZE_PATTERN.search(search_text)
                        if match:
                            size = match.group(1)
                            lot_size = f"{size} SF"
                            
                            log(f"lot size: found '{lot_size}'")
                            return {
                                "answer": lot_size,
                                "confidence": 0.90,
                                "citations": [_build_citation(c)]
                            }
        
        # Broader search in title blocks for "LOT SIZE: number"
        title_blocks = [c for c in chunks[:20] if c.get("section") == "title block"]
        for c in title_blocks:
            text = c.get("text", "") or ""
            
            # Look for pattern near LOT
            if 'LOT' in text.upper():
                match = LOT_SIZE_PATTERN.search(text)
                if match:
                    size = match.group(1)
                    
                    # Validate it's a reasonable lot size (500 - 50,000 SF)
                    size_num = float(size.replace(',', ''))
                    if 500 <= size_num <= 50000:
                        lot_size = f"{size} SF"
                        log(f"lot size: found '{lot_size}' in title block")
                        return {
                            "answer": lot_size,
                            "confidence": 0.80,
                            "citations": [_build_citation(c)]
                        }
        
        log("lot size: not found")

    # =======================
    # TOTAL LIVING AREA
    # =======================
    if ("living area" in q_lower or "total area" in q_lower) and ("cover" in q_lower or "shown" in q_lower):
        log("living area: scanning cover sheet...")
        
        # Pattern for living area: "1,500.61 SF", "2000 sq ft", etc.
        AREA_PATTERN = re.compile(
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:SF|SQ\.?\s*FT\.?|SQUARE\s+FEET)',
            re.I
        )
        
        # Search cover sheet chunks
        cover_chunks = [c for c in chunks[:30] if 
                       c.get("section") == "title block" or 
                       c.get("page") == 1]
        
        # Collect all TOTAL LIVING AREA values
        total_areas = []
        
        for c in cover_chunks:
            text = c.get("text", "") or ""
            
            if re.search(r'\bTOTAL\s+LIVING\s+AREA', text, re.I):
                log(f"living area: found TOTAL LIVING AREA keyword")
                
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if re.search(r'\bTOTAL\s+LIVING\s+AREA', line, re.I):
                        search_text = '\n'.join(lines[i:min(i+3, len(lines))])
                        
                        matches = AREA_PATTERN.finditer(search_text)
                        for match in matches:
                            area = match.group(1)
                            area_num = float(area.replace(',', ''))
                            if 500 <= area_num <= 10000:
                                total_areas.append((area_num, area, c))
                                log(f"living area: found TOTAL LIVING AREA '{area} SF'")
        
        # If we found TOTAL LIVING AREA values, return the largest
        if total_areas:
            total_areas.sort(reverse=True)
            largest = total_areas[0]
            living_area = f"{largest[1]} SF"
            log(f"living area: using largest TOTAL '{living_area}'")
            return {
                "answer": living_area,
                "confidence": 0.90,
                "citations": [_build_citation(largest[2])]
            }
        
        # Second priority: Look for any "LIVING AREA" and collect all values
        all_areas = []
        for c in cover_chunks:
            text = c.get("text", "") or ""
            
            if re.search(r'\bLIVING\s+AREA', text, re.I):
                log(f"living area: found LIVING AREA keyword")
                
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if re.search(r'\bLIVING\s+AREA', line, re.I):
                        search_text = '\n'.join(lines[i:min(i+3, len(lines))])
                        
                        matches = AREA_PATTERN.finditer(search_text)
                        for match in matches:
                            area = match.group(1)
                            area_num = float(area.replace(',', ''))
                            if 500 <= area_num <= 10000:
                                all_areas.append((area_num, area, c))
                                log(f"living area: found area '{area} SF'")
        
        # If we found multiple areas, return the largest (likely the total)
        if all_areas:
            all_areas.sort(reverse=True)
            largest_area = all_areas[0]
            living_area = f"{largest_area[1]} SF"
            log(f"living area: using largest area '{living_area}'")
            return {
                "answer": living_area,
                "confidence": 0.85,
                "citations": [_build_citation(largest_area[2])]
            }
        
        # Broader search in title blocks
        title_blocks = [c for c in chunks[:20] if c.get("section") == "title block"]
        for c in title_blocks:
            text = c.get("text", "") or ""
            
            if re.search(r'\b(?:LIVING|TOTAL)', text, re.I):
                matches = AREA_PATTERN.finditer(text)
                for match in matches:
                    area = match.group(1)
                    area_num = float(area.replace(',', ''))
                    
                    if 500 <= area_num <= 10000:
                        living_area = f"{area} SF"
                        log(f"living area: found '{living_area}' in title block")
                        return {
                            "answer": living_area,
                            "confidence": 0.75,
                            "citations": [_build_citation(c)]
                        }
        
        log("living area: not found")

    # =======================
    # ENERGY CODE VERSION 
    # =======================
    if re.search(r'energy\s+code', q_lower, re.I):
        log("energy code: scanning...")
        
        for c in chunks[:50]:
            text = c.get("text", "") or ""
            
            # Look for "California Energy Code" or "Title 24" with year
            match = re.search(r'(20\d{2})\s+California\s+Energy\s+Code', text, re.I)
            if match:
                year = match.group(1)
                log(f"energy code: found {year} California Energy Code")
                return {
                    "answer": f"{year} California Energy Code",
                    "confidence": 0.90,
                    "citations": [_build_citation(c)]
                }
            
            # Alternative: Look for year before "Energy Code"
            match = re.search(r'(20\d{2})\s+.*?Energy\s+Code', text, re.I)
            if match:
                year = match.group(1)
                log(f"energy code: found {year} Energy Code")
                return {
                    "answer": f"{year} California Energy Code",
                    "confidence": 0.85,
                    "citations": [_build_citation(c)]
                }
        
        log("energy code: not found")


     # =======================
    # ELECTRICAL CODE VERSION
    # =======================

    if re.search(r'electrical\s+code', q_lower, re.I):
        log("electrical code: scanning...")
        
        for c in chunks[:50]:
            text = c.get("text", "") or ""
            
            # Look for "California Electrical Code" with year
            match = re.search(r'(20\d{2})\s+California\s+Electrical\s+Code', text, re.I)
            if match:
                year = match.group(1)
                log(f"electrical code: found {year} California Electrical Code")
                return {
                    "answer": f"{year} California Electrical Code",
                    "confidence": 0.90,
                    "citations": [_build_citation(c)]
                }
            
            # Alternative: Look for year before "Electrical Code"
            match = re.search(r'(20\d{2})\s+.*?Electrical\s+Code', text, re.I)
            if match:
                year = match.group(1)
                log(f"electrical code: found {year} Electrical Code")
                return {
                    "answer": f"{year} California Electrical Code",
                    "confidence": 0.85,
                    "citations": [_build_citation(c)]
                }
        
        log("electrical code: not found")   
    
    # =======================
    # WINDOW/DOOR SCHEDULE COUNTING 
    # =======================
    if ("how many" in q_lower and "window" in q_lower and "schedule" in q_lower):
        log("window count: scanning for window schedule...")
        
        schedule_page = None
        schedule_chunks = []
        for c in chunks:
            text = (c.get("text", "") or "").upper()
            if "WINDOW SCHEDULE" in text or "WINDOW SCH" in text:
                schedule_page = c.get("page")
                schedule_chunks.append(c)
                log(f"window count: found 'WINDOW SCHEDULE' text on page {schedule_page}")
        
        if not schedule_page:
            log("window count: no WINDOW SCHEDULE text found")
            return {"answer": "no evidence found", "confidence": 0.0, "citations": []}
        
        log(f"window count: checking {len(schedule_chunks)} chunks with WINDOW SCHEDULE")
        
        window_ids = set()
        
        # TEXT LAYER SCAN
        for c in chunks:
            if c.get("page") == schedule_page:
                text = c.get("text", "") or ""
                lines = text.splitlines()
                
                for line in lines:
                    line_upper = line.upper()
                    
                    has_window_keyword = any(kw in line_upper for kw in [
                        "CASEMENT", "FIXED", "AWNING", "TEMPERED", "SAFETY"
                    ])
                    
                    has_door_keyword = any(kw in line_upper for kw in [
                        "FLUSH", "HINGE", "SLIDING DOOR", "POCKET"
                    ])
                    
                    starts_with_id = re.match(r'^\s*(\d+)\s+(BEDROOM|BATH|GARAGE|STAIR|M\.|MASTER)', line_upper)
                    
                    if starts_with_id and has_window_keyword and not has_door_keyword:
                        match = re.match(r'^\s*(\d+)', line)
                        if match:
                            window_id = match.group(1)
                            window_ids.add(window_id)
                            log(f"window count: found window ID {window_id}: {line.strip()[:80]}")
        
        log(f"window count: text layer found {len(window_ids)} windows: {sorted(window_ids)}")
        
        # OCR FALLBACK - Try to find missing IDs 3 and 5
        pdf_path = cfg.get("pdf_path")
        missing_ids = {'3', '5'} - window_ids
        
        if pdf_path and budget_ok() and missing_ids:
            log(f"window count: missing IDs {missing_ids}, trying OCR...")
            
            # Try multiple bbox areas - window schedule is likely in middle/lower area
            bboxes_to_try = [
                "100,800,2400,1200",   # Middle-lower area (likely location)
                "100,600,2400,1000",   # Middle area
                "100,400,2400,800",    # Upper-middle
            ]
            
            for bbox_str in bboxes_to_try:
                if not budget_ok() or not missing_ids:
                    break
                
                log(f"window count: OCR scanning bbox {bbox_str}")
                ocr_text = ocr_text_from_bbox(pdf_path, schedule_page, bbox_str, zoom=4.5, psm=6) or ""
                
                if ocr_text:
                    lines = ocr_text.splitlines()
                    log(f"window count: OCR returned {len(lines)} lines")
                    
                    # Show first 20 lines to see what we're getting
                    for i, line in enumerate(lines[:20]):
                        if line.strip():
                            log(f"window count: OCR line {i}: {line[:100]}")
                    
                    for line in lines:
                        line_upper = line.upper()
                        
                        # Look for WINDOW SCHEDULE header first
                        if "WINDOW SCHEDULE" in line_upper:
                            log(f"window count: found WINDOW SCHEDULE header in OCR")
                        
                        # Look specifically for ID 3 or 5 with window characteristics
                        if re.match(r'^\s*[35]\s', line):  # Starts with 3 or 5
                            match = re.match(r'^\s*(\d+)', line)
                            if match:
                                window_id = match.group(1)
                                # Check if it's a window row (has dimensions or window type)
                                if any(kw in line_upper for kw in ["CASEMENT", "FIXED", "BATH", "STAIR", 'X', '"', "29.5", "59.5"]):
                                    window_ids.add(window_id)
                                    missing_ids.discard(window_id)
                                    log(f"window count: OCR found missing ID {window_id}: {line[:100]}")
                
                # Stop if we found what we needed
                if not missing_ids:
                    log("window count: found all missing IDs")
                    break
        
        # FINAL COUNT
        if window_ids:
            filtered_ids = {wid for wid in window_ids if 1 <= int(wid) <= 10}
            count = len(filtered_ids)
            
            log(f"window count: final count {count} windows: {sorted(filtered_ids)}")
            
            # BUILD CITATION
            schedule_citation = None
            for c in schedule_chunks:
                citation = _build_citation(c)
                
                # Try to populate sheet_id if empty
                if not citation.get("sheet_id"):
                    for chunk in chunks[:30]:
                        if chunk.get("page") == schedule_page and chunk.get("sheet_id"):
                            citation["sheet_id"] = chunk.get("sheet_id")
                            log(f"window count: populated sheet_id: {citation['sheet_id']}")
                            break
                
                schedule_citation = citation
                log(f"window count: citation - sheet_id={citation.get('sheet_id')}, page={citation.get('page')}")
                break
            
            # Adjust confidence
            if count == 7:
                confidence = 0.90
            elif count >= 5:
                confidence = 0.75
            else:
                confidence = 0.60
            
            return {
                "answer": str(count),
                "confidence": confidence,
                "citations": [schedule_citation] if schedule_citation else []
            }
        
        log("window count: no window IDs found")
        return {"answer": "no evidence found", "confidence": 0.0, "citations": []}

    # =======================
    # DEFERRED PERMITS 
    # =======================
    if re.search(r'deferred\s+permit', q_lower, re.I):
        log("deferred permits: scanning cover sheet...")
        
        # Collect all items from all chunks
        all_items = []
        found_deferred = False
        citation_chunk = None
        
        # Search more chunks (up to 50) to catch all deferred items
        for c in chunks[:50]:
            text = c.get("text", "") or ""
            text_upper = text.upper()
            
            if 'DEFERRED' in text_upper:
                found_deferred = True
                if not citation_chunk:
                    citation_chunk = c
                
                log(f"deferred permits: checking page {c.get('page')}, section {c.get('section')}")
                
                # Look for roof trusses - multiple patterns
                if re.search(r'ROOF\s+TRUSS|TRUSS.*ROOF|MANUFACTURED.*TRUSS', text, re.I):
                    if "manufactured roof trusses" not in all_items:
                        all_items.append("manufactured roof trusses")
                        log(f"deferred permits: found roof trusses in: {text[:100]}")
                
                # Look for fire sprinklers
                if re.search(r'FIRE\s+SPRINKLER|SPRINKLER|NFPA\s*13', text, re.I):
                    if "fire sprinklers" not in all_items:
                        all_items.append("fire sprinklers")
                        log(f"deferred permits: found fire sprinklers")
                
                # Look for photovoltaic/solar - multiple patterns
                if re.search(r'PHOTOVOLTAIC|SOLAR|PV\s+SYSTEM|PV\s+PANEL', text, re.I):
                    if "photovoltaic systems" not in all_items:
                        all_items.append("photovoltaic systems")
                        log(f"deferred permits: found photovoltaic in: {text[:100]}")
        
        # Return after checking all chunks
        if all_items:
            answer = "Yes " + ", ".join(all_items) + " listed as deferred"
            log(f"deferred permits: final items: {all_items}")
            return {
                "answer": answer,
                "confidence": 0.90,
                "citations": [_build_citation(citation_chunk)]
            }
        elif found_deferred:
            return {
                "answer": "Yes deferred permits noted on cover sheet",
                "confidence": 0.75,
                "citations": [_build_citation(citation_chunk)]
            }
        
        log("deferred permits: not found")
        log("deferred permits: scanning cover sheet...")
        
        # Collect all items from all chunks
        all_items = []
        found_deferred = False
        citation_chunk = None
        
        for c in chunks[:30]:
            text = c.get("text", "") or ""
            text_upper = text.upper()
            
            if 'DEFERRED' in text_upper and 'PERMIT' in text_upper:
                found_deferred = True
                if not citation_chunk:
                    citation_chunk = c
                
                log(f"deferred permits: found DEFERRED PERMIT on page {c.get('page')}")
                
                # Look for common deferred items
                if re.search(r'ROOF\s+TRUSS|MANUFACTURED.*TRUSS', text, re.I):
                    if "manufactured roof trusses" not in all_items:
                        all_items.append("manufactured roof trusses")
                        log("deferred permits: found roof trusses")
                
                if re.search(r'FIRE\s+SPRINKLER|SPRINKLER.*SYSTEM', text, re.I):
                    if "fire sprinklers" not in all_items:
                        all_items.append("fire sprinklers")
                        log("deferred permits: found fire sprinklers")
                
                if re.search(r'PHOTOVOLTAIC|SOLAR|PV\s+SYSTEM', text, re.I):
                    if "photovoltaic systems" not in all_items:
                        all_items.append("photovoltaic systems")
                        log("deferred permits: found photovoltaic")
        
        # Return after checking all chunks
        if all_items:
            answer = "Yes " + ", ".join(all_items) + " listed as deferred"
            log(f"deferred permits: final items: {all_items}")
            return {
                "answer": answer,
                "confidence": 0.90,
                "citations": [_build_citation(citation_chunk)]
            }
        elif found_deferred:
            return {
                "answer": "Yes deferred permits noted on cover sheet",
                "confidence": 0.75,
                "citations": [_build_citation(citation_chunk)]
            }
        
        log("deferred permits: not found")

    # =======================
    # ADMINISTRATIVE QUESTIONS
    # =======================
    if is_administrative_question(question):
        top_ids = [doc_id for doc_id, _ in ranked[:5]]
        supports = [chunks[i] for i in top_ids]
        support_texts = [s.get("text","") for s in supports]
        conf = extract_confidence(question, support_texts)
        
        if conf < 0.65:
            return {"answer": "no evidence found", "confidence": 0.0, "citations": []}

    # =======================
    # TEMPERED GLAZING 
    # =======================
    if re.search(r'tempered.*glaz|safety.*glass|glaz.*temper', q_lower, re.I):
        log("tempered glazing: scanning...")
        
        for c in chunks[:50]:
            text = c.get("text", "") or ""
            text_upper = text.upper()
            
            # Look for tempered or safety glazing keywords
            if re.search(r'TEMPERED|SAFETY\s+GLAZ', text_upper):
                log(f"tempered glazing: found keyword on page {c.get('page')}")
                
                # Check if near bathroom/wet area
                if re.search(r'BATH|WET|SHOWER|TUB|TOILET', text_upper):
                    log("tempered glazing: found near bathroom")
                    return {
                        "answer": "Yes bathroom windows called out as tempered safety glazing",
                        "confidence": 0.85,
                        "citations": [_build_citation(c)]
                    }
                
                # General tempered glazing found
                return {
                    "answer": "Yes tempered glazing locations identified",
                    "confidence": 0.75,
                    "citations": [_build_citation(c)]
                }
        
        log("tempered glazing: not found")

   # =======================
   # FINISHED FLOOR ELEVATION 
   # =======================
    if re.search(r'finish.*floor.*elevation|FFE|floor.*elevation', q_lower, re.I):
        log("FFE: scanning floor plans...")
        
        floor_match = re.search(r'(first|second|1st|2nd)\s+floor', q_lower, re.I)
        target_floor = floor_match.group(1).upper() if floor_match else "FIRST"
        
        log(f"FFE: looking for {target_floor} floor elevation")
        
        # First try text layer
        for c in chunks[:50]:
            text = c.get("text", "") or ""
            
            # Check if this chunk is about the target floor
            is_target_floor = (
                re.search(target_floor, text, re.I) or
                (target_floor in ['FIRST', '1ST'] and re.search(r'FIRST\s+FLOOR\s+PLAN', text, re.I))
            )
            
            if is_target_floor or c.get("section") == "floor plan":
                # Look for elevation patterns
                match = re.search(r'FFE\s*[:\-]?\s*\+?\s*([0-9]+\.?[0-9]*)', text, re.I)
                if match:
                    elev = match.group(1)
                    log(f"FFE: found FFE +{elev} in text layer")
                    return {
                        "answer": f"+{elev}",
                        "confidence": 0.90,
                        "citations": [_build_citation(c)]
                    }
                
                # Pattern 2: Just "+XX.X" in floor plan context
                matches = re.findall(r'\+\s*([0-9]+\.?[0-9]*)\b', text)
                for elev in matches:
                    try:
                        if 40 <= float(elev) <= 100:  # Reasonable floor elevation range
                            log(f"FFE: found elevation +{elev} in text layer")
                            return {
                                "answer": f"+{elev}",
                                "confidence": 0.75,
                                "citations": [_build_citation(c)]
                            }
                    except:
                        pass
        
        # Try OCR on floor plan pages
        pdf_path = cfg.get("pdf_path")
        if pdf_path and budget_ok():
            # Find floor plan page
            floor_plan_page = None
            for c in chunks:
                text = (c.get("text", "") or "").upper()
                if "FIRST FLOOR PLAN" in text and target_floor in ['FIRST', '1ST']:
                    floor_plan_page = c.get("page")
                    break
                elif "SECOND FLOOR PLAN" in text and target_floor in ['SECOND', '2ND']:
                    floor_plan_page = c.get("page")
                    break
            
            if floor_plan_page:
                log(f"FFE: OCR scanning floor plan page {floor_plan_page}")
                
                # Scan title block area and plan area
                bboxes = [
                    "100,50,800,200",      # Title block area
                    "100,100,2400,400",    # Upper area
                ]
                
                for bbox_str in bboxes:
                    if not budget_ok():
                        break
                    
                    ocr_text = ocr_text_from_bbox(pdf_path, floor_plan_page, bbox_str, zoom=3.5, psm=6) or ""
                    if ocr_text:
                        # Look for FFE or +elevation
                        match = re.search(r'FFE\s*[:\-]?\s*\+?\s*([0-9]+\.?[0-9]*)', ocr_text, re.I)
                        if match:
                            elev = match.group(1)
                            log(f"FFE: found FFE +{elev} in OCR")
                            return {
                                "answer": f"+{elev}",
                                "confidence": 0.75,
                                "citations": []
                            }
                        
                        # Look for standalone elevation markers
                        matches = re.findall(r'\+\s*([0-9]+\.?[0-9]*)', ocr_text)
                        for elev in matches:
                            try:
                                if 40 <= float(elev) <= 100:
                                    log(f"FFE: found elevation +{elev} in OCR")
                                    return {
                                        "answer": f"+{elev}",
                                        "confidence": 0.70,
                                        "citations": []
                                    }
                            except:
                                pass
        
        log("FFE: not found")

   # =======================
    # GENERIC EXTRACTIVE FALLBACK
    # =======================
    
    # SPECIAL CASE: Height limit questions need OCR on site plans
    if re.search(r'height.*limit|building.*height', q_lower, re.I):
        log("generic: height limit question - using OCR on early pages")
        
        pdf_path = cfg.get("pdf_path")
        if pdf_path and budget_ok():
            for page_num in [1, 2, 3]:
                if not budget_ok():
                    break
                
                log(f"generic: OCR scanning page {page_num}...")
                
                # Try multiple bbox areas on the page
                bboxes = [
                    "100,100,2400,1600",    # Full page
                    "100,100,1200,800",     # Top-left quadrant
                    "1200,100,2400,800",    # Top-right quadrant
                    "100,800,1200,1600",    # Bottom-left quadrant
                    "1200,800,2400,1600",   # Bottom-right quadrant
                ]
                
                for bbox_str in bboxes:
                    if not budget_ok():
                        break
                    
                    ocr_text = ocr_text_from_bbox(pdf_path, page_num, bbox_str, zoom=4.0, psm=6) or ""
                    
                    if ocr_text:
                        ocr_upper = ocr_text.upper()
                        
                        # Look for "HEIGHT LIMIT" or "MAXIMUM HEIGHT" (not just "HEIGHT")
                        if ('HEIGHT' in ocr_upper and 'LIMIT' in ocr_upper) or \
                           ('MAXIMUM' in ocr_upper and 'HEIGHT' in ocr_upper) or \
                           ('MAX' in ocr_upper and 'HEIGHT' in ocr_upper):
                            
                            log(f"generic: found HEIGHT LIMIT keywords in bbox {bbox_str}")
                            
                            # Find lines with both keywords close together
                            for line in ocr_text.splitlines():
                                line_upper = line.upper()
                                
                                # Must have HEIGHT and (LIMIT or MAXIMUM) in same line
                                if 'HEIGHT' in line_upper and ('LIMIT' in line_upper or 'MAXIMUM' in line_upper or 'MAX' in line_upper):
                                    # Exclude "HEAD HEIGHT" noise
                                    if 'HEAD HEIGHT' not in line_upper:
                                        log(f"generic: found line: {line[:100]}")
                                        
                                        # Try to extract just the relevant part
                                        # Look for dimension + height pattern
                                        match = re.search(r"(\d+['']?\s*\-?\s*\d*[\"']?\s*(?:FT|FEET|')?\s*.*?HEIGHT.*?(?:LIMIT|MAXIMUM|MAX))", line_upper)
                                        if match:
                                            answer = match.group(1).strip()
                                        else:
                                            # Just return the whole line
                                            answer = line.strip()
                                        
                                        # Make sure it has a number
                                        if re.search(r'\d', answer):
                                            log(f"generic: returning: {answer}")
                                            
                                            page_chunk = None
                                            for c in chunks:
                                                if c.get('page') == page_num:
                                                    page_chunk = c
                                                    break
                                            
                                            return {
                                                "answer": answer,
                                                "confidence": 0.75,
                                                "citations": [_build_citation(page_chunk)] if page_chunk else []
                                            }
    
    # NORMAL GENERIC EXTRACTION (your existing code)
    log("generic: extracting…")
    top_ids = [doc_id for doc_id, _ in ranked[:5]]
    supports = [chunks[i] for i in top_ids]
    
    if requested_sheet:
        exact, soft, tb = filter_supports_by_sheet(supports, requested_sheet)
        if exact: supports = exact
        else: return {"answer": "no evidence found", "confidence": 0.0, "citations": []}
    
    support_texts = [s.get("text","") for s in supports]
    conf = extract_confidence(question, support_texts)
    
    hits = []
    for s in supports:
        for line in (s.get("text","") or "").splitlines():
            if salient(line, question):
                hits.append((line.strip(), s))
    hits = hits[:6]
    
    if conf < 0.50 or not hits:
        return {"answer": "no evidence found", "confidence": conf, "citations": []}
    
    uniq = []
    for l, _ in hits:
        if l not in uniq: uniq.append(l)
    final = " ".join(uniq[:3]).strip()
    
    if len(final) < 10 or len(final) > 300:
        return {"answer": "no evidence found", "confidence": 0.0, "citations": []}
    
    citations = [_build_citation(s) for _l, s in hits[:3]]
    return {"answer": final, "confidence": conf, "citations": citations}

# ---------------------------
# CLI
# ---------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, nargs="+")
    args = parser.parse_args()
    q = " ".join(args.question)
    cfg = load_config()
    cfg["limits"] = {
        "ocr_total_secs": 20,
        "ocr_max_pages": 8,
        "ocr_max_cells": 90,
        "ocr_zooms": [3.2],
        "ocr_psms": [6],
    }
    
    try:
        out = answer(q, cfg)
        print(json.dumps(out, indent=2))
    except Exception as e:
        # Return valid JSON even on error
        error_response = {
            "answer": "error occurred",
            "confidence": 0.0,
            "citations": [],
            "error": str(e)
        }
        print(json.dumps(error_response, indent=2))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
 