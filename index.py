import os, json, pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from utils import load_config

def load_chunks(path="./data/chunks.jsonl"):
    chunks = []
    with open(path) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def build_faiss(chunks, store_dir, model_name="all-MiniLM-L6-v2"):
    os.makedirs(store_dir, exist_ok=True)
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    print("[index] Encoding embeddings...")
    mat = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    mat = np.array(mat).astype("float32")
    print("[index] Building FAISS index...")
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, os.path.join(store_dir, "faiss.index"))
    with open(os.path.join(store_dir, "faiss_meta.pkl"), "wb") as f:
        pickle.dump({"model": model_name, "ids": list(range(len(texts)))}, f)
    with open(os.path.join(store_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    return mat.shape

def build_bm25(chunks, store_dir):
    print("[index] Building BM25 index...")
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(os.path.join(store_dir, "bm25.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25}, f)
    return len(tokenized_corpus)

def main():
    cfg = load_config()
    chunks = load_chunks("./data/chunks.jsonl")
    d = cfg["store_dir"]
    shape = build_faiss(chunks, d)
    n = build_bm25(chunks, d)
    print(f"[index] FAISS dim={shape[1]} on {shape[0]} vectors; BM25 on {n} docs.")

if __name__ == "__main__":
    main()
