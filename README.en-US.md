# RAG System for Architectural Drawing Question Answering

A Retrieval-Augmented Generation (RAG) system that answers natural language questions about architectural/construction drawing sets.

## Overview

This system processes large PDF architectural drawings (~60MB, 20-30 pages) and answers questions about floor plans, schedules, title blocks, and specifications. It uses hybrid retrieval (semantic + keyword search) with intelligent OCR fallback for information extraction from both text layers and drawings.

**Key Features:**
- Hybrid retrieval (FAISS vector search + BM25 keyword search)
- Intelligent OCR fallback for graphics-heavy content
- Grounded answers with citations (page, section, bounding box)
- Zero hallucination (returns "no evidence found" when uncertain)
- Fast (~2 seconds per query)
- Runs locally on CPU (no GPU or cloud required)

---

## Quick Start

### Prerequisites
- Python 3.8+
- Tesseract OCR
- 16GB RAM recommended

### Installation
```bash
# Clone the repository
git clone git@github.com:ishamishra0408/RAG-for-Design-Drawings.git
cd rag_drawings

# Install Python dependencies
pip install -r requirements.txt

# Install Tesseract OCR
# macOS:
brew install tesseract
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# Windows:
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Setup & CLI Usage Order

#### 1. Place your PDF file
Put your PDF in the `data/` directory and set the path in `config.yaml`:
```yaml
pdf_path: "./data/your_drawing.pdf"
```

#### 2. Run the following CLI commands in order

**(1) Data preprocessing**
```bash
python app.py ingest
```
> Parse the PDF and generate data chunks (chunks.jsonl)

**(2) Build index**
```bash
python app.py index
```
> Build FAISS/BM25 index from data chunks

**(3) Ask questions**
```bash
python app.py ask "What is the project address?"
python app.py ask "Who is the architect?"
python app.py ask "What is the lot area?"
```
> Supports natural language questions, returns answer, confidence, and citations

**(4) Demo (optional)**
```bash
python app.py demo
```
> Run built-in demo questions

#### Typical output format
```json
{
  "answer": "PROJECTADDRESS: 23KERLEYCOURT",
  "confidence": 0.85,
  "citations": [
    {
      "sheet_id": "S1",
      "page": 1,
      "section": "title block",
      "bbox": "..."
    }
  ]
}
```

---

## Notes
- Supports both English and Chinese natural language questions
- If you get "no evidence found", it means no relevant content was retrieved or confidence is too low
- For detailed architecture, tech stack, and evaluation, see the main README
