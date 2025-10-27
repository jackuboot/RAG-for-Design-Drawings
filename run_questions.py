#!/usr/bin/env python3
import subprocess, json, csv, sys, shlex, os, time
from pathlib import Path

# Config
QA_PY = os.environ.get("QA_SCRIPT_PATH", "qa_clean.py")  # allow override via env
QUESTIONS_FILE = os.environ.get("QUESTIONS_FILE", "questions.json")
OUT_JSON = os.environ.get("OUT_JSON", "qa_results.json")
OUT_CSV = os.environ.get("OUT_CSV", "qa_results.csv")

def run_qa(question: str):
    """Run qa.py with the given question and try to parse JSON from stdout."""
    cmd = f'python {shlex.quote(QA_PY)} {shlex.quote(question)}'
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"question": question, "error": "timeout", "raw": ""}
    raw = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # Try to extract JSON blob from the raw output
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start:end+1])
            return {"question": question, **data, "raw": raw}
        except Exception as e:
            return {"question": question, "error": f"json_parse_error: {e}", "raw": raw}
    return {"question": question, "error": "no_json_found", "raw": raw}

def main():
    # Load questions
    with open(QUESTIONS_FILE, "r") as f:
        questions = json.load(f)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q}", flush=True)
        res = run_qa(q)
        results.append(res)
        time.sleep(0.2)  # small pacing

    # Save JSON
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    fieldnames = ["question", "answer", "confidence", "citations", "error"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "question": r.get("question"),
                "answer": r.get("answer"),
                "confidence": r.get("confidence"),
                "citations": json.dumps(r.get("citations", [])),
                "error": r.get("error")
            }
            writer.writerow(row)

    print(f"Saved {len(results)} results to {OUT_JSON} and {OUT_CSV}")

if __name__ == "__main__":
    main()
