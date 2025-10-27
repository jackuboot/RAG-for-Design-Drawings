import subprocess, sys
import typer
from rich import print
from utils import load_config

app = typer.Typer(help="RAG for design drawings CLI")

@app.command()
def ingest(pdf: str = typer.Option(None, help="Path to PDF. Defaults to config.yaml")):
    cfg = load_config()
    if pdf:
        cfg["pdf_path"] = pdf
        print(f"[bold]Using PDF:[/] {pdf}")
    print("[bold]Ingesting PDF...[/]")
    subprocess.check_call([sys.executable, "ingest.py"])

@app.command()
def index():
    print("[bold]Building indexes...[/]")
    subprocess.check_call([sys.executable, "index.py"])

@app.command()
def ask(q: str):
    print(f"[bold]Q:[/] {q}")
    out = subprocess.check_output([sys.executable, "qa.py", q]).decode()
    print(out)

@app.command()
def demo():
    qs = [
        "What is the sheet scale for the floor plan on A1.1?",
        "How many windows are listed in the window schedule?",
        "What is the finish for the living room floor?",
        "What window type is used in Bedroom 2?",
        "List rooms and their ceiling heights on Sheet A2.0.",
        "What is the revision date in the title block on A1.0?"
    ]
    for q in qs:
        print("=" * 60)
        ask(q)

if __name__ == "__main__":
    app()
