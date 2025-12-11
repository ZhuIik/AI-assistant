from pathlib import Path

def load_documents(root: str) -> list[dict]:
    """Читает все .txt и возвращает список {id, text, metadata}."""
    root_path = Path(root)
    docs = []
    for path in root_path.rglob("*.txt"):
        text = path.read_text(encoding="utf-8")
        docs.append({
            "id": path.stem,
            "text": text,
            "source": str(path),
        })
    return docs
