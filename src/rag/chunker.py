def split_into_chunks(text: str, max_chars: int = 800, overlap: int = 200) -> list[str]:
    """Простой текстовый чанкер по символам (потом можно усложнить)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks