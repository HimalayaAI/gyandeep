from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from core.services.storage.embedding_service import EmbeddingService, index_embeddings

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


async def run(text_path: str | Path, source: str | None = None) -> None:
    text = Path(text_path).read_text(encoding="utf-8")
    chunks = chunk_text(text)

    embedder = EmbeddingService()
    embeddings = await embedder.get_embeddings(chunks)

    print(f"chunks: {len(chunks)}")
    print(f"embedding_dim: {len(embeddings[0]) if embeddings else 0}")

    index_embeddings(chunks, embeddings, source=source or Path(text_path).stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks from a text file.")
    parser.add_argument("text", help="Path to the text file.")
    parser.add_argument("--source", default=None, help="Source label for embeddings.")
    args = parser.parse_args()

    asyncio.run(run(args.text, args.source))


if __name__ == "__main__":
    main()
