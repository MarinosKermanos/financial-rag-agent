import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.fetcher import Article
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    metadata: dict  # title, url, source, published_at, chunk_index


def chunk_articles(articles: list[Article], chunk_size: int = 400, chunk_overlap: int = 50) -> list[Chunk]:
    """
    Split articles into chunks for embedding.
    
    Why 400 tokens? Large enough to carry context, small enough that
    retrieval stays precise. Overlap avoids cutting sentences mid-thought.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # try to split at natural boundaries
    )

    chunks = []
    for article in articles:
        # Combine title + content so every chunk carries the headline for context
        full_text = f"{article.title}\n\n{article.content}"
        splits = splitter.split_text(full_text)

        for i, split in enumerate(splits):
            chunks.append(Chunk(
                text=split,
                metadata={
                    "title": article.title,
                    "url": article.url,
                    "source": article.source,
                    "published_at": article.published_at,
                    "chunk_index": i,
                }
            ))

    print(f"Total chunks created: {len(chunks)}")
    return chunks


if __name__ == "__main__":
    from ingestion.fetcher import fetch_all_articles
    articles = fetch_all_articles()
    chunks = chunk_articles(articles)
    print(f"\nExample chunk:\n{chunks[0].text}")
    print(f"Metadata: {chunks[0].metadata}")