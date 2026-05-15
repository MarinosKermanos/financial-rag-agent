import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from ingestion.chunker import Chunk
from config import Config
import uuid

openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
qdrant_client = QdrantClient(
    host=Config.QDRANT_HOST,
    port=Config.QDRANT_PORT,
)


def ensure_collection():
    """Create the Qdrant collection if it doesn't exist."""
    existing = [c.name for c in qdrant_client.get_collections().collections]

    if Config.COLLECTION_NAME not in existing:
        qdrant_client.create_collection(
            collection_name=Config.COLLECTION_NAME,
            vectors_config=VectorParams(size=Config.VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection: {Config.COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {Config.COLLECTION_NAME}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using OpenAI.
    
    Batching is important — sending 1 request per chunk would be slow and expensive.
    OpenAI allows up to 2048 inputs per request for embedding models.
    """
    response = openai_client.embeddings.create(
        model=Config.EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def store_chunks(chunks: list[Chunk], batch_size: int = 50):
    """
    Embed and store chunks in Qdrant in batches.
    
    Why batch? Embedding 200 chunks in 4 requests is much faster
    than 200 individual API calls.
    """
    ensure_collection()
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk.text for chunk in batch]

        print(f"Embedding batch {i // batch_size + 1} ({len(batch)} chunks)...")
        embeddings = embed_texts(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={**chunk.metadata, "text": chunk.text},
            )
            for chunk, embedding in zip(batch, embeddings)
        ]

        qdrant_client.upsert(
            collection_name=Config.COLLECTION_NAME,
            points=points,
        )

    print(f"Stored {total} chunks in Qdrant.")


if __name__ == "__main__":
    from ingestion.fetcher import fetch_all_articles
    from ingestion.chunker import chunk_articles

    articles = fetch_all_articles()
    chunks = chunk_articles(articles)
    store_chunks(chunks)