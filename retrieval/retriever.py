import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from openai import OpenAI
from config import Config

openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
qdrant_client = QdrantClient(
    host=Config.QDRANT_HOST,
    port=Config.QDRANT_PORT,
)


def embed_query(query: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=Config.EMBEDDING_MODEL,
        input=[query],
    )
    return response.data[0].embedding


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Semantic search: embed the query, find the top_k most similar chunks.
    Returns a list of dicts with text + metadata.
    """
    query_vector = embed_query(query)

    results = qdrant_client.query_points(
    collection_name=Config.COLLECTION_NAME,
    query=query_vector,
    limit=top_k,
    with_payload=True,
).points

    return [
        {
            "text": hit.payload.get("text", ""),
            "score": hit.score,         # cosine similarity, higher = more relevant
            "title": hit.payload.get("title", ""),
            "source": hit.payload.get("source", ""),
            "published_at": hit.payload.get("published_at", ""),
            "url": hit.payload.get("url", ""),
        }
        for hit in results
    ]


if __name__ == "__main__":
    test_queries = [
        "What is happening with the US dollar?",
        "Latest news about oil prices",
        "Federal Reserve rate decision",
    ]

    for query in test_queries:
        print(f"\n=== Query: {query} ===")
        results = search(query, top_k=3)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['title']} ({r['source']})")
            print(f"  {r['text'][:150]}...")