import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Qdrant
    COLLECTION_NAME = "financial_news"
    VECTOR_SIZE = 1536
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-small"

    # NewsAPI
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    NEWS_API_URL = os.getenv("NEWS_API_GET_URL")
