# Financial RAG Agent 📈

An agentic AI system that answers financial questions by combining semantic search over recent news with live forex data — built as a portfolio project targeting production-grade AI engineering standards.


**Live UI:** `https://streamlit-ui-production-578d.up.railway.app/` 
**Live API:** `https://financial-rag-agent-production.up.railway.app/docs`

---

## What It Does

Ask questions like:
- *"What is the current EUR/USD rate and what news is affecting it?"*
- *"Summarize the latest Federal Reserve news"*
- *"What's happening with oil prices?"*

The agent reasons through which tools to use, fetches live data, searches recent news, and synthesizes a cited answer — all traced end-to-end in LangFuse.

---

## Architecture

```
User Question
      │
      ▼
 FastAPI /chat
      │
      ▼
 ReAct Agent (GPT-4o-mini)
      │
      ├──► Tool 1: search_financial_news ──► Qdrant (vector DB)
      │
      └──► Tool 2: get_forex_price ──────► exchangerate-api.com
      │
      ▼
 Final Answer
      │
      ▼
 LangFuse (traces every step)
```

The agent uses the **ReAct (Reason + Act)** pattern — it decides which tools to call and in what order, rather than blindly retrieving then answering like a plain RAG chain.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o-mini (OpenAI) |
| Agent framework | LangChain ReAct |
| Vector database | Qdrant |
| Embeddings | text-embedding-3-small |
| Observability | LangFuse |
| API | FastAPI |
| UI | Streamlit |
| Deployment | Railway + Docker |

---

## Project Structure

```
financial-rag-agent/
├── ingestion/
│   ├── fetcher.py       # NewsAPI ingestion with deduplication
│   ├── chunker.py       # Recursive text splitting (400 token chunks)
│   └── embedder.py      # Batch embedding + Qdrant upsert
├── retrieval/
│   └── retriever.py     # Semantic search (cosine similarity)
├── agent/
│   ├── tools.py         # LangChain tools: news search + forex price
│   ├── agent.py         # ReAct agent with LangFuse tracing
│   └── tracing.py       # LangFuse callback handler
├── api/
│   └── main.py          # FastAPI app with /chat endpoint
├── tests/
│   ├── test_tools.py
│   └── test_retriever.py
├── streamlit_app.py     # Chat UI
├── Dockerfile
├── docker-compose.yml
└── railway.toml
```

---

## Design Decisions

**Why ReAct over a plain RAG chain?**
A plain RAG chain always retrieves then answers. ReAct lets the agent decide *which* tools to use and *in what order* — so for "how is EUR/USD doing?" it fetches the live price AND searches news in one reasoning chain, then synthesizes both into a single answer.

**Why Qdrant over ChromaDB?**
Qdrant is production-grade, has a proper dashboard for inspecting vectors, supports metadata filtering (useful for filtering news by date or source), and is what teams actually run in production.

**Why batch embeddings?**
Each OpenAI API call has fixed latency overhead. Batching 50 chunks per request instead of 1 reduces ingestion time by ~50x with the same cost.

**Why LangFuse?**
LLM applications fail silently. Without observability you can't tell if the agent picked the wrong tool, retrieval returned irrelevant chunks, or the LLM hallucinated. LangFuse traces every step with token counts, latency, and tool I/O — essential for debugging and iterating in production.

**Why `temperature=0` for the agent LLM?**
Higher temperature causes the agent to hallucinate tool names or produce malformed ReAct output, breaking the parsing loop. Deterministic output is critical for reliable agent behavior.

---

## Local Setup

### Prerequisites
- Python 3.10+
- Docker
- `uv` package manager

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Clone and install

```bash
git clone https://github.com/MarinosKermanos/financial-rag-agent
cd financial-rag-agent
uv sync
source .venv/bin/activate
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your API keys in .env
```

Required keys:
```
OPENAI_API_KEY=sk-proj-...
NEWS_API_KEY=...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Start Qdrant

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### 4. Ingest financial news

```bash
python ingestion/embedder.py
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 6. Start the UI (separate terminal)

```bash
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` for the chat UI or `http://localhost:8000/docs` for the API.

---

## Running with Docker Compose

```bash
docker compose up --build
```

This starts both Qdrant and the FastAPI service together.

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## Observability

All agent traces are logged to LangFuse automatically, including:
- Full ReAct reasoning chain (Thought → Action → Observation loop)
- Tool inputs and outputs
- Token usage and cost per query
- End-to-end latency per request

Access your traces at [cloud.langfuse.com](https://cloud.langfuse.com).

---

## Deployment

Deployed on Railway with three services:
- **financial-rag-agent** — FastAPI app (Dockerfile-based)
- **streamlit-ui** — Streamlit chat UI (Nixpacks)
- **qdrant** — Qdrant vector DB (Docker image), private networking only

Environment variables are configured per-service in Railway. Qdrant is not exposed publicly — only the API service can reach it via `qdrant.railway.internal`.

After first deployment, trigger news ingestion once by calling `POST /ingest` via the API docs. This populates Qdrant with the financial news corpus.

---

## Known Limitations & Future Improvements

- **NewsAPI free tier** truncates article content to ~200 characters. Full ingestion would require a paid tier or direct RSS scraping.
- **No re-ingestion scheduling** — news is ingested once manually. A production version would run ingestion on a schedule (e.g. daily cron or a queue-based pipeline).
- **No conversation memory** — each question is stateless. A follow-up like "tell me more about that" loses context. Could be added with LangChain's `ConversationBufferMemory`.
- **Single collection** — all topics share one Qdrant collection. Filtering by topic or date range would improve retrieval precision.
