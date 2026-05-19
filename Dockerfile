FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files first (layer caching — faster rebuilds)
COPY pyproject.toml .
COPY uv.lock* .

# Install dependencies
RUN uv sync --frozen

# Copy the rest of the code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the FastAPI server
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]