import requests
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.tools import tool
from retrieval.retriever import search

# Tool 1: Financial News Search

@tool
def search_financial_news(query: str) -> str:
    """
    Search recent financial news relevant to the query.
    Use this to answer questions about market sentiment, recent events,
    macroeconomic news, or anything requiring up-to-date financial context.
    """
    results = search(query, top_k=4)

    if not results:
        return "No relevant news found for this query."

    # Format results into readable text the LLM can reason over
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[Article {i}]\n"
            f"Title: {r['title']}\n"
            f"Source: {r['source']} | Published: {r['published_at']}\n"
            f"Content: {r['text']}\n"
            f"URL: {r['url']}"
        )

    return "\n\n---\n\n".join(formatted)


# Tool 2: Forex Price Lookup

@tool
def get_forex_price(symbol: str) -> str:
    """
    Get the current exchange rate for a forex pair.
    Input must be a currency pair like 'EURUSD', 'GBPUSD', 'USDJPY'.
    Use this when the user asks about current prices or exchange rates.
    """
    # Normalize input: remove slashes, uppercase
    symbol = symbol.upper().replace("/", "").replace("-", "").strip().strip("'\"")

    if len(symbol) != 6:
        return f"Invalid symbol '{symbol}'. Use format like EURUSD, GBPUSD, USDJPY."

    base = symbol[:3]
    quote = symbol[3:]

    try:
        # exchangerate-api.com free tier, no key needed
        url = f"https://api.exchangerate-api.com/v4/latest/{base}"
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()

        rate = data["rates"].get(quote)
        if rate is None:
            return f"Could not find rate for {symbol}. Try a major pair like EURUSD."

        return (
            f"Current exchange rate: 1 {base} = {rate:.5f} {quote}\n"
            f"Data timestamp: {data.get('date', 'N/A')}"
        )

    except requests.RequestException as e:
        return f"Error fetching price for {symbol}: {str(e)}"


# Export all tools

TOOLS = [search_financial_news, get_forex_price]