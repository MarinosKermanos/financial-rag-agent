import pytest
from unittest.mock import patch, MagicMock


# Forex Tool Tests

def test_get_forex_price_valid_symbol():
    from agent.tools import get_forex_price
    result = get_forex_price.invoke("EURUSD")
    assert "EUR" in result
    assert "USD" in result
    # Should contain a numeric rate
    assert any(char.isdigit() for char in result)


def test_get_forex_price_invalid_symbol():
    from agent.tools import get_forex_price
    result = get_forex_price.invoke("INVALID")
    assert "Invalid" in result or "Could not find" in result


def test_get_forex_price_with_slash():
    """Ensure EUR/USD format is handled same as EURUSD."""
    from agent.tools import get_forex_price
    result = get_forex_price.invoke("EUR/USD")
    assert "EUR" in result


# News Search Tool Tests

def test_search_financial_news_returns_results():
    from agent.tools import search_financial_news
    result = search_financial_news.invoke("Federal Reserve interest rates")
    # Should return something — we ingested this topic on Day 1
    assert len(result) > 50
    assert "No relevant news found" not in result


def test_search_financial_news_empty_corpus_query():
    from agent.tools import search_financial_news
    # Very obscure query — might return no results, should not crash
    result = search_financial_news.invoke("xyzzy quantum banana market")
    assert isinstance(result, str)  # just shouldn't crash


# Retriever Unit Test

def test_retriever_returns_scored_results():
    from retrieval.retriever import search
    results = search("oil price", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    for r in results:
        assert "text" in r
        assert "score" in r
        assert 0.0 <= r["score"] <= 1.0