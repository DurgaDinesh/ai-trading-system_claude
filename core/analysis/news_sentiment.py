"""
News sentiment analysis using:
  1. RSS feeds (free, no key needed) — Economic Times, Moneycontrol, NDTV Profit
  2. newsapi.org — for broader headline search
  3. Claude AI (Anthropic) — final NLP scoring with financial context
"""

import time
from datetime import datetime, timedelta
from typing import Optional
import feedparser
import requests
import structlog
import yaml
import anthropic

from core.security.vault import vault

logger = structlog.get_logger(__name__)
_cfg = yaml.safe_load(open("config/settings.yaml", encoding="utf-8"))["news"]

_anthropic_client: Optional[anthropic.Anthropic] = None


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=vault.anthropic_api_key)
    return _anthropic_client


# ── RSS Feed Fetcher ───────────────────────────────────────────────────────────

def fetch_rss_headlines(max_age_hours: int = 12) -> list[dict]:
    """Fetch and deduplicate headlines from all configured RSS feeds."""
    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    headlines = []
    seen_titles = set()

    for url in _cfg["sources"]["rss"]:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                title = str(entry.get("title", "")).strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                published_str = entry.get("published", "")
                try:
                    pub_dt = datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") and entry.published_parsed else datetime.utcnow()
                except Exception:
                    pub_dt = datetime.utcnow()
                if pub_dt < cutoff:
                    continue
                headlines.append({
                    "title": title,
                    "summary": str(entry.get("summary", ""))[:300],
                    "published": pub_dt.isoformat(),
                    "source": str(feed.feed.get("title", url)),
                    "link": str(entry.get("link", "")),
                })
        except Exception as e:
            logger.warning("rss_fetch_failed", url=url, error=str(e))

    return headlines


# ── NewsAPI.org Fetcher ────────────────────────────────────────────────────────

def fetch_newsapi_headlines(query: str = "Nifty OR Indian stock market OR RBI OR NSE BSE", max_results: int = 20) -> list[dict]:
    """Fetch financial headlines from newsapi.org."""
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_results,
                "apiKey": vault.news_api_key,
            },
            timeout=10,
        )
        data = resp.json()
        articles = data.get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "summary": a.get("description", "")[:300],
                "published": a.get("publishedAt", ""),
                "source": a.get("source", {}).get("name", "newsapi"),
                "link": a.get("url", ""),
            }
            for a in articles
            if a.get("title")
        ]
    except Exception as e:
        logger.warning("newsapi_fetch_failed", error=str(e))
        return []


# ── Claude AI Sentiment Scorer ─────────────────────────────────────────────────

def score_headlines_with_claude(headlines: list[dict]) -> dict:
    """
    Send headlines to Claude for financial market sentiment scoring.
    Returns: { score: float (-1 to +1), summary: str, risk_events: list[str] }
    """
    if not headlines:
        return {"score": 0.0, "summary": "No headlines available.", "risk_events": []}

    headline_text = "\n".join(
        f"- [{h['source']}] {h['title']}" for h in headlines[:30]
    )

    prompt = f"""You are a senior Indian equity market analyst. Analyze these financial news headlines
from the past 12 hours and assess their impact on Nifty 50 and Indian stock markets.

Headlines:
{headline_text}

Respond with a JSON object containing:
1. "score": float from -1.0 (very bearish) to +1.0 (very bullish) for Indian markets today
2. "summary": 2-sentence summary of market-moving themes
3. "risk_events": list of specific risk events that could cause volatility (e.g., "Fed rate decision", "RBI policy", "earnings miss")
4. "sector_impacts": dict of sector → "bullish"/"bearish"/"neutral"

Be conservative — if signals are mixed, return score near 0. Only give ±0.7 or higher for very clear directional events.

Return ONLY the JSON object, no other text."""

    try:
        response = _get_anthropic().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        text = response.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        result["score"] = max(-1.0, min(1.0, float(result.get("score", 0))))
        logger.info("claude_sentiment_scored", score=result["score"])
        return result
    except Exception as e:
        logger.error("claude_sentiment_failed", error=str(e))
        return {"score": 0.0, "summary": str(e), "risk_events": []}


def _keyword_score(headlines: list[dict]) -> float:
    """Fast keyword-based fallback scorer."""
    neg_keywords = _cfg.get("negative_keywords", [])
    pos_keywords = ["bull", "rally", "gain", "surge", "buy", "growth", "record", "FII buying", "strong GDP"]

    score = 0.0
    for h in headlines:
        text = (str(h.get("title", "")) + " " + str(h.get("summary", ""))).lower()
        for kw in neg_keywords:
            if kw.lower() in text:
                score -= 0.15
        for kw in pos_keywords:
            if kw.lower() in text:
                score += 0.10

    return max(-1.0, min(1.0, score))


# ── Master Sentiment Fetch ─────────────────────────────────────────────────────

def get_market_sentiment() -> dict:
    """
    Full pipeline: fetch → aggregate → score.
    Returns unified sentiment context.
    """
    logger.info("fetching_news_sentiment")

    rss = fetch_rss_headlines(max_age_hours=12)
    newsapi = fetch_newsapi_headlines()
    all_headlines = rss + newsapi

    # Deduplicate by title
    seen = set()
    unique = []
    for h in all_headlines:
        t = str(h.get("title", "")).lower()[:80]
        if t not in seen:
            seen.add(t)
            unique.append(h)

    if not unique:
        logger.warning("no_headlines_found")
        return {"score": 0.0, "summary": "No news data", "headlines_count": 0, "risk_events": []}

    # Primary: Claude AI scoring (only if API key is configured)
    import os
    use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_claude:
        result = score_headlines_with_claude(unique)
    else:
        result = {"score": 0.0, "summary": "Claude API not configured.", "risk_events": []}

    # Fallback: keyword scoring if Claude is disabled or returns 0 with many headlines
    if result["score"] == 0.0 and len(unique) > 5:
        result["score"] = _keyword_score(unique)
        result["method"] = "keyword_fallback"
    else:
        result["method"] = "claude_ai" if use_claude else "keyword_fallback"

    result["headlines_count"] = len(unique)
    result["fetched_at"] = datetime.utcnow().isoformat()
    result["sample_headlines"] = [h["title"] for h in unique[:5]]

    logger.info("sentiment_ready", score=result["score"], method=result["method"], n=len(unique))
    return result
