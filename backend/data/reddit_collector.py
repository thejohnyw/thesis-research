"""
Live Reddit collection job — fetch recent posts, score sentiment, store in DB.

Called by the scheduler every 4 hours. Deduplicates by post_id so re-runs
are safe. Loads the sentiment model once per process (lazy singleton).

Can also be run manually:
    python -m backend.data.reddit_collector
    python -m backend.data.reddit_collector --team "Boston Celtics" --limit 100
"""
from __future__ import annotations

import argparse
import logging
import time

import requests

from src.user_features import TEAM_TO_SUBREDDIT
from backend.models.database import init_db, insert_reddit_post

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (thesis-bot) nba-reddit/1.0"}
_SENTIMENT_PIPE = None   # lazy singleton


def _get_sentiment_pipe():
    global _SENTIMENT_PIPE
    if _SENTIMENT_PIPE is None:
        from transformers import pipeline as hf_pipeline
        _SENTIMENT_PIPE = hf_pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            max_length=512,
            truncation=True,
        )
    return _SENTIMENT_PIPE


def _score(title: str, selftext: str) -> float:
    text = title + (" " + selftext[:300] if selftext else "")
    text = text[:512]
    try:
        pipe = _get_sentiment_pipe()
        res = pipe(text)[0]
        label, score = res["label"].lower(), res["score"]
        if "positive" in label:
            return round(score, 4)
        if "negative" in label:
            return round(-score, 4)
        return 0.0
    except Exception:
        return 0.0


def _fetch_subreddit(subreddit: str, limit: int = 50) -> list[dict]:
    url = f"https://www.reddit.com/r/{subreddit}/new.json"
    try:
        r = requests.get(url, headers=_HEADERS,
                         params={"limit": limit, "raw_json": 1}, timeout=15)
        if r.status_code == 429:
            log.warning(f"r/{subreddit} rate limited, skipping")
            return []
        r.raise_for_status()
        return [c["data"] for c in r.json().get("data", {}).get("children", [])]
    except Exception as e:
        log.warning(f"r/{subreddit} fetch error: {e}")
        return []


def collect_reddit_posts(
    teams: list[str] | None = None,
    limit_per_sub: int = 50,
    sleep_sec: float = 1.5,
) -> int:
    """
    Fetch, score, and store recent posts for all (or specified) teams.
    Returns number of new posts inserted.
    """
    targets = {t: TEAM_TO_SUBREDDIT[t]
               for t in (teams or TEAM_TO_SUBREDDIT)
               if t in TEAM_TO_SUBREDDIT}

    total_new = 0
    for team, subreddit in targets.items():
        posts = _fetch_subreddit(subreddit, limit=limit_per_sub)
        new = 0
        for p in posts:
            author   = p.get("author", "")
            title    = p.get("title", "")
            selftext = p.get("selftext", "") or ""
            post_id  = p.get("id", "")
            created  = int(p.get("created_utc", 0))

            if not post_id or author in ("[deleted]", "AutoModerator"):
                continue

            sentiment = _score(title, selftext)
            inserted  = insert_reddit_post(
                team=team, subreddit=subreddit, post_id=post_id,
                author=author, title=title, sentiment=sentiment,
                created_utc=created,
            )
            if inserted:
                new += 1

        log.info(f"r/{subreddit}: {len(posts)} fetched, {new} new")
        total_new += new
        time.sleep(sleep_sec)

    return total_new


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Collect and score Reddit posts")
    ap.add_argument("--team",  help="Specific team (e.g. 'Boston Celtics')")
    ap.add_argument("--limit", type=int, default=50, help="Posts per subreddit")
    args = ap.parse_args()

    init_db()
    teams = [args.team] if args.team else None

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    n = collect_reddit_posts(teams=teams, limit_per_sub=args.limit)
    print(f"Done — {n} new posts inserted")


if __name__ == "__main__":
    main()
