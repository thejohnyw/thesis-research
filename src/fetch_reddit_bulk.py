"""
Bulk fetch Reddit posts from NBA team subreddits using public JSON API.

No credentials required. Uses Reddit's /.json endpoints with pagination.
Fetches top/hot/new posts to maximize coverage across the season.

Output: JSONL file with one post per line, compatible with existing pipeline.

Usage:
  python src/fetch_reddit_bulk.py \
    --out_jsonl data/raw/reddit_team_posts_bulk.jsonl \
    --posts_per_sub 500 \
    --sorts top,hot,new
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

TEAM_SUBREDDITS = [
    "AtlantaHawks", "bostonceltics", "GoNets", "CharlotteHornets",
    "chicagobulls", "clevelandcavs", "Mavericks", "denvernuggets",
    "DetroitPistons", "warriors", "rockets", "pacers",
    "LAClippers", "lakers", "memphisgrizzlies", "heat",
    "MkeBucks", "timberwolves", "NOLAPelicans", "NYKnicks",
    "Thunder", "OrlandoMagic", "sixers", "suns",
    "ripcity", "kings", "NBASpurs", "torontoraptors",
    "UtahJazz", "washingtonwizards",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research-project) nba-thesis/1.0",
}


def fetch_listing(subreddit: str, sort: str, after: str | None = None,
                  limit: int = 100, time_filter: str = "year") -> tuple[list[dict], str | None]:
    """Fetch one page of posts from a subreddit."""
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    params: dict = {"limit": limit, "raw_json": 1}
    if sort == "top":
        params["t"] = time_filter
    if after:
        params["after"] = after

    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    posts = []
    listing = data.get("data", {})
    for child in listing.get("children", []):
        d = child.get("data", {})
        if d.get("is_self") is not None:
            posts.append({
                "id": d.get("id", ""),
                "subreddit": subreddit,
                "title": d.get("title", ""),
                "selftext": d.get("selftext", ""),
                "score": d.get("score", 0),
                "num_comments": d.get("num_comments", 0),
                "created_utc": d.get("created_utc", 0),
                "url": d.get("url", ""),
                "sort_source": sort,
            })

    next_after = listing.get("after")
    return posts, next_after


def fetch_subreddit(subreddit: str, sort: str, max_posts: int = 500,
                    sleep: float = 2.0) -> list[dict]:
    """Paginate through a subreddit listing."""
    all_posts = []
    after = None
    pages = 0

    while len(all_posts) < max_posts:
        try:
            posts, after = fetch_listing(subreddit, sort, after=after)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = max(sleep * 5, 30)
                print(f"    Rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
                try:
                    posts, after = fetch_listing(subreddit, sort, after=after)
                except Exception:
                    print(f"    Still rate limited, skipping")
                    break
            else:
                print(f"    Error on page {pages+1}: {e}")
                break
        except Exception as e:
            print(f"    Error on page {pages+1}: {e}")
            break

        if not posts:
            break

        all_posts.extend(posts)
        pages += 1

        if not after:
            break
        time.sleep(sleep)

    return all_posts[:max_posts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_jsonl", default="data/raw/reddit_team_posts_bulk.jsonl")
    ap.add_argument("--posts_per_sub", type=int, default=500,
                    help="Max posts per subreddit per sort")
    ap.add_argument("--sorts", default="top,new",
                    help="Comma-separated sort methods")
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="Seconds between API calls")
    args = ap.parse_args()

    sorts = args.sorts.split(",")
    seen_ids: set[str] = set()
    all_posts: list[dict] = []

    # Load existing posts to avoid duplicates
    out_path = Path(args.out_jsonl)
    if out_path.exists():
        print(f"Loading existing posts from {out_path}...")
        with out_path.open() as f:
            for line in f:
                post = json.loads(line)
                pid = post.get("id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_posts.append(post)
        print(f"  {len(all_posts)} existing posts loaded")

    for sort in sorts:
        print(f"\n--- Fetching {sort} posts ---")
        for idx, sub in enumerate(TEAM_SUBREDDITS, 1):
            print(f"  [{idx}/{len(TEAM_SUBREDDITS)}] r/{sub} ({sort})...", end=" ", flush=True)
            posts = fetch_subreddit(sub, sort, max_posts=args.posts_per_sub, sleep=args.sleep)
            new = 0
            for p in posts:
                pid = p.get("id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_posts.append(p)
                    new += 1
            print(f"{len(posts)} fetched, {new} new")
            time.sleep(args.sleep)

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for post in all_posts:
            f.write(json.dumps(post, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    sub_counts = Counter(p["subreddit"] for p in all_posts)
    print(f"\nTotal: {len(all_posts)} unique posts across {len(sub_counts)} subreddits")
    print(f"Saved to {out_path}")
    print(f"Per-subreddit range: {min(sub_counts.values())}-{max(sub_counts.values())}")


if __name__ == "__main__":
    main()
