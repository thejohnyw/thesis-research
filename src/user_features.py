"""
User-level sentiment aggregation for Reddit posts.

Core idea: aggregate sentiment by AUTHOR first, then across authors.
This preserves the disagreement signal (std of user means) that post-level
averaging destroys when some users post many times.

Falls back to post-level aggregation when author field is absent.

Canonical team → subreddit mapping lives here and is imported everywhere else.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import entropy as _entropy

# ── Team ↔ Subreddit mapping ─────────────────────────────────────────────────
# Keys match backend/config.py TEAM_ABBREV values exactly.

TEAM_TO_SUBREDDIT: dict[str, str] = {
    "Atlanta Hawks":          "AtlantaHawks",
    "Boston Celtics":         "bostonceltics",
    "Brooklyn Nets":          "GoNets",
    "Charlotte Hornets":      "CharlotteHornets",
    "Chicago Bulls":          "chicagobulls",
    "Cleveland Cavaliers":    "clevelandcavs",
    "Dallas Mavericks":       "Mavericks",
    "Denver Nuggets":         "denvernuggets",
    "Detroit Pistons":        "DetroitPistons",
    "Golden State Warriors":  "warriors",
    "Houston Rockets":        "rockets",
    "Indiana Pacers":         "pacers",
    "Los Angeles Clippers":   "LAClippers",
    "Los Angeles Lakers":     "lakers",
    "Memphis Grizzlies":      "memphisgrizzlies",
    "Miami Heat":             "heat",
    "Milwaukee Bucks":        "MkeBucks",
    "Minnesota Timberwolves": "timberwolves",
    "New Orleans Pelicans":   "NOLAPelicans",
    "New York Knicks":        "NYKnicks",
    "Oklahoma City Thunder":  "Thunder",
    "Orlando Magic":          "OrlandoMagic",
    "Philadelphia 76ers":     "sixers",
    "Phoenix Suns":           "suns",
    "Portland Trail Blazers": "ripcity",
    "Sacramento Kings":       "kings",
    "San Antonio Spurs":      "NBASpurs",
    "Toronto Raptors":        "torontoraptors",
    "Utah Jazz":              "UtahJazz",
    "Washington Wizards":     "washingtonwizards",
}

SUBREDDIT_TO_TEAM: dict[str, str] = {v: k for k, v in TEAM_TO_SUBREDDIT.items()}

_SKIP_AUTHORS = {"[deleted]", "AutoModerator", ""}

# ── Post loading ──────────────────────────────────────────────────────────────

def load_posts(path: str | Path) -> list[dict]:
    posts = []
    with open(path) as f:
        for line in f:
            try:
                posts.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return posts


def build_index(posts: list[dict]) -> dict[str, list[dict]]:
    """Index posts by subreddit for fast lookup."""
    idx: dict[str, list[dict]] = defaultdict(list)
    for p in posts:
        sub = p.get("subreddit", "")
        if sub:
            idx[sub].append(p)
    return dict(idx)


# ── Window filtering ──────────────────────────────────────────────────────────

def get_team_posts(
    index: dict[str, list[dict]],
    team: str,
    game_utc: float,      # Unix timestamp of game start
    hours: int = 48,
) -> list[dict]:
    """Posts from team's subreddit in the N hours before game_utc."""
    sub = TEAM_TO_SUBREDDIT.get(team)
    if not sub:
        return []
    window_start = game_utc - hours * 3600
    return [
        p for p in index.get(sub, [])
        if window_start <= p.get("created_utc", 0) <= game_utc
        and "sentiment" in p          # must have been scored
    ]


# ── Aggregation ───────────────────────────────────────────────────────────────

_EMPTY_FEATS: dict[str, float] = {
    "mean_sentiment":  0.0,
    "std_sentiment":   0.0,
    "mean_user_std":   0.0,
    "user_entropy":    0.0,
    "num_users":       0.0,
    "num_posts":       0.0,
}


def aggregate_by_user(posts: list[dict]) -> dict[str, float]:
    """
    Aggregate sentiment by author, then across authors.

    If no author field present, treats each post as a separate "user"
    (i.e., falls back to post-level aggregation).
    """
    if not posts:
        return dict(_EMPTY_FEATS)

    has_authors = any(p.get("author", "") not in _SKIP_AUTHORS for p in posts)

    if has_authors:
        user_sentiments: dict[str, list[float]] = defaultdict(list)
        for p in posts:
            author = p.get("author", "")
            if author in _SKIP_AUTHORS:
                continue
            user_sentiments[author].append(p["sentiment"])
    else:
        # No author data — treat every post as its own "user"
        user_sentiments = {str(i): [p["sentiment"]] for i, p in enumerate(posts)}

    if not user_sentiments:
        return dict(_EMPTY_FEATS)

    user_means = np.array([np.mean(v) for v in user_sentiments.values()])
    user_stds  = np.array([np.std(v) if len(v) > 1 else 0.0
                           for v in user_sentiments.values()])

    hist, _ = np.histogram(user_means, bins=5, range=(-1.0, 1.0))
    hist_norm = hist / (hist.sum() + 1e-10)
    entropy = float(_entropy(hist_norm + 1e-10))

    return {
        "mean_sentiment": float(np.mean(user_means)),
        "std_sentiment":  float(np.std(user_means)),
        "mean_user_std":  float(np.mean(user_stds)),
        "user_entropy":   entropy,
        "num_users":      float(len(user_sentiments)),
        "num_posts":      float(len(posts)),
    }


# ── Game-level features ───────────────────────────────────────────────────────

SENTIMENT_FEATURE_COLS = [
    "home_mean_sentiment", "home_std_sentiment", "home_mean_user_std", "home_user_entropy",
    "away_mean_sentiment", "away_std_sentiment", "away_mean_user_std", "away_user_entropy",
    "diff_mean_sentiment", "diff_std_sentiment", "diff_mean_user_std", "diff_user_entropy",
]


def get_game_sentiment_features(
    index: dict[str, list[dict]],
    home_team: str,
    away_team: str,
    game_utc: float,
    hours: int = 48,
) -> dict[str, float]:
    home_posts = get_team_posts(index, home_team, game_utc, hours)
    away_posts = get_team_posts(index, away_team, game_utc, hours)

    home = aggregate_by_user(home_posts)
    away = aggregate_by_user(away_posts)

    features: dict[str, float] = {}
    for k, v in home.items():
        features[f"home_{k}"] = v
    for k, v in away.items():
        features[f"away_{k}"] = v
    for k in ["mean_sentiment", "std_sentiment", "mean_user_std", "user_entropy"]:
        features[f"diff_{k}"] = home[k] - away[k]

    return features


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scored = Path("data/processed/reddit_with_sentiment.jsonl")
    if not scored.exists():
        print(f"{scored} not found — run: python -m src.sentiment")
    else:
        posts = load_posts(scored)
        idx   = build_index(posts)
        print(f"Loaded {len(posts)} posts across {len(idx)} subreddits")

        # Sample: Celtics posts
        sub = TEAM_TO_SUBREDDIT["Boston Celtics"]
        sample = idx.get(sub, [])
        print(f"\nr/{sub}: {len(sample)} posts")
        feats = aggregate_by_user(sample[:100])
        for k, v in feats.items():
            print(f"  {k}: {v:.3f}")
