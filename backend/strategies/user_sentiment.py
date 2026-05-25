"""
UserSentimentStrategy — direct Reddit sentiment comparison.

Signal rule (no ML model):
  home_sent - away_sent > +threshold  →  BUY home  (home fans more bullish)
  home_sent - away_sent < -threshold  →  SELL home  (away fans more bullish)

"buy" always means betting the HOME team wins (YES on the home-team contract).
"sell" always means betting the AWAY team wins (NO on the home-team contract).

Edge scales with sentiment delta: 0.1 diff → 3% edge, 0.3 diff → 9% edge.

Two modes, selected automatically:
  OFFLINE (backtest): reads pre-computed sentiment columns from training CSV.
  LIVE: reads recent reddit_posts from DB, aggregates per user in real-time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from backend.core.strategy import Strategy, Signal

TRAINING_DATA_PATH = "data/processed/training_data_with_sentiment.csv"


class UserSentimentStrategy(Strategy):
    """
    Trade when Reddit fan sentiment for one team significantly exceeds the other.

    threshold         : minimum |home_sent − away_sent| to issue a signal (default 0.1)
    reddit_hours      : look-back window for live Reddit posts (default 12)
    sharp_fallback_edge: fall back to sharp-vs-Kalshi if no sentiment signal (default 0.025)
    training_df       : pre-loaded DataFrame → activates offline/backtest mode
    """

    def __init__(
        self,
        threshold: float = 0.10,
        reddit_hours: int = 12,
        min_posts: int = 5,
        sharp_fallback_edge: float = 0.025,
        training_df: Optional[pd.DataFrame] = None,
    ):
        self.threshold           = threshold
        self.reddit_hours        = reddit_hours
        self.min_posts           = min_posts
        self.sharp_fallback_edge = sharp_fallback_edge
        self._offline            = training_df is not None
        self._training_df        = training_df

    # ── Sentiment helpers ─────────────────────────────────────────────────────

    def _live_sentiments(self, home: str, away: str) -> tuple[float, float, int, int]:
        """Returns (home_mean, away_mean, home_scored_posts, away_scored_posts).

        Only posts with a non-zero RoBERTa score count toward the minimum and
        the mean. Zero-scored posts are game threads / neutrals that add noise.
        """
        from backend.models.database import get_reddit_posts
        from src.user_features import aggregate_by_user

        since = int(datetime.now(timezone.utc).timestamp()) - self.reddit_hours * 3600

        def _agg(team: str) -> tuple[float, int]:
            rows = get_reddit_posts(team, since)
            # Only keep posts that were actually scored by RoBERTa (non-zero)
            posts = [
                {"author": r["author"], "sentiment": r["sentiment"]}
                for r in rows
                if r.get("sentiment") is not None and r["sentiment"] != 0.0
            ]
            feats = aggregate_by_user(posts)
            return feats["mean_sentiment"], len(posts)

        home_sent, home_n = _agg(home)
        away_sent, away_n = _agg(away)
        return home_sent, away_sent, home_n, away_n

    def _offline_sentiments(self, game: dict) -> Optional[tuple[float, float, int, int]]:
        """Look up pre-computed sentiment from training CSV row."""
        df   = self._training_df
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        date = (game.get("scheduled_time") or game.get("date") or "")[:10]

        mask = (
            (df["home_team"] == home) &
            (df["away_team"] == away) &
            (df["date"].astype(str).str[:10] == date)
        )
        rows = df[mask]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return (
            float(row.get("home_mean_sentiment", 0) or 0),
            float(row.get("away_mean_sentiment", 0) or 0),
            int(row.get("home_num_posts", 0) or 0),
            int(row.get("away_num_posts", 0) or 0),
        )

    # ── Core signal logic ─────────────────────────────────────────────────────

    def _sentiment_signal(
        self,
        home_sent: float,
        away_sent: float,
        home_team: str,
        away_team: str,
    ) -> Optional[Signal]:
        """
        Pure sentiment rule — no model involved.
        diff > 0 means home fans are more bullish.
        """
        diff = home_sent - away_sent

        if diff > self.threshold:
            # Home fans clearly more positive → BUY home
            edge = min(abs(diff) * 0.3, 0.12)
            return Signal(
                side="buy",
                confidence=min(1.0, abs(diff) / 0.30),
                edge_override=edge,
            )

        if diff < -self.threshold:
            # Away fans clearly more positive → SELL home (bet away wins)
            edge = min(abs(diff) * 0.3, 0.12)
            return Signal(
                side="sell",
                confidence=min(1.0, abs(diff) / 0.30),
                edge_override=edge,
            )

        return None

    # ── Strategy interface ────────────────────────────────────────────────────

    def signal(
        self,
        game: dict,
        kalshi_prob: float,
        sharp_prob: Optional[float],
    ) -> Optional[Signal]:
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        if self._offline:
            result = self._offline_sentiments(game)
            if result is None:
                return None
            home_sent, away_sent, home_n, away_n = result
        else:
            home_sent, away_sent, home_n, away_n = self._live_sentiments(home, away)

        if home_n < self.min_posts or away_n < self.min_posts:
            return None   # not enough scored posts to make a confident call

        sig = self._sentiment_signal(home_sent, away_sent, home, away)

        if sig is not None:
            return sig

        # Sharp-edge fallback when sentiment is neutral
        if sharp_prob is not None:
            sharp_edge = sharp_prob - kalshi_prob
            if abs(sharp_edge) >= self.sharp_fallback_edge:
                return Signal(
                    side="buy" if sharp_edge > 0 else "sell",
                    confidence=0.4,
                    edge_override=abs(sharp_edge),
                )

        return None

    def __repr__(self) -> str:
        mode = "offline" if self._offline else "live"
        return (f"DirectSentiment(threshold={self.threshold:.2f}, "
                f"min_posts={self.min_posts}, window={self.reddit_hours}h, mode={mode})")
