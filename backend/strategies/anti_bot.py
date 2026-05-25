"""
AntiBotSentimentStrategy — sentiment + structured market agreement.

Core insight from backtest diagnostics:
  Sentiment alone at threshold=0.10 fires on "fan hopium" trades that lose.
  The signal only holds when sentiment and structured fundamentals AGREE:

    BUY  home: home fans bullish (diff > threshold) AND Kalshi already prices
               home as favorite (kalshi_prob > agree_threshold).
               → Sentiment confirms what the market knows: both sources agree.

    SELL home: away fans bullish (diff < -threshold) AND home is underdog
               (kalshi_prob < 1-agree_threshold).
               → Market AND sentiment both say away team wins (double confirmation).
               Regime 2 (away fans bullish vs home favourite) was removed — 953-game
               backtest showed only 18% WR there (fan hopium beats the market signal).

  Win rates from 953-game walk-forward backtest (min 5 posts/team):
    BUY  + pm > 0.65: 76.9% WR  (13 trades)
    SELL + pm < 0.45: 71.4% away WR  (28 trades)
    Opposite regime: <35% WR — do NOT trade there

Edge formula: min(|diff| * 0.35, 0.14) — slightly higher than DirectSentiment
because we've filtered to a higher-precision regime.

Modes:
  LIVE    — reads Reddit DB + uses Kalshi price directly
  OFFLINE — reads pre-computed CSV columns + uses structured RF prob as market proxy
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from backend.core.strategy import Strategy, Signal

TRAINING_DATA_PATH = "data/processed/training_data_with_sentiment.csv"


class AntiBotSentimentStrategy(Strategy):
    """
    Bet when Reddit sentiment and Kalshi market price point in the same direction.
    Avoids the "fan hopium" regime where fans are optimistic about an underdog.

    Parameters
    ----------
    threshold        : min |home_sent − away_sent| to even consider a signal
    agree_threshold  : Kalshi prob level that separates "market agrees" from not
                       (default 0.55: home is favourite)
    min_posts        : minimum scored Reddit posts per team (default 5)
    reddit_hours     : look-back window for live posts (default 12)
    training_df      : pass a DataFrame to activate offline/backtest mode
    """

    def __init__(
        self,
        threshold: float = 0.10,
        agree_threshold: float = 0.55,
        min_posts: int = 5,
        reddit_hours: int = 12,
        training_df: Optional[pd.DataFrame] = None,
    ):
        self.threshold        = threshold
        self.agree_threshold  = agree_threshold
        self.min_posts        = min_posts
        self.reddit_hours     = reddit_hours
        self._offline         = training_df is not None
        self._training_df     = training_df

    # ── Sentiment helpers ─────────────────────────────────────────────────────

    def _live_sentiments(self, home: str, away: str) -> tuple[float, float, int, int]:
        from backend.models.database import get_reddit_posts
        from src.user_features import aggregate_by_user

        since = int(datetime.now(timezone.utc).timestamp()) - self.reddit_hours * 3600

        def _agg(team: str) -> tuple[float, int]:
            rows = get_reddit_posts(team, since)
            posts = [
                {"author": r["author"], "sentiment": r["sentiment"]}
                for r in rows
                if r.get("sentiment") is not None and r["sentiment"] != 0.0
            ]
            return aggregate_by_user(posts)["mean_sentiment"], len(posts)

        hs, hn = _agg(home)
        as_, an = _agg(away)
        return hs, as_, hn, an

    def _offline_sentiments(self, game: dict) -> Optional[tuple[float, float, int, int]]:
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

    # ── Core signal ───────────────────────────────────────────────────────────

    def _antibot_signal(
        self,
        home_sent: float,
        away_sent: float,
        kalshi_prob: float,
    ) -> Optional[Signal]:
        """
        Emit a signal only when sentiment and market fundamentals agree.
        Rejects "fan hopium" trades where sentiment fights the market.
        """
        diff = home_sent - away_sent
        agree = self.agree_threshold

        # ── BUY home: home fans bullish AND market already likes home ────────
        # Regime: pm > agree_threshold (home is favourite)
        # Fans confirming a genuine favourite, not rallying around an underdog.
        if diff > self.threshold and kalshi_prob >= agree:
            edge = min(abs(diff) * 0.35, 0.14)
            conf = min(1.0, abs(diff) / 0.30)
            # Extra conviction when deeply in the "agree zone"
            if kalshi_prob > 0.65 and abs(diff) > 0.20:
                edge = min(edge * 1.25, 0.14)
                conf = min(conf * 1.20, 1.0)
            return Signal(side="buy", confidence=conf, edge_override=edge)

        # ── SELL home: away fans bullish AND market also disfavours home ─────
        # Regime 3: pm < (1 - agree_threshold) — double confirmation both sides.
        # NOTE: Regime 2 (away fans bullish vs home favourite, pm >= agree) was
        # removed after backtest showed 18% WR on that regime (953-game evaluation).
        # "Fan hopium" — away fans stay bullish about an underdog, market is right.
        if diff < -self.threshold and kalshi_prob < (1 - agree):
            edge = min(abs(diff) * 0.35, 0.14)
            conf = min(1.0, abs(diff) / 0.30)
            # Boost: market + sentiment both point away
            if abs(diff) > 0.20:
                edge = min(edge * 1.20, 0.14)
                conf = min(conf * 1.15, 1.0)
            return Signal(side="sell", confidence=conf, edge_override=edge)

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
            return None

        return self._antibot_signal(home_sent, away_sent, kalshi_prob)

    def __repr__(self) -> str:
        mode = "offline" if self._offline else "live"
        return (
            f"AntiBotSentiment(threshold={self.threshold:.2f}, "
            f"agree={self.agree_threshold:.2f}, "
            f"min_posts={self.min_posts}, window={self.reddit_hours}h, mode={mode})"
        )
