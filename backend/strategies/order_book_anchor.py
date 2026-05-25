"""
OrderBookAnchorStrategy — exploit round-number limit orders placed by LLM bots.

How it works
------------
LLM bots (e.g. ryanfrigo/kalshi-ai-trading-bot) anchor limit orders at round
prices: 25¢, 50¢, 75¢, 80¢. They scan for YES asks above a threshold and park
resting NO orders there.  This creates two signals:

1. SPREAD COMPRESSION at a round number
   If yes_ask is stuck at exactly a round number (±1¢) for multiple consecutive
   polls, a large resting order is holding it there. The "true" price should move
   but can't until the order is filled or pulled.

2. DEPTH ASYMMETRY
   If the orderbook shows large depth on one side at a round-number price, that
   side is being artificially supplied by a bot.  The other side is thin — meaning
   the price should snap through when a real participant acts.

Signal logic
------------
We need two conditions to fire:

  A. Bot anchor detected:
       - yes_ask has been at a round number (±1¢) for >= min_anchor_polls cycles, OR
       - Orderbook has >= min_anchor_size contracts resting at a round-number price

  B. Our sentiment says the fair value is on the OTHER side of the anchor:
       - If anchor is at 75¢ YES ask, and sentiment says home team should be at 70¢
         → we SELL (buy NO at 25¢ effectively) — bot is selling YES too cheap relative
         to what we think is fair, but in the WRONG direction for us to buy

     More precisely:
       - Anchor on YES ask at price P, sentiment diff > threshold → fair value > P
         → BUY YES — the bot is offering YES too cheaply, we take it
       - Anchor on YES bid at price P, sentiment diff < -threshold → fair value < P
         → SELL YES — the bot is bidding YES too expensively, we fade it

Edge formula: base_edge + anchor_bonus
  base_edge   = min(|diff| × 0.35, 0.14)  (same as AntiBot)
  anchor_bonus= 0.02 per % the ask is below our fair-value estimate, capped at 0.04

Live vs backtest
----------------
LIVE: fetches real orderbook from Kalshi API each cycle
BACKTEST: falls back to AntiBot signal (no historical orderbook data)
          — set training_df to activate, anchor bonus is simulated from bid/ask spread
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from backend.core.strategy import Strategy, Signal
from backend.models.database import get_kalshi_ask_history

log = logging.getLogger(__name__)

ROUND_PRICES = [0.20, 0.25, 0.50, 0.75, 0.80]   # common bot anchor points
ROUND_TOL    = 0.01                                 # ±1¢ counts as "at" a round price


def _nearest_round(price: float) -> Optional[float]:
    """Return the nearest round anchor price if within tolerance, else None."""
    for r in ROUND_PRICES:
        if abs(price - r) <= ROUND_TOL:
            return r
    return None


def _anchor_score(history: list[dict], orderbook: Optional[dict] = None) -> dict:
    """
    Score how strongly a round-number bot anchor is present.

    Returns:
        anchored      : bool
        anchor_price  : float | None   (the round number being anchored)
        anchor_side   : 'ask'|'bid'|None
        depth_at_anchor: int            (contracts resting at anchor price)
        consecutive   : int             (how many polls the ask has been stuck)
        score         : 0.0–1.0
    """
    result = {
        "anchored": False, "anchor_price": None, "anchor_side": None,
        "depth_at_anchor": 0, "consecutive": 0, "score": 0.0,
    }

    if not history:
        return result

    # ── Check 1: ask price stuck at round number for multiple polls ───────────
    asks = [h["ask"] for h in history if h.get("ask") is not None]
    if len(asks) >= 3:
        latest_ask = asks[-1]
        rp = _nearest_round(latest_ask)
        if rp is not None:
            # Count consecutive polls where ask == same round price
            consec = 0
            for a in reversed(asks):
                if abs(a - rp) <= ROUND_TOL:
                    consec += 1
                else:
                    break
            if consec >= 3:
                result["anchored"]     = True
                result["anchor_price"] = rp
                result["anchor_side"]  = "ask"
                result["consecutive"]  = consec
                result["score"]        = min(1.0, 0.4 + consec * 0.1)

    # ── Check 2: deep resting orders at a round price in order book ───────────
    if orderbook:
        for side_key, side_label in [("yes", "ask"), ("no", "bid")]:
            for price_cents, size in orderbook.get(side_key, []):
                price = price_cents / 100.0
                rp = _nearest_round(price)
                if rp is not None and size >= 50:   # 50+ contracts = meaningful depth
                    result["anchored"]        = True
                    result["anchor_price"]    = rp
                    result["anchor_side"]     = side_label
                    result["depth_at_anchor"] = int(size)
                    depth_score = min(0.5, size / 500)   # scales to 0.5 at 500 contracts
                    result["score"] = max(result["score"], 0.5 + depth_score)
                    break

    return result


class OrderBookAnchorStrategy(Strategy):
    """
    Fire when a round-number bot anchor is detected AND sentiment disagrees
    with the anchored price direction.

    Parameters
    ----------
    threshold        : min |diff| to consider a sentiment signal (default 0.10)
    min_anchor_polls : consecutive polls ask must be stuck to count (default 3)
    min_anchor_size  : min contracts at round price in orderbook (default 50)
    min_posts        : min scored Reddit posts per team (default 5)
    reddit_hours     : look-back window for live posts (default 12)
    training_df      : DataFrame → activates backtest mode
    """

    def __init__(
        self,
        threshold: float = 0.10,
        min_anchor_polls: int = 3,
        min_anchor_size: int = 50,
        min_posts: int = 5,
        reddit_hours: int = 12,
        training_df: Optional[pd.DataFrame] = None,
    ):
        self.threshold        = threshold
        self.min_anchor_polls = min_anchor_polls
        self.min_anchor_size  = min_anchor_size
        self.min_posts        = min_posts
        self.reddit_hours     = reddit_hours
        self._offline         = training_df is not None
        self._training_df     = training_df

    # ── Sentiment helpers (same as AntiBot) ───────────────────────────────────

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
            (df["home_team"] == home) & (df["away_team"] == away) &
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

    # ── Anchor detection ──────────────────────────────────────────────────────

    def _get_anchor(self, game_id: int, kalshi_ticker: Optional[str]) -> dict:
        """Return anchor score dict for this game."""
        history   = get_kalshi_ask_history(game_id, hours=4)
        orderbook = None

        if kalshi_ticker and not self._offline:
            try:
                from backend.data.kalshi_client import KalshiClient
                orderbook = KalshiClient().get_orderbook(kalshi_ticker)
            except Exception as e:
                log.debug(f"Orderbook fetch failed for {kalshi_ticker}: {e}")

        return _anchor_score(history, orderbook)

    # ── Core signal ───────────────────────────────────────────────────────────

    def signal(
        self,
        game: dict,
        kalshi_prob: float,
        sharp_prob: Optional[float],
    ) -> Optional[Signal]:
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        # Sentiment
        if self._offline:
            result = self._offline_sentiments(game)
            if result is None:
                return None
            home_sent, away_sent, home_n, away_n = result
        else:
            home_sent, away_sent, home_n, away_n = self._live_sentiments(home, away)

        if home_n < self.min_posts or away_n < self.min_posts:
            return None

        diff = home_sent - away_sent
        if abs(diff) < self.threshold:
            return None

        # Anchor detection
        game_id = game.get("id")
        ticker  = game.get("kalshi_ticker")
        anchor  = self._get_anchor(game_id, ticker) if game_id else {"anchored": False, "score": 0.0}

        # Agreement-only regimes (Regime 2 removed — 18% WR in 953-game backtest).
        agree = 0.55
        if diff > self.threshold and kalshi_prob >= agree:
            side = "buy"
        elif diff < -self.threshold and kalshi_prob < (1 - agree):
            side = "sell"
        else:
            return None

        base_edge = min(abs(diff) * 0.35, 0.14)
        confidence = min(1.0, abs(diff) / 0.30)

        # Anchor bonus: if bot has a resting order at a round price AND
        # that price is on the same side we're trading against, we get better fill
        anchor_bonus = 0.0
        if anchor["anchored"] and anchor["anchor_price"] is not None:
            ap = anchor["anchor_price"]
            # BUY: anchor ask below our fair value estimate → cheaper entry
            if side == "buy" and ap < kalshi_prob:
                anchor_bonus = min(0.04, (kalshi_prob - ap) * 0.5)
            # SELL: anchor bid above our fair value → better exit price
            elif side == "sell" and ap > kalshi_prob:
                anchor_bonus = min(0.04, (ap - kalshi_prob) * 0.5)

            if anchor_bonus > 0:
                confidence = min(1.0, confidence + anchor["score"] * 0.2)
                log.info(
                    f"Anchor detected: {home} ticker={ticker} "
                    f"price={ap:.2f} depth={anchor['depth_at_anchor']} "
                    f"consecutive={anchor['consecutive']} bonus={anchor_bonus:.3f}"
                )

        edge = min(base_edge + anchor_bonus, 0.16)
        return Signal(side=side, confidence=confidence, edge_override=edge)

    def __repr__(self) -> str:
        mode = "offline" if self._offline else "live"
        return (
            f"OrderBookAnchor(threshold={self.threshold:.2f}, "
            f"min_polls={self.min_anchor_polls}, min_depth={self.min_anchor_size}, "
            f"min_posts={self.min_posts}, mode={mode})"
        )
