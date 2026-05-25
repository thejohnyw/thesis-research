"""
Strategy interface for the trading bot and backtester.

A Strategy receives current market data for one game and returns a Signal
(buy/sell + confidence) or None to skip. The bot calls this once per game
per scan; the backtester calls it once per game in the historical dataset.

Built-in strategies:
  SharpVsKalshiStrategy  — trade when sharp books diverge from Kalshi (default)
  RandomStrategy         — random orders, used to test infrastructure / baseline
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import random as _random


@dataclass
class Signal:
    side: str                           # 'buy' or 'sell'
    confidence: float = 1.0            # 0–1, scales Kelly fraction
    edge_override: Optional[float] = None  # force this edge value for Kelly sizing


class Strategy(ABC):
    """
    Base class for all trading strategies.

    signal() is called with:
      game       — dict with keys: id, home_team, away_team, kalshi_ticker, ...
      kalshi_prob — Kalshi yes_bid as implied prob (0–1), home team wins
      sharp_prob  — average of top sharp books (0–1), or None if unavailable

    Return a Signal to trade, None to skip.
    """

    @abstractmethod
    def signal(
        self,
        game: dict,
        kalshi_prob: float,
        sharp_prob: Optional[float],
    ) -> Optional[Signal]:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


class SharpVsKalshiStrategy(Strategy):
    """
    Default strategy: trade when sharp books and Kalshi disagree.

    Edge = |sharp_prob - kalshi_prob|.
    Buy YES when Kalshi is too cheap (sharp_prob > kalshi_prob).
    Sell YES when Kalshi is too expensive.

    Confidence scales with edge size: 10% edge → full confidence.
    """

    def __init__(self, min_edge: float = 0.03):
        self.min_edge = min_edge

    def signal(self, game, kalshi_prob, sharp_prob) -> Optional[Signal]:
        if sharp_prob is None:
            return None
        edge = sharp_prob - kalshi_prob
        if abs(edge) < self.min_edge:
            return None
        return Signal(
            side="buy" if edge > 0 else "sell",
            confidence=min(1.0, abs(edge) / 0.10),
        )

    def __repr__(self):
        return f"SharpVsKalshi(min_edge={self.min_edge:.0%})"


class RandomStrategy(Strategy):
    """
    Places random orders — for infrastructure testing and as a performance baseline.
    Deliberately ignores all market data.

    trade_prob: probability of entering any given game (default 30%).
    seed:       fix for reproducible runs.
    """

    def __init__(self, trade_prob: float = 0.30, seed: Optional[int] = None):
        self._rng = _random.Random(seed)
        self._trade_prob = trade_prob

    def signal(self, game, kalshi_prob, sharp_prob) -> Optional[Signal]:
        if self._rng.random() > self._trade_prob:
            return None
        return Signal(
            side=self._rng.choice(["buy", "sell"]),
            confidence=0.5,
            edge_override=0.08,  # needs to clear the 7% Kalshi fee; ~3.6% is break-even at p=0.5
        )

    def __repr__(self):
        return f"Random(p={self._trade_prob:.0%})"
