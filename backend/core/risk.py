"""
Risk manager — Kelly criterion position sizing with safety caps.
"""
from dataclasses import dataclass
from backend.config import (
    KELLY_FRACTION, MAX_POSITION_PCT, MAX_POSITION_DOLLARS,
    DAILY_LOSS_LIMIT, KALSHI_FEE,
)
from backend.models.database import get_bot_state


@dataclass
class PositionSize:
    size: float
    kelly_fraction: float
    capped: bool
    reason: str


class RiskManager:
    def __init__(self):
        state = get_bot_state()
        self.bankroll = state["bankroll"] if state else 1000
        self.daily_pnl = state["daily_pnl"] if state else 0

    def reload(self):
        state = get_bot_state()
        if state:
            self.bankroll = state["bankroll"]
            self.daily_pnl = state["daily_pnl"]

    def calculate_size(self, edge: float, kalshi_prob: float, side: str) -> PositionSize:
        """
        Kelly criterion for Kalshi binary contracts.

        Buy YES at price p:
          - win_prob = sharp_prob = kalshi_prob + edge
          - profit if win = (1 - p) * (1 - fee)
          - loss if lose = p
          - odds (b) = (1 - p) * (1 - fee) / p

        Sell YES at price p:
          - win_prob = 1 - sharp_prob = 1 - (kalshi_prob - edge)
          - profit if win = p * (1 - fee)
          - loss if lose = (1 - p)
        """
        # Circuit breaker
        if self.daily_pnl <= -DAILY_LOSS_LIMIT:
            return PositionSize(0, 0, True, "Daily loss limit hit")

        if edge <= 0:
            return PositionSize(0, 0, False, "No edge")

        if side == "buy":
            win_prob = kalshi_prob + edge
            profit_per_dollar = (1 - kalshi_prob) * (1 - KALSHI_FEE)
            loss_per_dollar = kalshi_prob
        else:
            win_prob = 1 - (kalshi_prob - edge)
            profit_per_dollar = kalshi_prob * (1 - KALSHI_FEE)
            loss_per_dollar = 1 - kalshi_prob

        win_prob = max(0.01, min(0.99, win_prob))
        lose_prob = 1 - win_prob

        # Kelly: f* = (p * b - q) / b where b = profit/loss ratio
        b = profit_per_dollar / loss_per_dollar if loss_per_dollar > 0 else 0
        if b <= 0:
            return PositionSize(0, 0, False, "No positive odds")

        kelly = (win_prob * b - lose_prob) / b
        kelly = max(0, kelly)

        # Fractional Kelly
        kelly_adj = kelly * KELLY_FRACTION
        size = self.bankroll * kelly_adj

        # Caps
        capped = False
        max_pct = self.bankroll * MAX_POSITION_PCT
        if size > max_pct:
            size = max_pct
            capped = True
        if size > MAX_POSITION_DOLLARS:
            size = MAX_POSITION_DOLLARS
            capped = True

        # Minimum trade size ($1)
        if size < 1:
            return PositionSize(0, kelly_adj, False, "Size below minimum")

        return PositionSize(
            size=round(size, 2),
            kelly_fraction=round(kelly_adj, 4),
            capped=capped,
            reason="OK",
        )
