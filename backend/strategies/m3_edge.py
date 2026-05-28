"""
M3EdgeStrategy — live implementation of the thesis M3 RF model edge signal.

Fires when:
  1. Kalshi pregame price ∈ [0.40, 0.60]   (coin-flip filter)
  2. |model_prob − kalshi_prob| > 0.05      (edge threshold)

Probabilities are pre-computed daily by backend.data.pregame_features and
stored in data/pregame_predictions.json. This file is reloaded automatically
when updated (mtime check on every signal call).

OOS backtest performance (953 regular-season games, compare_models.py):
  SELL side: 56.3% WR, +$258 PnL
  Overall:   52.3% WR, +$154, Sharpe 0.66, p=0.315
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from backend.core.strategy import Strategy, Signal

PREDICTIONS_PATH = "data/pregame_predictions.json"
MKT_LO    = 0.40
MKT_HI    = 0.60
THRESHOLD = 0.05

log = logging.getLogger(__name__)


class M3EdgeStrategy(Strategy):
    """
    Trade when the M3 RF model disagrees with Kalshi by >5% in coin-flip games.
    Predictions loaded from data/pregame_predictions.json (refreshed daily at noon).
    """

    def __init__(
        self,
        mkt_lo: float = MKT_LO,
        mkt_hi: float = MKT_HI,
        threshold: float = THRESHOLD,
        predictions_path: str = PREDICTIONS_PATH,
    ):
        self.mkt_lo    = mkt_lo
        self.mkt_hi    = mkt_hi
        self.threshold = threshold
        self._path     = predictions_path
        self._preds: dict[str, float] = {}
        self._mtime: float = 0.0

    def _refresh(self) -> None:
        if not os.path.exists(self._path):
            return
        mtime = os.path.getmtime(self._path)
        if mtime != self._mtime:
            with open(self._path) as f:
                self._preds = json.load(f)
            self._mtime = mtime
            log.debug(f"Loaded {len(self._preds)} predictions from {self._path}")

    def _lookup(self, home: str, away: str, sched: str) -> float | None:
        self._refresh()
        key = f"{home}|{away}|{sched[:10]}"
        return self._preds.get(key)

    def signal(
        self,
        game: dict,
        kalshi_prob: float,
        sharp_prob: Optional[float],
    ) -> Optional[Signal]:
        if not (self.mkt_lo <= kalshi_prob <= self.mkt_hi):
            return None

        home  = game.get("home_team", "")
        away  = game.get("away_team", "")
        sched = game.get("scheduled_time", "")

        model_prob = self._lookup(home, away, sched)
        if model_prob is None:
            log.debug(f"No M3 prediction for {away}@{home} {sched[:10]} — run pregame job")
            return None

        edge = model_prob - kalshi_prob
        if abs(edge) <= self.threshold:
            return None

        side = "buy" if edge > 0 else "sell"
        log.info(
            f"M3 signal: {away[:3]}@{home[:3]}  side={side}  "
            f"model={model_prob:.3f}  mkt={kalshi_prob:.3f}  edge={edge:+.3f}"
        )
        return Signal(
            side=side,
            confidence=min(1.0, abs(edge) / 0.15),
            edge_override=abs(edge),
        )

    def __repr__(self) -> str:
        return (
            f"M3Edge(mkt=[{self.mkt_lo},{self.mkt_hi}], "
            f"threshold=±{self.threshold}, "
            f"n_preds={len(self._preds)})"
        )
