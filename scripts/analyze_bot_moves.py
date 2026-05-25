"""
LLM bot move detector for Kalshi NBA markets.

llm_move_score(game_id) returns a 0–1 score of how likely a recent
Kalshi price move was driven by an LLM bot rather than a sharp participant.

Signal weights (empirically unvalidated — needs >>20 divergent events to tune):
  0.40  Kalshi moved, sharps didn't (decoupling from informed flow)
  0.20  Move toward a round number (25 / 50 / 75%)
  0.20  Move reverted >50% within 30 min (mean-reversion = informed pushback)
  0.20  Move happened after a long flat period (bot woke up on news, not flow)

Current data limitation: 5 weeks of NBA data shows zero pre-game divergent
moves — all big Kalshi moves are settlement moves where sharps also moved.
Scores should be interpreted as directional signals, not calibrated probabilities,
until more pre-game divergent events are observed.
"""
from __future__ import annotations

import os
import sys
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.config import DATABASE_PATH

SHARP_BOOKS = {"lowvig", "fanduel", "draftkings", "betmgm"}
ROUND_TARGETS = [0.25, 0.50, 0.75]


def _fetch_recent_prices(
    conn: sqlite3.Connection,
    game_id: int,
    hours: int = 2,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return (kalshi_prices, sharp_prices) as (timestamp, prob) lists."""
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")

    k = conn.execute("""
        SELECT timestamp, home_prob FROM odds_snapshots
        WHERE game_id=? AND source='kalshi' AND timestamp >= ?
        ORDER BY timestamp ASC
    """, (game_id, cutoff)).fetchall()

    s = conn.execute("""
        SELECT strftime('%Y-%m-%dT%H:%M', timestamp) as ts,
               AVG(home_prob) as sp
        FROM odds_snapshots
        WHERE game_id=? AND source IN ('lowvig','fanduel','draftkings','betmgm')
          AND timestamp >= ?
        GROUP BY ts ORDER BY ts ASC
    """, (game_id, cutoff)).fetchall()

    return (
        [(r["timestamp"], float(r["home_prob"])) for r in k],
        [(r["ts"], float(r["sp"])) for r in s],
    )


def _nearest(prices: list[tuple[str, float]], ts: str, max_gap_min: int = 15) -> Optional[float]:
    """Find price nearest in time to ts within max_gap_min minutes."""
    t = datetime.strptime(ts[:16], "%Y-%m-%d %H:%M")
    best, best_gap = None, float("inf")
    for pts, pv in prices:
        gap = abs((t - datetime.strptime(pts[:16], "%Y-%m-%d %H:%M")).total_seconds())
        if gap < best_gap and gap <= max_gap_min * 60:
            best, best_gap = pv, gap
    return best


def llm_move_score(game_id: int, window_minutes: int = 30) -> dict:
    """
    Score how LLM-bot-like the most recent Kalshi move looks.

    Returns a dict with:
      score        : 0.0–1.0 composite
      components   : breakdown of each signal
      k_move       : Kalshi price change over window
      s_move       : Sharp composite change over same window
      details      : human-readable explanation
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    k_prices, s_prices = _fetch_recent_prices(conn, game_id, hours=max(2, window_minutes // 30 + 1))
    conn.close()

    if len(k_prices) < 4:
        return {
            "score": 0.0,
            "components": {},
            "k_move": 0.0,
            "s_move": 0.0,
            "details": "insufficient data (< 4 Kalshi snapshots in window)",
        }

    # Use the full window: first vs last snapshot
    t0, k0 = k_prices[0]
    t1, k1 = k_prices[-1]
    k_move = k1 - k0

    s0 = _nearest(s_prices, t0)
    s1 = _nearest(s_prices, t1)
    s_move = (s1 - s0) if (s0 is not None and s1 is not None) else None

    components: dict[str, float] = {}

    # ── Signal 1: Kalshi moved, sharps didn't (weight 0.40) ──────────────────
    if abs(k_move) < 0.01:
        components["decoupling"] = 0.0
    elif s_move is None:
        components["decoupling"] = 0.5   # no sharp data; unknown
    else:
        ratio = abs(s_move) / abs(k_move) if k_move != 0 else 1.0
        # score=1.0 if sharps didn't move at all, 0.0 if sharps moved proportionally
        components["decoupling"] = max(0.0, min(1.0, 1.0 - ratio / 0.5))

    # ── Signal 2: Move toward round number (weight 0.20) ─────────────────────
    if abs(k_move) >= 0.01:
        toward = any(abs(k1 - r) < abs(k0 - r) for r in ROUND_TARGETS)
        components["round_number"] = 1.0 if toward else 0.0
    else:
        components["round_number"] = 0.0

    # ── Signal 3: Price reverted >50% (weight 0.20) ──────────────────────────
    if abs(k_move) >= 0.02 and len(k_prices) > 6:
        # Look at the second half of the window for reversion
        mid = len(k_prices) // 2
        future_prices = [p for _, p in k_prices[mid:]]
        if future_prices:
            max_rev = (k1 - min(future_prices)) if k_move > 0 else (max(future_prices) - k1)
            rev_pct = max_rev / abs(k_move)
            components["reversion"] = min(1.0, rev_pct)
        else:
            components["reversion"] = 0.0
    else:
        components["reversion"] = 0.0

    # ── Signal 4: Move after long flat period (bot woke up) (weight 0.20) ────
    if len(k_prices) >= 4:
        early_prices = [p for _, p in k_prices[:len(k_prices)//3]]
        flat_range = max(early_prices) - min(early_prices) if early_prices else 1.0
        # Score=1.0 if perfectly flat before move, 0.0 if already volatile
        components["flat_before"] = max(0.0, 1.0 - flat_range / 0.02)
    else:
        components["flat_before"] = 0.0

    weights = {
        "decoupling":   0.40,
        "round_number": 0.20,
        "reversion":    0.20,
        "flat_before":  0.20,
    }

    score = sum(components.get(k, 0.0) * w for k, w in weights.items())

    # Build human explanation
    lines = [f"Kalshi moved {k_move:+.3f} ({t0[:16]} → {t1[:16]})"]
    if s_move is not None:
        lines.append(f"Sharps moved {s_move:+.3f} (ratio={abs(s_move)/abs(k_move):.0%} of Kalshi)")
    else:
        lines.append("No sharp data in window")
    for sig, val in components.items():
        lines.append(f"  {sig:<16}: {val:.2f} (weight {weights[sig]:.0%})")
    lines.append(f"Composite score: {score:.2f}")
    if score >= 0.70:
        lines.append("⚠ HIGH: strong LLM-bot signature — consider fading")
    elif score >= 0.40:
        lines.append("~ MEDIUM: possible bot move — monitor for reversion")
    else:
        lines.append("✓ LOW: move looks informed (sharp-confirmed or small)")

    return {
        "score":      round(score, 3),
        "components": {k: round(v, 3) for k, v in components.items()},
        "k_move":     round(k_move, 4),
        "s_move":     round(s_move, 4) if s_move is not None else None,
        "details":    "\n".join(lines),
    }


if __name__ == "__main__":
    import sys
    gid = int(sys.argv[1]) if len(sys.argv) > 1 else 2635
    result = llm_move_score(gid)
    print(result["details"])
    print(f"\nRaw: {result}")
