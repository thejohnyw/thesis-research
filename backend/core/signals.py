"""
Signal generator — orchestrates data collection, edge finding, and trade execution.

Default strategy is M3EdgeStrategy (RF model prob vs Kalshi, coin-flip filter).
Requires data/pregame_predictions.json (written daily by the pregame scheduler job)
and models/m3_rf.pkl (written by scripts/train_m3_model.py).
Falls back to SharpVsKalshiStrategy if model files are missing.
"""
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from backend.config import MAX_TRADES_PER_SCAN, MAX_PENDING_TRADES, PAPER_TRADING
from backend.data.kalshi_client import KalshiClient
from backend.data.markets import match_game_to_kalshi, parse_kalshi_ticker
from backend.core.edge import find_edges, _is_stale, get_sharp_prob, calculate_confidence
from backend.core.risk import RiskManager
from backend.core.strategy import Strategy
from backend.models.database import (
    upsert_game, insert_odds, insert_signal, insert_trade,
    get_open_trades, update_bot_state,
    get_latest_odds_with_timestamps, get_upcoming_games,
)

PREGAME_CUTOFF_MINUTES  = 30   # don't trade within 30 min of tipoff
PREGAME_OPEN_HOURS      = 24   # don't trade more than 24h before tipoff
MARKET_DRIFT_HOURS      = 4    # look-back window for drift check
MARKET_DRIFT_MAX        = 0.03 # skip if Kalshi moved >3% in that window

log = logging.getLogger(__name__)


def collect_odds():
    """Fetch Kalshi NBA markets and store price snapshots. No Odds API needed."""
    kalshi_client = KalshiClient()
    markets = kalshi_client.get_nba_markets()

    seen = 0
    for km in markets:
        parsed = parse_kalshi_ticker(km.ticker)
        if not parsed:
            continue

        home_team = parsed["home_team"]
        away_team = parsed["away_team"]
        game_date = parsed["game_date"]

        scheduled_time = (
            f"{game_date.isoformat()}T00:00:00Z" if game_date else km.close_time
        )

        game_id = upsert_game(
            external_id=km.ticker,
            home_team=home_team,
            away_team=away_team,
            scheduled_time=scheduled_time,
            kalshi_ticker=km.ticker,
        )
        insert_odds(game_id, "kalshi", km.yes_price, yes_ask=km.yes_ask)
        seen += 1

    log.info(f"Collected Kalshi prices for {seen} NBA markets")
    return seen


def _default_strategy() -> Strategy:
    """M3EdgeStrategy if model files exist, else SharpVsKalshi with a warning."""
    from backend.strategies.m3_edge import M3EdgeStrategy
    model_files = ["models/m3_rf.pkl", "models/m3_scaler.pkl", "models/m3_feature_cols.json"]
    if all(os.path.exists(p) for p in model_files):
        return M3EdgeStrategy()
    log.warning(
        "M3 model files not found — falling back to SharpVsKalshiStrategy. "
        "Run: python scripts/train_m3_model.py"
    )
    from backend.core.strategy import SharpVsKalshiStrategy
    return SharpVsKalshiStrategy()


def scan_and_trade(strategy: Optional[Strategy] = None) -> list[dict]:
    """
    Find edges and execute trades (paper or live).

    Default strategy is M3EdgeStrategy (RF model prob vs Kalshi, coin-flip filter).
    Falls back to SharpVsKalshiStrategy if model files are missing.
    Pass an explicit strategy to override (e.g. RandomStrategy() for testing).
    """
    open_count = len(get_open_trades())
    if open_count >= MAX_PENDING_TRADES:
        log.info(f"At max pending trades ({open_count}), skipping scan")
        return []

    risk = RiskManager()
    executed = []

    if strategy is None:
        strategy = _default_strategy()

    candidates = _build_candidates(strategy)

    for c in candidates[:MAX_TRADES_PER_SCAN]:
        sizing = risk.calculate_size(c["edge"], c["kalshi_prob"], c["side"])

        if sizing.size <= 0:
            log.info(f"Skip {c['matchup']}: {sizing.reason}")
            continue

        insert_signal(
            game_id=c["game_id"],
            direction=c["side"],
            kalshi_prob=c["kalshi_prob"],
            sharp_prob=c["sharp_prob"],
            edge=c["edge"],
            confidence=c["confidence"],
            kelly_fraction=sizing.kelly_fraction,
            suggested_size=sizing.size,
        )

        entry_price = c["kalshi_prob"]
        trade_id = insert_trade(
            game_id=c["game_id"],
            side=c["side"],
            size=sizing.size,
            entry_price=entry_price,
            kalshi_prob=c["kalshi_prob"],
            sharp_prob=c["sharp_prob"],
            edge=c["edge"],
            kalshi_ticker=c["kalshi_ticker"],
        )

        if not PAPER_TRADING and c["kalshi_ticker"]:
            try:
                kalshi = KalshiClient()
                contracts = int(sizing.size)   # 1 contract = $1
                price_cents = int(entry_price * 100)
                kalshi_side = "yes" if c["side"] == "buy" else "no"
                kalshi.place_order(c["kalshi_ticker"], kalshi_side, contracts, price_cents)
                log.info(
                    f"LIVE order placed: {c['kalshi_ticker']} "
                    f"{kalshi_side} x{contracts} @ {price_cents}c"
                )
            except Exception as e:
                log.error(f"Order failed: {e}")

        update_bot_state(last_run=datetime.utcnow().isoformat())

        trade_info = {
            "trade_id": trade_id,
            "game": c["matchup"],
            "side": c["side"],
            "edge": f"{c['edge']:+.1%}",
            "size": f"${sizing.size:.2f}",
            "kelly": f"{sizing.kelly_fraction:.1%}",
            "mode": "PAPER" if PAPER_TRADING else "LIVE",
            "bpi_warning": c.get("bpi_warning"),
        }
        log.info(f"Trade: {trade_info}")
        executed.append(trade_info)

    return executed


def _pregame_ok(game: dict) -> tuple[bool, str]:
    """Return (ok, reason). Rejects games that have already started or are outside the trading window."""
    sched_str = game.get("scheduled_time", "")
    if not sched_str:
        return True, "ok"
    try:
        sched = datetime.fromisoformat(sched_str.replace("Z", "+00:00"))
    except ValueError:
        return True, "ok"

    now = datetime.now(timezone.utc)
    if now >= sched:
        return False, "game_started"
    if now > sched - timedelta(minutes=PREGAME_CUTOFF_MINUTES):
        return False, "too_close_to_tipoff"
    if now < sched - timedelta(hours=PREGAME_OPEN_HOURS):
        return False, "too_far_from_game"
    return True, "ok"


def _drift_ok(game_id: int, odds: dict) -> tuple[bool, float]:
    """
    Check Kalshi price drift over the last MARKET_DRIFT_HOURS hours.
    Returns (ok, drift). Rejects if drift > MARKET_DRIFT_MAX — market knows something we don't.
    """
    import sqlite3
    from backend.config import DATABASE_PATH
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=MARKET_DRIFT_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
    rows = conn.execute("""
        SELECT home_prob FROM odds_snapshots
        WHERE game_id = ? AND source = 'kalshi' AND timestamp >= ?
        ORDER BY id ASC
    """, (game_id, cutoff)).fetchall()
    conn.close()

    if len(rows) < 2:
        return True, 0.0
    drift = abs(rows[-1]["home_prob"] - rows[0]["home_prob"])
    return drift <= MARKET_DRIFT_MAX, round(drift, 4)


def _build_candidates(strategy: Strategy) -> list[dict]:
    """
    Run strategy against all live games that have fresh Kalshi odds.
    Skips: already-open positions, started games, games outside 24h window,
    and games where Kalshi moved >3% in the last 4h (market has info we don't).
    Returns a list of candidate dicts sorted by edge × confidence descending.
    """
    candidates = []

    open_game_ids = {t["game_id"] for t in get_open_trades()}

    for game in get_upcoming_games():
        if game["id"] in open_game_ids:
            continue

        ok, reason = _pregame_ok(game)
        if not ok:
            log.debug(f"Skip {game.get('home_team')}: {reason}")
            continue
        odds = get_latest_odds_with_timestamps(game["id"])
        kd = odds.get("kalshi")
        if kd is None or _is_stale(kd["timestamp"]):
            continue

        drift_ok, drift = _drift_ok(game["id"], odds)
        if not drift_ok:
            log.info(f"Skip {game.get('home_team')}: Kalshi drifted {drift:.1%} in {MARKET_DRIFT_HOURS}h — market has info")
            continue

        kalshi_prob = kd["prob"]

        sig = strategy.signal(game, kalshi_prob, sharp_prob=None)
        if sig is None:
            continue

        edge = sig.edge_override if sig.edge_override is not None else 0.01
        confidence = sig.confidence

        # ESPN BPI warning: flag when fading a heavy Kalshi favorite (>65%)
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        fading_home = sig.side == "sell" and kalshi_prob > 0.65
        fading_away = sig.side == "buy"  and kalshi_prob < 0.35
        bpi_warning = None
        if fading_home or fading_away:
            fav_team = home if fading_home else away
            fav_prob = kalshi_prob if fading_home else (1 - kalshi_prob)
            bpi_warning = f"fading {fav_team} ({fav_prob:.0%} Kalshi favorite)"
            log.warning(f"BPI ALERT: {bpi_warning} — edge={edge:.1%}")

        candidates.append({
            "game_id": game["id"],
            "matchup": f"{away} @ {home}",
            "side": sig.side,
            "edge": edge,
            "confidence": confidence,
            "kalshi_prob": kalshi_prob,
            "sharp_prob": kalshi_prob,   # no sharp book; store kalshi_prob as reference
            "kalshi_ticker": game.get("kalshi_ticker"),
            "bpi_warning": bpi_warning,
        })

    return sorted(candidates, key=lambda x: x["edge"] * x["confidence"], reverse=True)
