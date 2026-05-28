"""
Fetch real pre-game Kalshi prices for all 953 regular-season NBA training games.

Event ticker format: KXNBAGAME-{YY}{MON}{DD}{AWAY_ABBR}{HOME_ABBR}
  - YY: 25 (2025 games Oct-Dec) or 26 (2026 games Jan-Mar)
  - Home team YES contract ticker: {event_ticker}-{HOME_ABBR}

Strategy: candlestick endpoint with targeted time window
  - start_ts = market open_time
  - end_ts   = close_time - 4h  (pre-game proxy)
  - period   = 60 min candles
  - pre-game price = last candle close before end_ts
  - opening price  = first candle close after market opened

Skips games already present in the output CSV.
Saves checkpoints every 25 games.

Usage:
    python scripts/fetch_all_kalshi_prices.py
    python scripts/fetch_all_kalshi_prices.py --dry-run 10
    python scripts/fetch_all_kalshi_prices.py --delay 0.3
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE = "https://api.elections.kalshi.com/trade-api/v2"
TRAINING_CSV = "data/processed/training_data_with_sentiment.csv"
OUT_CSV = "data/kalshi_historical_prices.csv"

NAME_TO_ABBR: dict[str, str] = {
    "Atlanta Hawks":          "ATL",
    "Boston Celtics":         "BOS",
    "Brooklyn Nets":          "BKN",
    "Charlotte Hornets":      "CHA",
    "Chicago Bulls":          "CHI",
    "Cleveland Cavaliers":    "CLE",
    "Dallas Mavericks":       "DAL",
    "Denver Nuggets":         "DEN",
    "Detroit Pistons":        "DET",
    "Golden State Warriors":  "GSW",
    "Houston Rockets":        "HOU",
    "Indiana Pacers":         "IND",
    "LA Clippers":            "LAC",
    "Los Angeles Clippers":   "LAC",
    "Los Angeles Lakers":     "LAL",
    "Memphis Grizzlies":      "MEM",
    "Miami Heat":             "MIA",
    "Milwaukee Bucks":        "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans":   "NOP",
    "New York Knicks":        "NYK",
    "Oklahoma City Thunder":  "OKC",
    "Orlando Magic":          "ORL",
    "Philadelphia 76ers":     "PHI",
    "Phoenix Suns":           "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings":       "SAC",
    "San Antonio Spurs":      "SAS",
    "Toronto Raptors":        "TOR",
    "Utah Jazz":              "UTA",
    "Washington Wizards":     "WAS",
}

OUT_COLS = [
    "date", "home_team", "away_team", "home_abbr", "away_abbr", "home_won",
    "ticker", "open_time", "close_time", "expected_expiration_time",
    "kalshi_open_price", "kalshi_pregame_price", "n_candles",
]


def make_event_ticker(date_str: str, away_team: str, home_team: str) -> str | None:
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    yy  = dt.strftime("%y").upper()
    mon = dt.strftime("%b").upper()
    dd  = dt.strftime("%d")
    away = NAME_TO_ABBR.get(away_team)
    home = NAME_TO_ABBR.get(home_team)
    if not away or not home:
        return None
    return f"KXNBAGAME-{yy}{mon}{dd}{away}{home}"


def _get(url: str, params: dict | None = None, max_retries: int = 4) -> dict | None:
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == max_retries - 1:
                return None
            time.sleep(1)
    return None


def get_market_info(event_ticker: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Return (home_ticker, open_time, close_time, expected_expiration_time)."""
    data = _get(f"{BASE}/historical/markets", {"event_ticker": event_ticker, "limit": 10})
    if not data:
        return None, None, None, None
    markets = data.get("markets", [])
    if not markets:
        return None, None, None, None

    home_abbr = event_ticker[-3:]
    home_mkt = next((m for m in markets if m.get("ticker", "").endswith(f"-{home_abbr}")), None)
    if home_mkt is None:
        home_mkt = markets[0]

    return (
        home_mkt.get("ticker"),
        home_mkt.get("open_time"),
        home_mkt.get("close_time"),
        home_mkt.get("expected_expiration_time"),
    )


def fetch_pregame_price_candles(
    ticker: str,
    open_time_str: str,
    expected_expiration_str: str,
    period: int = 60,
) -> dict | None:
    """
    Fetch hourly candlesticks from market open to (expected_expiration_time - 3h).
    The expected_expiration_time is Kalshi's estimate of game end; subtracting 3h
    puts the cutoff ~30-60 minutes before tipoff regardless of tip-off time zone.
    Returns pre-game price = last candle close before that cutoff.
    Returns opening price  = first candle close after market opened.
    """
    open_dt  = datetime.fromisoformat(open_time_str.replace("Z", "+00:00"))
    exp_dt   = datetime.fromisoformat(expected_expiration_str.replace("Z", "+00:00"))
    target_dt = exp_dt - timedelta(hours=3)

    start_ts = int(open_dt.timestamp())
    end_ts   = int(target_dt.timestamp())

    if end_ts <= start_ts:
        end_ts = start_ts + 3600 * 48  # fallback: 48h window

    data = _get(
        f"{BASE}/historical/markets/{ticker}/candlesticks",
        {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period},
    )
    if not data:
        return None

    candles = data.get("candlesticks", [])
    if not candles:
        return None

    # Candles sorted by end_period_ts ascending
    candles_sorted = sorted(candles, key=lambda c: c.get("end_period_ts", 0))

    def _close(c: dict) -> float | None:
        p = c.get("price", {})
        val = p.get("close") or p.get("mean")
        return float(val) if val is not None else None

    # Opening price: first valid candle
    open_price: float | None = None
    for c in candles_sorted:
        v = _close(c)
        if v is not None:
            open_price = v
            break

    # Pre-game price: last valid candle at or before target_dt
    pregame_price: float | None = None
    for c in reversed(candles_sorted):
        ts = c.get("end_period_ts", 0)
        if ts <= end_ts:
            v = _close(c)
            if v is not None:
                pregame_price = v
                break

    if open_price is None:
        return None

    return {
        "kalshi_open_price":    round(open_price, 4),
        "kalshi_pregame_price": round(pregame_price if pregame_price is not None else open_price, 4),
        "n_candles":            len(candles),
    }


def load_existing(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {row["ticker"] for row in csv.DictReader(f) if row.get("ticker")}


def append_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", type=int, default=0,
                    help="Process only N games (for testing)")
    ap.add_argument("--delay",   type=float, default=0.5,
                    help="Seconds between API calls (default 0.5)")
    ap.add_argument("--period",  type=int, default=60,
                    help="Candlestick period in minutes (default 60)")
    args = ap.parse_args()

    df = pd.read_csv(TRAINING_CSV).sort_values("date").reset_index(drop=True)
    existing = load_existing(OUT_CSV)

    todo: list[dict] = []
    skipped_no_abbr = 0
    for _, row in df.iterrows():
        event_ticker = make_event_ticker(str(row["date"]), row["away_team"], row["home_team"])
        if event_ticker is None:
            skipped_no_abbr += 1
            continue
        home_abbr = NAME_TO_ABBR.get(row["home_team"], "")
        expected_ticker = f"{event_ticker}-{home_abbr}"
        if expected_ticker in existing:
            continue
        todo.append({
            "date":         str(row["date"])[:10],
            "home_team":    row["home_team"],
            "away_team":    row["away_team"],
            "home_won":     int(row["home_win"]),
            "event_ticker": event_ticker,
            "home_abbr":    home_abbr,
            "away_abbr":    NAME_TO_ABBR.get(row["away_team"], ""),
        })

    if args.dry_run:
        todo = todo[:args.dry_run]

    print(f"Training games total  : {len(df)}")
    print(f"Already in CSV        : {len(existing)}")
    print(f"To fetch              : {len(todo)}")
    if skipped_no_abbr:
        print(f"Skipped (no abbr map) : {skipped_no_abbr}")
    if not todo:
        print("Nothing to do.")
        return

    print(f"Date range            : {todo[0]['date']} → {todo[-1]['date']}")
    print()

    buffer: list[dict] = []
    n_ok, n_skip = 0, 0

    for i, g in enumerate(todo):
        et        = g["event_ticker"]
        home_abbr = g["home_abbr"]
        away_abbr = g["away_abbr"]

        print(f"[{i+1:4}/{len(todo)}] {g['date']}  {away_abbr} @ {home_abbr}  ... ",
              end="", flush=True)

        home_ticker, open_time, close_time, exp_time = get_market_info(et)
        time.sleep(args.delay)

        if not home_ticker or not open_time or not exp_time:
            print("SKIP (no market found)")
            n_skip += 1
            continue

        prices = fetch_pregame_price_candles(
            home_ticker, open_time, exp_time, period=args.period
        )
        time.sleep(args.delay)

        if not prices:
            print("SKIP (no candles)")
            n_skip += 1
            continue

        print(f"open={prices['kalshi_open_price']:.3f}  "
              f"pregame={prices['kalshi_pregame_price']:.3f}  "
              f"candles={prices['n_candles']}")
        n_ok += 1

        buffer.append({
            "date":                      g["date"],
            "home_team":                 g["home_team"],
            "away_team":                 g["away_team"],
            "home_abbr":                 home_abbr,
            "away_abbr":                 away_abbr,
            "home_won":                  g["home_won"],
            "ticker":                    home_ticker,
            "open_time":                 open_time,
            "close_time":                close_time or "",
            "expected_expiration_time":  exp_time,
            "kalshi_open_price":         prices["kalshi_open_price"],
            "kalshi_pregame_price":      prices["kalshi_pregame_price"],
            "n_candles":                 prices["n_candles"],
        })

        if (i + 1) % 25 == 0:
            append_rows(OUT_CSV, buffer)
            buffer = []
            print(f"  ── checkpoint: {n_ok} ok  {n_skip} skipped ──")

    append_rows(OUT_CSV, buffer)

    with open(OUT_CSV) as f:
        total = sum(1 for _ in csv.DictReader(f))
    print(f"\nDone. Fetched {n_ok}, skipped {n_skip}.")
    print(f"Total in {OUT_CSV}: {total} games")


if __name__ == "__main__":
    main()
