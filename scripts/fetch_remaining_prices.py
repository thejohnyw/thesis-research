"""
Fetch pre-game Kalshi prices for all settled games not yet in the CSV.
Uses /historical/trades?ticker= (unauthenticated, works for all settled markets).
"""
import json
import csv
import requests
import time
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE = "https://api.elections.kalshi.com/trade-api/v2"
CSV_PATH = "data/kalshi_historical_prices.csv"
SETTLED_PATH = "data/kalshi_settled_games.json"

HEADERS_OUT = [
    "date", "home_team", "away_team", "home_abbr", "away_abbr", "home_won",
    "ticker", "open_time", "close_time",
    "kalshi_open_price", "kalshi_pregame_price", "n_candles", "market_open", "n_trades"
]


def fetch_pregame_price(ticker: str, close_time_str: str) -> dict | None:
    """Fetch opening and pre-game YES price from historical trades."""
    close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
    target_dt = close_dt - timedelta(hours=4)
    target_str = target_dt.strftime("%Y-%m-%dT%H:%M")

    all_trades = []
    cursor = None
    for page in range(30):
        params = {"ticker": ticker, "limit": 100}
        if cursor:
            params["cursor"] = cursor
        for attempt in range(4):
            try:
                r = requests.get(f"{BASE}/historical/trades", params=params, timeout=15)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 3:
                    print(f"  Error fetching {ticker}: {e}")
                    return None
                time.sleep(1)

        data = r.json()
        trades = data.get("trades", [])
        if not trades:
            break
        all_trades.extend(trades)

        # Trades are newest-first; stop once oldest in batch is before target
        oldest = trades[-1]["created_time"][:16]
        if oldest <= target_str:
            break
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.4)

    if not all_trades:
        return None

    all_trades.sort(key=lambda t: t["created_time"])
    pre = [t for t in all_trades if t["created_time"][:16] <= target_str]

    opening_yes = round(1 - float(all_trades[0]["no_price_dollars"]), 2)
    pregame_yes = round(1 - float(pre[-1]["no_price_dollars"]), 2) if pre else opening_yes

    return {
        "kalshi_open_price": opening_yes,
        "kalshi_pregame_price": pregame_yes,
        "n_trades": len(all_trades),
    }


def main():
    with open(SETTLED_PATH) as f:
        all_games = json.load(f)

    existing = set()
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            existing.add(row["ticker"])

    to_fetch = [g for g in all_games if g["ticker"] not in existing]
    print(f"Fetching prices for {len(to_fetch)} games ({to_fetch[0]['date']} → {to_fetch[-1]['date']})")

    results = []
    for i, g in enumerate(to_fetch):
        ticker = g["ticker"]
        print(f"  [{i+1}/{len(to_fetch)}] {g['date']} {g.get('home_abbr','?')} vs {g.get('away_abbr','?')} ... ", end="", flush=True)

        prices = fetch_pregame_price(ticker, g["close_time"])
        if prices:
            print(f"open={prices['kalshi_open_price']:.2f} pregame={prices['kalshi_pregame_price']:.2f} trades={prices['n_trades']}")
            results.append({
                "date": g["date"],
                "home_team": g.get("home_team", ""),
                "away_team": g.get("away_team", ""),
                "home_abbr": g.get("home_abbr", ""),
                "away_abbr": g.get("away_abbr", ""),
                "home_won": g.get("home_won", ""),
                "ticker": ticker,
                "open_time": g.get("open_time", ""),
                "close_time": g.get("close_time", ""),
                "kalshi_open_price": prices["kalshi_open_price"],
                "kalshi_pregame_price": prices["kalshi_pregame_price"],
                "n_candles": "",
                "market_open": "",
                "n_trades": prices["n_trades"],
            })
        else:
            print("SKIP (no trades found)")

        time.sleep(0.5)

        # Write checkpoint every 20 games
        if (i + 1) % 20 == 0:
            _append_to_csv(results)
            results = []
            print(f"  -- checkpoint saved --")

    if results:
        _append_to_csv(results)

    with open(CSV_PATH) as f:
        total = sum(1 for _ in csv.DictReader(f))
    print(f"\nDone. Total real Kalshi prices: {total} games")


def _append_to_csv(rows: list[dict]):
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS_OUT)
        for row in rows:
            w.writerow(row)


if __name__ == "__main__":
    main()
