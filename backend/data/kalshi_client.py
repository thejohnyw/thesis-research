"""
Kalshi API client — RSA-PSS signed auth, NBA game markets.
"""
import httpx
import os
import time
import base64
from dataclasses import dataclass
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from backend.config import KALSHI_KEY_ID, KALSHI_KEY_PATH


@dataclass
class Market:
    ticker: str
    title: str
    yes_price: float   # yes_bid_dollars (implied prob)
    yes_ask: float      # yes_ask_dollars
    volume: float
    close_time: str


class KalshiClient:
    BASE = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self):
        if not KALSHI_KEY_ID:
            raise ValueError("Set KALSHI_KEY_ID in .env")
        with open(KALSHI_KEY_PATH, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)

    def _sign(self, method: str, path: str) -> dict:
        ts = int(time.time() * 1000)
        msg = f"{ts}{method}{path}".encode()
        sig = self.private_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
        }

    def get_nba_markets(self) -> list[Market]:
        path = "/trade-api/v2/markets"
        r = httpx.get(
            f"{self.BASE}/markets",
            params={"series_ticker": "KXNBAGAME", "status": "open", "limit": "200"},
            headers=self._sign("GET", path),
        )
        r.raise_for_status()

        markets = []
        for m in r.json().get("markets", []):
            markets.append(Market(
                ticker=m["ticker"],
                title=m["title"],
                yes_price=float(m.get("yes_bid_dollars", "0")),
                yes_ask=float(m.get("yes_ask_dollars", "0")),
                volume=float(m.get("volume_fp", "0")),
                close_time=m.get("close_time", ""),
            ))
        return markets

    def get_orderbook(self, ticker: str) -> dict:
        """
        Fetch order book depth for a market.
        Returns {"yes": [[price_cents, size], ...], "no": [[price_cents, size], ...]}
        Prices are in cents (1–99). Lists are sorted best-price-first.
        """
        path = f"/trade-api/v2/markets/{ticker}/orderbook"
        r = httpx.get(
            f"{self.BASE}/markets/{ticker}/orderbook",
            headers=self._sign("GET", path),
        )
        r.raise_for_status()
        return r.json().get("orderbook", {"yes": [], "no": []})

    def get_market(self, ticker: str) -> dict:
        path = f"/trade-api/v2/markets/{ticker}"
        r = httpx.get(
            f"{self.BASE}/markets/{ticker}",
            headers=self._sign("GET", path),
        )
        r.raise_for_status()
        return r.json().get("market", {})

    def place_order(self, ticker: str, side: str, size: int, price: int) -> dict:
        """
        Place a limit order on Kalshi.
        side: 'yes' or 'no'
        size: number of contracts
        price: price in cents (1-99)
        """
        path = "/trade-api/v2/portfolio/orders"
        r = httpx.post(
            f"{self.BASE}/portfolio/orders",
            headers=self._sign("POST", path),
            json={
                "ticker": ticker,
                "action": "buy",
                "side": side,
                "type": "limit",
                "count": size,
                "yes_price": price if side == "yes" else None,
                "no_price": price if side == "no" else None,
            },
        )
        r.raise_for_status()
        return r.json()
