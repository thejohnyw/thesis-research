"""
FastAPI server — dashboard API + bot control.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.models.database import init_db, get_bot_state, get_recent_trades, get_settled_trades
from backend.core.edge import find_edges
from backend.core.signals import collect_odds, scan_and_trade
from backend.core.settlement import settle_completed
from backend.core.scheduler import start_scheduler, stop_scheduler, scheduler
from backend.core.risk import RiskManager
from backend.config import PAPER_TRADING, MIN_EDGE
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logging.info("Database initialized")
    yield
    if scheduler.running:
        stop_scheduler()


app = FastAPI(title="Kalshi NBA Trading Bot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/dashboard")
def dashboard():
    """All dashboard data in one call."""
    state = get_bot_state()
    edges = find_edges()
    trades = get_recent_trades(20)
    return {
        "mode": "PAPER" if PAPER_TRADING else "LIVE",
        "bankroll": state["bankroll"] if state else 0,
        "daily_pnl": state["daily_pnl"] if state else 0,
        "total_pnl": state["total_pnl"] if state else 0,
        "total_trades": state["total_trades"] if state else 0,
        "scheduler_running": scheduler.running,
        "edges": [
            {
                "game": f"{e.away_team} @ {e.home_team}",
                "kalshi": f"{e.kalshi_prob:.0%}",
                "sharp": f"{e.sharp_prob:.0%}",
                "edge": f"{e.edge:+.1%}",
                "side": e.side,
                "confidence": f"{e.confidence:.0%}",
            }
            for e in edges
        ],
        "recent_trades": trades,
    }


@app.get("/api/edges")
def get_edges(min_edge: float = None):
    """Current trading opportunities."""
    edges = find_edges(min_edge=min_edge or MIN_EDGE)
    return [
        {
            "game_id": e.game_id,
            "game": f"{e.away_team} @ {e.home_team}",
            "kalshi_prob": e.kalshi_prob,
            "sharp_prob": e.sharp_prob,
            "edge": e.edge,
            "side": e.side,
            "confidence": e.confidence,
            "ticker": e.kalshi_ticker,
        }
        for e in edges
    ]


@app.get("/api/trades")
def get_trades(limit: int = 50):
    """Trade history."""
    return get_recent_trades(limit)


@app.get("/api/stats")
def get_stats():
    """Performance metrics."""
    trades = get_settled_trades()
    if not trades:
        return {"total_trades": 0, "message": "No settled trades yet"}

    pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
    wins = [t for t in trades if t["pnl"] and t["pnl"] > 0]
    clvs = [t["clv"] for t in trades if t["clv"] is not None]

    sharpe = 0
    if pnls and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)

    return {
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0,
        "total_pnl": sum(pnls),
        "avg_pnl": np.mean(pnls) if pnls else 0,
        "sharpe": round(sharpe, 2),
        "avg_clv": np.mean(clvs) if clvs else 0,
        "avg_edge": np.mean([t["edge_estimate"] for t in trades if t["edge_estimate"]]),
    }


@app.post("/api/collect")
def trigger_collect():
    """Manual odds collection."""
    n = collect_odds()
    return {"status": "ok", "games": n}


@app.post("/api/scan")
def trigger_scan():
    """Manual scan + trade."""
    trades = scan_and_trade()
    return {"status": "ok", "trades": trades}


@app.post("/api/settle")
def trigger_settle():
    """Manual settlement."""
    settled = settle_completed()
    return {"status": "ok", "settled": settled}


@app.post("/api/bot/start")
def start_bot():
    """Start the background scheduler."""
    if not scheduler.running:
        start_scheduler()
    return {"status": "running"}


@app.post("/api/bot/stop")
def stop_bot():
    """Stop the background scheduler."""
    if scheduler.running:
        stop_scheduler()
    return {"status": "stopped"}


@app.get("/api/health")
def health():
    return {"status": "ok", "scheduler": scheduler.running}
