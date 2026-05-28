"""
Background scheduler — periodic Kalshi price collection, edge scanning, and settlement.

Collection and scanning both run every 5 minutes during the NBA game window
(~6pm–1am ET). Settlement runs every 2 minutes. Reddit posts are collected
every 4 hours. Pre-game M3 predictions run daily at 16:00 UTC (noon ET).
"""
import logging
from datetime import datetime, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from backend.config import (
    SCAN_INTERVAL_SECONDS, COLLECT_INTERVAL_SECONDS, SETTLE_INTERVAL_SECONDS,
    GAME_WINDOW_START_UTC, GAME_WINDOW_END_UTC,
)
from backend.core.signals import collect_odds, scan_and_trade
from backend.core.settlement import settle_completed
from backend.models.database import save_daily_stats, update_bot_state, get_bot_state
from backend.data.reddit_collector import collect_reddit_posts

log = logging.getLogger(__name__)

scheduler = BackgroundScheduler()


def _in_game_window() -> bool:
    """Check if current time is within the NBA game window."""
    hour = datetime.now(timezone.utc).hour
    if GAME_WINDOW_START_UTC > GAME_WINDOW_END_UTC:
        # Window spans midnight (e.g., 22:00 - 05:00)
        return hour >= GAME_WINDOW_START_UTC or hour < GAME_WINDOW_END_UTC
    return GAME_WINDOW_START_UTC <= hour < GAME_WINDOW_END_UTC


def collect_job():
    if not _in_game_window():
        log.debug("Outside game window, skipping collection")
        return
    try:
        n = collect_odds()
        log.info(f"Collected odds for {n} games")
    except Exception as e:
        log.error(f"Collection failed: {e}")


def scan_job():
    try:
        trades = scan_and_trade()
        if trades:
            log.info(f"Executed {len(trades)} trades")
    except Exception as e:
        log.error(f"Scan failed: {e}")


def settle_job():
    try:
        settled = settle_completed()
        if settled:
            for s in settled:
                log.info(f"Settled: {s['game']} → {s['outcome']} PnL=${s['pnl']:+.2f} CLV={s['clv']:+.4f}")
    except Exception as e:
        log.error(f"Settlement failed: {e}")


def heartbeat_job():
    state = get_bot_state()
    if state:
        log.info(
            f"Heartbeat: bankroll=${state['bankroll']:.2f} "
            f"daily_pnl=${state['daily_pnl']:+.2f} "
            f"trades={state['total_trades']} "
            f"win_rate={state['winning_trades']/max(1,state['total_trades']):.0%}"
        )


def reddit_job():
    try:
        n = collect_reddit_posts()
        log.info(f"Collected {n} new Reddit posts")
    except Exception as e:
        log.error(f"Reddit collection failed: {e}")


def pregame_job():
    """Compute M3 model probabilities for today's games. Runs at noon ET (16:00 UTC)."""
    try:
        from backend.data.pregame_features import run_and_save
        run_and_save()
        log.info("Pre-game M3 predictions computed")
    except Exception as e:
        log.error(f"Pregame prediction failed: {e}")


def daily_reset_job():
    save_daily_stats()
    update_bot_state(daily_pnl=0)
    log.info("Daily stats saved, PnL reset")


def start_scheduler():
    scheduler.add_job(collect_job, "interval", seconds=COLLECT_INTERVAL_SECONDS, id="collect")
    scheduler.add_job(scan_job, "interval", seconds=SCAN_INTERVAL_SECONDS, id="scan")
    scheduler.add_job(settle_job, "interval", seconds=SETTLE_INTERVAL_SECONDS, id="settle")
    scheduler.add_job(heartbeat_job, "interval", seconds=60, id="heartbeat")
    scheduler.add_job(reddit_job, "interval", hours=4, id="reddit")
    scheduler.add_job(pregame_job, "cron", hour=16, minute=0, id="pregame")
    scheduler.add_job(daily_reset_job, "cron", hour=0, id="daily_reset")
    scheduler.start()
    log.info(
        f"Scheduler started (collect every {COLLECT_INTERVAL_SECONDS}s during game window, "
        f"scan every {SCAN_INTERVAL_SECONDS}s)"
    )


def stop_scheduler():
    scheduler.shutdown(wait=False)
    log.info("Scheduler stopped")
