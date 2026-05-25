import sqlite3
import os
from datetime import datetime
from backend.config import DATABASE_PATH


def get_conn():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            external_id TEXT UNIQUE,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            scheduled_time TEXT,
            kalshi_ticker TEXT,
            outcome INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            timestamp TEXT DEFAULT (datetime('now')),
            source TEXT NOT NULL,
            home_prob REAL NOT NULL,
            yes_ask REAL,
            FOREIGN KEY (game_id) REFERENCES games(id)
        );

        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            timestamp TEXT DEFAULT (datetime('now')),
            side TEXT NOT NULL,
            size REAL NOT NULL,
            entry_price REAL NOT NULL,
            kalshi_prob REAL NOT NULL,
            sharp_prob REAL NOT NULL,
            edge_estimate REAL NOT NULL,
            kalshi_ticker TEXT,
            closing_price REAL,
            outcome INTEGER,
            pnl REAL,
            clv REAL,
            status TEXT DEFAULT 'open',
            FOREIGN KEY (game_id) REFERENCES games(id)
        );

        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            timestamp TEXT DEFAULT (datetime('now')),
            direction TEXT NOT NULL,
            kalshi_prob REAL NOT NULL,
            sharp_prob REAL NOT NULL,
            edge REAL NOT NULL,
            confidence REAL NOT NULL,
            kelly_fraction REAL,
            suggested_size REAL,
            executed INTEGER DEFAULT 0,
            actual_outcome INTEGER,
            outcome_correct INTEGER,
            FOREIGN KEY (game_id) REFERENCES games(id)
        );

        CREATE TABLE IF NOT EXISTS bot_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            bankroll REAL NOT NULL,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            daily_pnl REAL DEFAULT 0,
            last_run TEXT,
            is_running INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            bankroll REAL NOT NULL,
            daily_pnl REAL NOT NULL,
            trades_count INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            avg_edge REAL
        );

        CREATE TABLE IF NOT EXISTS reddit_posts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            team        TEXT    NOT NULL,
            subreddit   TEXT    NOT NULL,
            post_id     TEXT    UNIQUE NOT NULL,
            author      TEXT,
            title       TEXT,
            sentiment   REAL,
            created_utc INTEGER,
            fetched_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_odds_game ON odds_snapshots(game_id);
        CREATE INDEX IF NOT EXISTS idx_odds_ts ON odds_snapshots(timestamp);
        CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        CREATE INDEX IF NOT EXISTS idx_trades_game ON trades(game_id);
        CREATE INDEX IF NOT EXISTS idx_reddit_team ON reddit_posts(team, created_utc);
    """)
    conn.commit()

    # Ensure bot_state row exists
    row = conn.execute("SELECT id FROM bot_state WHERE id = 1").fetchone()
    if not row:
        from backend.config import INITIAL_BANKROLL
        conn.execute(
            "INSERT INTO bot_state (id, bankroll) VALUES (1, ?)",
            (INITIAL_BANKROLL,),
        )
        conn.commit()
    conn.close()


# ── Game operations ──

def upsert_game(external_id, home_team, away_team, scheduled_time=None, kalshi_ticker=None):
    conn = get_conn()
    conn.execute("""
        INSERT INTO games (external_id, home_team, away_team, scheduled_time, kalshi_ticker)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(external_id) DO UPDATE SET
            kalshi_ticker = COALESCE(excluded.kalshi_ticker, kalshi_ticker),
            scheduled_time = COALESCE(excluded.scheduled_time, scheduled_time)
    """, (external_id, home_team, away_team, scheduled_time, kalshi_ticker))
    conn.commit()
    game_id = conn.execute(
        "SELECT id FROM games WHERE external_id = ?", (external_id,)
    ).fetchone()["id"]
    conn.close()
    return game_id


def get_upcoming_games():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM games WHERE outcome IS NULL ORDER BY scheduled_time"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_game_outcome(game_id, outcome):
    conn = get_conn()
    conn.execute("UPDATE games SET outcome = ? WHERE id = ?", (outcome, game_id))
    conn.commit()
    conn.close()


# ── Odds operations ──

def insert_odds(game_id, source, home_prob, yes_ask=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO odds_snapshots (game_id, source, home_prob, yes_ask) VALUES (?, ?, ?, ?)",
        (game_id, source, home_prob, yes_ask),
    )
    conn.commit()
    conn.close()


def get_kalshi_ask_history(game_id: int, hours: int = 4) -> list[dict]:
    """Return recent (timestamp, yes_bid, yes_ask) rows for a Kalshi market."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT home_prob, yes_ask, timestamp FROM odds_snapshots
        WHERE game_id = ? AND source = 'kalshi' AND yes_ask IS NOT NULL
          AND timestamp >= datetime('now', ?)
        ORDER BY id ASC
    """, (game_id, f"-{hours} hours")).fetchall()
    conn.close()
    return [{"bid": r["home_prob"], "ask": r["yes_ask"], "ts": r["timestamp"]} for r in rows]


def get_latest_odds(game_id):
    """Get most recent odds per source for a game."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT source, home_prob, timestamp
        FROM odds_snapshots
        WHERE game_id = ?
        AND id IN (
            SELECT MAX(id) FROM odds_snapshots WHERE game_id = ? GROUP BY source
        )
    """, (game_id, game_id)).fetchall()
    conn.close()
    return {r["source"]: r["home_prob"] for r in rows}


def get_latest_odds_with_timestamps(game_id):
    """Get most recent odds per source with timestamps for staleness checks."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT source, home_prob, timestamp
        FROM odds_snapshots
        WHERE game_id = ?
        AND id IN (
            SELECT MAX(id) FROM odds_snapshots WHERE game_id = ? GROUP BY source
        )
    """, (game_id, game_id)).fetchall()
    conn.close()
    return {r["source"]: {"prob": r["home_prob"], "timestamp": r["timestamp"]} for r in rows}


# ── Trade operations ──

def insert_trade(game_id, side, size, entry_price, kalshi_prob, sharp_prob, edge, kalshi_ticker=None):
    conn = get_conn()
    conn.execute("""
        INSERT INTO trades (game_id, side, size, entry_price, kalshi_prob, sharp_prob,
                           edge_estimate, kalshi_ticker)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (game_id, side, size, entry_price, kalshi_prob, sharp_prob, edge, kalshi_ticker))
    conn.commit()
    trade_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return trade_id


def get_open_trades():
    conn = get_conn()
    rows = conn.execute("""
        SELECT t.*, g.home_team, g.away_team, g.outcome as game_outcome
        FROM trades t JOIN games g ON t.game_id = g.id
        WHERE t.status = 'open'
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def settle_trade(trade_id, pnl, clv=None, outcome=None):
    conn = get_conn()
    conn.execute("""
        UPDATE trades SET pnl = ?, clv = ?, outcome = ?, status = 'settled'
        WHERE id = ?
    """, (pnl, clv, outcome, trade_id))
    conn.commit()
    conn.close()


def get_recent_trades(limit=50):
    conn = get_conn()
    rows = conn.execute("""
        SELECT t.*, g.home_team, g.away_team
        FROM trades t JOIN games g ON t.game_id = g.id
        ORDER BY t.timestamp DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_settled_trades():
    conn = get_conn()
    rows = conn.execute("""
        SELECT t.*, g.home_team, g.away_team
        FROM trades t JOIN games g ON t.game_id = g.id
        WHERE t.status = 'settled'
        ORDER BY t.timestamp
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Signal operations ──

def insert_signal(game_id, direction, kalshi_prob, sharp_prob, edge, confidence,
                  kelly_fraction=None, suggested_size=None):
    conn = get_conn()
    conn.execute("""
        INSERT INTO signals (game_id, direction, kalshi_prob, sharp_prob, edge,
                           confidence, kelly_fraction, suggested_size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (game_id, direction, kalshi_prob, sharp_prob, edge, confidence,
          kelly_fraction, suggested_size))
    conn.commit()
    conn.close()


# ── Bot state ──

def get_bot_state():
    conn = get_conn()
    row = conn.execute("SELECT * FROM bot_state WHERE id = 1").fetchone()
    conn.close()
    return dict(row) if row else None


def update_bot_state(**kwargs):
    conn = get_conn()
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values())
    conn.execute(f"UPDATE bot_state SET {sets} WHERE id = 1", vals)
    conn.commit()
    conn.close()


def insert_reddit_post(team: str, subreddit: str, post_id: str,
                       author: str, title: str, sentiment: float,
                       created_utc: int) -> bool:
    """Insert a Reddit post. Returns True if inserted, False if duplicate."""
    conn = get_conn()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO reddit_posts
                (team, subreddit, post_id, author, title, sentiment, created_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (team, subreddit, post_id, author, title, sentiment, created_utc))
        conn.commit()
        inserted = conn.execute("SELECT changes()").fetchone()[0] > 0
    finally:
        conn.close()
    return inserted


def get_reddit_posts(team: str, since_utc: int) -> list[dict]:
    """Get Reddit posts for a team after a Unix timestamp."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT author, title, sentiment, created_utc
        FROM reddit_posts
        WHERE team = ? AND created_utc >= ?
        ORDER BY created_utc DESC
    """, (team, since_utc)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_daily_stats():
    state = get_bot_state()
    if not state:
        return
    today = datetime.utcnow().strftime("%Y-%m-%d")
    conn = get_conn()
    trades = conn.execute(
        "SELECT COUNT(*) as cnt, SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, "
        "AVG(edge_estimate) as avg_edge "
        "FROM trades WHERE date(timestamp) = ? AND status = 'settled'",
        (today,),
    ).fetchone()
    conn.execute("""
        INSERT OR REPLACE INTO daily_stats (date, bankroll, daily_pnl, trades_count, wins, avg_edge)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (today, state["bankroll"], state["daily_pnl"],
          trades["cnt"] or 0, trades["wins"] or 0, trades["avg_edge"]))
    conn.commit()
    conn.close()
