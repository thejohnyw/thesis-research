import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──
KALSHI_KEY_ID = os.getenv("KALSHI_KEY_ID")
KALSHI_KEY_PATH = os.getenv("KALSHI_KEY_PATH", "kalshi_key.pem")

# ── Trading ──
MIN_EDGE = 0.03                # 3% minimum edge to trade
KELLY_FRACTION = 0.25          # quarter Kelly
MAX_POSITION_PCT = 0.05        # 5% max of bankroll per trade
MAX_POSITION_DOLLARS = 100     # hard cap per trade
DAILY_LOSS_LIMIT = 200         # stop trading after $200 daily loss
MAX_TRADES_PER_SCAN = 3        # max trades per scan cycle
MAX_PENDING_TRADES = 20        # max open positions
KALSHI_FEE = 0.07              # 7% fee on profit

# ── Scheduling ──
COLLECT_INTERVAL_SECONDS = 300   # 5 min  (Kalshi price refresh)
SCAN_INTERVAL_SECONDS = 300      # 5 min  (signal check)
SETTLE_INTERVAL_SECONDS = 120    # 2 min

# NBA game window (UTC) — roughly 6pm-1am ET = 22:00-05:00 UTC
GAME_WINDOW_START_UTC = 22   # hour
GAME_WINDOW_END_UTC = 5      # hour (next day)

# ── Mode ──
PAPER_TRADING = True
INITIAL_BANKROLL = 1000.0

# ── Database ──
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trading.db")

# ── Sharp books (in priority order) ──
# Pinnacle is the gold standard but isn't always available in the API.
# Fall back through increasingly soft books.
SHARP_BOOKS = ["pinnacle", "lowvig", "betonlineag", "fanduel", "draftkings", "betrivers"]

# ── Team abbreviation mapping ──
TEAM_ABBREV = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}
TEAM_TO_ABBREV = {v: k for k, v in TEAM_ABBREV.items()}
