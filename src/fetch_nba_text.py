from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


TEAM_ALIASES = {
    "Atlanta Hawks": ["Atlanta", "Hawks"],
    "Boston Celtics": ["Boston", "Celtics"],
    "Brooklyn Nets": ["Brooklyn", "Nets"],
    "Charlotte Hornets": ["Charlotte", "Hornets"],
    "Chicago Bulls": ["Chicago", "Bulls"],
    "Cleveland Cavaliers": ["Cleveland", "Cavaliers", "Cavs"],
    "Dallas Mavericks": ["Dallas", "Mavericks", "Mavs"],
    "Denver Nuggets": ["Denver", "Nuggets"],
    "Detroit Pistons": ["Detroit", "Pistons"],
    "Golden State Warriors": ["Golden State", "Warriors"],
    "Houston Rockets": ["Houston", "Rockets"],
    "Indiana Pacers": ["Indiana", "Pacers"],
    "LA Clippers": ["LA Clippers", "Los Angeles Clippers", "Clippers"],
    "Los Angeles Clippers": ["LA Clippers", "Los Angeles Clippers", "Clippers"],
    "Los Angeles Lakers": ["LA Lakers", "Los Angeles Lakers", "Lakers"],
    "Memphis Grizzlies": ["Memphis", "Grizzlies"],
    "Miami Heat": ["Miami", "Heat"],
    "Milwaukee Bucks": ["Milwaukee", "Bucks"],
    "Minnesota Timberwolves": ["Minnesota", "Timberwolves", "Wolves"],
    "New Orleans Pelicans": ["New Orleans", "Pelicans"],
    "New York Knicks": ["New York", "Knicks"],
    "Oklahoma City Thunder": ["Oklahoma City", "Thunder", "OKC"],
    "Orlando Magic": ["Orlando", "Magic"],
    "Philadelphia 76ers": ["Philadelphia", "76ers", "Sixers"],
    "Phoenix Suns": ["Phoenix", "Suns"],
    "Portland Trail Blazers": ["Portland", "Trail Blazers", "Blazers"],
    "Sacramento Kings": ["Sacramento", "Kings"],
    "San Antonio Spurs": ["San Antonio", "Spurs"],
    "Toronto Raptors": ["Toronto", "Raptors"],
    "Utah Jazz": ["Utah", "Jazz"],
    "Washington Wizards": ["Washington", "Wizards"],
}


def build_gdelt_url(query: str, start: str, end: str, maxrecords: int = 75) -> str:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "startdatetime": start,
        "enddatetime": end,
        "maxrecords": maxrecords,
        "sort": "datedesc",
    }
    return "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)


def team_query_group(team_name: str) -> str:
    aliases = TEAM_ALIASES.get(team_name, [team_name])
    # GDELT only allows parentheses around OR groups.
    return "( " + " OR ".join(f'"{token}"' for token in aliases) + " )"


def build_queries(away: str, home: str, include_espn: bool) -> list[str]:
    away_group = team_query_group(away)
    home_group = team_query_group(home)
    league_group = '( NBA OR "National Basketball Association" OR basketball )'
    base = f"{away_group} AND {home_group} AND {league_group}"
    queries = [base]
    if include_espn:
        queries.append(f"{base} AND domain:espn.com")
    return queries


def fetch_articles(url: str, timeout: int = 12) -> tuple[list[dict], str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        return payload.get("articles", []) or [], None
    except Exception as exc:
        # Some environments can resolve GDELT but fail TLS handshake on HTTPS.
        if url.startswith("https://"):
            fallback_url = "http://" + url[len("https://") :]
            try:
                with urllib.request.urlopen(fallback_url, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                return payload.get("articles", []) or [], f"https_failed_used_http: {exc}"
            except Exception as fallback_exc:
                return [], f"{exc} | http_fallback_failed: {fallback_exc}"
        return [], str(exc)


def dedupe_articles(articles: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for article in articles:
        key = str(article.get("url") or article.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(article)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch NBA-related GDELT text per game")
    ap.add_argument("--games_csv", default="data/processed/games_api.csv")
    ap.add_argument("--out_jsonl", default="data/raw/nba_gdelt_articles.jsonl")
    ap.add_argument("--window_days", type=int, default=5)
    ap.add_argument("--limit_games", type=int, default=0, help="0 means all games")
    ap.add_argument("--maxrecords", type=int, default=75)
    ap.add_argument("--include_espn", action="store_true")
    ap.set_defaults(include_espn=True)
    ap.add_argument("--request_timeout", type=int, default=12)
    ap.add_argument("--progress_every", type=int, default=25)
    ap.add_argument("--sleep", type=float, default=0.4, help="Seconds between API requests")
    args = ap.parse_args()

    games_path = Path(args.games_csv)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    games = pd.read_csv(games_path, parse_dates=["date"]).sort_values(["date", "game_id"])
    if args.limit_games > 0:
        games = games.head(args.limit_games)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    total = len(games)
    with tmp_path.open("w", encoding="utf-8") as f:
        for idx, (_, row) in enumerate(games.iterrows(), start=1):
            game_id = int(row["game_id"])
            away = str(row["away_team"])
            home = str(row["home_team"])
            game_date = pd.to_datetime(row["date"]).to_pydatetime()

            start_dt = (game_date - pd.Timedelta(days=args.window_days)).strftime("%Y%m%d000000")
            end_dt = game_date.strftime("%Y%m%d000000")

            urls = []
            articles = []
            errors = []
            for query in build_queries(away=away, home=home, include_espn=args.include_espn):
                url = build_gdelt_url(query=query, start=start_dt, end=end_dt, maxrecords=args.maxrecords)
                urls.append(url)
                batch, err = fetch_articles(url, timeout=args.request_timeout)
                articles.extend(batch)
                if err:
                    errors.append(err)
                time.sleep(args.sleep)

            articles = dedupe_articles(articles)

            record = {
                "game_id": game_id,
                "away_team": away,
                "home_team": home,
                "game_date": game_date.strftime("%Y-%m-%d"),
                "window_days": args.window_days,
                "gdelt_urls": urls,
                "errors": errors,
                "articles": articles,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if idx % max(1, args.progress_every) == 0 or idx == total:
                print(f"[{idx}/{total}] game_id={game_id} articles={len(articles)} errors={len(errors)}")

    tmp_path.replace(out_path)

    print(f"Wrote -> {out_path}")


if __name__ == "__main__":
    main()
