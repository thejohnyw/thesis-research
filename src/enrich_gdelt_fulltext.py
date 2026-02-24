"""
Try to enrich GDELT article records with full-text body/summary from article URLs.

Writes a new JSONL file with `body_text` and `summary` fields per article when available.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import urllib.request
from html import unescape
from pathlib import Path


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
JSONLD_RE = re.compile(r"<script[^>]+type=[\"']application/ld\\+json[\"'][^>]*>(.*?)</script>", re.I | re.S)
META_DESC_RE = re.compile(
    r'<meta[^>]+(?:name=["\']description["\']|property=["\']og:description["\'])[^>]+content=["\'](.*?)["\']',
    re.I | re.S,
)
PARA_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.I | re.S)


def clean_text(value: str) -> str:
    text = TAG_RE.sub(" ", value)
    text = unescape(text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def parse_jsonld_body(html: str) -> str:
    for block in JSONLD_RE.findall(html):
        raw = block.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("articleBody", "text"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return clean_text(value)
    return ""


def parse_meta_summary(html: str) -> str:
    m = META_DESC_RE.search(html)
    if not m:
        return ""
    return clean_text(m.group(1))


def parse_paragraph_text(html: str) -> str:
    chunks: list[str] = []
    for raw in PARA_RE.findall(html):
        txt = clean_text(raw)
        if len(txt) >= 40:
            chunks.append(txt)
        if len(chunks) >= 12:
            break
    return " ".join(chunks).strip()


def fetch_html(url: str, timeout: int) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def enrich_article(article: dict, timeout: int) -> dict:
    out = dict(article)
    url = str(article.get("url", "") or "").strip()
    if not url:
        return out
    try:
        html = fetch_html(url, timeout=timeout)
    except Exception:
        return out

    body = parse_jsonld_body(html)
    if not body:
        body = parse_paragraph_text(html)
    summary = parse_meta_summary(html)
    if body:
        out["body_text"] = body
    if summary:
        out["summary"] = summary
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich GDELT JSONL with full article text/summary")
    parser.add_argument("--in_jsonl", default="data/raw/nba_gdelt_articles.jsonl")
    parser.add_argument("--out_jsonl", default="data/raw/nba_gdelt_articles_enriched.jsonl")
    parser.add_argument("--limit_games", type=int, default=0, help="0 means all games")
    parser.add_argument("--max_articles_per_game", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--progress_every", type=int, default=25)
    args = parser.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    enriched_count = 0
    game_count = 0
    total_articles = 0

    with in_path.open("r", encoding="utf-8", errors="replace") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                record = json.loads(s)
            except json.JSONDecodeError:
                continue
            game_count += 1
            if args.limit_games > 0 and game_count > args.limit_games:
                break

            articles = record.get("articles", []) or []
            updated = list(articles)
            job_idx: list[int] = []
            jobs: list[dict] = []
            for idx, article in enumerate(articles):
                total_articles += 1
                if idx >= args.max_articles_per_game:
                    continue
                if article.get("body_text") or article.get("summary"):
                    continue
                job_idx.append(idx)
                jobs.append(article)

            if jobs:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                    futs = [ex.submit(enrich_article, article, args.timeout) for article in jobs]
                    for idx, fut in zip(job_idx, futs):
                        try:
                            new_article = fut.result()
                        except Exception:
                            new_article = articles[idx]
                        if new_article.get("body_text") or new_article.get("summary"):
                            enriched_count += 1
                        updated[idx] = new_article

            record["articles"] = updated
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            if game_count % max(1, args.progress_every) == 0:
                print(
                    f"Processed {game_count} games, articles={total_articles}, enriched={enriched_count}"
                )

    print(f"Wrote -> {out_path}")
    print(f"Games processed: {game_count}")
    print(f"Articles processed: {total_articles}")
    print(f"Articles enriched with body/summary: {enriched_count}")


if __name__ == "__main__":
    main()
