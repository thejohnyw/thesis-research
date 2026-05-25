"""
Sentiment scoring for Reddit posts.

Reads the raw JSONL (from fetch_reddit_bulk.py), adds a `sentiment` float in
[-1, 1] to every post using a RoBERTa-based Twitter sentiment model, and
writes a new JSONL file.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest
  Labels: negative / neutral / positive
  Output: negative → −score, neutral → 0, positive → +score

Usage:
    python -m src.sentiment
    python -m src.sentiment --input data/raw/reddit_team_posts_bulk.jsonl \
                            --output data/processed/reddit_with_sentiment.jsonl \
                            --batch 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_IN  = Path("data/raw/reddit_team_posts_bulk.jsonl")
DEFAULT_OUT = Path("data/processed/reddit_with_sentiment.jsonl")


def _label_to_float(label: str, score: float) -> float:
    label = label.lower()
    if "positive" in label:
        return score
    if "negative" in label:
        return -score
    return 0.0


def load_pipeline(device: str | int = "cpu"):
    import torch
    from transformers import pipeline

    # Auto-select best available device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = 0
        else:
            device = "cpu"

    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        max_length=512,
        truncation=True,
        device=device,
    )


def score_posts(
    input_path: Path = DEFAULT_IN,
    output_path: Path = DEFAULT_OUT,
    batch_size: int = 128,
    device: str | int = "auto",
) -> int:
    """Add sentiment scores to all posts. Returns number of posts processed."""
    print(f"Loading model {MODEL_NAME} ...")
    pipe = load_pipeline(device=device)

    posts = []
    with input_path.open() as f:
        for line in f:
            posts.append(json.loads(line))
    print(f"Loaded {len(posts)} posts from {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    texts = []
    for p in posts:
        raw = p.get("title", "")
        body = p.get("selftext", "")
        if body and body not in ("[deleted]", "[removed]"):
            raw = raw + " " + body[:400]
        texts.append(raw[:512])

    print(f"Scoring {len(texts)} posts (batch_size={batch_size}) ...")
    try:
        from tqdm import tqdm
        iterator = tqdm(range(0, len(texts), batch_size), unit="batch")
    except ImportError:
        iterator = range(0, len(texts), batch_size)

    results = []
    for start in iterator:
        batch = texts[start : start + batch_size]
        try:
            preds = pipe(batch)
        except Exception:
            preds = [{"label": "neutral", "score": 0.0}] * len(batch)
        results.extend(preds)

    with output_path.open("w", encoding="utf-8") as f:
        for post, pred in zip(posts, results):
            post["sentiment"] = round(_label_to_float(pred["label"], pred["score"]), 4)
            post["sentiment_label"] = pred["label"]
            f.write(json.dumps(post, ensure_ascii=False) + "\n")

    print(f"Done. Saved to {output_path}")
    return len(posts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=str(DEFAULT_IN))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--device", default="auto",
                    help="Device: auto (default), mps, cpu, 0 (cuda)")
    args = ap.parse_args()
    device = args.device
    try:
        device = int(device)
    except (ValueError, TypeError):
        pass
    score_posts(Path(args.input), Path(args.output), args.batch, device)


if __name__ == "__main__":
    main()
