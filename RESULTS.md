# NBA Game Prediction – Baseline Results

## Experimental Setup

- **Dataset:** 1,309 NBA regular-season games (2024–25 season)
- **Text coverage:** 405 games have pre-game news articles (GDELT, Oct–Dec 2024)
- **Text model:** `sentence-transformers/all-MiniLM-L6-v2` (frozen, 384-dim)
- **Evaluation:** Time-based expanding-window backtest, 5 splits, 20% test size
- **Classifier:** Logistic Regression (Baseline 2) / MLP (Baseline 1, 3)

---

## Baseline Comparison

| Model | Features | Dimensions | Accuracy | ROC AUC | Log Loss |
|---|---|---|---|---|---|
| **Baseline 1** – Structured MLP | Team form & history stats (win rate, points avg, streak, rest days, etc.) | ~25 | 0.593 ± 0.064 | 0.663 ± 0.043 | 0.660 ± 0.041 |
| **Baseline 2** – Text-only BERT + LR | Frozen sentence embeddings from pre-game GDELT news articles | 384 | 0.439 ± 0.047 | 0.391 ± 0.029 | 1.680 ± 0.166 |
| **Baseline 3** – Fusion (Structured + Text) | PCA-compressed structured features + text embeddings, concatenated | 32 + 384 = 416 | 0.485 ± 0.028 | 0.456 ± 0.028 | 2.110 ± 0.115 |

---

## Notes

- **Baseline 1** uses all 1,309 games with full structured feature coverage.
- **Baselines 2 & 3** are restricted to the 325 games that have both structured features and matched GDELT news articles (`--require_text`).
- Text embeddings add noise rather than signal at this data scale: 384-dimensional logistic regression with ~80–160 training samples per split is heavily underdetermined.
- The fusion model (B3) performs worse than the structured-only model (B1), suggesting text signal is not yet strong enough to complement structured features.
- **Alternative text sources tested:** Google News RSS (near-zero historical coverage for Oct 2024 data), Guardian Open Platform API (100% coverage but returned generic league-wide news, not game-specific previews).

---

## Pipeline

```
fetch_games_api.py          →  data/processed/games_api.csv
prepare_games.py            →  data/processed/games_api.csv (cleaned)
feature_engineering.py      →  data/processed/features.csv
[GDELT data: data/raw/nba_gdelt_articles_enriched.jsonl]

train_baseline1.py          →  data/processed/baseline1_results.csv
train_baseline2.py          →  data/processed/baseline2_results_first405.csv
train_baseline3.py          →  data/processed/baseline3_results_first405.csv
```
