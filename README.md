# Baseline Reproduction & Audit — Week 6
**NBA Game Outcome Prediction Using Structured and Text Features**

---

## 1. Overview
All models are evaluated under a **time-based expanding-window backtest** to respect the temporal structure of sports data: each test fold is always strictly after all training data, preventing any look-ahead bias. Five chronological splits are used with 20% test size per split.

The three baselines span a spectrum of feature modalities:

| | Baseline 1 | Baseline 2 | Baseline 3 |
|---|---|---|---|
| **Model** | MLP | Logistic Regression | MLP |
| **Features** | Structured stats | BERT text embeddings | Structured + text (fusion) |
| **Games used** | 1,309 | 325 (text-covered) | 325 (text-covered) |
| **Input dim** | ~25 | 384 | 32 + 384 = 416 |

---

## 2. Experimental Setup

**Dataset.** 1,309 NBA regular-season games from the 2024–25 season (October 2024 – June 2025), sourced from the official NBA API. Labels are binary: 1 = home win, 0 = away win (historical home-win rate ≈ 56%).

**Structured features (Baseline 1).** Engineered per-team rolling statistics computed strictly before each game's tip-off, including: win rate (last 10 games), average points scored and allowed, point differential, win streak, back-to-back flag, rest days, season win percentage, and home/away splits. All features are lag-safe by construction.

**Text features (Baselines 2 & 3).** Pre-game news articles fetched from the GDELT Global Knowledge Graph and enriched with full article text. Articles are restricted to a 5-day window before each game. Sentence embeddings are produced by `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, frozen weights). Multiple articles per game are mean-pooled. GDELT coverage spans October–December 2024 only, yielding **325 of 1,309 games** (24.8%) with at least one matched article.

**Fusion (Baseline 3).** Structured features are first PCA-compressed to 32 dimensions, then concatenated with the 384-dimensional text embedding, giving a 416-dimensional input to an MLP classifier.

**Evaluation protocol.** All splits use an expanding training window starting from October 22, 2024. The minimum training set is 100 games (Baseline 1) or scaled proportionally for Baselines 2/3 given the smaller text-covered set. Metrics reported: accuracy, ROC AUC, log loss, MSE, and calibration error, averaged across all valid splits.

---

## 3. Results

### 3.1 Aggregate Metrics

| Model | Accuracy | ROC AUC | Log Loss | MSE | Cal. Error |
|---|---|---|---|---|---|
| **Baseline 1** – Structured MLP | 0.593 ± 0.064 | **0.663 ± 0.043** | 0.660 ± 0.041 | 0.230 ± 0.027 | 0.074 ± 0.044 |
| **Baseline 2** – Text-only BERT + LR | 0.439 ± 0.047 | 0.391 ± 0.029 | 1.680 ± 0.166 | 0.432 ± 0.006 | 0.400 ± 0.012 |
| **Baseline 3** – Fusion MLP | 0.485 ± 0.028 | 0.456 ± 0.028 | 2.110 ± 0.115 | 0.430 ± 0.029 | 0.430 ± 0.020 |

### 3.2 Per-Split Detail: Baseline 1

| Split | Train Games | Test Games | Accuracy | ROC AUC |
|---|---|---|---|---|
| 1 | 261 | 262 | 0.500 | 0.601 |
| 2 | 523 | 262 | 0.622 | 0.670 |
| 3 | 785 | 262 | 0.645 | 0.688 |
| 4 | 1,047 | 262 | 0.603 | 0.694 |

Baseline 1 improves monotonically in ROC AUC (0.60 → 0.69) as the training window grows — a healthy learning curve indicating that the structured features carry genuine predictive signal.

### 3.3 Per-Split Detail: Baselines 2 & 3 (text-covered games only)

| Split | Train | Test | B2 ROC AUC | B3 ROC AUC |
|---|---|---|---|---|
| 1 | 129 | 66 | 0.360 | 0.446 |
| 2 | 195 | 65 | 0.398 | 0.435 |
| 3 | 260 | 65 | 0.416 | 0.488 |

Both text-based models remain below the 0.5 chance line throughout. No consistent improvement trend is visible over the three available splits.

---

## 4. Error Analysis

**4.1 Baseline 1: calibration drift in early splits.**
Split 1 (smallest training set, 261 games) yields accuracy exactly at 50% — the model defaults to predicting the majority class. Calibration error (0.149) is elevated, reflecting poorly tuned probability estimates on a small sample. From Split 2 onward the model calibrates and accuracy and ROC AUC rise consistently. The primary remaining errors cluster around back-to-back games and games involving closely matched opponents, where rolling statistics alone provide insufficient discriminating power.

**4.2 Baselines 2 & 3: sub-random discrimination.**
The ROC AUCs of 0.391 and 0.456 are *below* 0.5, meaning these models systematically predict the wrong class more often than chance. Three compounding causes:

1. **Dimensionality–sample mismatch.** Logistic regression on 384-dimensional inputs with 129–260 training games is severely underdetermined (~1.5–2 parameters per sample). The model overfits to noise in the embedding space, and regularisation cannot fully compensate.

2. **Overconfident incorrect predictions.** Log losses of 1.68 (B2) and 2.11 (B3) far exceed the 0.693 log loss of a constant-probability predictor. This indicates the model assigns high confidence to incorrect predictions rather than hedging — a symptom of overfitting to non-predictive embedding dimensions.

3. **Text content does not encode game outcome.** Inspection of fetched articles reveals sports desk roundups, injury updates, and team standings — useful contextual information but not strongly predictive of a specific binary outcome on a given night. The mean-pooled embedding averages away any game-specific signal.

**4.3 Fusion makes results worse.**
Baseline 3 has a higher log loss (2.11) than Baseline 2 (1.68) despite having access to structured features. The structured features add complexity without adding discriminating power at this scale, and the noise from the 384-dimensional text component dominates the gradient signal during MLP training.

---

## 5. Baseline Gaps

The following gaps represent concrete deficiencies in the current baselines that motivate the next phase of the project.

**Gap 1 — Text coverage (24.8%).** GDELT articles are available for only the first two months of the season (Oct–Dec 2024). The remaining 984 games have no text signal. Any text-augmented model trained on this subset risks selection bias: early-season games may differ systematically from playoff-adjacent late-season games.

**Gap 2 — Text quality and relevance.** GDELT articles are league-wide news pieces, not game-specific previews. They do not capture injury reports, individual player matchups, or coach decisions announced hours before tip-off — the kind of pre-game information that bettors and analysts actually use. A higher-quality source (e.g., beat reporter articles, official injury reports, or structured lineup data) would be more informative.

**Gap 3 — Embedding dimensionality vs. dataset size.** 384 frozen dimensions are well-suited for large corpora (tens of thousands of examples). With 325 games, the embedding space is too high-dimensional for either logistic regression or a shallow MLP to fit reliably. Possible remediations: aggressive PCA compression (to ~16–32 dims), direct fine-tuning on a classification objective, or a much larger game dataset.

**Gap 4 — Structured features lack player-level resolution.** Baseline 1 uses only team-aggregate rolling statistics. It cannot represent: (a) whether a star player is sitting out a back-to-back, (b) recent individual performance trends, or (c) head-to-head matchup history between specific players. Player-level box score features are available from the NBA API and represent the clearest upgrade path for the structured model.

---

## 6. Next Steps

The audit motivates three concrete directions:

1. **Richer structured features** — incorporate player-level statistics (per-player rolling averages, starter lineup quality) from the NBA stats API. This is the highest-expected-value improvement for Baseline 1.
2. **Larger or higher-quality text corpus** — if text is retained, either source full-season GDELT data or replace with structured pre-game reports (official injury designations, lineup announcements) encoded as categorical/numerical features rather than free text.
3. **Dimensionality-aware text integration** — compress embeddings to 16–32 PCA components before fusion, or explore attention-based pooling to weight articles by relevance rather than mean-pooling.
