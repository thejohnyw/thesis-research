# Baseline Audit: NBA Game Prediction
Repo Link: https://github.com/thejohnyw/thesis-research

## Setup

**Data:** 1,309 NBA games (2024–25 season), binary label = home win (about 56% base rate)

**Evaluation:** Time-based expanding window, 5 splits for B1; 3 for B2/B3 (limited by text coverage), 20% test each. Training always before test.

| Baseline | Model | Features | Games | Dims |
|----------|-------|----------|-------|------|
| B1 | MLP | Structured stats | 1,309 | ~25 |
| B2 | Logistic Reg | BERT embeddings | 325 | 384 |
| B3 | MLP | Struct + Text fusion | 325 | 416 |

**Structured features:** Rolling team stats (win rate, pts scored/allowed, streak, rest days, home/away splits) — all computed strictly before tip-off.

**Text features:** GDELT articles from 5-day pre-game window → `all-MiniLM-L6-v2` (384d, frozen) → mean pool across articles. Only 325 games have text coverage (Oct–Dec 2024).

---

## Results

| Model | Accuracy | ROC AUC | Log Loss |
|-------|----------|---------|----------|
| B1 – Structured | **0.593** | **0.663** | 0.660 |
| B2 – Text only | 0.439 | 0.391 | 1.680 |
| B3 – Fusion | 0.485 | 0.456 | 2.110 |

**B1** improves with more training data (AUC: 0.60 → 0.69 across splits). Structured features carry actual, useful signal.

**B2 & B3** perform *worse than random* (AUC < 0.5). Text embeddings hurt rather than help.

---

## Why Text Failed

1. **Dimensionality mismatch** — 384 dims with only 130–260 training games. Logistic regression is massively underdetermined.

2. **Overconfident wrong predictions** — Log loss of 1.68–2.11 vs 0.693 for a constant predictor. Model assigns high confidence to incorrect outcomes.

3. **Text content isn't predictive** — GDELT returns generic sports desk roundups, not game-specific intel. Mean-pooling washes out any signal that might exist.

4. **Fusion amplifies noise** — B3 has worse log loss than B2. The 384d text noise dominates the gradient; structured features can't compensate.

---

## Gaps

- **Coverage:** Only 24.8% of games have text (Oct–Dec). Selection bias risk.
- **Text quality:** GDELT gives league news, little to no injury reports or lineup announcements.
- **Embedding dims:** 384d is overkill for 325 samples. Need aggressive PCA (16–32d) or fine-tuning.
- **Structured features lack player-level info:** No starter lineups, individual player form, or injury status.

---

## Next Steps

1. **Richer structured features** — player-level stats from NBA API (starter quality, individual rolling averages). Highest expected value (EV).
2. **Better text source** — official injury reports, lineup announcements, or beat reporter articles instead of GDELT.
3. **Compress embeddings** — PCA to 16–32d before fusion, or attention-weighted pooling instead of mean.

## References
GPT Codex and Claude Code were used for implementation and analysis
