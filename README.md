# NBA Game Prediction: Gated Modality Fusion
Repo Link: https://github.com/thejohnyw/thesis-research

## Current Results — Gated Modality Fusion (2025–26 Season)

**Data:** 741 NBA games (2025–26 season, Oct 2025 – Mar 2026) with bilateral Reddit text coverage. 28,070 Reddit posts from 30 team subreddits (Mar 2025 – Mar 2026).

**Evaluation:** 4-fold temporal cross-validation (expanding window, 20% test). 5 random seeds per model per fold, predictions averaged. All models train/test on identical game sets.

**Structured features (38 dims):** Rolling team stats (win rate, point differential, streak, rest days, home/away splits, ELO) — all computed strictly before tip-off.

**Text features (32 dims):** Reddit posts from team subreddits within 48h pre-game window → `all-MiniLM-L6-v2` (384d) → score-weighted average per team → `[home_emb; away_emb; diff_emb]` (1152d) → PCA to 32d (38.2% variance explained).

| Model | Dims | Accuracy | ROC AUC | Log Loss |
|-------|------|----------|---------|----------|
| Dummy (home-win %) | 0 | 0.516 | 0.500 | 0.694 |
| Structured Only | 38 | 0.620 | 0.689 | 0.754 |
| **Concatenation** | 70 | **0.625** | **0.692** | **0.723** |
| Gated Fusion | 38+32 | 0.614 | 0.680 | 0.912 |
| Cross-Attention | 38+32 | 0.610 | 0.683 | 0.932 |

**Concatenation** performs best overall. All learned models substantially beat the dummy baseline (AUC 0.50 → 0.68–0.69).

**Gated Fusion** gate mean = 0.526 (balanced), indicating the model uses both modalities adaptively. However, higher log loss suggests overfitting on the small dataset.

---

## Architecture Details

**Gated Fusion (novel):**
- Separate encoders for structured and text features
- Gate conditioned on structured representation: `gate = sigmoid(gate_net(h_s))`
- Fused: `h = gate * h_s + (1 - gate) * h_t`
- When structured features are confident, gate suppresses noisy text

**Cross-Attention (novel):**
- Structured features produce a query that attends over text
- `context = attention_weight * h_text`
- Output from `[h_struct; context]`

---

## Prior Baseline Audit (2024–25 Season)

**Data:** 1,309 NBA games (2024–25 season), binary label = home win (~56% base rate)

**Evaluation:** Time-based expanding window, 5 splits for B1; 3 for B2/B3 (limited by text coverage), 20% test each.

| Baseline | Model | Features | Games | Dims |
|----------|-------|----------|-------|------|
| B1 | MLP | Structured stats | 1,309 | ~25 |
| B2 | Logistic Reg | BERT embeddings | 325 | 384 |
| B3 | MLP | Struct + Text fusion | 325 | 416 |

**Text features:** GDELT articles from 5-day pre-game window → `all-MiniLM-L6-v2` (384d, frozen) → mean pool. Only 325 games had text coverage.

| Model | Accuracy | ROC AUC | Log Loss |
|-------|----------|---------|----------|
| B1 – Structured | **0.593** | **0.663** | 0.660 |
| B2 – Text only | 0.439 | 0.391 | 1.680 |
| B3 – Fusion | 0.485 | 0.456 | 2.110 |

### Why Text Failed in Baselines

1. **Dimensionality mismatch** — 384 dims with only 130–260 training games.
2. **Overconfident wrong predictions** — Log loss of 1.68–2.11 vs 0.693 for a constant predictor.
3. **Text content isn't predictive** — GDELT returns generic sports desk roundups, not game-specific intel.
4. **Fusion amplifies noise** — B3 has worse log loss than B2. 384d text noise dominates the gradient.

### What Changed for Gated Fusion

1. **Better text source** — Reddit team subreddits instead of GDELT. Fan discussion captures injuries, lineup changes, team sentiment.
2. **Aggressive dimensionality reduction** — PCA 1152 → 32 dims instead of raw 384d.
3. **Score-weighted embeddings** — High-engagement posts weighted more heavily (instead of mean pool).
4. **Bilateral filter** — Only games where both teams have Reddit posts are used, ensuring fair comparison.
5. **Seed averaging** — 5 seeds per model per fold to reduce variance.

## References
Claude Code was used for implementation and analysis
