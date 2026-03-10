"""
Gated Modality Fusion for NBA Game Prediction

Novel architecture: structured features dynamically control how much
text features contribute, learning WHEN Reddit signal is informative.

Text representation: pre-game Reddit posts are embedded with a sentence
transformer (all-MiniLM-L6-v2, 384-d). Per-game embedding is a score-
weighted average of matched posts from each team's subreddit within
a 48-hour pre-game window.  Text vector = [home_emb; away_emb; diff_emb]
projected down to a compact representation via PCA.

Four experiments compared under strict temporal cross-validation:
  1. Structured Only     — MLP on box-score / rolling stats
  2. Concatenation       — naive joint MLP on [struct || text_embed]
  3. Gated Fusion        — gate(h_s) * h_s + (1-gate) * h_t   [NOVEL]
  4. Cross-Attention     — structured queries attend to text    [NOVEL]

Run:
  python src/gated_fusion.py \
    --reddit_jsonl data/raw/reddit_team_posts_top_year_public.jsonl \
    --games_csv    data/processed/games_api.csv \
    --features_csv data/processed/features.csv \
    --out_csv      data/processed/gated_fusion_results.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from backtest import TimeBasedSplit, validate_time_split

TEAM_SUBREDDITS = {
    "Atlanta Hawks": "AtlantaHawks", "Boston Celtics": "bostonceltics",
    "Brooklyn Nets": "GoNets", "Charlotte Hornets": "CharlotteHornets",
    "Chicago Bulls": "chicagobulls", "Cleveland Cavaliers": "clevelandcavs",
    "Dallas Mavericks": "Mavericks", "Denver Nuggets": "denvernuggets",
    "Detroit Pistons": "DetroitPistons", "Golden State Warriors": "warriors",
    "Houston Rockets": "rockets", "Indiana Pacers": "pacers",
    "LA Clippers": "LAClippers", "Los Angeles Clippers": "LAClippers",
    "Los Angeles Lakers": "lakers", "Memphis Grizzlies": "memphisgrizzlies",
    "Miami Heat": "heat", "Milwaukee Bucks": "MkeBucks",
    "Minnesota Timberwolves": "timberwolves", "New Orleans Pelicans": "NOLAPelicans",
    "New York Knicks": "NYKnicks", "Oklahoma City Thunder": "Thunder",
    "Orlando Magic": "OrlandoMagic", "Philadelphia 76ers": "sixers",
    "Phoenix Suns": "suns", "Portland Trail Blazers": "ripcity",
    "Sacramento Kings": "kings", "San Antonio Spurs": "NBASpurs",
    "Toronto Raptors": "torontoraptors", "Utah Jazz": "UtahJazz",
    "Washington Wizards": "washingtonwizards",
}


# ============================================================================
# Text Embedding Pipeline
# ============================================================================

def load_posts_index(reddit_jsonl: Path) -> dict[str, list[dict]]:
    """Load Reddit posts grouped by subreddit, sorted by time."""
    index: dict[str, list[dict]] = {}
    with reddit_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            post = json.loads(line)
            sub = post.get("subreddit", "")
            if sub:
                index.setdefault(sub, []).append(post)
    for sub in index:
        index[sub].sort(key=lambda p: float(p.get("created_utc", 0) or 0))
    return index


def embed_all_posts(
    posts_index: dict[str, list[dict]],
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, dict[str, np.ndarray]]:
    """Pre-compute embeddings for every post. Returns {subreddit: {post_id: emb}}."""
    encoder = SentenceTransformer(model_name)

    all_texts: list[str] = []
    all_keys: list[tuple[str, str]] = []
    for sub, posts in posts_index.items():
        for p in posts:
            title = str(p.get("title", "") or "")
            selftext = str(p.get("selftext", "") or "").strip()
            text = title if not selftext or selftext == "[removed]" else f"{title}. {selftext[:300]}"
            all_texts.append(text)
            all_keys.append((sub, p["id"]))

    print(f"  Encoding {len(all_texts)} posts with {model_name}...")
    embeddings = encoder.encode(all_texts, show_progress_bar=True, batch_size=128)

    result: dict[str, dict[str, np.ndarray]] = {}
    for (sub, pid), emb in zip(all_keys, embeddings):
        result.setdefault(sub, {})[pid] = emb
    return result


def build_game_embeddings(
    games_df: pd.DataFrame,
    posts_index: dict[str, list[dict]],
    emb_index: dict[str, dict[str, np.ndarray]],
    window_hours: int = 48,
    top_k: int = 25,
) -> np.ndarray:
    """
    Build per-game text embedding vectors.

    For each game, finds posts from home/away subreddits within the
    pre-game window, computes a score-weighted average embedding for
    each side, and returns [home_emb; away_emb; home_emb - away_emb].
    """
    emb_dim = next(iter(next(iter(emb_index.values())).values())).shape[0]
    rows = []

    for _, row in games_df.iterrows():
        game_ts = pd.to_datetime(row["date"]).timestamp()
        cutoff = game_ts - (window_hours * 3600)

        game_emb = np.zeros(emb_dim * 3)
        for side_idx, team_col in enumerate(["home_team", "away_team"]):
            sub = TEAM_SUBREDDITS.get(str(row[team_col]), "")
            posts = [
                p for p in posts_index.get(sub, [])
                if cutoff <= float(p.get("created_utc", 0) or 0) <= game_ts
            ]
            posts.sort(
                key=lambda p: (float(p.get("score", 0) or 0), float(p.get("num_comments", 0) or 0)),
                reverse=True,
            )
            if top_k > 0:
                posts = posts[:top_k]

            if posts:
                embs = np.array([emb_index.get(sub, {}).get(p["id"], np.zeros(emb_dim)) for p in posts])
                scores = np.array([max(1, float(p.get("score", 1) or 1)) for p in posts], dtype=float)
                weights = scores / scores.sum()
                weighted_emb = (embs.T * weights).sum(axis=1)
                game_emb[side_idx * emb_dim : (side_idx + 1) * emb_dim] = weighted_emb

        # Diff embedding
        game_emb[2 * emb_dim :] = game_emb[:emb_dim] - game_emb[emb_dim : 2 * emb_dim]
        rows.append(game_emb)

    return np.array(rows)


def load_and_embed(
    reddit_jsonl: Path,
    games_csv: Path,
    features_csv: Path,
    window_hours: int,
    top_k: int,
    pca_dim: int,
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Load games, structured features, and compute text embeddings."""
    # Posts
    print("Loading Reddit posts...")
    posts_index = load_posts_index(reddit_jsonl)
    total_posts = sum(len(v) for v in posts_index.values())
    print(f"  {total_posts} posts from {len(posts_index)} subreddits")

    # Embeddings
    emb_index = embed_all_posts(posts_index)

    # Games
    games_df = pd.read_csv(games_csv, parse_dates=["date"])
    games_df = games_df[~games_df["home_win"].isna()].copy()

    # Restrict to games after earliest post
    ts_all = []
    for sub_posts in posts_index.values():
        for p in sub_posts:
            ts = float(p.get("created_utc", 0) or 0)
            if ts > 0:
                ts_all.append(ts)
    text_start = pd.to_datetime(min(ts_all), unit="s", utc=True).tz_convert(None)
    games_df = games_df[pd.to_datetime(games_df["date"]) >= text_start].copy()

    # Structured features
    struct_df = pd.read_csv(features_csv)
    exclude = {"game_id", "date", "home_win", "away_team", "home_team"}
    struct_cols = [c for c in struct_df.columns if c not in exclude]

    add_cols = [c for c in struct_cols if c not in games_df.columns]
    df = games_df.merge(struct_df[["game_id", *add_cols]], on="game_id", how="left")

    # Game-level embeddings
    print("Building game embeddings...")
    raw_embs = build_game_embeddings(games_df, posts_index, emb_index, window_hours, top_k)

    # Filter to games where BOTH teams have text (fair comparison)
    emb_dim = raw_embs.shape[1] // 3
    home_has = np.any(raw_embs[:, :emb_dim] != 0, axis=1)
    away_has = np.any(raw_embs[:, emb_dim:2*emb_dim] != 0, axis=1)
    both_have = home_has & away_has
    df = df[both_have].reset_index(drop=True)
    raw_embs = raw_embs[both_have]
    print(f"  {len(df)} games with bilateral text coverage")
    print(f"  Raw embedding dim: {raw_embs.shape[1]}")

    # PCA to reduce dimensionality (384*3=1152 -> pca_dim)
    actual_pca_dim = min(pca_dim, raw_embs.shape[0], raw_embs.shape[1])
    pca = PCA(n_components=actual_pca_dim, random_state=42)
    text_embs = pca.fit_transform(raw_embs)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {raw_embs.shape[1]} -> {actual_pca_dim} dims ({var_explained:.1%} variance)")

    return df, struct_cols, text_embs


# ============================================================================
# Models
# ============================================================================

class DummyModel:
    """Baseline: always predicts training home-win rate (no learning)."""

    def __init__(self):
        self.home_win_rate = 0.5

    def fit(self, y_train: np.ndarray) -> None:
        self.home_win_rate = float(y_train.mean())

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.home_win_rate)


class StructuredOnlyModel(nn.Module):
    """Baseline: MLP on structured features only."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_s: torch.Tensor, _x_t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x_s))


class ConcatModel(nn.Module):
    """Baseline: naive concatenation of both modalities."""

    def __init__(self, struct_dim: int, text_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(struct_dim + text_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([x_s, x_t], dim=1)))


class GatedFusionModel(nn.Module):
    """
    Novel: Gated Modality Fusion.

    Architecture:
        h_s  = struct_encoder(x_s)
        h_t  = text_encoder(x_t)
        gate = sigmoid(gate_net(h_s))        # ∈ [0,1], per sample
        h    = gate * h_s + (1-gate) * h_t   # adaptive fusion
        out  = sigmoid(head(h))

    The gate is conditioned on the structured representation only:
    when structured features are confident, the gate closes and
    suppresses potentially noisy text; when uncertain, it opens.
    """

    def __init__(self, struct_dim: int, text_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.struct_encoder = nn.Sequential(
            nn.Linear(struct_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x_s: torch.Tensor,
        x_t: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h_s = self.struct_encoder(x_s)
        h_t = self.text_encoder(x_t)
        gate = self.gate_net(h_s)
        h_fused = gate * h_s + (1.0 - gate) * h_t
        prob = torch.sigmoid(self.head(h_fused))
        if return_gate:
            return prob, gate
        return prob


class CrossAttentionModel(nn.Module):
    """
    Novel: Cross-Attention Fusion.

    Structured features produce a query that attends over text features,
    allowing the model to selectively extract relevant text signal.

        h_s = struct_encoder(x_s)              # query
        h_t = text_encoder(x_t)                # key/value
        attn_weight = softmax(h_s @ h_t / sqrt(d))
        context = attn_weight * h_t
        h = [h_s; context]
        out = sigmoid(head(h))
    """

    def __init__(self, struct_dim: int, text_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.struct_encoder = nn.Sequential(
            nn.Linear(struct_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.attn_scale = hidden ** 0.5
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        h_s = self.struct_encoder(x_s)
        h_t = self.text_encoder(x_t)
        # Scalar attention weight per sample
        attn = torch.sigmoid((h_s * h_t).sum(dim=1, keepdim=True) / self.attn_scale)
        context = attn * h_t
        h = torch.cat([h_s, context], dim=1)
        return torch.sigmoid(self.head(h))


# ============================================================================
# Training
# ============================================================================

def _make_tensors(X_s: np.ndarray, X_t: np.ndarray, y: np.ndarray):
    return torch.FloatTensor(X_s), torch.FloatTensor(X_t), torch.FloatTensor(y)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for x_s, x_t, y in loader:
        optimizer.zero_grad()
        criterion(model(x_s, x_t).squeeze(), y).backward()
        optimizer.step()


@torch.no_grad()
def predict(model, x_s, x_t):
    model.eval()
    return model(x_s, x_t).squeeze().numpy()


@torch.no_grad()
def predict_with_gate(model, x_s, x_t):
    model.eval()
    prob, gate = model(x_s, x_t, return_gate=True)
    return prob.squeeze().numpy(), gate.squeeze().numpy()


def fit(model, X_s_tr, X_t_tr, y_tr, X_s_val, X_t_val, y_val,
        epochs=150, lr=1e-3, patience=20, batch_size=32):
    """Train with early stopping on validation AUC."""
    ts, tt, ty = _make_tensors(X_s_tr, X_t_tr, y_tr)
    vs, vt, vy = _make_tensors(X_s_val, X_t_val, y_val)
    loader = DataLoader(TensorDataset(ts, tt, ty), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_auc, no_improve, best_state = 0.0, 0, None
    for _ in range(epochs):
        train_epoch(model, loader, optimizer, criterion)
        val_pred = predict(model, vs, vt)
        if len(np.unique(vy.numpy())) > 1:
            auc = roc_auc_score(vy.numpy(), val_pred)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)


def metrics_dict(y_true, p):
    return {
        "accuracy": float(accuracy_score(y_true, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan"),
        "log_loss": float(log_loss(y_true, p)),
    }


# ============================================================================
# Cross-validation
# ============================================================================

def run_cv(
    df: pd.DataFrame,
    struct_cols: list[str],
    text_embs: np.ndarray,
    n_splits: int,
    test_size: float,
    min_train_games: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    # Drop all-NaN structured columns
    all_nan = [c for c in struct_cols if df[c].isna().all()]
    if all_nan:
        print(f"  Dropping all-NaN columns: {all_nan}")
    struct_cols = [c for c in struct_cols if c not in all_nan]

    splitter = TimeBasedSplit(
        date_col="date", n_splits=n_splits,
        test_size=test_size, min_train_games=min_train_games,
    )
    splits = splitter.split(df)

    experiments = {
        "Dummy (home-win %)": [],
        "Structured Only": [],
        "Concatenation": [],
        "Gated Fusion": [],
        "Cross-Attention": [],
    }
    gate_vals, gate_stds = [], []

    for fold_idx, split in enumerate(splits):
        if not validate_time_split(df, split, date_col="date"):
            continue
        train_df = df.iloc[split.train_indices]
        test_df = df.iloc[split.test_indices]
        if train_df.empty or test_df.empty:
            continue

        y_te = test_df["home_win"].values.astype(float)

        # Inner train/val split
        val_cut = int(len(train_df) * 0.8)
        tr_idx = split.train_indices[:val_cut]
        val_idx = split.train_indices[val_cut:]
        y_tr = df.iloc[tr_idx]["home_win"].values.astype(float)
        y_val = df.iloc[val_idx]["home_win"].values.astype(float)

        # Scale structured
        imp_s = SimpleImputer(strategy="mean")
        sc_s = StandardScaler()
        X_tr_s = sc_s.fit_transform(imp_s.fit_transform(df.iloc[tr_idx][struct_cols]))
        X_val_s = sc_s.transform(imp_s.transform(df.iloc[val_idx][struct_cols]))
        X_te_s = sc_s.transform(imp_s.transform(test_df[struct_cols]))

        # Scale text embeddings
        sc_t = StandardScaler()
        X_tr_t = sc_t.fit_transform(text_embs[tr_idx])
        X_val_t = sc_t.transform(text_embs[val_idx])
        X_te_t = sc_t.transform(text_embs[split.test_indices])

        s_dim = X_tr_s.shape[1]
        t_dim = X_tr_t.shape[1]
        ts_te, tt_te, _ = _make_tensors(X_te_s, X_te_t, y_te)

        print(f"  Fold {fold_idx+1}: train={len(tr_idx)} val={len(val_idx)} test={len(y_te)}")

        # Dummy baseline
        dummy = DummyModel()
        dummy.fit(y_tr)
        dummy_pred = dummy.predict(len(y_te))
        experiments["Dummy (home-win %)"].append(metrics_dict(y_te, dummy_pred))

        # Seed-averaged predictions (5 seeds per model to reduce variance)
        n_seeds = 5
        model_specs = [
            ("Structured Only", lambda: StructuredOnlyModel(s_dim, hidden_dim)),
            ("Concatenation", lambda: ConcatModel(s_dim, t_dim, hidden_dim)),
            ("Gated Fusion", lambda: GatedFusionModel(s_dim, t_dim, hidden_dim)),
            ("Cross-Attention", lambda: CrossAttentionModel(s_dim, t_dim, hidden_dim)),
        ]

        for name, make_model in model_specs:
            preds_all, gates_all = [], []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                m = make_model()
                fit(m, X_tr_s, X_tr_t, y_tr, X_val_s, X_val_t, y_val, epochs=epochs, lr=lr)
                if name == "Gated Fusion":
                    p, g = predict_with_gate(m, ts_te, tt_te)
                    preds_all.append(p)
                    gates_all.append(g)
                else:
                    preds_all.append(predict(m, ts_te, tt_te))

            avg_pred = np.mean(preds_all, axis=0)
            experiments[name].append(metrics_dict(y_te, avg_pred))
            if name == "Gated Fusion" and gates_all:
                avg_gate = np.mean(gates_all, axis=0)
                gate_vals.append(float(avg_gate.mean()))
                gate_stds.append(float(avg_gate.std()))

    # Build results table
    dims_map = {
        "Dummy (home-win %)": 0,
        "Structured Only": s_dim,
        "Concatenation": s_dim + t_dim,
        "Gated Fusion": f"{s_dim}+{t_dim}",
        "Cross-Attention": f"{s_dim}+{t_dim}",
    }
    rows = []
    for name, folds in experiments.items():
        if not folds:
            continue
        row = {
            "Experiment": name,
            "Dims": dims_map[name],
            "Accuracy": round(float(np.nanmean([m["accuracy"] for m in folds])), 3),
            "ROC AUC": round(float(np.nanmean([m["roc_auc"] for m in folds])), 3),
            "Log Loss": round(float(np.nanmean([m["log_loss"] for m in folds])), 3),
            "Splits": len(folds),
        }
        if name == "Gated Fusion" and gate_vals:
            row["Gate Mean"] = round(float(np.mean(gate_vals)), 3)
            row["Gate Std"] = round(float(np.mean(gate_stds)), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Gated Modality Fusion for NBA prediction")
    ap.add_argument("--reddit_jsonl", required=True)
    ap.add_argument("--games_csv", required=True)
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--out_csv", default="data/processed/gated_fusion_results.csv")
    ap.add_argument("--window_hours", type=int, default=48)
    ap.add_argument("--top_k_posts", type=int, default=25)
    ap.add_argument("--pca_dim", type=int, default=32)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--min_train_games", type=int, default=80)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    df, struct_cols, text_embs = load_and_embed(
        Path(args.reddit_jsonl),
        Path(args.games_csv),
        Path(args.features_csv),
        args.window_hours,
        args.top_k_posts,
        args.pca_dim,
    )
    print(f"  Structured: {len(struct_cols)} dims")
    print(f"  Text embed: {text_embs.shape[1]} dims (PCA)")

    print(f"\nRunning {args.n_splits}-fold temporal CV...")
    results = run_cv(
        df, struct_cols, text_embs,
        n_splits=args.n_splits, test_size=args.test_size,
        min_train_games=args.min_train_games, hidden_dim=args.hidden_dim,
        epochs=args.epochs, lr=args.lr,
    )

    print("\n" + "=" * 75)
    print("RESULTS")
    print("=" * 75)
    print(results.to_string(index=False))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    gated = results[results["Experiment"] == "Gated Fusion"]
    if not gated.empty and "Gate Mean" in gated.columns:
        g = float(gated["Gate Mean"].values[0])
        print("\n" + "=" * 75)
        print("GATE INTERPRETATION")
        print("=" * 75)
        if g > 0.7:
            print(f"Gate mean = {g:.3f} (HIGH): model mostly trusts structured features.")
            print("Text adds noise for most games — the gate suppresses it.")
        elif g < 0.3:
            print(f"Gate mean = {g:.3f} (LOW): model mostly trusts text features.")
        else:
            print(f"Gate mean = {g:.3f} (BALANCED): model uses both modalities adaptively.")
            print("Text helps for a subset of games — the ideal outcome.")


if __name__ == "__main__":
    main()
