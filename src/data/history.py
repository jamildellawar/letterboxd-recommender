"""Recommendation history tracking and feedback detection."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.config import HISTORY_DIR


def save_run(recs_df: pd.DataFrame, genre_picks_df: pd.DataFrame) -> None:
    """Save this run's recommendations to a dated parquet file."""
    today = date.today().isoformat()
    rows = []

    for _, row in recs_df.iterrows():
        rows.append({
            "tmdb_id": row["tmdb_id"],
            "title": row.get("title", ""),
            "year": row.get("year"),
            "similarity_score": row.get("similarity_score", 0.0),
            "rec_type": "for_you",
            "picked_genre": None,
            "run_date": today,
        })

    for _, row in genre_picks_df.iterrows():
        rows.append({
            "tmdb_id": row["tmdb_id"],
            "title": row.get("title", ""),
            "year": row.get("year"),
            "similarity_score": row.get("similarity_score", 0.0),
            "rec_type": "genre_pick",
            "picked_genre": row.get("picked_genre"),
            "run_date": today,
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    path = HISTORY_DIR / f"recs_{today}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} recommendations to {path.name}")


def load_all_history() -> pd.DataFrame:
    """Load all past recommendation runs into a single DataFrame."""
    files = sorted(HISTORY_DIR.glob("recs_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def detect_feedback(
    current_rated_df: pd.DataFrame,
    previous_rated_ids: set,
) -> pd.DataFrame:
    """Find recommended movies that the user has since rated.

    Compares past recommendations against newly rated movies (movies rated
    since the last run) to detect hits, misses, and neutral outcomes.

    Returns DataFrame with columns:
        tmdb_id, title, rec_date, user_rating, similarity_score, outcome
    """
    history = load_all_history()
    if history.empty:
        return pd.DataFrame()

    rec_ids = set(history["tmdb_id"].values)
    rated_ids = set(current_rated_df["tmdb_id"].values)

    # Movies that were recommended AND have been newly rated
    new_rated_ids = rated_ids - previous_rated_ids
    feedback_ids = rec_ids & new_rated_ids

    if not feedback_ids:
        return pd.DataFrame()

    # Build feedback rows
    rows = []
    for tmdb_id in feedback_ids:
        rec_row = history[history["tmdb_id"] == tmdb_id].iloc[-1]
        rated_row = current_rated_df[current_rated_df["tmdb_id"] == tmdb_id].iloc[0]
        user_rating = rated_row["memberRating"]

        if user_rating >= 4.0:
            outcome = "hit"
        elif user_rating <= 2.5:
            outcome = "miss"
        else:
            outcome = "neutral"

        rows.append({
            "tmdb_id": tmdb_id,
            "title": rec_row.get("title", ""),
            "rec_date": rec_row.get("run_date", ""),
            "user_rating": user_rating,
            "similarity_score": rec_row.get("similarity_score", 0.0),
            "outcome": outcome,
        })

    return pd.DataFrame(rows)


def save_feedback(feedback_df: pd.DataFrame) -> None:
    """Append feedback to the log, deduplicating by tmdb_id."""
    if feedback_df.empty:
        return

    path = HISTORY_DIR / "feedback_log.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, feedback_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="tmdb_id", keep="last")
    else:
        combined = feedback_df

    combined.to_parquet(path, index=False)
    print(f"Saved {len(feedback_df)} feedback entries ({len(combined)} total)")


def feedback_metrics() -> dict:
    """Compute aggregate feedback metrics."""
    path = HISTORY_DIR / "feedback_log.parquet"
    if not path.exists():
        return {}

    df = pd.read_parquet(path)
    if df.empty:
        return {}

    total = len(df)
    hits = (df["outcome"] == "hit").sum()
    misses = (df["outcome"] == "miss").sum()
    watched = total  # all feedback entries are watched recs

    history = load_all_history()
    total_recs = history["tmdb_id"].nunique() if not history.empty else total

    return {
        "watch_rate": watched / total_recs if total_recs > 0 else 0.0,
        "avg_rating": df["user_rating"].mean(),
        "hit_rate": hits / total if total > 0 else 0.0,
        "miss_rate": misses / total if total > 0 else 0.0,
    }
