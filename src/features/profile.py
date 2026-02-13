"""Build user taste profile from rated movies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_list(val) -> list:
    """Convert a value to a Python list (handles numpy arrays, None, etc.)."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    try:
        return list(val)
    except TypeError:
        return []


def build_taste_profile(
    feature_matrix: np.ndarray,
    ratings: pd.Series,
) -> np.ndarray:
    """Build a taste profile using positive/negative profile separation.

    Instead of a single weighted average, builds two profiles:
    - Positive: weighted average of movies rated >= 3.5 (what you like)
    - Negative: weighted average of movies rated <= 2.5 (what you don't)

    Final profile = positive - 0.5 * negative

    This explicitly captures "avoid movies like the bad ones", which
    improves discrimination vs a single weighted average.
    """
    ratings_arr = ratings.values

    # Positive profile: movies rated 3.5+
    pos_mask = ratings_arr >= 3.5
    if pos_mask.sum() > 0:
        pos_weights = ratings_arr[pos_mask] - 3.0  # 5.0→2.0, 4.0→1.0, 3.5→0.5
        pos_weighted = feature_matrix[pos_mask] * pos_weights[:, np.newaxis]
        pos_profile = pos_weighted.sum(axis=0)
        pos_norm = np.linalg.norm(pos_profile)
        if pos_norm > 0:
            pos_profile = pos_profile / pos_norm
    else:
        pos_profile = np.zeros(feature_matrix.shape[1])

    # Negative profile: movies rated <= 2.5
    neg_mask = ratings_arr <= 2.5
    if neg_mask.sum() > 0:
        neg_weights = 3.0 - ratings_arr[neg_mask]  # 0.5→2.5, 1.5→1.5, 2.5→0.5
        neg_weighted = feature_matrix[neg_mask] * neg_weights[:, np.newaxis]
        neg_profile = neg_weighted.sum(axis=0)
        neg_norm = np.linalg.norm(neg_profile)
        if neg_norm > 0:
            neg_profile = neg_profile / neg_norm
    else:
        neg_profile = np.zeros(feature_matrix.shape[1])

    # Combined: attract to liked, repel from disliked
    profile = pos_profile - 0.5 * neg_profile

    # L2 normalize
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm

    return profile


def profile_summary(df: pd.DataFrame) -> dict:
    """Compute a summary of the user's taste profile for display."""
    total = len(df)
    avg_rating = df["memberRating"].mean()

    # Top genres by weighted frequency (rating as weight)
    genre_scores: dict[str, float] = {}
    for _, row in df.iterrows():
        rating = row.get("memberRating", 3.0)
        for genre in _safe_list(row.get("genres")):
            genre_scores[genre] = genre_scores.get(genre, 0) + rating
    top_genres = sorted(genre_scores, key=genre_scores.get, reverse=True)[:5]

    # Top directors by avg rating (min 2 films)
    director_ratings: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        rating = row.get("memberRating", 3.0)
        for d in _safe_list(row.get("director")):
            director_ratings.setdefault(d, []).append(rating)

    director_avg = {
        d: sum(r) / len(r)
        for d, r in director_ratings.items()
        if len(r) >= 2
    }
    top_directors = sorted(director_avg, key=director_avg.get, reverse=True)[:5]

    return {
        "total_rated": total,
        "avg_rating": round(avg_rating, 2),
        "top_genres": top_genres,
        "top_directors": top_directors,
    }
