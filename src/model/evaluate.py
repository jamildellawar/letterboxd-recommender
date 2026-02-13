"""Evaluation metrics for the recommendation model."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from src.features.vectorize import MovieVectorizer
from src.features.profile import build_taste_profile


def leave_one_out_eval(
    rated_df: pd.DataFrame,
    vectorizer: MovieVectorizer,
    min_rating: float = 4.0,
    top_k: int = 20,
) -> dict:
    """Leave-one-out evaluation for highly-rated movies.

    For each movie rated >= min_rating:
    1. Remove it from the profile
    2. Score it against the remaining profile
    3. Check if it would appear in top-K

    Returns dict with hit_rate, mean_reciprocal_rank, n_evaluated.
    """
    high_rated = rated_df[rated_df["memberRating"] >= min_rating]
    all_vectors = vectorizer.transform(rated_df)

    hits = 0
    reciprocal_ranks = []
    n_evaluated = 0

    for idx in high_rated.index:
        # Leave out this movie
        mask = rated_df.index != idx
        remaining_vectors = all_vectors[mask]
        remaining_ratings = rated_df.loc[mask, "memberRating"]

        if len(remaining_ratings) < 5:
            continue

        # Build profile without this movie
        profile = build_taste_profile(remaining_vectors, remaining_ratings)

        # Score all movies (including the held-out one)
        scores = cosine_similarity(all_vectors, profile.reshape(1, -1)).flatten()

        # Rank all movies by score descending
        ranked_indices = np.argsort(scores)[::-1]

        # Find rank of the held-out movie
        pos = np.where(ranked_indices == idx)[0]
        if len(pos) > 0:
            rank = pos[0] + 1  # 1-indexed
            if rank <= top_k:
                hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        n_evaluated += 1

    hit_rate = hits / n_evaluated if n_evaluated > 0 else 0.0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return {
        "hit_rate": round(hit_rate, 4),
        "mean_reciprocal_rank": round(mrr, 4),
        "n_evaluated": n_evaluated,
        "top_k": top_k,
    }


def rating_correlation(
    rated_df: pd.DataFrame,
    vectorizer: MovieVectorizer,
) -> dict:
    """Compute correlation between similarity scores and actual ratings.

    For each rated movie, compute its similarity to the profile built
    from all OTHER movies, then correlate with actual rating.

    Returns dict with spearman_r, p_value.
    """
    all_vectors = vectorizer.transform(rated_df)
    similarities = []

    for idx in rated_df.index:
        mask = rated_df.index != idx
        remaining_vectors = all_vectors[mask]
        remaining_ratings = rated_df.loc[mask, "memberRating"]

        profile = build_taste_profile(remaining_vectors, remaining_ratings)
        sim = cosine_similarity(
            all_vectors[idx].reshape(1, -1),
            profile.reshape(1, -1),
        )[0, 0]
        similarities.append(sim)

    actual_ratings = rated_df["memberRating"].values
    r, p = spearmanr(similarities, actual_ratings)

    return {
        "spearman_r": round(r, 4),
        "p_value": round(p, 6),
        "n_movies": len(rated_df),
    }
