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


def evaluate_cf_model(
    model,
    rated_df: pd.DataFrame,
    candidate_tmdb_ids: set[int],
    n_folds: int = 5,
    top_k: int = 20,
    min_rating: float = 4.0,
) -> dict:
    """K-fold cross-validation for CF models (KNN, LightFM, Two-Tower).

    Holds out a fold of rated movies, uses the rest as "guest input",
    scores candidates, and checks if held-out movies rank in top-K.

    Args:
        model: Any model with a predict(guest_ratings, candidate_tmdb_ids) method.
        rated_df: User's rated movies DataFrame with tmdb_id and memberRating.
        candidate_tmdb_ids: Set of tmdb_ids in the candidate pool.
        n_folds: Number of folds for cross-validation.
        top_k: Top-K cutoff for hit rate.
        min_rating: Minimum rating for a movie to count as a "hit" target.

    Returns dict with hit_rate, mrr, ndcg, coverage.
    """
    # Only evaluate highly-rated movies as targets
    high_rated = rated_df[rated_df["memberRating"] >= min_rating].copy()
    mappable = high_rated[high_rated["tmdb_id"].isin(candidate_tmdb_ids)]

    if len(mappable) < n_folds:
        return {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0, "coverage": 0.0, "n_evaluated": 0}

    # Shuffle and split into folds
    indices = mappable.index.tolist()
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = len(indices) // n_folds

    hits = 0
    reciprocal_ranks = []
    dcg_scores = []
    all_recommended = set()
    n_evaluated = 0

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(indices)
        holdout_indices = indices[start:end]
        train_mask = ~rated_df.index.isin(holdout_indices)

        # Build guest ratings from training set
        train_df = rated_df[train_mask]
        guest_ratings = dict(zip(
            train_df["tmdb_id"].astype(int),
            train_df["memberRating"].astype(float),
        ))

        if len(guest_ratings) < 3:
            continue

        # Get model predictions
        scores = model.predict(guest_ratings, candidate_tmdb_ids)
        if not scores:
            continue

        # Rank by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_ids = [tid for tid, _ in ranked[:top_k]]
        all_recommended.update(ranked_ids)

        # Check each held-out movie
        for idx in holdout_indices:
            tmdb_id = int(rated_df.loc[idx, "tmdb_id"])
            n_evaluated += 1

            if tmdb_id in set(ranked_ids):
                rank = ranked_ids.index(tmdb_id) + 1
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
                dcg_scores.append(1.0 / np.log2(rank + 1))
            else:
                # Check full ranking for MRR
                all_ranked_ids = [tid for tid, _ in ranked]
                if tmdb_id in set(all_ranked_ids):
                    rank = all_ranked_ids.index(tmdb_id) + 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)
                dcg_scores.append(0.0)

    hit_rate = hits / n_evaluated if n_evaluated > 0 else 0.0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    ndcg = np.mean(dcg_scores) if dcg_scores else 0.0
    coverage = len(all_recommended) / len(candidate_tmdb_ids) if candidate_tmdb_ids else 0.0

    return {
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "ndcg": round(ndcg, 4),
        "coverage": round(coverage, 4),
        "n_evaluated": n_evaluated,
    }
