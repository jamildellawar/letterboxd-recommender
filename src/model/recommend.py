"""Cosine similarity scoring and recommendation generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import TOP_N_RECOMMENDATIONS
from src.features.vectorize import MovieVectorizer
from src.features.profile import build_taste_profile
from src.features.feedback import apply_feedback_to_profile


# Quality thresholds
MIN_VOTE_COUNT = 50       # filter out niche films with inflated ratings
QUALITY_BLEND = 0.25      # 25% TMDB quality, 75% taste similarity


def _bayesian_rating(vote_avg: np.ndarray, vote_count: np.ndarray) -> np.ndarray:
    """IMDB-style weighted rating that penalizes films with few votes.

    Formula: WR = (v / (v + m)) * R + (m / (v + m)) * C
    where m = median vote count, C = mean vote average.

    This pulls niche films with inflated ratings (e.g. Gabriel's Inferno
    with 1K votes and 8.4 avg) toward the pool mean, while films with
    10K+ votes keep their actual rating.
    """
    m = np.median(vote_count[vote_count > 0]) if (vote_count > 0).any() else 1000.0
    c = np.mean(vote_avg[vote_avg > 0]) if (vote_avg > 0).any() else 6.5
    weighted = (vote_count / (vote_count + m)) * vote_avg + (m / (vote_count + m)) * c
    return weighted


def generate_recommendations(
    rated_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    vectorizer: MovieVectorizer,
    top_n: int = TOP_N_RECOMMENDATIONS,
    mmr_lambda: float = 0.7,
    feedback_adjustment: np.ndarray | None = None,
) -> pd.DataFrame:
    """Generate movie recommendations.

    Pipeline:
    1. Filter candidates by minimum vote count (removes niche/inflated-rating films)
    2. Score by cosine similarity to taste profile
    3. Blend with TMDB quality signal (vote_average * log(vote_count))
    4. MMR re-ranking for diversity
    """
    # Build user taste profile
    rated_vectors = vectorizer.transform(rated_df)
    taste_profile = build_taste_profile(rated_vectors, rated_df["memberRating"])
    taste_profile = apply_feedback_to_profile(taste_profile, feedback_adjustment)

    # Filter out already-watched movies
    watched_ids = set(rated_df["tmdb_id"].values)
    unseen = candidates_df[~candidates_df["tmdb_id"].isin(watched_ids)].copy()

    if unseen.empty:
        print("Warning: No unseen candidates to score.")
        return pd.DataFrame()

    # Quality filter: remove niche films with inflated ratings
    vote_counts = unseen["vote_count"].fillna(0) if "vote_count" in unseen.columns else pd.Series(0, index=unseen.index)
    quality_mask = vote_counts >= MIN_VOTE_COUNT
    unseen = unseen[quality_mask].copy()
    print(f"  After quality filter (>={MIN_VOTE_COUNT} votes): {len(unseen)} candidates")

    if unseen.empty:
        print("Warning: No candidates passed quality filter.")
        return pd.DataFrame()

    # Vectorize candidates and compute taste similarity
    candidate_vectors = vectorizer.transform(unseen)
    taste_scores = cosine_similarity(candidate_vectors, taste_profile.reshape(1, -1)).flatten()

    # Quality signal: Bayesian weighted rating (penalizes niche inflated ratings)
    vote_avg = unseen["vote_average"].fillna(0).values if "vote_average" in unseen.columns else np.zeros(len(unseen))
    vote_cnt = unseen["vote_count"].fillna(0).values if "vote_count" in unseen.columns else np.ones(len(unseen))
    quality_raw = _bayesian_rating(vote_avg, vote_cnt)
    q_min, q_max = quality_raw.min(), quality_raw.max()
    if q_max > q_min:
        quality_norm = (quality_raw - q_min) / (q_max - q_min)
    else:
        quality_norm = np.ones_like(quality_raw)

    # Normalize taste scores to [0, 1]
    t_min, t_max = taste_scores.min(), taste_scores.max()
    if t_max > t_min:
        taste_norm = (taste_scores - t_min) / (t_max - t_min)
    else:
        taste_norm = np.ones_like(taste_scores)

    # Blended score
    blended_scores = (1 - QUALITY_BLEND) * taste_norm + QUALITY_BLEND * quality_norm

    # Pre-filter to top 100 by blended score
    shortlist_size = min(100, len(blended_scores))
    top_indices = np.argsort(blended_scores)[::-1][:shortlist_size]

    shortlist_vectors = candidate_vectors[top_indices]
    shortlist_scores = blended_scores[top_indices]

    # MMR re-ranking for diversity
    selected_indices = _mmr_rerank(
        shortlist_scores, shortlist_vectors, top_n=top_n, lambda_param=mmr_lambda
    )

    # Map back to original indices
    final_indices = top_indices[selected_indices]
    recs = unseen.iloc[final_indices].copy()
    recs["similarity_score"] = taste_scores[final_indices]  # report pure taste similarity

    # Generate explanations
    recs["explanation"] = recs.apply(
        lambda row: _explain_recommendation(row, rated_df), axis=1
    )

    return recs.reset_index(drop=True)


def generate_genre_picks(
    rated_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    vectorizer: MovieVectorizer,
    n_genres: int = 10,
    exclude_ids: set | None = None,
    feedback_adjustment: np.ndarray | None = None,
) -> pd.DataFrame:
    """Generate one best recommendation per genre for the user's top genres.

    Args:
        rated_df: DataFrame of rated movies (with TMDB metadata).
        candidates_df: DataFrame of candidate movies to score.
        vectorizer: Fitted MovieVectorizer.
        n_genres: Number of top genres to pick from.
        exclude_ids: Set of tmdb_ids to exclude (e.g. from main recs).

    Returns:
        DataFrame with one rec per genre, including a 'picked_genre' column.
    """
    from src.features.profile import _safe_list

    # Build taste profile and score all candidates
    rated_vectors = vectorizer.transform(rated_df)
    taste_profile = build_taste_profile(rated_vectors, rated_df["memberRating"])
    taste_profile = apply_feedback_to_profile(taste_profile, feedback_adjustment)

    watched_ids = set(rated_df["tmdb_id"].values)
    skip_ids = watched_ids | (exclude_ids or set())
    unseen = candidates_df[~candidates_df["tmdb_id"].isin(skip_ids)].copy()

    if unseen.empty:
        return pd.DataFrame()

    # Quality filter
    vote_counts = unseen["vote_count"].fillna(0) if "vote_count" in unseen.columns else pd.Series(0, index=unseen.index)
    unseen = unseen[vote_counts >= MIN_VOTE_COUNT].copy()

    if unseen.empty:
        return pd.DataFrame()

    candidate_vectors = vectorizer.transform(unseen)
    taste_scores = cosine_similarity(candidate_vectors, taste_profile.reshape(1, -1)).flatten()

    # Quality blend (same as main model) — favors well-known films over niche fan-favorites
    vote_avg = unseen["vote_average"].fillna(0).values if "vote_average" in unseen.columns else np.zeros(len(unseen))
    vote_cnt = unseen["vote_count"].fillna(0).values if "vote_count" in unseen.columns else np.ones(len(unseen))
    quality_raw = _bayesian_rating(vote_avg, vote_cnt)
    q_min, q_max = quality_raw.min(), quality_raw.max()
    quality_norm = (quality_raw - q_min) / (q_max - q_min) if q_max > q_min else np.ones_like(quality_raw)

    t_min, t_max = taste_scores.min(), taste_scores.max()
    taste_norm = (taste_scores - t_min) / (t_max - t_min) if t_max > t_min else np.ones_like(taste_scores)

    blended = (1 - QUALITY_BLEND) * taste_norm + QUALITY_BLEND * quality_norm

    unseen = unseen.copy()
    unseen["similarity_score"] = taste_scores
    unseen["_blended_score"] = blended

    # Get user's top genres by rating-weighted frequency
    genre_scores: dict[str, float] = {}
    for _, row in rated_df.iterrows():
        rating = row.get("memberRating", 3.0)
        for g in _to_list(row.get("genres")):
            genre_scores[g] = genre_scores.get(g, 0) + rating
    top_genres = sorted(genre_scores, key=genre_scores.get, reverse=True)[:n_genres]

    # Pick best candidate per genre (by blended score), no duplicates
    used_ids = set()
    picks = []

    for genre in top_genres:
        # Filter to candidates that have this genre
        genre_mask = unseen["genres"].apply(lambda gs: genre in _to_list(gs))
        genre_candidates = unseen[genre_mask & ~unseen["tmdb_id"].isin(used_ids)]

        if genre_candidates.empty:
            continue

        best_idx = genre_candidates["_blended_score"].idxmax()
        best_row = genre_candidates.loc[best_idx].copy()
        best_row["picked_genre"] = genre
        picks.append(best_row)
        used_ids.add(best_row["tmdb_id"])

    if not picks:
        return pd.DataFrame()

    result = pd.DataFrame(picks)

    # Generate explanations
    result["explanation"] = result.apply(
        lambda row: _explain_recommendation(row, rated_df), axis=1
    )

    return result.reset_index(drop=True)


def get_top_candidate_ids(
    rated_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    vectorizer: MovieVectorizer,
    top_n: int = 150,
    feedback_adjustment: np.ndarray | None = None,
) -> list[int]:
    """Return top-N candidate tmdb_ids by blended taste+quality score.

    Used to build a shortlist for streaming availability checks,
    avoiding API calls on low-scoring movies.
    """
    rated_vectors = vectorizer.transform(rated_df)
    taste_profile = build_taste_profile(rated_vectors, rated_df["memberRating"])
    taste_profile = apply_feedback_to_profile(taste_profile, feedback_adjustment)

    watched_ids = set(rated_df["tmdb_id"].values)
    unseen = candidates_df[~candidates_df["tmdb_id"].isin(watched_ids)].copy()

    if unseen.empty:
        return []

    vote_counts = unseen["vote_count"].fillna(0) if "vote_count" in unseen.columns else pd.Series(0, index=unseen.index)
    unseen = unseen[vote_counts >= MIN_VOTE_COUNT].copy()

    if unseen.empty:
        return []

    candidate_vectors = vectorizer.transform(unseen)
    taste_scores = cosine_similarity(candidate_vectors, taste_profile.reshape(1, -1)).flatten()

    vote_avg = unseen["vote_average"].fillna(0).values if "vote_average" in unseen.columns else np.zeros(len(unseen))
    vote_cnt = unseen["vote_count"].fillna(0).values if "vote_count" in unseen.columns else np.ones(len(unseen))
    quality_raw = _bayesian_rating(vote_avg, vote_cnt)
    q_min, q_max = quality_raw.min(), quality_raw.max()
    quality_norm = (quality_raw - q_min) / (q_max - q_min) if q_max > q_min else np.ones_like(quality_raw)

    t_min, t_max = taste_scores.min(), taste_scores.max()
    taste_norm = (taste_scores - t_min) / (t_max - t_min) if t_max > t_min else np.ones_like(taste_scores)

    blended = (1 - QUALITY_BLEND) * taste_norm + QUALITY_BLEND * quality_norm

    top_indices = np.argsort(blended)[::-1][:top_n]
    return [int(unseen.iloc[i]["tmdb_id"]) for i in top_indices]


def generate_streaming_picks(
    rated_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    vectorizer: MovieVectorizer,
    streaming_map: dict[int, list[str]],
    top_n: int = 20,
    feedback_adjustment: np.ndarray | None = None,
) -> pd.DataFrame:
    """Generate top streaming picks from movies available on Netflix, Max, or Prime.

    Scores all candidates with the same taste+quality blend, then filters
    to those present in streaming_map. No MMR — the pool is already small.
    """
    rated_vectors = vectorizer.transform(rated_df)
    taste_profile = build_taste_profile(rated_vectors, rated_df["memberRating"])
    taste_profile = apply_feedback_to_profile(taste_profile, feedback_adjustment)

    watched_ids = set(rated_df["tmdb_id"].values)
    unseen = candidates_df[~candidates_df["tmdb_id"].isin(watched_ids)].copy()

    if unseen.empty:
        return pd.DataFrame()

    vote_counts = unseen["vote_count"].fillna(0) if "vote_count" in unseen.columns else pd.Series(0, index=unseen.index)
    unseen = unseen[vote_counts >= MIN_VOTE_COUNT].copy()

    if unseen.empty:
        return pd.DataFrame()

    # Filter to movies that are actually streaming
    streaming_ids = set(streaming_map.keys())
    unseen = unseen[unseen["tmdb_id"].isin(streaming_ids)].copy()

    if unseen.empty:
        print("  No streaming candidates found after filtering.")
        return pd.DataFrame()

    candidate_vectors = vectorizer.transform(unseen)
    taste_scores = cosine_similarity(candidate_vectors, taste_profile.reshape(1, -1)).flatten()

    vote_avg = unseen["vote_average"].fillna(0).values if "vote_average" in unseen.columns else np.zeros(len(unseen))
    vote_cnt = unseen["vote_count"].fillna(0).values if "vote_count" in unseen.columns else np.ones(len(unseen))
    quality_raw = _bayesian_rating(vote_avg, vote_cnt)
    q_min, q_max = quality_raw.min(), quality_raw.max()
    quality_norm = (quality_raw - q_min) / (q_max - q_min) if q_max > q_min else np.ones_like(quality_raw)

    t_min, t_max = taste_scores.min(), taste_scores.max()
    taste_norm = (taste_scores - t_min) / (t_max - t_min) if t_max > t_min else np.ones_like(taste_scores)

    blended = (1 - QUALITY_BLEND) * taste_norm + QUALITY_BLEND * quality_norm

    unseen = unseen.copy()
    unseen["similarity_score"] = taste_scores
    unseen["_blended_score"] = blended
    unseen["streaming_services"] = unseen["tmdb_id"].map(streaming_map)

    # Sort by blended score, take top_n
    unseen = unseen.sort_values("_blended_score", ascending=False).head(top_n)

    # Generate explanations
    unseen["explanation"] = unseen.apply(
        lambda row: _explain_recommendation(row, rated_df), axis=1
    )

    unseen = unseen.drop(columns=["_blended_score"])
    print(f"  Generated {len(unseen)} streaming picks")
    return unseen.reset_index(drop=True)


def _mmr_rerank(
    scores: np.ndarray,
    vectors: np.ndarray,
    top_n: int,
    lambda_param: float = 0.7,
) -> list[int]:
    """Maximal Marginal Relevance re-ranking for diversity."""
    n = len(scores)
    top_n = min(top_n, n)

    selected: list[int] = []
    remaining = set(range(n))

    score_min, score_max = scores.min(), scores.max()
    if score_max > score_min:
        norm_scores = (scores - score_min) / (score_max - score_min)
    else:
        norm_scores = np.ones_like(scores)

    while len(selected) < top_n and remaining:
        best_idx = -1
        best_mmr = float("-inf")

        for i in remaining:
            relevance = norm_scores[i]

            if selected:
                sims = cosine_similarity(
                    vectors[i : i + 1], vectors[selected]
                ).flatten()
                max_sim = sims.max()
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


def _to_list(val) -> list:
    """Safely convert a value to a list (handles None, numpy arrays, etc.)."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    try:
        return list(val)
    except TypeError:
        return []


def _explain_recommendation(rec_row: pd.Series, rated_df: pd.DataFrame) -> list[str]:
    """Generate human-readable explanation factors for a recommendation."""
    reasons = []

    # Check director overlap
    rec_directors = set(_to_list(rec_row.get("director")))
    highly_rated = rated_df[rated_df["memberRating"] >= 4.0]
    fav_directors = set(
        d for dirs in highly_rated["director"] for d in _to_list(dirs)
    )
    shared_directors = rec_directors & fav_directors
    if shared_directors:
        reasons.append(f"Director: {', '.join(list(shared_directors)[:2])}")

    # Check genre overlap with top-rated genre preferences
    rec_genres = set(_to_list(rec_row.get("genres")))
    genre_scores: dict[str, float] = {}
    for _, r in rated_df.iterrows():
        for g in _to_list(r.get("genres")):
            genre_scores[g] = genre_scores.get(g, 0) + r.get("memberRating", 0)
    top_genres = sorted(genre_scores, key=genre_scores.get, reverse=True)[:5]
    matching_genres = [g for g in rec_genres if g in top_genres]
    if matching_genres:
        reasons.append(f"Genres: {', '.join(matching_genres[:3])}")

    # Check cast overlap
    rec_cast = set(_to_list(rec_row.get("cast")))
    fav_cast = set(
        c for cast_list in highly_rated["cast"] for c in _to_list(cast_list)
    )
    shared_cast = rec_cast & fav_cast
    if shared_cast:
        reasons.append(f"Stars: {', '.join(list(shared_cast)[:2])}")

    # Check keyword similarity
    rec_keywords = set(_to_list(rec_row.get("keywords")))
    all_keywords: dict[str, int] = {}
    for _, r in highly_rated.iterrows():
        for kw in _to_list(r.get("keywords")):
            all_keywords[kw] = all_keywords.get(kw, 0) + 1
    top_kws = sorted(all_keywords, key=all_keywords.get, reverse=True)[:20]
    shared_kws = [kw for kw in rec_keywords if kw in top_kws]
    if shared_kws and len(reasons) < 3:
        reasons.append(f"Themes: {', '.join(shared_kws[:2])}")

    # Fallback: TMDB rating
    if len(reasons) < 3 and rec_row.get("vote_average", 0) >= 7.0:
        reasons.append(f"TMDB: {rec_row['vote_average']:.1f}/10")

    return reasons[:3]
