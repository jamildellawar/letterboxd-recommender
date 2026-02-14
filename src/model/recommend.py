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
QUALITY_BLEND = 0.20      # 20% quality signal in three-way blend (50/30/20)


def _get_cf_scores(cf_model, rated_df: pd.DataFrame, unseen: pd.DataFrame) -> np.ndarray | None:
    """Get normalized CF scores for unseen movies, or None if unavailable."""
    if cf_model is None:
        return None

    guest_ratings = dict(zip(
        rated_df["tmdb_id"].astype(int),
        rated_df["memberRating"].astype(float),
    ))
    candidate_ids = set(int(x) for x in unseen["tmdb_id"].values)

    try:
        cf_raw = cf_model.predict(guest_ratings, candidate_ids)
    except Exception as e:
        print(f"  CF prediction failed: {e}")
        return None

    if not cf_raw:
        return None

    cf_scores = np.zeros(len(unseen))
    for i, tid in enumerate(unseen["tmdb_id"].values):
        cf_scores[i] = cf_raw.get(int(tid), 0.0)

    c_min, c_max = cf_scores.min(), cf_scores.max()
    if c_max > c_min:
        cf_scores = (cf_scores - c_min) / (c_max - c_min)
    else:
        return None

    scored = (cf_scores > 0).sum()
    print(f"  CF scores: {scored}/{len(unseen)} candidates scored")
    return cf_scores


def _bayesian_rating(vote_avg: np.ndarray, vote_count: np.ndarray) -> np.ndarray:
    """IMDB-style weighted rating that penalizes films with few votes."""
    m = np.median(vote_count[vote_count > 0]) if (vote_count > 0).any() else 1000.0
    c = np.mean(vote_avg[vote_avg > 0]) if (vote_avg > 0).any() else 6.5
    weighted = (vote_count / (vote_count + m)) * vote_avg + (m / (vote_count + m)) * c
    return weighted


def _quality_scores(unseen: pd.DataFrame) -> np.ndarray:
    """Compute quality scores, preferring Letterboxd community ratings over TMDB.

    Letterboxd ratings (0-5) are normalized to [0, 1].
    Falls back to TMDB Bayesian rating for movies without Letterboxd data.
    """
    n = len(unseen)
    scores = np.zeros(n)

    has_lb = "letterboxd_rating" in unseen.columns
    lb_ratings = unseen["letterboxd_rating"].values if has_lb else np.full(n, np.nan)

    vote_avg = unseen["vote_average"].fillna(0).values if "vote_average" in unseen.columns else np.zeros(n)
    vote_cnt = unseen["vote_count"].fillna(0).values if "vote_count" in unseen.columns else np.ones(n)

    tmdb_bayesian = _bayesian_rating(vote_avg, vote_cnt)

    for i in range(n):
        lb = lb_ratings[i] if has_lb else np.nan
        if not np.isnan(lb) and lb > 0:
            scores[i] = lb / 5.0
        else:
            scores[i] = tmdb_bayesian[i] / 10.0

    return scores


def generate_recommendations(
    rated_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    vectorizer: MovieVectorizer,
    top_n: int = TOP_N_RECOMMENDATIONS,
    mmr_lambda: float = 0.7,
    feedback_adjustment: np.ndarray | None = None,
    cf_model=None,
) -> pd.DataFrame:
    """Generate movie recommendations.

    Pipeline:
    1. Filter candidates by minimum vote count (removes niche/inflated-rating films)
    2. Score by cosine similarity to taste profile
    3. Optionally blend with CF scores (collaborative filtering)
    4. Blend with TMDB quality signal
    5. MMR re-ranking for diversity
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

    # Drop movies with low Letterboxd community ratings
    if "letterboxd_rating" in unseen.columns:
        before = len(unseen)
        lb_mask = unseen["letterboxd_rating"].isna() | (unseen["letterboxd_rating"] >= 3.0)
        unseen = unseen[lb_mask].copy()
        dropped = before - len(unseen)
        if dropped > 0:
            print(f"  Dropped {dropped} candidates with Letterboxd rating < 3.0")

    print(f"  After quality filter: {len(unseen)} candidates")

    if unseen.empty:
        print("Warning: No candidates passed quality filter.")
        return pd.DataFrame()

    # Vectorize candidates and compute taste similarity
    candidate_vectors = vectorizer.transform(unseen)
    taste_scores = cosine_similarity(candidate_vectors, taste_profile.reshape(1, -1)).flatten()

    # Quality signal: prefer Letterboxd community ratings, fall back to TMDB Bayesian
    quality_norm = _quality_scores(unseen)

    # Normalize taste scores to [0, 1]
    t_min, t_max = taste_scores.min(), taste_scores.max()
    if t_max > t_min:
        taste_norm = (taste_scores - t_min) / (t_max - t_min)
    else:
        taste_norm = np.ones_like(taste_scores)

    # Get CF scores if model provided
    cf_norm = _get_cf_scores(cf_model, rated_df, unseen)

    if cf_norm is not None:
        # Three-way blend: 50% taste + 30% CF + 20% quality
        blended_scores = 0.50 * taste_norm + 0.30 * cf_norm + QUALITY_BLEND * quality_norm
        print("  Using three-way blend: 50% taste + 30% CF + 20% quality")
    else:
        # Original: 75% taste + 25% quality
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
    cf_model=None,
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

    # Drop movies with low Letterboxd community ratings
    if "letterboxd_rating" in unseen.columns:
        lb_mask = unseen["letterboxd_rating"].isna() | (unseen["letterboxd_rating"] >= 3.0)
        unseen = unseen[lb_mask].copy()

    if unseen.empty:
        return pd.DataFrame()

    candidate_vectors = vectorizer.transform(unseen)
    taste_scores = cosine_similarity(candidate_vectors, taste_profile.reshape(1, -1)).flatten()

    quality_norm = _quality_scores(unseen)

    t_min, t_max = taste_scores.min(), taste_scores.max()
    taste_norm = (taste_scores - t_min) / (t_max - t_min) if t_max > t_min else np.ones_like(taste_scores)

    cf_norm = _get_cf_scores(cf_model, rated_df, unseen)
    if cf_norm is not None:
        blended = 0.50 * taste_norm + 0.30 * cf_norm + QUALITY_BLEND * quality_norm
    else:
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
    cf_model=None,
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

    quality_norm = _quality_scores(unseen)

    t_min, t_max = taste_scores.min(), taste_scores.max()
    taste_norm = (taste_scores - t_min) / (t_max - t_min) if t_max > t_min else np.ones_like(taste_scores)

    cf_norm = _get_cf_scores(cf_model, rated_df, unseen)
    if cf_norm is not None:
        blended = 0.50 * taste_norm + 0.30 * cf_norm + QUALITY_BLEND * quality_norm
    else:
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
    cf_model=None,
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

    # Drop movies with low Letterboxd community ratings
    if "letterboxd_rating" in unseen.columns:
        before = len(unseen)
        lb_mask = unseen["letterboxd_rating"].isna() | (unseen["letterboxd_rating"] >= 3.0)
        unseen = unseen[lb_mask].copy()
        dropped = before - len(unseen)
        if dropped > 0:
            print(f"  Dropped {dropped} streaming candidates with Letterboxd rating < 3.0")

    # Filter to movies that are actually streaming
    streaming_ids = set(streaming_map.keys())
    unseen = unseen[unseen["tmdb_id"].isin(streaming_ids)].copy()

    if unseen.empty:
        print("  No streaming candidates found after filtering.")
        return pd.DataFrame()

    candidate_vectors = vectorizer.transform(unseen)
    taste_scores = cosine_similarity(candidate_vectors, taste_profile.reshape(1, -1)).flatten()

    quality_norm = _quality_scores(unseen)

    t_min, t_max = taste_scores.min(), taste_scores.max()
    taste_norm = (taste_scores - t_min) / (t_max - t_min) if t_max > t_min else np.ones_like(taste_scores)

    cf_norm = _get_cf_scores(cf_model, rated_df, unseen)
    if cf_norm is not None:
        blended = 0.50 * taste_norm + 0.30 * cf_norm + QUALITY_BLEND * quality_norm
    else:
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
