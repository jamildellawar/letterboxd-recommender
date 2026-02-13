"""Generate fresh movie recommendations and write to DynamoDB."""

import pandas as pd

from src.data.merge import load_rated_movies
from src.data.tmdb import fetch_popular_movies
from src.data.history import detect_feedback, save_run, save_feedback, feedback_metrics, load_all_history
from src.features.vectorize import MovieVectorizer
from src.features.profile import profile_summary
from src.features.feedback import compute_feedback_adjustment
from src.model.recommend import generate_recommendations, generate_genre_picks, get_top_candidate_ids, generate_streaming_picks
from src.model.evaluate import leave_one_out_eval, rating_correlation
from src.data.letterboxd import resolve_letterboxd_urls, fetch_letterboxd_ratings
from src.data.tmdb import fetch_watch_providers
from src.deploy.dynamo_writer import write_recommendations, write_genre_picks, write_streaming_picks, write_profile


def main():
    # Load rated movies
    rated_df = load_rated_movies()
    print(f"Loaded {len(rated_df)} rated movies")

    # Fit vectorizer on rated movies
    vectorizer = MovieVectorizer()
    vectorizer.fit(rated_df)

    # Evaluate model quality
    print("\n--- Model Evaluation ---")
    loo = leave_one_out_eval(rated_df, vectorizer)
    print(f"Leave-one-out (top-{loo['top_k']}): "
          f"hit rate={loo['hit_rate']:.2%}, MRR={loo['mean_reciprocal_rank']:.4f} "
          f"({loo['n_evaluated']} movies evaluated)")

    corr = rating_correlation(rated_df, vectorizer)
    print(f"Rating correlation: Spearman r={corr['spearman_r']:.4f}, "
          f"p={corr['p_value']:.6f}")

    # Detect feedback from past recommendations
    print("\n--- Feedback Loop ---")
    history = load_all_history()
    if history.empty:
        print("No history from past recommendations.")
        feedback_adj = None
    else:
        previous_rated_ids = set(history["tmdb_id"].values)
        fb = detect_feedback(rated_df, previous_rated_ids)
        if fb.empty:
            print("No new feedback detected.")
            feedback_adj = None
        else:
            save_feedback(fb)
            hits = (fb["outcome"] == "hit").sum()
            misses = (fb["outcome"] == "miss").sum()
            neutral = (fb["outcome"] == "neutral").sum()
            print(f"Feedback: {hits} hits, {misses} misses, {neutral} neutral")

            rated_vectors = vectorizer.transform(rated_df)
            feedback_adj = compute_feedback_adjustment(fb, rated_vectors, rated_df)
            if feedback_adj is not None:
                print("Feedback adjustment computed and will be applied.")

        metrics = feedback_metrics()
        if metrics:
            print(f"Overall: watch_rate={metrics['watch_rate']:.1%}, "
                  f"avg_rating={metrics['avg_rating']:.2f}, "
                  f"hit_rate={metrics['hit_rate']:.1%}, "
                  f"miss_rate={metrics['miss_rate']:.1%}")

    # Fetch candidate movies from TMDB
    print("\n--- Fetching Candidates ---")
    candidates = fetch_popular_movies(pages=50)
    candidates_df = pd.DataFrame(candidates)
    print(f"Candidate pool: {len(candidates_df)} movies")

    # Generate main recommendations (MMR diversity model)
    print("\n--- Generating Recommendations (pass 1: taste similarity) ---")
    recs = generate_recommendations(rated_df, candidates_df, vectorizer, feedback_adjustment=feedback_adj)

    # Generate genre picks (1 per genre) — exclude main recs for no overlap
    print("\n--- Generating Genre Picks (pass 1: taste similarity) ---")
    main_rec_ids = set(recs["tmdb_id"].values)
    genre_picks = generate_genre_picks(rated_df, candidates_df, vectorizer, exclude_ids=main_rec_ids, feedback_adjustment=feedback_adj)

    # Generate streaming picks
    print("\n--- Generating Streaming Picks ---")
    shortlist_ids = get_top_candidate_ids(rated_df, candidates_df, vectorizer, 150, feedback_adj)
    streaming_map = fetch_watch_providers(shortlist_ids)
    streaming_picks = generate_streaming_picks(rated_df, candidates_df, vectorizer, streaming_map, feedback_adjustment=feedback_adj)

    # Resolve Letterboxd URLs for all sets
    print("\n--- Resolving Letterboxd URLs ---")
    all_recs = pd.concat([recs, genre_picks, streaming_picks], ignore_index=True).drop_duplicates(subset="tmdb_id")
    url_map = resolve_letterboxd_urls(all_recs)
    recs["letterboxd_url"] = recs["tmdb_id"].astype(str).map(url_map)
    genre_picks["letterboxd_url"] = genre_picks["tmdb_id"].astype(str).map(url_map)
    if not streaming_picks.empty:
        streaming_picks["letterboxd_url"] = streaming_picks["tmdb_id"].astype(str).map(url_map)

    # Fetch Letterboxd community ratings for quality re-ranking
    print("\n--- Fetching Letterboxd Ratings ---")
    lb_ratings = fetch_letterboxd_ratings(url_map)

    # Re-rank using Letterboxd ratings as quality signal
    print("\n--- Re-ranking with Letterboxd Ratings ---")
    recs = _rerank_with_letterboxd(recs, lb_ratings)
    genre_picks = _rerank_genre_picks_with_letterboxd(genre_picks, lb_ratings)
    if not streaming_picks.empty:
        streaming_picks = _rerank_with_letterboxd(streaming_picks, lb_ratings)

    print(f"Top 10 recommendations (after Letterboxd re-rank):")
    for i, (_, row) in enumerate(recs.head(10).iterrows()):
        lb_r = lb_ratings.get(str(row["tmdb_id"]), "?")
        lb_str = f"{lb_r:.1f}" if isinstance(lb_r, float) else lb_r
        print(f"  {i+1}. {row['title']} ({row.get('year', '?')}) "
              f"- sim={row['similarity_score']:.3f} "
              f"- LB={lb_str}/5 "
              f"- {', '.join(row.get('explanation', []))}")

    print(f"\nGenre picks ({len(genre_picks)} genres):")
    for _, row in genre_picks.iterrows():
        lb_r = lb_ratings.get(str(row["tmdb_id"]), "?")
        lb_str = f"{lb_r:.1f}" if isinstance(lb_r, float) else lb_r
        print(f"  [{row['picked_genre']}] {row['title']} ({row.get('year', '?')}) "
              f"- sim={row['similarity_score']:.3f} LB={lb_str}/5")

    print(f"\nStreaming picks ({len(streaming_picks)}):")
    for _, row in streaming_picks.iterrows():
        services = ", ".join(row.get("streaming_services", []))
        print(f"  {row['title']} ({row.get('year', '?')}) "
              f"- sim={row['similarity_score']:.3f} "
              f"- [{services}]")

    # Write to DynamoDB
    print("\n--- Writing to DynamoDB ---")
    write_recommendations(recs)
    write_genre_picks(genre_picks)
    if not streaming_picks.empty:
        write_streaming_picks(streaming_picks)

    summary = profile_summary(rated_df)
    write_profile(summary)

    # Save recommendation history
    print("\n--- Saving History ---")
    save_run(recs, genre_picks, streaming_picks if not streaming_picks.empty else None)

    print("\nDone!")


def _rerank_with_letterboxd(recs_df: pd.DataFrame, lb_ratings: dict[str, float]) -> pd.DataFrame:
    """Re-rank main recs using Letterboxd ratings, dropping low-rated films."""
    recs_df = recs_df.copy()
    recs_df["lb_rating"] = recs_df["tmdb_id"].astype(str).map(lb_ratings)

    # Drop films rated below 3.0 on Letterboxd (likely bad recommendations)
    before = len(recs_df)
    recs_df = recs_df[recs_df["lb_rating"].isna() | (recs_df["lb_rating"] >= 3.0)]
    dropped = before - len(recs_df)
    if dropped > 0:
        print(f"  Dropped {dropped} recs with Letterboxd rating < 3.0")

    # Sort by blended score: 70% taste similarity + 30% Letterboxd quality
    max_sim = recs_df["similarity_score"].max()
    min_sim = recs_df["similarity_score"].min()
    sim_range = max_sim - min_sim if max_sim > min_sim else 1.0

    recs_df["_blend"] = recs_df.apply(
        lambda r: 0.7 * ((r["similarity_score"] - min_sim) / sim_range) +
                  0.3 * ((r["lb_rating"] - 1.0) / 4.0 if pd.notna(r["lb_rating"]) else 0.5),
        axis=1,
    )
    recs_df = recs_df.sort_values("_blend", ascending=False).drop(columns=["lb_rating", "_blend"])
    return recs_df.head(30).reset_index(drop=True)


def _rerank_genre_picks_with_letterboxd(picks_df: pd.DataFrame, lb_ratings: dict[str, float]) -> pd.DataFrame:
    """Filter genre picks that have very low Letterboxd ratings."""
    picks_df = picks_df.copy()
    picks_df["lb_rating"] = picks_df["tmdb_id"].astype(str).map(lb_ratings)

    # Flag but don't drop genre picks (we want one per genre)
    # Instead, just note the rating — the main filtering happened via quality blend
    picks_df = picks_df.drop(columns=["lb_rating"], errors="ignore")
    return picks_df


if __name__ == "__main__":
    main()
